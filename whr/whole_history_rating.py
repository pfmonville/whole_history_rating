from __future__ import annotations

import ast
import math
import pickle
import time
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np

from whr.game import Game
from whr.player import Player

# _compute_drift allocates arrays sized to the CALENDAR SPAN of day values
# (max_day - min_day), not to the number of games. If `time_step` is an epoch
# timestamp instead of a compact day index, this guard prevents a silent
# hang/OOM.
_MAX_DRIFT_DAY_SPAN = 1_000_000

# Converts a natural-rating (r = ln(gamma)) variance/quantity to elo units:
# elo = 400/ln(10) * r.
_ELO_PER_NAT = 400.0 / math.log(10)


class WHR:
    def __init__(self, config: dict[str, Any] | None = None):
        # Copy the caller's dict so we never mutate it and instances never
        # share the same config object.
        self.config = dict(config) if config is not None else {}
        self.config.setdefault("w2", 300.0)
        self.config.setdefault("uncased", False)
        self.config.setdefault("initial_prior_wins", 0.5)
        self.config.setdefault("hessian_damping", 1.0)
        self.config.setdefault("drift_kernel_radius", 100)
        self.config.setdefault("pinned_handicap", {})
        self.config.setdefault("pinned_komi", {})
        self.config.setdefault("estimate_handicap_zero", False)
        self.config.setdefault("pinned_draw", None)
        self._has_draws = False
        self.nu = 0.0
        self.games: list[Game] = []
        self.players: dict[str, Player] = {}
        self.handicap_gamma: dict[Any, float] = {}
        self.komi_gamma: dict[Any, float] = {}
        self._pinned_handicap_keys: set[Any] = set()
        self._pinned_komi_keys: set[Any] = set()
        for key, elo in self.config["pinned_handicap"].items():
            self.handicap_gamma[key] = 10 ** (elo / 400.0)
            self._pinned_handicap_keys.add(key)
        for key, elo in self.config["pinned_komi"].items():
            self.komi_gamma[key] = 10 ** (elo / 400.0)
            self._pinned_komi_keys.add(key)
        if not self.config["estimate_handicap_zero"] and 0 not in self.handicap_gamma:
            self.handicap_gamma[0] = 1.0
            self._pinned_handicap_keys.add(0)

    def _ensure_advantage_keys(self, handicap: Any, komi: Any) -> None:
        """Ensure the advantage tables have an entry (default gamma 1.0) for a
        game's handicap and komi keys, without overwriting existing/pinned ones."""
        if handicap not in self.handicap_gamma:
            self.handicap_gamma[handicap] = 1.0
        if komi not in self.komi_gamma:
            self.komi_gamma[komi] = 1.0

    def _accumulate_handicap_komi(
        self,
    ) -> tuple[
        dict[Any, float],
        dict[Any, float],
        dict[Any, float],
        dict[Any, float],
        dict[Any, int],
        dict[Any, int],
        dict[Any, int],
        dict[Any, int],
    ]:
        """Accumulates the per-key Newton gradient/Hessian terms (and raw
        game/win counts) for the handicap and komi advantage gammas, from the
        current player gammas and advantage tables (Coulom's
        NewtonKomiHandicap accumulation step).

        Returns:
            (h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins)
        """
        h_grad: dict[Any, float] = {}
        h_hess: dict[Any, float] = {}
        k_grad: dict[Any, float] = {}
        k_hess: dict[Any, float] = {}
        h_games: dict[Any, int] = {}
        h_wins: dict[Any, int] = {}
        k_games: dict[Any, int] = {}
        k_wins: dict[Any, int] = {}
        for g in self.games:
            if g.bpd is None or g.wpd is None:
                continue
            # A draw credits neither a handicap (black) win nor a komi
            # (white) win, and the plain Bradley-Terry gradient/Hessian
            # denominator accumulated below does not apply to it. Skipping
            # draws here means handicap/komi advantages are estimated from
            # DECISIVE games only when draws are present -- a deliberate
            # simplification; full Davidson-aware handicap/komi estimation
            # is out of scope for this phase.
            if g.winner == "D":
                continue
            h = g.handicap
            k = g.extras["komi"]
            gh = self.handicap_gamma[h]
            gk = self.komi_gamma[k]
            gb = g.bpd.gamma()
            gw = g.wpd.gamma()
            c_komi = gw
            d_komi = gb * gh
            c_handicap = gb
            d_handicap = gw * gk
            div = 1.0 / (d_komi + d_handicap)
            h_grad[h] = h_grad.get(h, 0.0) + c_handicap * div
            h_hess[h] = h_hess.get(h, 0.0) + c_handicap * d_handicap * div * div
            k_grad[k] = k_grad.get(k, 0.0) + c_komi * div
            k_hess[k] = k_hess.get(k, 0.0) + c_komi * d_komi * div * div
            h_games[h] = h_games.get(h, 0) + 1
            k_games[k] = k_games.get(k, 0) + 1
            if g.winner == "B":
                h_wins[h] = h_wins.get(h, 0) + 1
            else:
                k_wins[k] = k_wins.get(k, 0) + 1
        return h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins

    def _eligible_advantage_updates(
        self,
        table: dict[Any, float],
        pinned: set[Any],
        grad_terms: dict[Any, float],
        games: dict[Any, int],
        wins: dict[Any, int],
    ) -> Iterator[tuple[Any, float, float]]:
        """Yield ``(key, gamma, grad)`` for every non-pinned advantage key that
        a Newton step would actually update: those seen in a game with a
        non-degenerate ``0 < wins < games`` outcome. ``grad`` is the Newton
        gradient ``wins - gamma * grad_term`` w.r.t. ``log(gamma_key)``.

        Both the Newton updater (``_newton_handicap_komi``) and the convergence
        gauge (``_handicap_komi_gradient_norm``) iterate through here, so they
        can never disagree on which keys are eligible — a disagreement was the
        phase-3 convergence bug.
        """
        for key in list(table):
            if key in pinned:
                continue
            n_games = games.get(key, 0)
            n_wins = wins.get(key, 0)
            if n_games > 0 and 0 < n_wins < n_games:
                gamma = table[key]
                grad = n_wins - gamma * grad_terms.get(key, 0.0)
                yield key, gamma, grad

    def _newton_handicap_komi(self) -> None:
        """One Newton step on each non-pinned handicap/komi advantage gamma
        (Coulom's NewtonKomiHandicap)."""
        h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins = (
            self._accumulate_handicap_komi()
        )
        damping = self.config["hessian_damping"]
        for key, gamma, grad in self._eligible_advantage_updates(
            self.handicap_gamma, self._pinned_handicap_keys, h_grad, h_games, h_wins
        ):
            hess = -gamma * h_hess.get(key, 0.0) - damping
            self.handicap_gamma[key] = gamma * math.exp(-grad / hess)
        for key, gamma, grad in self._eligible_advantage_updates(
            self.komi_gamma, self._pinned_komi_keys, k_grad, k_games, k_wins
        ):
            hess = -gamma * k_hess.get(key, 0.0) - damping
            self.komi_gamma[key] = gamma * math.exp(-grad / hess)

    def _handicap_komi_gradient_norm(self) -> float:
        """Max absolute Newton gradient ``|wins - gamma * grad|`` over the
        non-pinned handicap/komi keys that would actually be updated by
        ``_newton_handicap_komi`` (those with games and ``0 < wins < games``).

        This gradient is w.r.t. ``log(gamma_key)`` — the same units as the
        player gradients (w.r.t. ``r = log(gamma_player)``) — so it is
        directly comparable to them in a single infinity-norm. Returns 0.0
        if there is no such key.
        """
        h_grad, _h_hess, k_grad, _k_hess, h_games, h_wins, k_games, k_wins = (
            self._accumulate_handicap_komi()
        )
        norm = 0.0
        for _key, _gamma, grad in self._eligible_advantage_updates(
            self.handicap_gamma, self._pinned_handicap_keys, h_grad, h_games, h_wins
        ):
            norm = max(norm, abs(grad))
        for _key, _gamma, grad in self._eligible_advantage_updates(
            self.komi_gamma, self._pinned_komi_keys, k_grad, k_games, k_wins
        ):
            norm = max(norm, abs(grad))
        return norm

    def _game_descriptions(
        self,
    ) -> list[tuple[str, str, str, int, float, dict[str, Any]]]:
        """Replayable (black, white, winner, day, handicap, extras) tuples.

        ``extras`` is copied so replaying into a fresh WHR cannot mutate this
        instance's game state.
        """
        return [
            (
                g.black_player.name,
                g.white_player.name,
                g.winner,
                g.day,
                g.handicap,
                dict(g.extras),
            )
            for g in self.games
        ]

    def _temporal_folds(self, n_splits: int) -> list[tuple[list[tuple], list[tuple]]]:
        """Expanding-window temporal (train, test) folds, split on distinct days.

        Distinct days are cut into ``n_splits + 1`` contiguous groups; fold ``i``
        trains on the first ``i`` groups and tests on group ``i`` (1-indexed), so
        every train day is strictly earlier than every test day.
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        descs = self._game_descriptions()
        days = sorted({d[3] for d in descs})
        if len(days) < n_splits + 1:
            raise ValueError(
                f"need at least {n_splits + 1} distinct days for "
                f"n_splits={n_splits}, got {len(days)}"
            )
        n_days = len(days)
        cuts = [round(n_days * i / (n_splits + 1)) for i in range(n_splits + 2)]
        day_index = {day: idx for idx, day in enumerate(days)}
        folds = []
        for i in range(1, n_splits + 1):
            train_end = cuts[i]
            test_end = cuts[i + 1]
            train = [d for d in descs if day_index[d[3]] < train_end]
            test = [d for d in descs if train_end <= day_index[d[3]] < test_end]
            folds.append((train, test))
        return folds

    def _predict_black_win_probability(
        self, black_name: str, white_name: str, handicap: float, komi: Any
    ) -> float | None:
        """P(black wins) from the current ratings, or None if either player is
        unknown / unrated (cold start)."""
        black = self._existing_player(black_name)
        white = self._existing_player(white_name)
        if black is None or white is None or not black.days or not white.days:
            return None
        gb = black.days[-1].gamma() * self.handicap_gamma.get(handicap, 1.0)
        gw = white.days[-1].gamma() * self.komi_gamma.get(komi, 1.0)
        return gb / (gb + gw)

    def fit_w2(
        self,
        candidates: list[float] | None = None,
        n_splits: int = 5,
        iterations: int = 50,
    ) -> dict[str, Any]:
        """Pick w2 by temporal expanding-window CV predictive log-loss.

        Trains a fresh model (this instance's config, each candidate w2) on each
        fold's earlier games and scores pooled predictive log-loss on the fold's
        held-out later games. Does NOT mutate this instance. See the design spec
        for details. Raises ValueError if a temporal split is impossible, or if
        ``candidates`` is an explicitly empty list. Warns (``UserWarning``) if
        every test game is cold-start (none can be scored), in which case
        ``log_loss`` is all-``inf`` and ``best_w2`` is not meaningful.
        """
        if candidates is None:
            candidates = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
        if not candidates:
            raise ValueError("candidates must be a non-empty list of w2 values")
        folds = self._temporal_folds(n_splits)
        eps = 1e-15

        # Whether a test game is cold-start (skipped) depends only on whether
        # its players appear in that fold's training games -- it does not
        # depend on w2. Compute the scored/skipped totals once, up front, so
        # every candidate shares identical counts (previously these were
        # reassigned per-candidate and only the LAST candidate's counts were
        # kept, which happened to be correct but was undocumented and fragile
        # if a future change made skip depend on w2).
        n_scored = 0
        n_skipped = 0
        for train, test in folds:
            trained_names = {d[0] for d in train} | {d[1] for d in train}
            for black, white, winner, *_rest in test:
                if winner == "D":
                    # A win/loss predictive log-loss has no correct value for
                    # a draw (see the scoring loop below) -- never counted as
                    # scored, regardless of cold-start status.
                    n_skipped += 1
                elif black in trained_names and white in trained_names:
                    n_scored += 1
                else:
                    n_skipped += 1

        log_loss: dict[float, float] = {}
        for w2 in candidates:
            sub_config = {**self.config, "w2": w2}
            total = 0.0
            for train, test in folds:
                model = WHR(sub_config)
                for black, white, winner, day, handicap, extras in train:
                    model.create_game(black, white, winner, day, handicap, extras)
                model.iterate(iterations)
                for black, white, winner, _day, handicap, extras in test:
                    if winner == "D":
                        # A win/loss predictive log-loss has no correct value
                        # for a draw -- skip it (already counted in
                        # n_test_skipped above), do not score it as a white
                        # win.
                        continue
                    komi = extras.get("komi", 6.5)
                    p_black = model._predict_black_win_probability(
                        black, white, handicap, komi
                    )
                    if p_black is None:
                        continue
                    p_actual = p_black if winner == "B" else 1.0 - p_black
                    p_actual = min(max(p_actual, eps), 1.0 - eps)
                    total += -math.log(p_actual)
            log_loss[w2] = total / n_scored if n_scored else float("inf")
        best_w2 = min(candidates, key=lambda w: log_loss[w])
        if n_scored == 0:
            warnings.warn(
                "fit_w2: no test games could be scored across any fold/candidate "
                "(all cold-start) -- log_loss is all inf and best_w2 is not "
                "meaningful. Check n_test_scored/n_test_skipped; try fewer "
                "n_splits or provide more historical data.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "best_w2": best_w2,
            "log_loss": log_loss,
            "n_splits": n_splits,
            "n_test_scored": n_scored,
            "n_test_skipped": n_skipped,
        }

    def print_ordered_ratings(self, current: bool = False) -> None:
        """Displays all ratings for each player (for each of their playing days), ordered.

        Args:
            current (bool, optional): If True, displays only the latest elo rating. If False, displays all elo ratings for each day played.
        """
        players = [x for x in self.players.values() if len(x.days) > 0]
        players.sort(key=lambda x: x.days[-1].gamma())
        for p in players:
            if len(p.days) > 0:
                if current:
                    print(f"{p.name} => {p.days[-1].elo}")
                else:
                    print(f"{p.name} => {[x.elo for x in p.days]}")

    def get_ordered_ratings(
        self, current: bool = False, compact: bool = False
    ) -> (
        list[float]
        | list[tuple[str, float]]
        | list[list[float]]
        | list[tuple[str, list[float]]]
    ):
        """Retrieves all ratings for each player (for each of their playing days), ordered.

        Args:
            current (bool, optional): If True, retrieves only the latest elo rating estimation. If False, retrieves all elo rating estimations for each day played.
            compact (bool, optional): If True, returns only a list of elo ratings. If False, includes the player's name before their elo ratings.

        Returns:
            The elo ratings for each player, ordered. The exact shape depends on ``current`` and ``compact``: a plain list of elos, a list of ``(name, elo)`` tuples, a list of per-day elo lists, or a list of ``(name, per-day elos)`` tuples.
        """
        result: list[Any] = []
        players = [x for x in self.players.values() if len(x.days) > 0]
        players.sort(key=lambda x: x.days[-1].gamma())
        for p in players:
            if len(p.days) > 0:
                if current and compact:
                    result.append(p.days[-1].elo)
                elif current:
                    result.append((p.name, p.days[-1].elo))
                elif compact:
                    result.append([x.elo for x in p.days])
                else:
                    result.append((p.name, [x.elo for x in p.days]))
        return result

    def log_likelihood(self) -> float:
        """Calculates the likelihood of the current state.

        The likelihood increases with more iterations.

        Returns:
            float: The likelihood.
        """
        score = 0.0
        for p in self.players.values():
            if len(p.days) > 0:
                score += p.log_likelihood()
        return score

    def player_by_name(self, name: str) -> Player:
        """Retrieves the player object corresponding to the given name.

        Args:
            name (str): The name of the player.

        Returns:
            Player: The corresponding player object.
        """
        if self.config["uncased"]:
            name = name.lower()
        if self.players.get(name, None) is None:
            self.players[name] = Player(name, self.config)
        return self.players[name]

    def _existing_player(self, name: str) -> Player | None:
        """Returns the player with the given name, or None, without creating it."""
        if self.config["uncased"]:
            name = name.lower()
        return self.players.get(name)

    @staticmethod
    def _player_day(player: Player, day: int | None) -> Any:
        """The player's given rated day, or its last rated day if ``day`` is
        None. Raises ValueError if ``day`` is given but not a rated day."""
        if day is None:
            return player.days[-1]
        for d in player.days:
            if d.day == day:
                return d
        raise ValueError(f"player {player.name!r} has no rated day {day}")

    def rating_difference(
        self,
        name_a: str,
        name_b: str,
        day_a: int | None = None,
        day_b: int | None = None,
    ) -> dict[str, Any]:
        """Elo difference (a - b) between two players and its uncertainty.

        Uses each player's given day (else their last day). The difference
        variance uses the INDEPENDENCE APPROXIMATION Var(a-b) ~= Var(a)+Var(b);
        WHR computes no cross-player covariance, so this ignores the correlated
        errors of players who have faced each other. Returns
        {"difference", "std_error", "confidence_interval_95"} in elo. Raises
        ValueError for an unknown/unrated player or day, or if name_a and
        name_b refer to the same player (use rating_change() instead, since
        the independence approximation above is invalid for one player
        compared against themselves across days).
        """
        cmp_a, cmp_b = name_a, name_b
        if self.config["uncased"]:
            cmp_a, cmp_b = cmp_a.lower(), cmp_b.lower()
        if cmp_a == cmp_b:
            raise ValueError("use rating_change() to compare one player across days")
        pa = self._existing_player(name_a)
        pb = self._existing_player(name_b)
        if pa is None or not pa.days:
            raise ValueError(f"No ratings available for player {name_a!r}")
        if pb is None or not pb.days:
            raise ValueError(f"No ratings available for player {name_b!r}")
        da = self._player_day(pa, day_a)
        db = self._player_day(pb, day_b)
        if da.uncertainty < 0 or db.uncertainty < 0:
            raise ValueError("uncertainties not computed; call iterate() first")
        difference = da.elo - db.elo
        se = math.sqrt(da.uncertainty + db.uncertainty) * _ELO_PER_NAT
        return {
            "difference": difference,
            "std_error": se,
            "confidence_interval_95": (difference - 1.96 * se, difference + 1.96 * se),
        }

    def rating_covariance(self, name: str) -> tuple[list[int], np.ndarray]:
        """Full within-player covariance of a player's day ratings, in elo^2.

        Returns (days, matrix) where matrix[i][j] = Cov(elo on days[i], elo on
        days[j]) — the exact inverse of the player's negative tridiagonal
        Hessian scaled to elo^2. The diagonal equals the per-day marginal
        variance. Raises ValueError for an unknown/unrated player.

        Evaluated at the player's CURRENT rating state: call iterate() or
        auto_iterate() first for meaningful values. Unlike rating_difference(),
        this does not raise if uncertainties were never computed (i.e. before
        any iterate() call) — it will simply invert whatever Hessian the
        current (possibly un-iterated) state produces. Inverts a dense n x n
        matrix, where n is the player's number of distinct rated days, so the
        cost grows with that count.
        """
        player = self._existing_player(name)
        if player is None or not player.days:
            raise ValueError(f"No ratings available for player {name!r}")
        n = len(player.days)
        sigma2 = player.compute_sigma2()
        diagonal, sub_diagonal = Player.hessian(
            player.days, sigma2, player.hessian_damping
        )
        neg_h = np.zeros((n, n))
        for i in range(n):
            neg_h[i, i] = -diagonal[i]
        for i in range(n - 1):
            neg_h[i, i + 1] = -sub_diagonal[i]
            neg_h[i + 1, i] = -sub_diagonal[i]
        cov = np.linalg.inv(neg_h) * (_ELO_PER_NAT**2)
        days = [d.day for d in player.days]
        return days, cov

    def rating_change(self, name: str, day_from: int, day_to: int) -> dict[str, Any]:
        """Elo change of one player between two of their days, with uncertainty.

        Var(change) = C[to,to] + C[from,from] - 2*C[from,to] from the WITHIN-
        player covariance (exact; consecutive days are positively correlated via
        the Wiener prior, so a change is more certain than summing marginals).
        Returns {"change", "std_error", "confidence_interval_95"} in elo. Raises
        ValueError for an unknown player or an unknown day.

        Evaluated at the player's CURRENT rating state: call iterate() or
        auto_iterate() first for meaningful values. Unlike rating_difference(),
        this does not raise if uncertainties were never computed. It delegates
        to rating_covariance(), which inverts a dense n x n matrix (n = the
        player's number of distinct rated days), so cost grows with that count.
        """
        player = self._existing_player(name)
        if player is None or not player.days:
            raise ValueError(f"No ratings available for player {name!r}")
        days, cov = self.rating_covariance(name)
        index = {d: i for i, d in enumerate(days)}
        if day_from not in index or day_to not in index:
            raise ValueError(f"player {name!r} has no rated day {day_from} / {day_to}")
        i, j = index[day_from], index[day_to]
        change = player.days[j].elo - player.days[i].elo
        var = cov[j, j] + cov[i, i] - 2 * cov[i, j]
        se = math.sqrt(max(var, 0.0))
        return {
            "change": change,
            "std_error": se,
            "confidence_interval_95": (change - 1.96 * se, change + 1.96 * se),
        }

    def ratings_for_player(
        self, name, current: bool = False
    ) -> list[tuple[int, float, float]] | tuple[float, float]:
        """Retrieves all ratings for each day played by the specified player.

        Args:
            name (str): The name of the player.
            current (bool, optional): If True, retrieves only the latest elo rating and uncertainty. If False, retrieves all elo ratings and uncertainties for each day played.

        Returns:
            list[tuple[int, float, float]] | tuple[float, float]: For each day, includes the time step, the elo rating, and the uncertainty if current is False, else just return the elo and uncertainty of the last day

        Raises:
            ValueError: If the player is unknown or has no rated day.
        """
        player = self._existing_player(name)
        if player is None or len(player.days) == 0:
            raise ValueError(f"No ratings available for unknown player {name!r}")
        if current:
            return (
                round(player.days[-1].elo),
                round(player.days[-1].uncertainty, 2),
            )
        return [(d.day, round(d.elo), round(d.uncertainty, 2)) for d in player.days]

    def _setup_game(
        self,
        black: str,
        white: str,
        winner: str,
        time_step: int,
        handicap: float,
        extras: dict[str, Any] | None = None,
    ) -> Game:
        if extras is None:
            extras = {}
        if black == white:
            raise AttributeError("Invalid game (black player == white player)")
        if winner.upper() not in ("B", "W", "D"):
            raise ValueError(
                f"Invalid winner {winner!r}: must be 'B', 'W', or 'D' "
                "(case-insensitive)"
            )
        white_player = self.player_by_name(white)
        black_player = self.player_by_name(black)
        game = Game(
            black_player,
            white_player,
            winner,
            time_step,
            handicap,
            extras,
            handicap_gamma=self.handicap_gamma,
            komi_gamma=self.komi_gamma,
        )
        return game

    def create_game(
        self,
        black: str,
        white: str,
        winner: str,
        time_step: int,
        handicap: float,
        extras: dict[str, Any] | None = None,
    ) -> Game:
        """Creates a new game to be added to the base.

        Args:
            black (str): The name of the black player.
            white (str): The name of the white player.
            winner (str): "B" if black won, "W" if white won.
            time_step (int): The day of the match from the origin.
            handicap (float): The handicap category key (e.g. a stone count).
                Its advantage is estimated from the data (or pinned to a
                known elo value via the ``pinned_handicap`` config), not a
                fixed elo amount itself — see "Handicap and komi" in the
                README.
            extras (dict[str, Any] | None, optional): Extra parameters.

        Returns:
            Game: The newly added game.
        """
        if extras is None:
            extras = {}
        if self.config["uncased"]:
            black = black.lower()
            white = white.lower()
        game = self._setup_game(black, white, winner, time_step, handicap, extras)
        self._ensure_advantage_keys(game.handicap, game.extras["komi"])
        return self._add_game(game)

    def _add_game(self, game: Game) -> Game:
        game.white_player.add_game(game)
        game.black_player.add_game(game)
        if game.bpd is None:
            raise RuntimeError(
                "Game could not be attached to the black player's playing day"
            )
        if game.winner == "D":
            self._has_draws = True
            pinned_draw = self.config["pinned_draw"]
            if pinned_draw is not None:
                self.nu = pinned_draw
            elif self.nu == 0.0:
                self.nu = 1.0
        self.games.append(game)
        return game

    @property
    def draw_tendency(self) -> float:
        return self.nu

    def iterate(self, count: int) -> None:
        """Performs a specified number of iterations of the algorithm.

        Args:
            count (int): The number of iterations to perform.
        """
        for _ in range(count):
            self._run_one_iteration()
        for player in self.players.values():
            player.update_uncertainty()

    def max_gradient_norm(self) -> float:
        """Largest gradient infinity-norm across all players, non-pinned
        handicap/komi advantage keys, and the draw tendency nu (stationarity
        gauge). The handicap/komi Newton gradient is w.r.t.
        ``log(gamma_key)``, and the nu gradient is w.r.t. ``log(nu)``, both
        the same units as the player gradients w.r.t. ``r = log(gamma_player)``,
        so all are directly comparable in this single infinity-norm.

        Without folding in the nu gradient, ``auto_iterate`` could declare
        convergence while nu (Davidson's draw tendency) was still moving."""
        norm = 0.0
        for p in self.players.values():
            if len(p.days) > 0:
                norm = max(norm, p.gradient_infinity_norm())
        norm = max(norm, self._handicap_komi_gradient_norm())
        if self._has_draws and self.config["pinned_draw"] is None:
            nu_gradient, _nu_hessian = self._nu_gradient_hessian()
            norm = max(norm, abs(nu_gradient))
        return norm

    def auto_iterate(
        self,
        time_limit: int | None = None,
        precision: float = 1e-3,
        batch_size: int = 10,
    ) -> tuple[int, bool]:
        """Iterate until the gradient infinity-norm drops below ``precision``.

        Args:
            time_limit: max seconds before giving up. None means no timeout.
            precision: convergence tolerance on the max absolute gradient
                component (natural-rating units).
            batch_size: iterations per convergence/timeout check.

        Returns:
            (iterations performed, whether convergence was reached).
        """
        start = time.time()
        i = 0
        while True:
            self.iterate(batch_size)
            i += batch_size
            if self.max_gradient_norm() < precision:
                return i, True
            if time_limit is not None and time.time() - start > time_limit:
                return i, False

    def _compute_drift(self) -> dict[int, float]:
        """Smoothed per-day drift in elo (Coulom's ComputeDrift).

        For every game, accumulate ``elo_black + elo_white`` and a game count
        on its day; convolve both with a Gaussian kernel (radius
        ``drift_kernel_radius``, sigma = radius * 0.5, centre half-weighted);
        the per-day drift is ``filtered_elo / (2 * filtered_count)``. Days with
        no smoothing support (or a non-finite result) get a 0 drift.
        """
        if not self.games:
            return {}
        radius = self.config["drift_kernel_radius"]
        if not isinstance(radius, int) or radius < 1:
            raise ValueError(f"drift_kernel_radius must be an int >= 1, got {radius!r}")
        days = [g.day for g in self.games]
        min_day, max_day = min(days), max(days)
        span = max_day - min_day
        if span > _MAX_DRIFT_DAY_SPAN:
            raise ValueError(
                f"remove_drift() cost scales with the day span "
                f"(max_day - min_day = {span}); this exceeds "
                f"{_MAX_DRIFT_DAY_SPAN}. `time_step` must be a compact day index "
                f"from an origin (e.g. day number), not an epoch timestamp."
            )
        n = max_day - min_day + 1

        total_elo = [0.0] * (n + 2 * radius)
        game_count = [0.0] * (n + 2 * radius)
        for g in self.games:
            if g.bpd is None or g.wpd is None:
                continue
            j = g.day - min_day + radius
            total_elo[j] += g.bpd.elo + g.wpd.elo
            game_count[j] += 1.0

        sigma = radius * 0.5
        kernel = [0.0] * radius
        total = 1.0
        for k in range(1, radius):
            x = math.exp(-(k * k) / (2.0 * sigma * sigma))
            kernel[k] = x
            total += 2.0 * x
        norm = 1.0 / total
        kernel[0] = norm * 0.5
        for k in range(1, radius):
            kernel[k] *= norm

        drift: dict[int, float] = {}
        for i in range(n):
            j = i + radius
            filtered_elo = 0.0
            filtered_count = 0.0
            for k in range(radius):
                filtered_elo += (total_elo[j + k] + total_elo[j - k]) * kernel[k]
                filtered_count += (game_count[j + k] + game_count[j - k]) * kernel[k]
            if filtered_count > 0:
                d = filtered_elo / (2.0 * filtered_count)
                drift[min_day + i] = d if math.isfinite(d) else 0.0
            else:
                drift[min_day + i] = 0.0
        return drift

    def remove_drift(self) -> dict[int, float]:
        """Cancel global rating drift over time (Coulom's RemoveDrift).

        Call after iterate()/auto_iterate() — and call it last, since a
        subsequent iterate()/auto_iterate() call would revert the correction.
        Shifts every player-day's rating by the negated smoothed per-day drift
        so the average player strength per day is recentred near 0 elo, making
        ratings comparable across eras. Mutates the stored ratings in place and
        returns the applied per-day corrections ({day: correction_elo}). Because
        the shift is uniform per day, within-day rating differences (hence
        same-day win probabilities) are unchanged. Uncertainties are not
        recomputed; this is only approximate, since the first-day anchor
        curvature is not exactly invariant under the shift, but the effect is
        output-only and has no downstream effect on iteration.

        `time_step` (the day index used when creating games) must be a compact
        day index counted from some origin (e.g. a day number), not an epoch
        timestamp: this method's cost scales with the CALENDAR SPAN of day
        values (max day - min day), not with the number of games.

        Raises:
            ValueError: If `drift_kernel_radius` is not an int >= 1, or if the
                day span (max day - min day) is implausibly large (see above).
        """
        drift = self._compute_drift()
        factor = math.log(10) / 400.0
        applied: dict[int, float] = {}
        for player in self.players.values():
            for pd in player.days:
                correction_elo = -drift.get(pd.day, 0.0)
                pd.r += correction_elo * factor
                applied[pd.day] = correction_elo
        return applied

    def _match_player_days(
        self, name1: str, name2: str
    ) -> tuple[Player | None, Player | None, float, float, float, float]:
        """Resolves two players by name for a hypothetical match, without
        creating persistent entries (a pure query).

        Returns ``(player1, player2, bpd_gamma, bpd_elo, wpd_gamma, wpd_elo)``:
        the two ``Player`` objects (``None`` if unknown) and, from each
        player's last day, the gamma and elo used as black (name1) / white
        (name2) respectively. Unknown or unrated players default to
        gamma=1.0, elo=0.0 (an even reference).

        Raises:
            AttributeError: If name1 and name2 are equal.
        """
        if self.config["uncased"]:
            name1 = name1.lower()
            name2 = name2.lower()
        if name1 == name2:
            raise AttributeError("Invalid game (black == white)")
        player1 = self._existing_player(name1)
        player2 = self._existing_player(name2)
        bpd_gamma = 1.0
        bpd_elo = 0.0
        wpd_gamma = 1.0
        wpd_elo = 0.0
        if player1 is not None and len(player1.days) > 0:
            bpd = player1.days[-1]
            bpd_gamma = bpd.gamma()
            bpd_elo = bpd.elo
        if player2 is not None and len(player2.days) > 0:
            wpd = player2.days[-1]
            wpd_gamma = wpd.gamma()
            wpd_elo = wpd.elo
        return player1, player2, bpd_gamma, bpd_elo, wpd_gamma, wpd_elo

    def _resolve_advantage_gammas(
        self, handicap_key: Any, komi_key: Any
    ) -> tuple[float, float]:
        """Resolves the (handicap, komi) category-key advantage gammas.

        Unseen or omitted (``None``) keys default to gamma 1.0 (no
        advantage).

        Raises:
            AttributeError: If a supplied key resolves to a non-finite or
                non-positive gamma.
        """
        gh = 1.0 if handicap_key is None else self.handicap_gamma.get(handicap_key, 1.0)
        gk = 1.0 if komi_key is None else self.komi_gamma.get(komi_key, 1.0)
        if not math.isfinite(gh) or gh <= 0 or not math.isfinite(gk) or gk <= 0:
            raise AttributeError("bad advantage gamma")
        return gh, gk

    def probability_future_match(
        self,
        name1: str,
        name2: str,
        handicap: float = 0,
        handicap_key: Any = None,
        komi_key: Any = None,
        account_for_uncertainty: bool = False,
        uncertainty_steps: int = 4,
    ) -> tuple[float, float]:
        """Calculates the winning probability for a hypothetical match between two players.

        name1 plays the black role and name2 the white role, matching
        ``create_game(black, white, ...)``.

        Two independent, stackable advantage inputs are supported:

        * ``handicap`` — a raw elo adjustment (NOT a category key) that shifts
          the effective elo gap in name1's favour by ``handicap`` points.
          Equivalently (and as the formula does it), name2's elo is lowered by
          ``handicap`` when scoring name1 and name1's elo is raised by
          ``handicap`` when scoring name2; both express the same gap shift.
        * ``handicap_key`` / ``komi_key`` — *category keys* (as passed to
          ``create_game``/``load_games``) whose learned/pinned advantage gammas
          (``handicap_gamma`` boosting black = name1, ``komi_gamma`` boosting
          white = name2) are folded in exactly as in a real game. Unseen keys
          default to gamma 1.0 (no advantage). Leaving them ``None`` applies no
          learned advantage.

        When keys are supplied the raw ``handicap`` elo shift stacks on top of
        the learned advantages. When both keys are ``None`` this is a pure
        raw-elo what-if, independent of the estimated advantages learned from
        the game history (backward-compatible with earlier releases).

        Args:
            name1 (str): The name of the first player.
            name2 (str): The name of the second player.
            handicap (float, optional): A raw elo adjustment favouring name1 by
                shifting the effective elo gap ``handicap`` points in name1's
                favour, not a handicap category key.
            handicap_key (Any, optional): A handicap *category key* whose
                learned/pinned ``handicap_gamma`` advantage (favouring name1)
                is folded in. ``None`` (default) applies no handicap advantage.
            komi_key (Any, optional): A komi *category key* whose learned/pinned
                ``komi_gamma`` advantage (favouring name2) is folded in.
                ``None`` (default) applies no komi advantage.
            account_for_uncertainty (bool, optional): When ``False`` (default,
                backward-compatible), returns the point probability computed
                from the players' current ratings only. When ``True``,
                integrates the point probability's logit over the Gaussian
                implied by the players' rating variances (Coulom's
                ``Predict``), hedging the result toward 0.5 when ratings are
                uncertain.
            uncertainty_steps (int, optional): Number of quadrature steps on
                each side of the integration grid used for the Gaussian
                quadrature when ``account_for_uncertainty`` is ``True``; the
                grid spans +/-0.5 * uncertainty_steps standard deviations
                (nodes at x_i = 0.5*i). Ignored otherwise. Defaults to 4. Must
                be >= 1 when ``account_for_uncertainty`` is ``True``.

        Returns:
            tuple[float, float]: The winning probabilities for name1 and name2 respectively. Unknown players are treated as an even (gamma = 1) reference without being added to the base.

        Raises:
            AttributeError: Raised if name1 and name2 are equal, or if a
                supplied category key resolves to a non-finite/non-positive
                advantage gamma.
            ValueError: Raised if ``account_for_uncertainty`` is ``True`` and
                ``uncertainty_steps`` is less than 1.
        """
        player1, player2, bpd_gamma, bpd_elo, wpd_gamma, wpd_elo = (
            self._match_player_days(name1, name2)
        )
        if handicap_key is None and komi_key is None:
            # Backward-compatible raw-elo path (byte-identical to prior
            # releases): no learned advantages are consulted.
            player1_proba = bpd_gamma / (
                bpd_gamma + 10 ** ((wpd_elo - handicap) / 400.0)
            )
            player2_proba = wpd_gamma / (
                wpd_gamma + 10 ** ((bpd_elo + handicap) / 400.0)
            )
        else:
            # Learned-advantage path: fold the category-key advantage gammas
            # into the opponent's gamma exactly as
            # ``Game.opponents_adjusted_gamma`` does (handicap boosts black =
            # name1, komi boosts white = name2), and stack the raw
            # ``handicap`` elo shift on top as a further name1 boost.
            gh, gk = self._resolve_advantage_gammas(handicap_key, komi_key)
            h_shift = 10 ** (handicap / 400.0)
            opponent_of_name1 = wpd_gamma * gk / gh / h_shift
            opponent_of_name2 = bpd_gamma * gh / gk * h_shift
            player1_proba = bpd_gamma / (bpd_gamma + opponent_of_name1)
            player2_proba = wpd_gamma / (wpd_gamma + opponent_of_name2)

        if not account_for_uncertainty:
            return player1_proba, player2_proba

        if uncertainty_steps < 1:
            raise ValueError("uncertainty_steps must be >= 1")

        var1 = player1.days[-1].uncertainty if (player1 and player1.days) else 0.0
        var2 = player2.days[-1].uncertainty if (player2 and player2.days) else 0.0
        var1 = max(var1, 0.0)  # uncertainty is -1 before iterate()
        var2 = max(var2, 0.0)
        sigma = math.sqrt(var1 + var2)
        if sigma == 0.0:
            return player1_proba, player2_proba

        eps = 1e-12
        p1c = min(max(player1_proba, eps), 1.0 - eps)
        delta_r = math.log(p1c / (1.0 - p1c))  # logit of the point probability
        total_weight = 0.0
        integral = 0.0
        for i in range(-uncertainty_steps, uncertainty_steps + 1):
            x = 0.5 * i
            weight = math.exp(-x * x / 2.0)
            integral += weight / (1.0 + math.exp(-(delta_r + sigma * x)))
            total_weight += weight
        p1 = integral / total_weight
        return p1, 1.0 - p1

    def win_draw_loss_probabilities(
        self,
        name1: str,
        name2: str,
        handicap: float = 0,
        handicap_key: Any = None,
        komi_key: Any = None,
    ) -> tuple[float, float, float]:
        """(P(name1 wins), P(draw), P(name2 wins)) for a hypothetical match,
        under the fitted Davidson model.

        name1 plays the black role and name2 the white role, matching
        ``create_game(black, white, ...)``. The player lookup and the
        handicap/komi advantage handling are identical to
        ``probability_future_match`` (see its docstring for the full
        semantics of ``handicap``, ``handicap_key``, and ``komi_key``).

        The two players' effective gammas ``s1`` (name1) and ``s2`` (name2)
        at their last day are combined with the fitted draw tendency
        ``self.nu`` (Davidson's model): ``t = nu * sqrt(s1 * s2)``,
        ``z = s1 + s2 + t``, giving ``(s1/z, t/z, s2/z)``. When ``nu == 0``
        (no draws observed/fitted), ``t == 0``, the draw probability is 0,
        and the win/loss pair reduces to the Bradley-Terry split used by
        ``probability_future_match``.

        Args:
            name1 (str): The name of the first (black) player.
            name2 (str): The name of the second (white) player.
            handicap (float, optional): A raw elo adjustment favouring name1,
                as in ``probability_future_match``.
            handicap_key (Any, optional): A handicap category key whose
                learned/pinned advantage gamma is folded in, as in
                ``probability_future_match``.
            komi_key (Any, optional): A komi category key whose
                learned/pinned advantage gamma is folded in, as in
                ``probability_future_match``.

        Returns:
            tuple[float, float, float]: ``(P(name1 wins), P(draw), P(name2
            wins))``, non-negative and summing to 1. Unknown players are
            treated as an even (gamma = 1) reference without being added to
            the base.

        Raises:
            AttributeError: Raised if name1 and name2 are equal, or if a
                supplied category key resolves to a non-finite/non-positive
                advantage gamma.
        """
        _player1, _player2, bpd_gamma, _bpd_elo, wpd_gamma, _wpd_elo = (
            self._match_player_days(name1, name2)
        )
        gh, gk = self._resolve_advantage_gammas(handicap_key, komi_key)
        h_shift = 10 ** (handicap / 400.0)
        s1 = bpd_gamma * gh * h_shift
        s2 = wpd_gamma * gk
        t = self.nu * math.sqrt(s1 * s2)
        z = s1 + s2 + t
        return s1 / z, t / z, s2 / z

    def _nu_gradient_hessian(self) -> tuple[float, float]:
        """(gradient, Hessian) of the global draw tendency nu (Davidson), in
        log-nu space, summed over all games -- the per-game accumulation
        shared by ``_newton_draw`` (the Newton step) and
        ``max_gradient_norm`` (the convergence gauge), so they can never
        disagree on it (the same disagreement risk ``_eligible_advantage_updates``
        guards against for handicap/komi)."""
        gradient = 0.0
        hessian = 0.0
        for game in self.games:
            if game.bpd is None or game.wpd is None:
                continue
            s, o = game.effective_gammas(game.black_player)
            t = self.nu * math.sqrt(s * o)
            z = s + o + t
            ratio = t / z
            gradient += (1.0 if game.winner == "D" else 0.0) - ratio
            hessian += -ratio * (1.0 - ratio)
        return gradient, hessian

    def _newton_draw(self) -> None:
        """One Newton step on the global draw tendency nu (Davidson), in log-nu
        space. Skipped when there are no draws or nu is pinned."""
        if not self._has_draws or self.config["pinned_draw"] is not None:
            return
        gradient, hessian = self._nu_gradient_hessian()
        hessian -= self.config["hessian_damping"]
        v = math.log(self.nu) - gradient / hessian
        self.nu = math.exp(v)

    def _run_one_iteration(self) -> None:
        """Runs one iteration of the WHR algorithm."""
        for player in self.players.values():
            player.draw_tendency = self.nu
            player.run_one_newton_iteration()
        self._newton_handicap_komi()
        self._newton_draw()

    def load_games(self, games: list[str], separator: str = " ") -> None:
        """Loads all games at once.

        Each game string must follow the format: "black_name white_name winner time_step handicap extras",
        where handicap and extras are optional. Handicap defaults to 0 if not provided, and extras must be a valid dictionary.

        Args:
            games (list[str]): A list of strings representing games.
            separator (str, optional): The separator used between elements of a game, defaulting to a space.

        Raises:
            ValueError: If any game string does not comply with the expected format or if parsing fails.
        """
        for line in games:
            parts = [part.strip() for part in line.split(separator)]
            if len(parts) < 4 or len(parts) > 6:
                raise ValueError(f"Invalid game format: '{line}'")

            black, white, winner, time_step, *rest = parts
            handicap = 0
            extras = {}

            if len(rest) == 1:
                try:
                    handicap = int(rest[0])
                except ValueError:
                    try:
                        extras = ast.literal_eval(rest[0])
                        if not isinstance(extras, dict):
                            raise ValueError()
                    except (ValueError, SyntaxError):
                        raise ValueError(
                            f"Invalid handicap or extra value in: '{line}'"
                        ) from None

            if len(rest) == 2:
                try:
                    handicap = int(rest[0])
                except ValueError:
                    raise ValueError(f"Invalid handicap value in: '{line}'") from None
                try:
                    extras = ast.literal_eval(rest[1])
                    if not isinstance(extras, dict):
                        raise ValueError()
                except (ValueError, SyntaxError):
                    raise ValueError(
                        f"Invalid extras dictionary in: '{line}'"
                    ) from None

            if self.config["uncased"]:
                black, white = black.lower(), white.lower()

            self.create_game(black, white, winner, int(time_step), handicap, extras)

    def save_base(self, path: str) -> None:
        """Saves the current state of the base to a specified path.

        Instead of pickling the interconnected object graph (players, days and
        games all reference each other), a flat description is saved: the config,
        the list of games, and the computed ratings for every player day. This
        avoids the deep recursive traversal that pickle would otherwise perform,
        which overflows the stack on large histories (see issue #12), while still
        preserving the computed state so the history does not have to be
        re-rated after loading.

        Args:
            path (str): The path where the base will be saved.
        """
        games = [
            (
                game.black_player.name,
                game.white_player.name,
                game.winner,
                game.day,
                game.handicap,
                game.extras,
            )
            for game in self.games
        ]
        ratings = {
            name: [(day.day, day.r, day.uncertainty) for day in player.days]
            for name, player in self.players.items()
        }
        config = self.config
        try:
            pickle.dumps(config)
        except Exception:
            config = {
                k: v
                for k, v in self.config.items()
                if k
                in [
                    "w2",
                    "uncased",
                    "initial_prior_wins",
                    "hessian_damping",
                    "drift_kernel_radius",
                    "pinned_handicap",
                    "pinned_komi",
                    "estimate_handicap_zero",
                    "pinned_draw",
                ]
            }
            warnings.warn(
                "Some elements in config cannot be pickled; only 'w2', "
                "'uncased', 'initial_prior_wins', 'hessian_damping', "
                "'drift_kernel_radius', 'pinned_handicap', 'pinned_komi', "
                "'estimate_handicap_zero' and 'pinned_draw' will be saved.",
                stacklevel=2,
            )
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": config,
                    "games": games,
                    "ratings": ratings,
                    "handicap_gamma": dict(self.handicap_gamma),
                    "komi_gamma": dict(self.komi_gamma),
                    "nu": self.nu,
                },
                f,
            )

    @staticmethod
    def load_base(path: str) -> WHR:
        """Loads a saved base from a specified path.

        Args:
            path (str): The path to the saved base.

        Returns:
            WHR: The loaded base.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            # Legacy format: a pickled object graph as [players, games, config].
            players, games, config = data
            # Reconstruct through the constructor so config defaults (including
            # keys added in later versions) are applied and the dict is copied.
            result = WHR(config)
            result.games, result.players = games, players
            # Players pickled by older versions predate these attributes;
            # backfill them so the loaded base can still be iterated.
            for player in result.players.values():
                player.initial_prior_wins = result.config["initial_prior_wins"]
                player.hessian_damping = result.config["hessian_damping"]
            # Preserve advantages carried by pickled games (phase-3+ bases saved in
            # the legacy shape); genuinely-old games have no such attribute.
            for game in games:
                old_h = getattr(game, "handicap_gamma", None)
                old_k = getattr(game, "komi_gamma", None)
                if isinstance(old_h, dict):
                    result.handicap_gamma.update(old_h)
                if isinstance(old_k, dict):
                    result.komi_gamma.update(old_k)
            # Games pickled by older versions predate the handicap_gamma/
            # komi_gamma tables (phase-3): rewire each game onto this
            # instance's shared tables (discarding any stale per-game dict
            # from the pickle, now preserved above) and backfill an entry for
            # every key so _newton_handicap_komi can look them up.
            for game in result.games:
                game.handicap_gamma = result.handicap_gamma
                game.komi_gamma = result.komi_gamma
                result._ensure_advantage_keys(game.handicap, game.extras["komi"])
            return result
        result = WHR(data["config"])
        for black, white, winner, time_step, handicap, extras in data["games"]:
            result.create_game(black, white, winner, time_step, handicap, extras)
        result.handicap_gamma.update(data.get("handicap_gamma", {}))
        result.komi_gamma.update(data.get("komi_gamma", {}))
        for name, days in data["ratings"].items():
            # player_by_name (re)creates players that have no games, so those
            # queried for predictions are preserved rather than dropped.
            player = result.player_by_name(name)
            day_by_time_step = {day.day: day for day in player.days}
            for time_step, r, uncertainty in days:
                player_day = day_by_time_step[time_step]
                player_day.r = r
                player_day.uncertainty = uncertainty
        # Restore the fitted draw tendency after the replay above: replaying
        # the games through create_game re-seeds nu (via _add_game) on the
        # first "D" game, which would otherwise discard a fitted value. Guard
        # with .get so a base saved before this fix (lacking the key) keeps
        # whatever the replay seeded.
        result.nu = data.get("nu", result.nu)
        return result


class Base(WHR):
    """Deprecated alias for :class:`WHR`, kept for backward compatibility."""

    def __init__(self, config: dict[str, Any] | None = None):
        warnings.warn(
            "Base has been renamed to WHR; the Base alias will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)
