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
from whr.utils import (
    HandicapBaselineWarning,
    NoDrawsWarning,
    UncomputedUncertaintyWarning,
)

# _compute_drift allocates arrays sized to the CALENDAR SPAN of day values
# (max_day - min_day), not to the number of games. If `time_step` is an epoch
# timestamp instead of a compact day index, this guard prevents a silent
# hang/OOM.
_MAX_DRIFT_DAY_SPAN = 1_000_000

# Converts a natural-rating (r = ln(gamma)) variance/quantity to elo units:
# elo = 400/ln(10) * r.
_ELO_PER_NAT = 400.0 / math.log(10)

# Largest half-gap h = (r1 - r2)/2, in nats, that the uncertainty-integrated
# Davidson predictor exponentiates. math.exp overflows just above 709.78, and
# the sum e^h + e^-h + nu needs headroom on top; 700 leaves that while being
# far beyond the point where the win/draw/loss split has already saturated to
# (1, 0, 0) in double precision (h = 700 is ~486,000 elo).
_MAX_HALF_GAP = 700.0

# Trust-region cap on a single handicap/komi/draw-tendency Newton step, in
# natural-log (log-gamma / log-nu) units. The scalar advantage updates take a
# step of the form ``value *= exp(-grad / hess)``; a degenerate or
# ill-conditioned key can make ``-grad/hess`` enormous and overflow ``math.exp``
# (``OverflowError: math range error``). This cap is far larger than any step a
# well-conditioned fit ever takes (those are < 1), so it never binds in normal
# use, but exp(10) stays finite, so a pathological key can no longer crash the
# iteration. The per-key 1-D sub-problem is concave in log-space, so the clamped
# step still converges rather than growing unboundedly across iterations.
_MAX_ADVANTAGE_LOG_STEP = 10.0


def _validated_time_step(time_step: Any) -> int | float:
    """Check a ``time_step`` early, and normalise an integral float to ``int``.

    Fractional day values are meaningful to the model -- a day is just an index
    and the Wiener prior uses ``|delta| * w2`` -- so they are allowed. But they
    used to be accepted silently while ``load_games`` could not express them and
    ``remove_drift`` crashed on them; both now handle them, and the checks here
    turn the remaining nonsense into an error at the point of entry rather than a
    cryptic ``TypeError`` from inside the maths.

    ``1.0`` is narrowed to ``1`` so that a float and an int spelling of the same
    day are the *same* playing day rather than two, which is what a caller
    building days from arithmetic almost always means.
    """
    if isinstance(time_step, bool):
        raise TypeError(
            f"time_step must be a number, got a bool ({time_step!r}). "
            "A bool would silently be read as day 0 or 1."
        )
    if not isinstance(time_step, (int, float)):
        raise TypeError(
            f"time_step must be an int or a float, got "
            f"{type(time_step).__name__} ({time_step!r}). It is a day index "
            "counted from an origin of your choosing."
        )
    if not math.isfinite(time_step):
        raise ValueError(f"time_step must be finite, got {time_step!r}")
    if isinstance(time_step, float) and time_step.is_integer():
        return int(time_step)
    return time_step


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
        self.config.setdefault("draw_rate", None)
        self._has_draws = False
        self._warned_no_draws = False
        self._warned_baseline = False
        self._warned_uncomputed_uncertainty = False
        self.nu = self._resolve_pinned_draw()
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
        game's handicap and komi keys, without overwriting existing/pinned ones.

        A ``komi`` of ``None`` means the game has no komi (opt-in, 3.1.0): no
        komi key is registered, so it is never estimated."""
        if handicap not in self.handicap_gamma:
            self.handicap_gamma[handicap] = 1.0
        if komi is not None and komi not in self.komi_gamma:
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
        # A draw credits neither a handicap (black) win nor a komi (white)
        # win, and the plain Bradley-Terry gradient/Hessian denominator
        # accumulated below does not apply to it. Skipping draws here means
        # handicap/komi advantages are estimated from DECISIVE games only
        # when draws are present -- a deliberate simplification; full
        # Davidson-aware handicap/komi estimation is out of scope for this
        # phase.
        #
        # This first pass (gathering each decisive game's raw handicap/komi
        # keys and gammas into plain lists) stays a Python loop because
        # self.games is a list of Game objects, not already-batched arrays;
        # only the per-key ACCUMULATION below is vectorized with numpy.
        handicaps: list[Any] = []
        komis: list[Any] = []
        gb_vals: list[float] = []
        gw_vals: list[float] = []
        black_win: list[bool] = []
        for g in self.games:
            if g.bpd is None or g.wpd is None:
                continue
            if g.winner == "D":
                continue
            handicaps.append(g.handicap)
            # komi is opt-in: a game with no komi contributes gamma 1.0 to the
            # math (below) but its ``None`` komi key is never scattered into the
            # per-key results, so it is never estimated.
            komis.append(g.extras.get("komi"))
            gb_vals.append(g.bpd.gamma())
            gw_vals.append(g.wpd.gamma())
            black_win.append(g.winner == "B")

        h_grad: dict[Any, float] = {}
        h_hess: dict[Any, float] = {}
        k_grad: dict[Any, float] = {}
        k_hess: dict[Any, float] = {}
        h_games: dict[Any, int] = {}
        h_wins: dict[Any, int] = {}
        k_games: dict[Any, int] = {}
        k_wins: dict[Any, int] = {}
        if not handicaps:
            return h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins

        # Map each game's handicap/komi key to a contiguous integer index
        # (in first-seen order, matching the insertion order the old
        # dict-accumulation loop produced) so np.bincount can accumulate
        # per key.
        h_index_of: dict[Any, int] = {}
        k_index_of: dict[Any, int] = {}
        hi = np.empty(len(handicaps), dtype=np.int64)
        ki = np.empty(len(komis), dtype=np.int64)
        for i, (h, k) in enumerate(zip(handicaps, komis, strict=True)):
            hi[i] = h_index_of.setdefault(h, len(h_index_of))
            ki[i] = k_index_of.setdefault(k, len(k_index_of))
        nh = len(h_index_of)
        nk = len(k_index_of)

        gb = np.array(gb_vals, dtype=np.float64)
        gw = np.array(gw_vals, dtype=np.float64)
        gh = np.array([self.handicap_gamma[h] for h in handicaps], dtype=np.float64)
        gk = np.array([self.komi_gamma.get(k, 1.0) for k in komis], dtype=np.float64)
        black_win_arr = np.array(black_win, dtype=np.float64)

        c_komi = gw
        d_komi = gb * gh
        c_handicap = gb
        d_handicap = gw * gk
        # Guard the vectorized division: these gammas are read directly (no
        # opponents_adjusted_gamma positivity guard upstream), so a 0
        # denominator is theoretically reachable if a gamma underflows to 0.0
        # (only around -129000 elo, so unreachable in normal ranges). Raise a
        # loud, deterministic FloatingPointError then, rather than emitting
        # numpy's silent inf + divide-by-zero RuntimeWarning.
        with np.errstate(divide="raise", invalid="raise"):
            div = 1.0 / (d_komi + d_handicap)
            h_grad_g = c_handicap * div
            h_hess_g = c_handicap * d_handicap * div**2
            k_grad_g = c_komi * div
            k_hess_g = c_komi * d_komi * div**2

        h_grad_arr = np.bincount(hi, weights=h_grad_g, minlength=nh)
        h_hess_arr = np.bincount(hi, weights=h_hess_g, minlength=nh)
        h_games_arr = np.bincount(hi, minlength=nh)
        h_wins_arr = np.bincount(hi, weights=black_win_arr, minlength=nh)

        k_grad_arr = np.bincount(ki, weights=k_grad_g, minlength=nk)
        k_hess_arr = np.bincount(ki, weights=k_hess_g, minlength=nk)
        k_games_arr = np.bincount(ki, minlength=nk)
        k_wins_arr = np.bincount(ki, weights=1.0 - black_win_arr, minlength=nk)

        for key, idx in h_index_of.items():
            h_grad[key] = float(h_grad_arr[idx])
            h_hess[key] = float(h_hess_arr[idx])
            h_games[key] = int(h_games_arr[idx])
            # h_wins_arr is a bincount of 0/1 win-count weights, so each entry
            # is an exact integer count well under 2^53; round()+int() recovers
            # it exactly (guarding only against float bincount rounding dust).
            wins = int(round(h_wins_arr[idx]))
            if wins:
                h_wins[key] = wins
        for key, idx in k_index_of.items():
            if key is None:
                continue  # the no-komi sentinel: never estimated
            k_grad[key] = float(k_grad_arr[idx])
            k_hess[key] = float(k_hess_arr[idx])
            k_games[key] = int(k_games_arr[idx])
            # Exact integer bincount, as for h_wins_arr above.
            wins = int(round(k_wins_arr[idx]))
            if wins:
                k_wins[key] = wins
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

    @staticmethod
    def _clamped_log_step(grad: float, hess: float) -> float:
        """A scalar Newton step ``-grad / hess`` in log space, clamped to the
        ``_MAX_ADVANTAGE_LOG_STEP`` trust region so it can never overflow a
        subsequent ``math.exp``. Returns 0.0 (no update) for a non-finite step
        (e.g. a zero Hessian)."""
        if hess == 0.0:
            return 0.0
        step = -grad / hess
        if not math.isfinite(step):
            return 0.0
        return max(-_MAX_ADVANTAGE_LOG_STEP, min(_MAX_ADVANTAGE_LOG_STEP, step))

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
            self.handicap_gamma[key] = gamma * math.exp(
                self._clamped_log_step(grad, hess)
            )
        for key, gamma, grad in self._eligible_advantage_updates(
            self.komi_gamma, self._pinned_komi_keys, k_grad, k_games, k_wins
        ):
            hess = -gamma * k_hess.get(key, 0.0) - damping
            self.komi_gamma[key] = gamma * math.exp(self._clamped_log_step(grad, hess))

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
    ) -> list[tuple[str, str, str, int | float, float, dict[str, Any]]]:
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
        no test game can be scored (every one is cold-start or a draw), in
        which case ``log_loss`` is all-``inf`` and ``best_w2`` is not
        meaningful.
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
                    model.create_game(
                        black, white, winner, day, handicap, extras=extras
                    )
                model.iterate(iterations)
                for black, white, winner, _day, handicap, extras in test:
                    if winner == "D":
                        # A win/loss predictive log-loss has no correct value
                        # for a draw -- skip it (already counted in
                        # n_test_skipped above), do not score it as a white
                        # win.
                        continue
                    komi = extras.get("komi")  # None -> no komi (opt-in)
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
                "(cold-start or all-draw) -- log_loss is all inf and best_w2 is "
                "not meaningful. Check n_test_scored/n_test_skipped; try fewer "
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

    def rating_covariance(self, name: str) -> tuple[list[int | float], np.ndarray]:
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
    ) -> list[tuple[int | float, float, float]] | tuple[float, float]:
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
        self._warn_if_uncertainty_uncomputed(player)
        if current:
            return (
                round(player.days[-1].elo),
                round(player.days[-1].uncertainty, 2),
            )
        return [(d.day, round(d.elo), round(d.uncertainty, 2)) for d in player.days]

    def _warn_if_uncertainty_uncomputed(self, player: Player) -> None:
        """Flag the ``-1`` uncertainty sentinel being read as if it were a value.

        ``rating_difference`` and friends raise in this state; this method has
        always returned the raw sentinel instead, so an un-rated base stays
        inspectable. The warning keeps that behaviour while making the ``-1``
        impossible to mistake for a standard deviation. Once per instance.
        """
        if self._warned_uncomputed_uncertainty:
            return
        if not any(day.uncertainty < 0 for day in player.days):
            return
        self._warned_uncomputed_uncertainty = True
        warnings.warn(
            f"ratings_for_player({player.name!r}) is reporting the uncertainty "
            "sentinel -1, which means uncertainties have not been computed -- it "
            "is not a standard deviation. Call iterate() or auto_iterate() "
            "first. (rating_difference/rating_covariance raise a ValueError in "
            "this same state.)",
            UncomputedUncertaintyWarning,
            stacklevel=3,
        )

    def _setup_game(
        self,
        black: str,
        white: str,
        winner: str,
        time_step: int | float,
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
        time_step: int | float,
        handicap: float,
        komi: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> Game:
        """Creates a new game to be added to the base.

        Args:
            black (str): The name of the black player.
            white (str): The name of the white player.
            winner (str): "B" if black won, "W" if white won, "D" for a draw.
            time_step (int): The day of the match from the origin.
            handicap (float): The handicap category key (e.g. a stone count).
                Its advantage is estimated from the data (or pinned to a
                known elo value via the ``pinned_handicap`` config), not a
                fixed elo amount itself — see "Handicap and komi" in the
                README.
            komi (Any, optional): The komi category key (a white-side
                advantage). **Opt-in**: ``None`` (the default) models no komi at
                all — nothing is estimated. Pass a value (e.g. ``6.5``) to have
                that komi category's advantage estimated (or pinned via
                ``pinned_komi``). Prior to 3.1.0 a komi of ``6.5`` was assumed
                and estimated for every game; pass ``komi=6.5`` to reproduce
                that.
            extras (dict[str, Any] | None, optional): Extra per-game metadata.
                A ``"komi"`` entry here is still honoured (equivalent to the
                ``komi`` argument) for backward compatibility.

        Returns:
            Game: The newly added game.

        Raises:
            TypeError: If ``time_step`` is not a real number (a ``bool`` is
                rejected too: ``True`` silently meant day 1).
            ValueError: If ``time_step`` is NaN or infinite.
            AttributeError: If ``black`` and ``white`` are the same player, or
                if ``winner`` is not one of "B", "W", "D".
        """
        extras = dict(extras) if extras else {}
        if komi is not None:
            extras["komi"] = komi
        if self.config["uncased"]:
            black = black.lower()
            white = white.lower()
        time_step = _validated_time_step(time_step)
        game = self._setup_game(black, white, winner, time_step, handicap, extras)
        self._ensure_advantage_keys(game.handicap, game.extras.get("komi"))
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
            # A declared draw tendency is already in self.nu (resolved in
            # __init__) and must not be re-applied here; all this branch owes
            # the fit is a non-zero starting point, since nu is updated
            # multiplicatively in log-space and would stay pinned at 0.
            if not self.draws_declared and self.nu == 0.0:
                self.nu = 1.0
        self.games.append(game)
        return game

    @staticmethod
    def nu_from_draw_rate(draw_rate: float) -> float:
        """Davidson's draw tendency ``nu`` implied by a target draw rate.

        Between two players of *equal* strength ``s``, Davidson's split gives
        ``T = nu * s`` and ``Z = (2 + nu) * s``, so ``P(draw) = nu / (2 + nu)``.
        Inverting: ``nu = 2p / (1 - p)``.

        The even-matchup qualifier matters. Draws are likeliest between equals,
        so over a real fixture list -- where most pairings are lopsided -- the
        observed rate comes out *below* what this returns. European big-five
        football fits ``nu = 0.79``, i.e. a 28% even-matchup rate, against 25%
        draws observed overall. Treat this as a starting point for
        ``draw_rate=``, not as a substitute for fitting ``nu`` on real draws.

        Args:
            draw_rate (float): Target draw probability between equals, in
                ``[0, 1)``.

        Returns:
            float: The corresponding ``nu`` (``0.0`` for a rate of 0).

        Raises:
            ValueError: If ``draw_rate`` is not a finite value in ``[0, 1)``.
        """
        if not math.isfinite(draw_rate) or not 0.0 <= draw_rate < 1.0:
            raise ValueError(
                f"draw_rate must be a finite value in [0, 1), got {draw_rate!r}. "
                "A rate of 1 would need an infinite draw tendency."
            )
        return 2.0 * draw_rate / (1.0 - draw_rate)

    @staticmethod
    def draw_rate_from_nu(nu: float) -> float:
        """The even-matchup draw rate implied by a draw tendency ``nu``.

        The inverse of :meth:`nu_from_draw_rate`: ``P(draw) = nu / (2 + nu)``.
        Useful for reading a fitted ``draw_tendency`` back in a unit that can be
        compared against an observed draw percentage.

        Args:
            nu (float): A non-negative draw tendency.

        Returns:
            float: The implied draw probability between equal players.

        Raises:
            ValueError: If ``nu`` is negative or not finite.
        """
        if not math.isfinite(nu) or nu < 0.0:
            raise ValueError(f"nu must be finite and non-negative, got {nu!r}")
        return nu / (2.0 + nu)

    def _resolve_pinned_draw(self) -> float:
        """The draw tendency the caller declared, or 0.0 if they declared none.

        ``pinned_draw`` and ``draw_rate`` are two spellings of one decision, so
        setting both is an error rather than a precedence puzzle. Resolved once
        in ``__init__`` -- it used to be applied inside ``create_game``'s draw
        branch, which silently ignored it on data containing no draws, i.e. in
        exactly the case a caller reaches for it.
        """
        pinned, rate = self.config["pinned_draw"], self.config["draw_rate"]
        if pinned is not None and rate is not None:
            raise ValueError(
                "config sets both pinned_draw and draw_rate; they are two ways "
                "to state the same thing. Use draw_rate to express a draw "
                "percentage, or pinned_draw to set Davidson's nu directly."
            )
        if rate is not None:
            return self.nu_from_draw_rate(float(rate))
        if pinned is not None:
            pinned = float(pinned)
            if not math.isfinite(pinned) or pinned < 0.0:
                raise ValueError(
                    f"pinned_draw must be finite and non-negative, got {pinned!r}"
                )
            return pinned
        return 0.0

    _ONE_SIDED_SHARE = 0.05  # a player playing <5% or >95% of games as one colour
    _IMBALANCED_GAME_SHARE = 0.5  # ...and such players carrying most of the games

    def one_sided_game_share(self) -> float:
        """Share of games involving a player who almost never changes colour.

        The statistic behind :class:`~whr.utils.HandicapBaselineWarning`. A player
        is "one-sided" when at most ``5%`` of their games are on one side of the
        board. The fraction returned is games with at least one such player,
        over all games -- so a league where every team plays home and away
        scores ~0, while a base where one competitor is always "black" scores 1.
        """
        if not self.games:
            return 0.0
        black_count: dict[str, int] = {}
        total_count: dict[str, int] = {}
        for game in self.games:
            for name, is_black in (
                (game.black_player.name, True),
                (game.white_player.name, False),
            ):
                total_count[name] = total_count.get(name, 0) + 1
                black_count[name] = black_count.get(name, 0) + (1 if is_black else 0)
        one_sided = {
            name
            for name, total in total_count.items()
            if not (
                self._ONE_SIDED_SHARE
                <= black_count[name] / total
                <= 1.0 - self._ONE_SIDED_SHARE
            )
        }
        if not one_sided:
            return 0.0
        affected = sum(
            1
            for game in self.games
            if game.black_player.name in one_sided
            or game.white_player.name in one_sided
        )
        return affected / len(self.games)

    def _warn_if_baseline_unidentified(self) -> None:
        """Flag ``estimate_handicap_zero`` on data that cannot identify the
        freed baseline. Fires at most once per instance, at the start of a fit.
        """
        if self._warned_baseline or not self.config["estimate_handicap_zero"]:
            return
        share = self.one_sided_game_share()
        if share <= self._IMBALANCED_GAME_SHARE:
            return
        self._warned_baseline = True
        warnings.warn(
            f"estimate_handicap_zero=True frees the handicap key 0 baseline, but "
            f"{share:.0%} of games involve a player who almost always takes the "
            "same colour, so that baseline is not separately identifiable from "
            "player strength. Differences between handicap keys stay correct "
            "while the overall level leaks into the ratings: equal players can "
            "be reported tens of elo apart, and probability_future_match without "
            "a handicap_key is wrong by the same amount. Leave "
            "estimate_handicap_zero at False (the default), or anchor the scale "
            "with pinned_handicap. Inspect the statistic via "
            "one_sided_game_share().",
            HandicapBaselineWarning,
            stacklevel=3,
        )

    def _warn_if_draws_undeclared(self) -> None:
        """Flag a three-outcome prediction whose draw probability is 0 by default
        rather than by decision. Fires at most once per instance so a scoring
        loop over a whole season stays quiet after the first match.
        """
        if self._has_draws or self.draws_declared or self._warned_no_draws:
            return
        self._warned_no_draws = True
        warnings.warn(
            "win_draw_loss_probabilities was called but no draw has been "
            "recorded and no draw tendency was declared, so P(draw) is exactly "
            "0.0. That is correct for a domain that cannot draw and wrong for "
            "one that simply has not drawn yet -- and a P(draw) of 0 makes "
            "log-loss infinite if a draw does occur. Declare the intent: "
            "WHR({'pinned_draw': 0.0}) for no draws (or just use "
            "probability_future_match), or WHR({'draw_rate': 0.25}) to assume a "
            "draw rate until real draws are available to fit.",
            NoDrawsWarning,
            stacklevel=3,
        )

    @property
    def draws_declared(self) -> bool:
        """Whether the caller stated an intent about draws, either way.

        ``True`` when ``pinned_draw`` or ``draw_rate`` is set -- including to
        ``0``, which declares "this domain has no draws" and is an answer, not
        the absence of one.
        """
        return (
            self.config["pinned_draw"] is not None
            or self.config["draw_rate"] is not None
        )

    @property
    def draw_tendency(self) -> float:
        return self.nu

    def iterate(self, count: int) -> None:
        """Performs a specified number of iterations of the algorithm.

        Args:
            count (int): The number of iterations to perform.

        Warns:
            HandicapBaselineWarning: Once per instance, if
                ``estimate_handicap_zero`` is on while colour assignment is too
                one-sided to identify the freed baseline. See the warning class.
        """
        self._warn_if_baseline_unidentified()
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
        if self._has_draws and not self.draws_declared:
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
        # The smoothing runs on a dense integer grid, so fractional days are
        # floored into whole-day bins. That is a non-issue in practice: the
        # kernel already averages over +/- `radius` days (100 by default), so
        # sub-day resolution cannot survive it -- and before this, a single
        # non-integer day made the whole method raise a bare TypeError.
        bins = [math.floor(g.day) for g in self.games]
        min_day, max_day = min(bins), max(bins)
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
            j = math.floor(g.day) - min_day + radius
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

    def remove_drift(self) -> dict[int | float, float]:
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
        applied: dict[int | float, float] = {}
        for player in self.players.values():
            for pd in player.days:
                # drift is keyed by whole-day bin (see _compute_drift), so a
                # fractional day reads the correction for the day it falls in
                correction_elo = -drift.get(math.floor(pd.day), 0.0)
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

    @staticmethod
    def _prediction_sigma(player1: Player | None, player2: Player | None) -> float:
        """Standard deviation of the two players' rating *gap*, in natural
        (``r``) units, for an uncertainty-integrated prediction.

        Uses the independence approximation ``Var(r1 - r2) ~= Var(r1) +
        Var(r2)`` (Coulom's, since WHR computes no cross-player covariance).
        Unknown or dayless players contribute 0, and each per-day variance is
        clamped at 0 because ``uncertainty`` is ``-1`` before ``iterate()``
        runs. A return of 0.0 means "nothing to integrate over" and callers
        short-circuit to the point estimate.

        Shared by ``probability_future_match`` and
        ``win_draw_loss_probabilities`` so the two predictors can never
        disagree on the width of the Gaussian they hedge over.
        """
        var1 = player1.days[-1].uncertainty if (player1 and player1.days) else 0.0
        var2 = player2.days[-1].uncertainty if (player2 and player2.days) else 0.0
        return math.sqrt(max(var1, 0.0) + max(var2, 0.0))

    @staticmethod
    def _gaussian_quadrature(uncertainty_steps: int) -> list[tuple[float, float]]:
        """The ``(x, weight)`` nodes of the unnormalised Gaussian quadrature
        grid used by the uncertainty-integrated predictors: nodes at
        ``x_i = 0.5 * i`` for ``i`` in ``-steps..steps`` (so the grid spans
        +/-0.5 * uncertainty_steps standard deviations) with weights
        ``exp(-x^2 / 2)``.

        Returns a list rather than a generator so that an invalid
        ``uncertainty_steps`` is rejected when the grid is requested, not
        lazily when a caller first iterates it.

        Weights are deliberately left unnormalised; callers divide by their
        accumulated total, which is what makes a set of per-node outcome
        probabilities summing to 1 average to exactly 1.

        Raises:
            ValueError: If ``uncertainty_steps`` is less than 1.
        """
        if uncertainty_steps < 1:
            raise ValueError("uncertainty_steps must be >= 1")
        return [
            (0.5 * i, math.exp(-((0.5 * i) ** 2) / 2.0))
            for i in range(-uncertainty_steps, uncertainty_steps + 1)
        ]

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
            handicap (float, optional): A raw elo adjustment (not a handicap
                category key) that shifts the effective elo gap ``handicap``
                points in name1's favour.
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

        nodes = self._gaussian_quadrature(uncertainty_steps)
        sigma = self._prediction_sigma(player1, player2)
        if sigma == 0.0:
            return player1_proba, player2_proba

        eps = 1e-12
        p1c = min(max(player1_proba, eps), 1.0 - eps)
        delta_r = math.log(p1c / (1.0 - p1c))  # logit of the point probability
        total_weight = 0.0
        integral = 0.0
        for x, weight in nodes:
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
        account_for_uncertainty: bool = False,
        uncertainty_steps: int = 4,
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
            account_for_uncertainty (bool, optional): When ``False`` (default,
                backward-compatible), returns the point estimate computed from
                the players' current ratings only. When ``True``, integrates
                all three outcome probabilities over the Gaussian implied by
                the players' rating variances -- the three-outcome counterpart
                of ``probability_future_match``'s Coulom ``Predict``. See the
                note on hedging direction below: the effect is a compression of
                the win/loss odds, NOT a move toward an even three-way split.
            uncertainty_steps (int, optional): Number of quadrature steps on
                each side of the integration grid, as in
                ``probability_future_match``; the grid spans +/-0.5 *
                uncertainty_steps standard deviations (nodes at x_i = 0.5*i).
                Ignored when ``account_for_uncertainty`` is ``False``. Defaults
                to 4. Must be >= 1 when ``account_for_uncertainty`` is ``True``.

        Dividing the Davidson split through by ``sqrt(s1 * s2)`` shows that all
        three outcomes depend on the ratings only through the scalar gap
        ``d = ln(s1) - ln(s2)``::

            P(win) = e^(d/2) / Z,  P(draw) = nu / Z,  P(loss) = e^(-d/2) / Z
            with Z = e^(d/2) + e^(-d/2) + nu

        so ``account_for_uncertainty=True`` integrates that single scalar over
        ``d ~ N(d_hat, sigma^2)`` with ``sigma^2 = Var(r1) + Var(r2)``, on the
        same grid ``probability_future_match`` uses. Because every node
        contributes a triple summing to 1, the weighted average sums to 1
        identically -- normalisation is a property of the quadrature, not a
        renormalisation step. At ``nu == 0`` the resulting ``(win, loss)`` pair
        is exactly ``probability_future_match(..., account_for_uncertainty=True)``.

        Note on hedging direction: uncertainty compresses the win/loss odds
        ``P(win)/P(loss)`` toward 1 and never lowers the underdog's
        probability, but it does NOT simply move mass toward the draw or toward
        an even split. Davidson's draw curve ``nu/(2*cosh(d/2) + nu)`` is
        concave near ``d == 0`` and convex in the tails, so spreading ``d``
        *drains* the draw for a near-even matchup and *feeds* it for a lopsided
        one. Consequently a barely-favoured player's win probability can rise
        (the drained draw mass splits to both sides); only a clear favourite's
        falls.

        Returns:
            tuple[float, float, float]: ``(P(name1 wins), P(draw), P(name2
            wins))``, non-negative and summing to 1. Unknown players are
            treated as an even (gamma = 1) reference without being added to
            the base.

        Raises:
            AttributeError: Raised if name1 and name2 are equal, or if a
                supplied category key resolves to a non-finite/non-positive
                advantage gamma.
            ValueError: Raised if ``account_for_uncertainty`` is ``True`` and
                ``uncertainty_steps`` is less than 1.

        Warns:
            NoDrawsWarning: Once per instance, if no draw was ever recorded and
                no draw tendency was declared via ``pinned_draw`` /
                ``draw_rate``. The returned draw probability is then exactly
                ``0.0``, which is right for a domain that cannot draw and wrong
                for one that simply has not drawn yet -- and the library cannot
                tell which. Declare it either way to silence this.
        """
        self._warn_if_draws_undeclared()
        player1, player2, bpd_gamma, _bpd_elo, wpd_gamma, _wpd_elo = (
            self._match_player_days(name1, name2)
        )
        gh, gk = self._resolve_advantage_gammas(handicap_key, komi_key)
        h_shift = 10 ** (handicap / 400.0)
        s1 = bpd_gamma * gh * h_shift
        s2 = wpd_gamma * gk
        nu = self.nu
        t = nu * math.sqrt(s1 * s2)
        z = s1 + s2 + t
        if not account_for_uncertainty:
            return s1 / z, t / z, s2 / z

        nodes = self._gaussian_quadrature(uncertainty_steps)
        sigma = self._prediction_sigma(player1, player2)
        if sigma == 0.0:
            return s1 / z, t / z, s2 / z

        # Integrate the split over the Gaussian on the rating gap d, working in
        # the half-gap h = d/2 so both exponentials stay the same magnitude.
        #
        # h is clamped to +/-_MAX_HALF_GAP because an extreme gap (or a modest
        # gap with a huge sigma) would otherwise overflow math.exp, raising
        # where the point-estimate path returns fine. At the clamp the split is
        # already (1, 0, 0) to full double precision, so this is invisible for
        # every gap that carries information -- and leaves the arithmetic
        # untouched in the entire useful range, including the exact agreement
        # with ``probability_future_match`` at nu == 0.
        d_hat = math.log(s1) - math.log(s2)
        total_weight = 0.0
        win = draw = loss = 0.0
        for x, weight in nodes:
            h = 0.5 * (d_hat + sigma * x)
            h = max(-_MAX_HALF_GAP, min(_MAX_HALF_GAP, h))
            e_win = math.exp(h)
            e_loss = math.exp(-h)
            node_z = e_win + e_loss + nu
            win += weight * e_win / node_z
            draw += weight * nu / node_z
            loss += weight * e_loss / node_z
            total_weight += weight
        return win / total_weight, draw / total_weight, loss / total_weight

    def _nu_gradient_hessian(self) -> tuple[float, float]:
        """(gradient, Hessian) of the global draw tendency nu (Davidson), in
        log-nu space, summed over all games -- the per-game accumulation
        shared by ``_newton_draw`` (the Newton step) and
        ``max_gradient_norm`` (the convergence gauge), so they can never
        disagree on it (the same disagreement risk ``_eligible_advantage_updates``
        guards against for handicap/komi)."""
        # Gathering each game's raw effective gammas into plain lists stays
        # a Python loop (self.games is a list of Game objects, not
        # already-batched arrays); only the summation below is vectorized.
        s_vals: list[float] = []
        o_vals: list[float] = []
        is_draw_vals: list[float] = []
        for game in self.games:
            if game.bpd is None or game.wpd is None:
                continue
            s, o = game.effective_gammas(game.black_player)
            s_vals.append(s)
            o_vals.append(o)
            is_draw_vals.append(1.0 if game.winner == "D" else 0.0)

        if not s_vals:
            return 0.0, 0.0

        s_arr = np.array(s_vals, dtype=np.float64)
        o_arr = np.array(o_vals, dtype=np.float64)
        is_draw_arr = np.array(is_draw_vals, dtype=np.float64)

        t_arr = self.nu * np.sqrt(s_arr * o_arr)
        z_arr = s_arr + o_arr + t_arr
        # Guard the vectorized division as in davidson_derivatives: z = S+O+T
        # comes from the unguarded effective_gammas, so a 0 denominator is
        # only reachable via a gamma underflow to 0.0 (unreachable in normal
        # ranges). Raise a loud FloatingPointError rather than a silent inf/nan.
        with np.errstate(divide="raise", invalid="raise"):
            ratio = t_arr / z_arr
        gradient = float(np.sum(is_draw_arr) - np.sum(ratio))
        hessian = float(-np.sum(ratio * (1.0 - ratio)))
        return gradient, hessian

    def _newton_draw(self) -> None:
        """One Newton step on the global draw tendency nu (Davidson), in log-nu
        space. Skipped when there are no draws, or when the caller declared the
        draw tendency via ``pinned_draw`` / ``draw_rate``."""
        if not self._has_draws or self.draws_declared:
            return
        gradient, hessian = self._nu_gradient_hessian()
        hessian -= self.config["hessian_damping"]
        v = math.log(self.nu) + self._clamped_log_step(gradient, hessian)
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
            # strip the line before splitting, so a stray leading/trailing space
            # is not read as an extra (empty) field
            parts = [part.strip() for part in line.strip().split(separator)]
            if len(parts) < 4 or len(parts) > 6:
                raise ValueError(
                    f"Invalid game format: '{line}' -- expected 4 to 6 "
                    f"{separator!r}-separated fields "
                    "(black white winner time_step [handicap] [extras]), "
                    f"got {len(parts)}"
                )
            if any(part == "" for part in parts):
                raise ValueError(
                    f"Empty field in: '{line}' -- a repeated {separator!r} "
                    "separator leaves a blank field and shifts every field "
                    "after it. Collapse the repeat, or pass a separator that "
                    "does not occur inside your names."
                )

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

            # `create_game` accepts fractional days, so the parser must too --
            # it used to reject them with a bare int() error while the
            # programmatic path let them through.
            day: int | float
            try:
                day = int(time_step)
            except ValueError:
                try:
                    day = float(time_step)
                except ValueError:
                    raise ValueError(
                        f"Invalid time_step {time_step!r} in: '{line}' -- must "
                        "be a number (a day index counted from an origin)"
                    ) from None
            self.create_game(black, white, winner, day, handicap, extras=extras)

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
                    "draw_rate",
                ]
            }
            warnings.warn(
                "Some elements in config cannot be pickled; only 'w2', "
                "'uncased', 'initial_prior_wins', 'hessian_damping', "
                "'drift_kernel_radius', 'pinned_handicap', 'pinned_komi', "
                "'estimate_handicap_zero', 'pinned_draw' and 'draw_rate' will "
                "be saved.",
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
                result._ensure_advantage_keys(game.handicap, game.extras.get("komi"))
            return result
        result = WHR(data["config"])
        for black, white, winner, time_step, handicap, extras in data["games"]:
            # extras carries komi (when set); pass it by keyword since `komi`
            # now precedes `extras` in create_game's signature.
            result.create_game(black, white, winner, time_step, handicap, extras=extras)
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
