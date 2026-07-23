from __future__ import annotations

import ast
import math
import pickle
import time
import warnings
from typing import Any

from whr.game import Game
from whr.player import Player

# _compute_drift allocates arrays sized to the CALENDAR SPAN of day values
# (max_day - min_day), not to the number of games. If `time_step` is an epoch
# timestamp instead of a compact day index, this guard prevents a silent
# hang/OOM.
_MAX_DRIFT_DAY_SPAN = 1_000_000


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
        self.games: list[Game] = []
        self.players: dict[str, Player] = {}

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
        white_player = self.player_by_name(white)
        black_player = self.player_by_name(black)
        game = Game(black_player, white_player, winner, time_step, handicap, extras)
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
            handicap (float): The handicap (in elo points).
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
        return self._add_game(game)

    def _add_game(self, game: Game) -> Game:
        game.white_player.add_game(game)
        game.black_player.add_game(game)
        if game.bpd is None:
            raise RuntimeError(
                "Game could not be attached to the black player's playing day"
            )
        self.games.append(game)
        return game

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
        """Largest gradient infinity-norm across all players (stationarity gauge)."""
        norm = 0.0
        for p in self.players.values():
            if len(p.days) > 0:
                norm = max(norm, p.gradient_infinity_norm())
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

    def probability_future_match(
        self, name1: str, name2: str, handicap: float = 0
    ) -> tuple[float, float]:
        """Calculates the winning probability for a hypothetical match between two players.

        Args:
            name1 (str): The name of the first player.
            name2 (str): The name of the second player.
            handicap (float, optional): The handicap (in elo points).

        Returns:
            tuple[float, float]: The winning probabilities for name1 and name2 respectively. Unknown players are treated as an even (gamma = 1) reference without being added to the base.

        Raises:
            AttributeError: Raised if name1 and name2 are equal
        """
        # Avoid self-played games (no info)
        if self.config["uncased"]:
            name1 = name1.lower()
            name2 = name2.lower()
        if name1 == name2:
            raise AttributeError("Invalid game (black == white)")
        # Pure query: look players up without creating persistent entries.
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
        player1_proba = bpd_gamma / (bpd_gamma + 10 ** ((wpd_elo - handicap) / 400.0))
        player2_proba = wpd_gamma / (wpd_gamma + 10 ** ((bpd_elo + handicap) / 400.0))
        return player1_proba, player2_proba

    def _run_one_iteration(self) -> None:
        """Runs one iteration of the WHR algorithm."""
        for player in self.players.values():
            player.run_one_newton_iteration()

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
                ]
            }
            warnings.warn(
                "Some elements in config cannot be pickled; only 'w2', "
                "'uncased', 'initial_prior_wins', 'hessian_damping' and "
                "'drift_kernel_radius' will be saved.",
                stacklevel=2,
            )
        with open(path, "wb") as f:
            pickle.dump({"config": config, "games": games, "ratings": ratings}, f)

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
            return result
        result = WHR(data["config"])
        for black, white, winner, time_step, handicap, extras in data["games"]:
            result.create_game(black, white, winner, time_step, handicap, extras)
        for name, days in data["ratings"].items():
            # player_by_name (re)creates players that have no games, so those
            # queried for predictions are preserved rather than dropped.
            player = result.player_by_name(name)
            day_by_time_step = {day.day: day for day in player.days}
            for time_step, r, uncertainty in days:
                player_day = day_by_time_step[time_step]
                player_day.r = r
                player_day.uncertainty = uncertainty
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
