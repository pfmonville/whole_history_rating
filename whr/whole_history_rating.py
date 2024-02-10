from __future__ import annotations

import time
import ast
import pickle
from typing import Any

from whr.utils import test_stability
from whr.player import Player
from whr.game import Game


class Base:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config if config is not None else {}
        self.config.setdefault("debug", False)
        self.config.setdefault("w2", 300.0)
        self.config.setdefault("uncased", False)
        self.games = []
        self.players = {}

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
    ) -> list[list[float]]:
        """Retrieves all ratings for each player (for each of their playing days), ordered.

        Args:
            current (bool, optional): If True, retrieves only the latest elo rating estimation. If False, retrieves all elo rating estimations for each day played.
            compact (bool, optional): If True, returns only a list of elo ratings. If False, includes the player's name before their elo ratings.

        Returns:
            list[list[float]]: A list containing the elo ratings for each player and each of their playing days.
        """
        result = []
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

    def ratings_for_player(
        self, name, current: bool = False
    ) -> list[tuple[int, float, float]] | tuple[float, float]:
        """Retrieves all ratings for each day played by the specified player.

        Args:
            name (str): The name of the player.
            current (bool, optional): If True, retrieves only the latest elo rating and uncertainty. If False, retrieves all elo ratings and uncertainties for each day played.

        Returns:
            list[tuple[int, float, float]] | tuple[float, float]: For each day, includes the time step, the elo rating, and the uncertainty if current is False, else just return the elo and uncertainty of the last day
        """
        if self.config["uncased"]:
            name = name.lower()
        player = self.player_by_name(name)
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
            print("Bad game")
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

    def auto_iterate(
        self,
        time_limit: int | None = None,
        precision: float = 1e-3,
        batch_size: int = 10,
    ) -> tuple[int, bool]:
        """Automatically iterates until the algorithm converges or reaches the time limit.

        Args:
            time_limit (int | None, optional): The maximum time, in seconds, after which no more iterations will be launched. If None, no timeout is set
            precision (float, optional): The desired precision of stability.
            batch_size (int, optional): The number of iterations to perform at each step, with precision and timeout checks after each batch.

        Returns:
            tuple[int, bool]: The number of iterations performed and a boolean indicating whether stability was reached.
        """
        start = time.time()
        a = None
        i = 0
        while True:
            self.iterate(batch_size)
            i += batch_size
            b = self.get_ordered_ratings(compact=True)
            if a is not None and test_stability(a, b, precision):
                return i, True
            if time_limit is not None and time.time() - start > time_limit:
                return i, False
            a = b

    def probability_future_match(
        self, name1: str, name2: str, handicap: float = 0
    ) -> tuple[float, float]:
        """Calculates the winning probability for a hypothetical match between two players.

        Args:
            name1 (str): The name of the first player.
            name2 (str): The name of the second player.
            handicap (float, optional): The handicap (in elo points).

        Returns:
            tuple[float, float]: The winning probabilities for name1 and name2, respectively, as percentages rounded to the second decimal.

        Raises:
            AttributeError: Raised if name1 and name2 are equal
        """
        # Avoid self-played games (no info)
        if self.config["uncased"]:
            name1 = name1.lower()
            name2 = name2.lower()
        if name1 == name2:
            raise AttributeError("Invalid game (black == white)")
        player1 = self.player_by_name(name1)
        player2 = self.player_by_name(name2)
        bpd_gamma = 1
        bpd_elo = 0
        wpd_gamma = 1
        wpd_elo = 0
        if len(player1.days) > 0:
            bpd = player1.days[-1]
            bpd_gamma = bpd.gamma()
            bpd_elo = bpd.elo
        if len(player2.days) != 0:
            wpd = player2.days[-1]
            wpd_gamma = wpd.gamma()
            wpd_elo = wpd.elo
        player1_proba = bpd_gamma / (bpd_gamma + 10 ** ((wpd_elo - handicap) / 400.0))
        player2_proba = wpd_gamma / (wpd_gamma + 10 ** ((bpd_elo + handicap) / 400.0))
        print(
            f"win probability: {name1}:{player1_proba*100:.2f}%; {name2}:{player2_proba*100:.2f}%"
        )
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
                        )

            if len(rest) == 2:
                try:
                    handicap = int(rest[0])
                except ValueError:
                    raise ValueError(f"Invalid handicap value in: '{line}'")
                try:
                    extras = ast.literal_eval(rest[1])
                    if not isinstance(extras, dict):
                        raise ValueError()
                except (ValueError, SyntaxError):
                    raise ValueError(f"Invalid extras dictionary in: '{line}'")

            if self.config["uncased"]:
                black, white = black.lower(), white.lower()

            self.create_game(black, white, winner, int(time_step), handicap, extras)

    def save_base(self, path: str) -> None:
        """Saves the current state of the base to a specified path.

        Args:
            path (str): The path where the base will be saved.
        """
        try:
            pickle.dump([self.players, self.games, self.config], open(path, "wb"))
        except pickle.PicklingError:
            pickle.dump(
                [
                    self.players,
                    self.games,
                    {
                        k: v
                        for k, v in self.config.items()
                        if k in ["w2", "debug", "uncased"]
                    },
                ],
                open(path, "wb"),
            )
            print(
                "WARNING: some elements in self.config you configured can't be pickled, only 'w2', 'debug' and 'uncased' parameters will be saved for self.config"
            )

    @staticmethod
    def load_base(path: str) -> Base:
        """Loads a saved base from a specified path.

        Args:
            path (str): The path to the saved base.

        Returns:
            Base: The loaded base.
        """
        players, games, config = pickle.load(open(path, "rb"))
        result = Base()
        result.config, result.games, result.players = config, games, players
        return result
