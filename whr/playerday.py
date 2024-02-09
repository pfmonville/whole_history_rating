from __future__ import annotations

import math
import sys

from whr import player as P
from whr import game as G


class PlayerDay:
    def __init__(self, player: P.Player, day: int):
        self.day = day
        self.player = player
        self.is_first_day = False
        self.won_games = []
        self.lost_games = []
        self._won_game_terms = None
        self._lost_game_terms = None
        self.uncertainty: float = -1

    def set_gamma(self, value: float) -> None:
        """Sets the player's performance rating (gamma) for this day.

        Args:
            value (float): The new gamma value.
        """
        self.r = math.log(value)

    def gamma(self) -> float:
        """Calculates the player's performance rating (gamma) based on their rating.

        Returns:
            float: The player's gamma value.
        """
        return math.exp(self.r)

    @property
    def elo(self) -> float:
        """Calculates the ELO rating from the player's gamma value.

        Returns:
            float: The ELO rating.
        """
        return (self.r * 400) / (math.log(10))

    @elo.setter
    def elo(self, value: float) -> None:
        """Sets the player's ELO rating, adjusting their internal rating accordingly.

        Args:
            value (float): The new ELO rating.
        """
        self.r = value * (math.log(10) / 400)

    def clear_game_terms_cache(self) -> None:
        """Clears the cached terms for games won and lost, forcing recalculation."""
        self._won_game_terms = None
        self._lost_game_terms = None

    def won_game_terms(self) -> list[list[float]]:
        """Calculates terms for games won by the player on this day.

        Returns:
            list[list[float]]: A list of terms used for calculations, including the opponent's adjusted gamma.
        """
        if self._won_game_terms is None:
            self._won_game_terms = []
            for g in self.won_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                if other_gamma == 0 or other_gamma is None or other_gamma > sys.maxsize:
                    print(
                        f"other_gamma ({g.opponent(self.player).__str__()}) = {other_gamma}"
                    )
                self._won_game_terms.append([1.0, 0.0, 1.0, other_gamma])
            if self.is_first_day:
                # win against virtual player ranked with gamma = 1.0
                self._won_game_terms.append([1.0, 0.0, 1.0, 1.0])
        return self._won_game_terms

    def lost_game_terms(self) -> list[list[float]]:
        """Calculates terms for games lost by the player on this day.

        Returns:
            list[list[float]]: A list of terms used for calculations, including the opponent's adjusted gamma.
        """
        if self._lost_game_terms is None:
            self._lost_game_terms = []
            for g in self.lost_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                if other_gamma == 0 or other_gamma is None or other_gamma > sys.maxsize:
                    print(
                        f"other_gamma ({g.opponent(self.player).__str__()}) = {other_gamma}"
                    )
                self._lost_game_terms.append([0.0, other_gamma, 1.0, other_gamma])
            if self.is_first_day:
                # win against virtual player ranked with gamma = 1.0
                self._lost_game_terms.append([0.0, 1.0, 1.0, 1.0])
        return self._lost_game_terms

    def log_likelihood_second_derivative(self) -> float:
        """Calculates the second derivative of the log likelihood of the player's rating.

        Returns:
            float: The second derivative of the log likelihood.
        """
        result = 0.0
        for _, _, c, d in self.won_game_terms() + self.lost_game_terms():
            result += (c * d) / ((c * self.gamma() + d) ** 2.0)
        return -1 * self.gamma() * result

    def log_likelihood_derivative(self) -> float:
        """Calculates the derivative of the log likelihood of the player's rating.

        Returns:
            float: The derivative of the log likelihood.
        """
        tally = 0.0
        for _, _, c, d in self.won_game_terms() + self.lost_game_terms():
            tally += c / (c * self.gamma() + d)
        return len(self.won_game_terms()) - self.gamma() * tally

    def log_likelihood(self) -> float:
        """Calculates the log likelihood of the player's rating based on games played.

        Returns:
            float: The log likelihood.
        """
        tally = 0.0
        for a, b, c, d in self.won_game_terms():
            tally += math.log(a * self.gamma())
            tally -= math.log(c * self.gamma() + d)
        for a, b, c, d in self.lost_game_terms():
            tally += math.log(b)
            tally -= math.log(c * self.gamma() + d)
        return tally

    def add_game(self, game: G.Game) -> None:
        """Adds a game to this player's record, categorizing it as won or lost.

        Args:
            game (G.Game): The game to add.
        """
        if (game.winner == "W" and game.white_player == self.player) or (
            game.winner == "B" and game.black_player == self.player
        ):
            self.won_games.append(game)
        else:
            self.lost_games.append(game)

    def update_by_1d_newtons_method(self) -> None:
        """Updates the player's rating using one-dimensional Newton's method."""
        dlogp = self.log_likelihood_derivative()
        d2logp = self.log_likelihood_second_derivative()
        dr = dlogp / d2logp
        new_r = self.r - dr
        self.r = new_r
