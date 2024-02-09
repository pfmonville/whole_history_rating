from __future__ import annotations

import sys
from typing import Any

from whr import player as P
from whr import playerday as PD


class Game:
    def __init__(
        self,
        black: P.Player,
        white: P.Player,
        winner: str,
        time_step: int,
        handicap: float = 0,
        extras: dict[str, Any] | None = None,
    ):
        self.day = time_step
        self.white_player = white
        self.black_player = black
        self.winner = winner.upper()
        self.handicap = handicap
        self.handicap_proc = handicap
        self.bpd: PD.PlayerDay | None = None
        self.wpd: PD.PlayerDay | None = None
        if extras is None:
            self.extras = {"komi": 6.5}
        else:
            self.extras = extras
            self.extras.setdefault("komi", 6.5)

    def __str__(self) -> str:
        return f"W:{self.white_player.name}(r={self.wpd.r if self.wpd is not None else '?'}) B:{self.black_player.name}(r={self.bpd.r if self.bpd is not None else '?'}) winner = {self.winner}, komi = {self.extras['komi']}, handicap = {self.handicap}"

    def opponents_adjusted_gamma(self, player: P.Player) -> float:
        """
        Calculates the adjusted gamma value of a player's opponent. This is based on the opponent's
        Elo rating adjusted for the game's handicap.

        Parameters:
            player (P.Player): The player for whom to calculate the opponent's adjusted gamma.

        Returns:
            float: The adjusted gamma value of the opponent.

        Raises:
            AttributeError: If the player days are not set or the player is not part of the game.
        """
        if self.bpd is None or self.wpd is None:
            raise AttributeError("black player day and white player day must be set")
        if player == self.white_player:
            opponent_elo = self.bpd.elo + self.handicap
        elif player == self.black_player:
            opponent_elo = self.wpd.elo - self.handicap
        else:
            raise (
                AttributeError(
                    f"No opponent for {player.__str__()}, since they're not in this game: {self.__str__()}."
                )
            )
        rval = 10 ** (opponent_elo / 400.0)
        if rval == 0 or rval > sys.maxsize:
            raise AttributeError("bad adjusted gamma")
        return rval

    def opponent(self, player: P.Player) -> P.Player:
        """
        Returns the opponent of the specified player in this game.

        Parameters:
            player (P.Player): The player whose opponent is to be found.

        Returns:
            P.Player: The opponent player.
        """
        if player == self.white_player:
            return self.black_player
        return self.white_player

    def prediction_score(self) -> float:
        """
        Calculates the accuracy of the prediction for the game's outcome.
        Returns a score based on the actual outcome compared to the predicted probabilities:
        - Returns 1.0 if the prediction matches the actual outcome (white or black winning as predicted).
        - Returns 0.5 if the win probability is exactly 0.5, indicating uncertainty.
        - Returns 0.0 if the prediction does not match the actual outcome.

        Returns:
            float: The prediction score of the game.
        """
        if self.white_win_probability() == 0.5:
            return 0.5
        return (
            1.0
            if (
                (self.winner == "W" and self.white_win_probability() > 0.5)
                or (self.winner == "B" and self.white_win_probability() < 0.5)
            )
            else 0.0
        )

    def white_win_probability(self) -> float:
        """
        Calculates the win probability for the white player based on their gamma value and
        the adjusted gamma value of their opponent.

        Returns:
            float: The win probability for the white player.

        Raises:
            AttributeError: If the white player day is not set.
        """
        if self.wpd is None:
            raise AttributeError("white player day must be set")

        return self.wpd.gamma() / (
            self.wpd.gamma() + self.opponents_adjusted_gamma(self.white_player)
        )

    def black_win_probability(self) -> float:
        """
        Calculates the win probability for the black player based on their gamma value and
        the adjusted gamma value of their opponent.

        Returns:
            float: The win probability for the black player.

        Raises:
            AttributeError: If the black player day is not set.
        """
        if self.bpd is None:
            raise AttributeError("black player day must be set")
        return self.bpd.gamma() / (
            self.bpd.gamma() + self.opponents_adjusted_gamma(self.black_player)
        )
