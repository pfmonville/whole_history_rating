from __future__ import annotations

import math
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
        handicap_gamma: dict[Any, float] | None = None,
        komi_gamma: dict[Any, float] | None = None,
    ):
        self.day = time_step
        self.white_player = white
        self.black_player = black
        self.winner = winner.upper()
        self.handicap = handicap
        self.handicap_gamma = handicap_gamma
        self.komi_gamma = komi_gamma
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
        """Opponent's gamma folding in the handicap/komi advantages.

        With handicap boosting black (γ_h) and komi boosting white (γ_k):
        the opponent of white is black with effective gamma γ_b·γ_h/γ_k, and
        the opponent of black is white with effective gamma γ_w·γ_k/γ_h. When
        the tables are absent (direct construction) advantages are 1.
        """
        if self.bpd is None or self.wpd is None:
            raise AttributeError("black player day and white player day must be set")
        gh = (
            1.0
            if self.handicap_gamma is None
            else self.handicap_gamma.get(self.handicap, 1.0)
        )
        gk = (
            1.0
            if self.komi_gamma is None
            else self.komi_gamma.get(self.extras["komi"], 1.0)
        )
        # Resolve the opponent first: a player who isn't in this game must raise
        # the specific "No opponent" error rather than being pre-empted by the
        # gamma-divisor guard below. ``numerator``/``denominator`` are the
        # opponent's gamma folded with the advantage gammas (γ_b·γ_h/γ_k for
        # white's opponent, γ_w·γ_k/γ_h for black's).
        if player == self.white_player:
            numerator, denominator = self.bpd.gamma() * gh, gk
        elif player == self.black_player:
            numerator, denominator = self.wpd.gamma() * gk, gh
        else:
            raise AttributeError(
                f"No opponent for {player.__str__()}, since they're not in this game: {self.__str__()}."
            )
        # Validate the divisors before dividing: an underflowed/non-positive
        # gamma (e.g. from an extreme pinned elo value) must raise the
        # intended AttributeError below rather than a raw ZeroDivisionError.
        if not math.isfinite(gh) or gh <= 0 or not math.isfinite(gk) or gk <= 0:
            raise AttributeError("bad adjusted gamma")
        rval = numerator / denominator
        if not math.isfinite(rval) or rval <= 0:
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
