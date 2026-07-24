from __future__ import annotations

import math

from whr import game as G
from whr import player as P


class PlayerDay:
    def __init__(self, player: P.Player, day: int):
        self.day = day
        self.player = player
        self.won_games: list[G.Game] = []
        self.lost_games: list[G.Game] = []
        self.drawn_games: list[G.Game] = []
        self._won_game_terms: list[list[float]] | None = None
        self._lost_game_terms: list[list[float]] | None = None
        self.uncertainty: float = -1
        # natural-rating (log-gamma); overwritten by set_gamma / the elo setter
        self.r: float = 0.0

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
                # opponents_adjusted_gamma already raises on a non-finite gamma.
                other_gamma = g.opponents_adjusted_gamma(self.player)
                self._won_game_terms.append([1.0, 0.0, 1.0, other_gamma])
        return self._won_game_terms

    def lost_game_terms(self) -> list[list[float]]:
        """Calculates terms for games lost by the player on this day.

        Returns:
            list[list[float]]: A list of terms used for calculations, including the opponent's adjusted gamma.
        """
        if self._lost_game_terms is None:
            self._lost_game_terms = []
            for g in self.lost_games:
                # opponents_adjusted_gamma already raises on a non-finite gamma.
                other_gamma = g.opponents_adjusted_gamma(self.player)
                self._lost_game_terms.append([0.0, other_gamma, 1.0, other_gamma])
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
        for a, _b, c, d in self.won_game_terms():
            tally += math.log(a * self.gamma())
            tally -= math.log(c * self.gamma() + d)
        for _a, b, c, d in self.lost_game_terms():
            tally += math.log(b)
            tally -= math.log(c * self.gamma() + d)
        return tally

    def anchor_gradient(self) -> float:
        """First-day Bradley-Terry prior gradient (Coulom's InitialPriorWins)."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return k * (1.0 - 2.0 * gamma / (1.0 + gamma))

    def anchor_hessian(self) -> float:
        """Second derivative of the first-day prior."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return -2.0 * k * gamma / ((1.0 + gamma) ** 2)

    def anchor_log_likelihood(self) -> float:
        """Log-likelihood contribution of the first-day prior."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return k * (math.log(gamma) - 2.0 * math.log(1.0 + gamma))

    def add_game(self, game: G.Game) -> None:
        """Adds a game to this player's record, categorizing it as won or lost.

        Args:
            game (G.Game): The game to add.
        """
        if game.winner == "D":
            self.drawn_games.append(game)
        elif (game.winner == "W" and game.white_player == self.player) or (
            game.winner == "B" and game.black_player == self.player
        ):
            self.won_games.append(game)
        else:
            self.lost_games.append(game)

    def _weighted_games(self):
        """Yield (game, outcome_weight): 1.0 won, 0.0 lost, 0.5 drawn."""
        for g in self.won_games:
            yield g, 1.0
        for g in self.lost_games:
            yield g, 0.0
        for g in self.drawn_games:
            yield g, 0.5

    def davidson_log_likelihood(self, nu: float) -> float:
        """This day's game log-likelihood under the Davidson win/draw/loss
        model, mirroring ``_weighted_games``/``davidson_derivatives``.

        Sum over the day's games of ``log(num) - log(Z)``, where ``S, O =
        game.effective_gammas(self.player)``, ``T = nu*sqrt(S*O)``, ``Z = S +
        O + T``, and ``num`` is ``S`` (player won), ``O`` (player lost), or
        ``T`` (drawn). Reduces exactly to ``log_likelihood()`` at ``nu=0``
        (see ``test_davidson_log_likelihood_matches_bt_at_nu_zero``).
        """
        total = 0.0
        for game, weight in self._weighted_games():
            s, o = game.effective_gammas(self.player)
            t = nu * math.sqrt(s * o)
            z = s + o + t
            if weight == 1.0:
                num = s
            elif weight == 0.0:
                num = o
            else:
                num = t
            total += math.log(num) - math.log(z)
        return total

    def davidson_derivatives(self, nu: float) -> tuple[float, float]:
        """(gradient, Hessian) of this day's game log-likelihood under Davidson.

        Reduces to the plain Bradley-Terry win/loss derivatives at nu=0.
        """
        gradient = 0.0
        hessian = 0.0
        for game, weight in self._weighted_games():
            s, o = game.effective_gammas(self.player)
            t = nu * math.sqrt(s * o)
            z = s + o + t
            n = s + t / 2.0
            n_prime = s + t / 4.0
            ratio = n / z
            gradient += weight - ratio
            hessian += ratio * ratio - n_prime / z
        return gradient, hessian

    def update_by_1d_newtons_method(self) -> None:
        """Updates the player's rating using one-dimensional Newton's method.

        The Hessian damping (Coulom's ``HessianEpsilon``) is subtracted from the
        second derivative exactly as ``Player.hessian`` does for multi-day
        players, so a single-day player is damped consistently with the
        covariance computation that reads back through ``Player.hessian``.
        """
        if self.player.draw_tendency > 0.0:
            game_grad, game_hess = self.davidson_derivatives(self.player.draw_tendency)
        else:
            game_grad = self.log_likelihood_derivative()
            game_hess = self.log_likelihood_second_derivative()
        dlogp = game_grad + self.anchor_gradient()
        d2logp = game_hess + self.anchor_hessian() - self.player.hessian_damping
        dr = dlogp / d2logp
        self.r = self.r - dr
