from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from whr import game as G
from whr import playerday as PD
from whr.utils import UnstableRatingException


class Player:
    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.debug = config["debug"]
        self.w2 = (math.sqrt(config["w2"]) * math.log(10) / 400) ** 2
        self.initial_prior_wins = config["initial_prior_wins"]
        self.hessian_damping = config["hessian_damping"]
        self.days: list[PD.PlayerDay] = []

    def log_likelihood(self) -> float:
        """Log-posterior contribution of this player.

        Sum of the per-day game log-likelihoods, the first-day anchor prior,
        and the Gaussian Wiener prior log-density over consecutive days.
        """
        result = 0.0
        for day in self.days:
            result += day.log_likelihood()
        if self.days:
            result += self.days[0].anchor_log_likelihood()
        sigma2 = self.compute_sigma2()
        for i, s2 in enumerate(sigma2):
            rd = self.days[i + 1].r - self.days[i].r
            result += -(rd**2) / (2 * s2) - 0.5 * math.log(2 * math.pi * s2)
        return result

    @staticmethod
    def hessian(
        days: list[PD.PlayerDay], sigma2: list[float], damping: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Computes the Hessian matrix for the log likelihood function.

        Args:
            days (list[PD.PlayerDay]): A list of PD.PlayerDay instances for the player.
            sigma2 (list[float]): A list of variance values between consecutive days.
            damping (float): Diagonal damping term subtracted for numerical stability
                (Coulom's HessianEpsilon).

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A tuple containing the diagonal and sub-diagonal elements of the Hessian matrix.
        """
        n = len(days)
        diagonal = np.zeros((n,))
        sub_diagonal = np.zeros((n - 1,))
        for row in range(n):
            prior = 0.0
            if row < (n - 1):
                prior += -1 / sigma2[row]
            if row > 0:
                prior += -1 / sigma2[row - 1]
            diagonal[row] = (
                days[row].log_likelihood_second_derivative() + prior - damping
            )
        diagonal[0] += days[0].anchor_hessian()
        for i in range(n - 1):
            sub_diagonal[i] = 1 / sigma2[i]
        return (diagonal, sub_diagonal)

    def gradient(
        self, r: list[float], days: list[PD.PlayerDay], sigma2: list[float]
    ) -> list[float]:
        """Computes the gradient of the log likelihood function.

        Args:
            r (list[float]): A list of rating values for the player on different days.
            days (list[PD.PlayerDay]): A list of PD.PlayerDay instances for the player.
            sigma2 (list[float]): A list of variance values between consecutive days.

        Returns:
            list[float]: A list containing the gradient of the log likelihood function.
        """
        g = []
        n = len(days)
        for idx, day in enumerate(days):
            prior = 0.0
            if idx < (n - 1):
                prior += -(r[idx] - r[idx + 1]) / sigma2[idx]
            if idx > 0:
                prior += -(r[idx] - r[idx - 1]) / sigma2[idx - 1]
            term = day.log_likelihood_derivative() + prior
            if idx == 0:
                term += day.anchor_gradient()
            g.append(term)
        return g

    def run_one_newton_iteration(self) -> None:
        """Runs a single iteration of Newton's method to update player ratings."""
        for day in self.days:
            day.clear_game_terms_cache()
        if len(self.days) == 1:
            self.days[0].update_by_1d_newtons_method()
        elif len(self.days) > 1:
            self.update_by_ndim_newton()

    def gradient_infinity_norm(self) -> float:
        """Max absolute gradient component over this player's days."""
        if not self.days:
            return 0.0
        r = [d.r for d in self.days]
        sigma2 = self.compute_sigma2()
        return max(abs(gi) for gi in self.gradient(r, self.days, sigma2))

    def compute_sigma2(self) -> list[float]:
        """Computes the variance values used as the prior for rating changes.

        Returns:
            list[float]: A list of variance values between consecutive rating days.
        """
        sigma2 = []
        for d1, d2 in zip(self.days, self.days[1:], strict=False):
            sigma2.append(abs(d2.day - d1.day) * self.w2)
        return sigma2

    def update_by_ndim_newton(self) -> None:
        """Updates the player's ratings using a multidimensional Newton-Raphson method."""
        # r
        r = [d.r for d in self.days]

        # sigma squared (used in the prior)
        sigma2 = self.compute_sigma2()

        diag, sub_diag = Player.hessian(self.days, sigma2, self.hessian_damping)
        g = self.gradient(r, self.days, sigma2)
        n = len(r)
        a = np.zeros((n,))
        d = np.zeros((n,))
        b = np.zeros((n,))
        d[0] = diag[0]
        b[0] = sub_diag[0] if sub_diag.size > 0 else 0

        for i in range(1, n):
            a[i] = sub_diag[i - 1] / d[i - 1]
            d[i] = diag[i] - a[i] * b[i - 1]
            if i < n - 1:
                b[i] = sub_diag[i]

        y = np.zeros((n,))
        y[0] = g[0]
        for i in range(1, n):
            y[i] = g[i] - a[i] * y[i - 1]

        x = np.zeros((n,))
        x[n - 1] = y[n - 1] / d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - b[i] * x[i + 1]) / d[i]

        for idx, day in enumerate(self.days):
            new_r = float(day.r - x[idx])
            if not math.isfinite(new_r):
                raise UnstableRatingException(
                    f"Non-finite rating for {self.name} on day {day.day}"
                )
            day.r = new_r

    def covariance(self) -> npt.NDArray[np.float64]:
        """Computes the covariance matrix of the player's rating estimations.

        Returns:
            The covariance matrix for the player's ratings.
        """
        r = [d.r for d in self.days]

        sigma2 = self.compute_sigma2()
        diag, sub_diag = Player.hessian(self.days, sigma2, self.hessian_damping)
        n = len(r)

        a = np.zeros((n,))
        d = np.zeros((n,))
        b = np.zeros((n,))
        d[0] = diag[0]
        b[0] = sub_diag[0] if sub_diag.size > 0 else 0

        for i in range(1, n):
            a[i] = sub_diag[i - 1] / d[i - 1]
            d[i] = diag[i] - a[i] * b[i - 1]
            if i < n - 1:
                b[i] = sub_diag[i]

        dp = np.zeros((n,))
        dp[n - 1] = diag[n - 1]
        bp = np.zeros((n,))
        bp[n - 1] = sub_diag[n - 2] if sub_diag.size >= 2 else 0
        ap = np.zeros((n,))
        for i in range(n - 2, -1, -1):
            ap[i] = sub_diag[i] / dp[i + 1]
            dp[i] = diag[i] - ap[i] * bp[i + 1]
            if i > 0:
                bp[i] = sub_diag[i - 1]

        v = np.zeros((n,))
        for i in range(n - 1):
            v[i] = dp[i + 1] / (b[i] * bp[i + 1] - d[i] * dp[i + 1])
        v[n - 1] = -1 / d[n - 1]

        mat = np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                if row == col:
                    mat[row, col] = v[row]
                elif row == col - 1:
                    mat[row, col] = -1 * a[col] * v[col]
                else:
                    mat[row, col] = 0

        return mat

    def update_uncertainty(self) -> None:
        """Updates the uncertainty measure for each day based on the covariance matrix.

        For each day the variance is read from the diagonal of the covariance
        matrix and stored as that day's uncertainty. Players with no recorded
        day are left untouched.
        """
        if len(self.days) == 0:
            return
        c = self.covariance()
        u = [c[i, i] for i in range(len(self.days))]  # u = variance
        for i, d in enumerate(self.days):
            d.uncertainty = float(u[i])

    def add_game(self, game: G.Game) -> None:
        """Adds a game to the player's record, updating or creating a new PD.PlayerDay instance as necessary.

        Args:
            game (G.Game): The game to add to the player's record.
        """
        all_days = [x.day for x in self.days]
        if game.day not in all_days:
            day_index = bisect.bisect_right(all_days, game.day)
            new_pday = PD.PlayerDay(self, game.day)
            if len(self.days) == 0:
                new_pday.is_first_day = True
                new_pday.set_gamma(1)
            else:
                # still not perfect because gamma of day index can more farther if more games were not added in order
                new_pday.set_gamma(self.days[day_index - 1].gamma())
            self.days.insert(day_index, new_pday)
        else:
            day_index = all_days.index(game.day)
        if game.white_player == self:
            game.wpd = self.days[day_index]
        else:
            game.bpd = self.days[day_index]
        self.days[day_index].add_game(game)
