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
        self.w2 = (math.sqrt(config["w2"]) * math.log(10) / 400) ** 2
        self.initial_prior_wins = config["initial_prior_wins"]
        self.hessian_damping = config["hessian_damping"]
        self.days: list[PD.PlayerDay] = []
        self.draw_tendency: float = 0.0

    def log_likelihood(self) -> float:
        """Log-posterior contribution of this player.

        Sum of the per-day game log-likelihoods, the first-day anchor prior,
        and the Gaussian Wiener prior log-density over consecutive days. When
        ``draw_tendency > 0`` the game part uses the Davidson (win/draw/loss)
        formula instead of the plain Bradley-Terry win/loss one, so a drawn
        game's contribution isn't silently dropped (a drawn game never
        appears in ``won_games``/``lost_games``, so the BT ``log_likelihood``
        term would otherwise credit it as nothing at all).
        """
        result = 0.0
        if self.draw_tendency > 0.0:
            for day in self.days:
                result += day.davidson_log_likelihood(self.draw_tendency)
        else:
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
    ) -> tuple[list[float], list[float]]:
        """Computes the Hessian matrix for the log likelihood function.

        Args:
            days (list[PD.PlayerDay]): A list of PD.PlayerDay instances for the player.
            sigma2 (list[float]): A list of variance values between consecutive days.
            damping (float): Diagonal damping term subtracted for numerical stability
                (Coulom's HessianEpsilon).

        Returns:
            tuple[list[float], list[float]]: The diagonal and sub-diagonal
            elements of the Hessian matrix.
        """
        n = len(days)
        # Plain lists, not numpy: every consumer walks these one scalar at a time
        # (a tridiagonal recursion cannot be vectorised), so numpy only adds
        # per-element indexing cost -- measured 2-4.7x slower at every realistic
        # day count, for bit-identical results.
        diagonal = [0.0] * n
        sub_diagonal = [0.0] * (n - 1)
        for row in range(n):
            prior = 0.0
            if row < (n - 1):
                prior += -1 / sigma2[row]
            if row > 0:
                prior += -1 / sigma2[row - 1]
            player = days[row].player
            if player.draw_tendency > 0.0:
                _, game_hess = days[row].davidson_derivatives(player.draw_tendency)
            else:
                game_hess = days[row].log_likelihood_second_derivative()
            diagonal[row] = game_hess + prior - damping
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
            if self.draw_tendency > 0.0:
                game_grad, _ = day.davidson_derivatives(self.draw_tendency)
            else:
                game_grad = day.log_likelihood_derivative()
            term = game_grad + prior
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
        """Max absolute gradient component over this player's days.

        Clears each day's game-term cache first so the gradient reflects the
        current opponent gammas (including handicap/komi advantages updated
        later in the same iteration), rather than a value cached from before
        that update.
        """
        if not self.days:
            return 0.0
        for day in self.days:
            day.clear_game_terms_cache()
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
        a = [0.0] * n
        d = [0.0] * n
        b = [0.0] * n
        d[0] = diag[0]
        b[0] = sub_diag[0] if n > 1 else 0.0

        for i in range(1, n):
            a[i] = sub_diag[i - 1] / d[i - 1]
            d[i] = diag[i] - a[i] * b[i - 1]
            if i < n - 1:
                b[i] = sub_diag[i]

        y = [0.0] * n
        y[0] = g[0]
        for i in range(1, n):
            y[i] = g[i] - a[i] * y[i - 1]

        x = [0.0] * n
        x[n - 1] = y[n - 1] / d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - b[i] * x[i + 1]) / d[i]

        # Compute every new rating and validate them all before mutating any
        # day, so a mid-list non-finite value cannot leave earlier days already
        # updated (the update is all-or-nothing).
        new_rs = [float(day.r - x[idx]) for idx, day in enumerate(self.days)]
        for day, new_r in zip(self.days, new_rs, strict=True):
            if not math.isfinite(new_r):
                raise UnstableRatingException(
                    f"Non-finite rating for {self.name} on day {day.day}"
                )
        for day, new_r in zip(self.days, new_rs, strict=True):
            day.r = new_r

    def _banded_covariance(self) -> tuple[list[float], list[float]]:
        """The tridiagonal band of the posterior covariance, in nat² units.

        Returns ``(variances, adjacent)``: the exact diagonal of ``inv(-H)``, and
        the exact first off-diagonal (``adjacent[i]`` is the covariance of day
        ``i`` with day ``i+1``, so ``len(adjacent) == n - 1``). Coulom's
        forward/backward recursion gives both in O(n); nothing beyond the band is
        computed, and nothing beyond it is needed -- ``update_uncertainty`` reads
        only the diagonal, and ``WHR.rating_covariance`` inverts the Hessian
        densely when the full matrix is genuinely wanted.
        """
        sigma2 = self.compute_sigma2()
        diag, sub_diag = Player.hessian(self.days, sigma2, self.hessian_damping)
        n = len(self.days)

        a = [0.0] * n
        d = [0.0] * n
        b = [0.0] * n
        d[0] = diag[0]
        b[0] = sub_diag[0] if n > 1 else 0.0

        for i in range(1, n):
            a[i] = sub_diag[i - 1] / d[i - 1]
            d[i] = diag[i] - a[i] * b[i - 1]
            if i < n - 1:
                b[i] = sub_diag[i]

        dp = [0.0] * n
        dp[n - 1] = diag[n - 1]
        bp = [0.0] * n
        # The guard is on the validity of the INDEX n-2, not on the length of
        # sub_diag. `sub_diag.size >= 2` wrongly required two sub-diagonal
        # entries to read entry n-2 == 0, so a player with exactly two rated days
        # got bp[1] = 0 and a first-day variance ~25x too small (17 elo reported
        # against a true 90). n >= 3 was unaffected, since there size == n-1 >= 2.
        bp[n - 1] = sub_diag[n - 2] if n >= 2 else 0
        ap = [0.0] * n
        for i in range(n - 2, -1, -1):
            ap[i] = sub_diag[i] / dp[i + 1]
            dp[i] = diag[i] - ap[i] * bp[i + 1]
            if i > 0:
                bp[i] = sub_diag[i - 1]

        v = [0.0] * n
        for i in range(n - 1):
            v[i] = dp[i + 1] / (b[i] * bp[i + 1] - d[i] * dp[i + 1])
        v[n - 1] = -1 / d[n - 1]

        # cov(day i, day i+1), from the same recursion
        adjacent = [-a[i] * v[i] for i in range(1, n)]
        return v, adjacent

    def covariance(self) -> npt.NDArray[np.float64]:
        """The tridiagonal band of this player's posterior covariance, in nat².

        **This is a band, not the full covariance matrix.** The true ``inv(-H)``
        is dense: consecutive days are correlated through the Wiener prior, and so
        are distant ones, just weakly. What this returns is exact on the diagonal
        and on the two adjacent off-diagonals, and **zero everywhere else** -- the
        far entries are not computed, not zero. Reading them as covariances would
        understate every non-adjacent correlation.

        Two consequences worth knowing:

        * Until 3.5.0 the matrix was also *asymmetric*: the super-diagonal was
          filled and the sub-diagonal left at zero. It is symmetric now.
        * For a genuine dense covariance in **elo²**, use
          ``WHR.rating_covariance(name)``, which inverts the Hessian properly.
          That is what ``WHR.rating_change`` consumes.

        Returns:
            An ``n x n`` array holding the exact tridiagonal band of the posterior
            covariance in natural log-gamma units, zero outside it.
        """
        v, adjacent = self._banded_covariance()
        n = len(v)
        mat = np.zeros((n, n))
        np.fill_diagonal(mat, v)
        if n > 1:
            idx = np.arange(n - 1)
            mat[idx, idx + 1] = adjacent
            mat[idx + 1, idx] = adjacent
        return mat

    def update_uncertainty(self) -> None:
        """Updates the uncertainty measure for each day based on the covariance matrix.

        For each day the variance is read from the diagonal of the covariance
        matrix and stored as that day's uncertainty. Players with no recorded
        day are left untouched.

        The per-day game-term caches are cleared first, for the same reason
        ``gradient_infinity_norm`` does it: a cache populated during this
        player's own Newton step holds opponent gammas from *before* the
        opponents were updated later in the same iteration. Reading it left the
        stored variance a whisker off the true posterior variance (~2e-5
        relative, i.e. well under a thousandth of an elo, but needlessly
        inexact).
        """
        if len(self.days) == 0:
            return
        for day in self.days:
            day.clear_game_terms_cache()
        # Only the diagonal is wanted, so take the O(n) band rather than building
        # the n x n matrix: that used to run an n^2 *Python* double loop, which
        # cost 0.48 s per pass over the 37 NBA teams (451 rated days each) against
        # 3.94 s for fifty whole iterations.
        variances, _adjacent = self._banded_covariance()
        for day, variance in zip(self.days, variances, strict=True):
            day.uncertainty = float(variance)

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
