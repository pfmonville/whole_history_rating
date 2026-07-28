"""The maths, checked against finite differences rather than against itself.

Every analytic derivative in the model is compared to a numerical derivative of
the very log-likelihood it claims to differentiate, and every linear-algebra
shortcut (the tridiagonal solve, the banded inverse) to a dense reference. These
are the tests that would have caught the two defects fixed alongside them:

* the covariance backward pass mis-guarded its first index, so a player with
  exactly two rated days got a first-day variance ~25x too small;
* handicap/komi advantages were accumulated from decisive games only, so with
  draws present the fit was not a maximum of the likelihood at all.
"""

import math

import numpy as np
import pytest

from whr.player import Player
from whr.whole_history_rating import WHR

FD_H = 1e-6  # central-difference step for first derivatives
FD_H2 = 1e-4  # ...and for second derivatives


def _base(draws=False, handicap=False, komi=False, iters=80, seed=3, names="abcd"):
    import random

    random.seed(seed)
    w = WHR({"w2": 300})
    people = list(names)
    for day in range(1, 13):
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                w.create_game(
                    people[i],
                    people[j],
                    random.choice(["B", "W", "D"] if draws else ["B", "W"]),
                    day,
                    random.choice([0, 2]) if handicap else 0,
                    komi=random.choice([6.5, 7.5]) if komi else None,
                )
    w.iterate(iters)
    return w


def _clear(w):
    for p in w.players.values():
        for d in p.days:
            d.clear_game_terms_cache()


def _dense_hessian(p):
    sigma2 = p.compute_sigma2()
    diag, sub = Player.hessian(p.days, sigma2, p.hessian_damping)
    n = len(diag)
    h = np.zeros((n, n))
    for i in range(n):
        h[i, i] = diag[i]
    for i in range(n - 1):
        h[i, i + 1] = sub[i]
        h[i + 1, i] = sub[i]
    return h


# --------------------------------------------------------------------------- #
# player-day derivatives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"draws": True},
        {"handicap": True},
        {"draws": True, "handicap": True, "komi": True},
    ],
)
def test_player_gradient_matches_finite_differences(kwargs):
    w = _base(**kwargs)
    for p in w.players.values():
        _clear(w)
        analytic = p.gradient([d.r for d in p.days], p.days, p.compute_sigma2())
        for i, day in enumerate(p.days):
            r0 = day.r

            def at(value, day=day, p=p):
                day.r = value
                for d in p.days:
                    d.clear_game_terms_cache()
                return p.log_likelihood()

            numeric = (at(r0 + FD_H) - at(r0 - FD_H)) / (2 * FD_H)
            at(r0)
            assert analytic[i] == pytest.approx(numeric, abs=1e-6)


@pytest.mark.parametrize("kwargs", [{}, {"draws": True}])
def test_player_hessian_matches_finite_differences(kwargs):
    w = _base(**kwargs)
    for p in w.players.values():
        _clear(w)
        sigma2 = p.compute_sigma2()
        diag, sub = Player.hessian(p.days, sigma2, p.hessian_damping)
        for i, day in enumerate(p.days):
            r0 = day.r

            def at(value, day=day, p=p):
                day.r = value
                for d in p.days:
                    d.clear_game_terms_cache()
                return p.log_likelihood()

            numeric = (at(r0 + FD_H2) - 2 * at(r0) + at(r0 - FD_H2)) / FD_H2**2
            at(r0)
            # the stored diagonal carries -hessian_damping, which the
            # log-likelihood itself does not
            assert diag[i] + p.hessian_damping == pytest.approx(numeric, abs=1e-4)
        for i in range(len(p.days) - 1):
            di, dj = p.days[i], p.days[i + 1]
            ri, rj = di.r, dj.r

            def at2(a, b, di=di, dj=dj, p=p, ri=ri, rj=rj):
                di.r, dj.r = ri + a, rj + b
                for d in p.days:
                    d.clear_game_terms_cache()
                return p.log_likelihood()

            mixed = (
                at2(FD_H2, FD_H2)
                - at2(FD_H2, -FD_H2)
                - at2(-FD_H2, FD_H2)
                + at2(-FD_H2, -FD_H2)
            ) / (4 * FD_H2**2)
            at2(0.0, 0.0)
            assert sub[i] == pytest.approx(mixed, abs=1e-4)


def test_first_day_anchor_derivatives_match_its_own_log_likelihood():
    w = _base()
    day = w.player_by_name("a").days[0]
    r0 = day.r

    def at(value):
        day.r = value
        day.clear_game_terms_cache()
        return day.anchor_log_likelihood()

    grad = (at(r0 + FD_H) - at(r0 - FD_H)) / (2 * FD_H)
    hess = (at(r0 + FD_H2) - 2 * at(r0) + at(r0 - FD_H2)) / FD_H2**2
    at(r0)
    assert day.anchor_gradient() == pytest.approx(grad, abs=1e-7)
    assert day.anchor_hessian() == pytest.approx(hess, abs=1e-4)


# --------------------------------------------------------------------------- #
# linear algebra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [{}, {"draws": True}])
def test_tridiagonal_newton_step_equals_a_dense_solve(kwargs):
    """The Thomas recursion must reproduce -H^-1 g exactly."""
    w = _base(**kwargs)
    for p in w.players.values():
        _clear(w)
        sigma2 = p.compute_sigma2()
        h = _dense_hessian(p)
        g = np.array(p.gradient([d.r for d in p.days], p.days, sigma2))
        expected = -np.linalg.solve(h, g)
        before = [d.r for d in p.days]
        p.update_by_ndim_newton()
        applied = np.array([d.r for d in p.days]) - np.array(before)
        for d, r0 in zip(p.days, before, strict=True):
            d.r = r0
        _clear(w)
        assert applied == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n_days", [1, 2, 3, 4, 7, 12])
def test_uncertainty_is_the_true_posterior_variance_at_every_day_count(n_days):
    """The two-day case is the regression: the backward pass guarded on
    ``sub_diag.size >= 2`` when reading index ``n - 2``, so with exactly two days
    it read 0 and reported a first-day variance ~25x too small (17 elo against a
    true 90)."""
    w = WHR({"w2": 300})
    for day in range(1, n_days + 1):
        for _ in range(3):
            w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
    w.iterate(200)
    for p in w.players.values():
        _clear(w)
        true_cov = np.linalg.inv(-_dense_hessian(p))
        for i, day in enumerate(p.days):
            # rel, not abs: Coulom's forward/backward recursion and numpy's dense
            # inverse are different float paths through the same algebra. They
            # agree to float precision only because update_uncertainty now
            # refreshes the game-term cache first; the defect this guards against
            # was 25x.
            assert day.uncertainty == pytest.approx(true_cov[i, i], rel=1e-12)


def test_two_rated_days_report_a_plausible_standard_error():
    """A guard-rail in elo terms, independent of the matrix algebra: with two
    days of identical evidence the two standard errors must be similar."""
    w = WHR({"w2": 300})
    for day in (1, 2):
        for _ in range(3):
            w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
    w.iterate(200)
    factor = 400.0 / math.log(10)
    se = [math.sqrt(d.uncertainty) * factor for d in w.player_by_name("a").days]
    assert se[0] == pytest.approx(se[1], rel=0.05)


def test_rating_covariance_is_the_dense_symmetric_inverse_in_elo_units():
    w = _base(names="abcd")
    _, cov = w.rating_covariance("a")
    p = w.player_by_name("a")
    _clear(w)
    expected = np.linalg.inv(-_dense_hessian(p)) * (400.0 / math.log(10)) ** 2
    assert np.allclose(cov, cov.T)
    assert cov == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# structural identities
# --------------------------------------------------------------------------- #
def test_wiener_prior_variance_is_gap_times_w2_in_nat_units():
    w = WHR({"w2": 300})
    for day in (1, 4, 14, 15):
        w.create_game("a", "b", "B", day, 0)
    w.iterate(5)
    p = w.player_by_name("a")
    per_step = (math.sqrt(300) * math.log(10) / 400) ** 2
    expected = [
        (p.days[i + 1].day - p.days[i].day) * per_step for i in range(len(p.days) - 1)
    ]
    assert p.compute_sigma2() == pytest.approx(expected)


def test_elo_gamma_and_r_are_consistent():
    w = _base(iters=20)
    for p in w.players.values():
        for d in p.days:
            assert d.elo == pytest.approx(d.r * 400 / math.log(10), abs=1e-12)
            assert d.gamma() == pytest.approx(math.exp(d.r), rel=1e-12)


# --------------------------------------------------------------------------- #
# advantage estimation must be a maximum of the likelihood, draws included
# --------------------------------------------------------------------------- #
def _advantage_gradient(w, table, key):
    """d(total log-likelihood) / d log(gamma_key), numerically."""
    g0 = table[key]

    def at(factor):
        table[key] = g0 * factor
        _clear(w)
        return w.log_likelihood()

    value = (at(math.exp(FD_H)) - at(math.exp(-FD_H))) / (2 * FD_H)
    table[key] = g0
    _clear(w)
    return value


@pytest.mark.parametrize("draws", [False, True])
def test_fitted_advantages_are_stationary_points_of_the_likelihood(draws):
    """With draws present this used to fail badly: the accumulator skipped drawn
    games, leaving a gradient of ~0.2-0.6 at convergence."""
    w = _base(draws=draws, handicap=True, komi=True, iters=2000, seed=7)
    assert w.max_gradient_norm() < 1e-6
    for table, pinned in (
        (w.handicap_gamma, w._pinned_handicap_keys),
        (w.komi_gamma, w._pinned_komi_keys),
    ):
        for key in list(table):
            if key in pinned:
                continue
            assert _advantage_gradient(w, table, key) == pytest.approx(0.0, abs=1e-5)


def test_draws_carry_advantage_information_rather_than_being_discarded():
    """A handicap key whose decisive games are all black wins used to be skipped
    entirely (0 < wins < games failed once draws were dropped); its draws now
    make it estimable."""
    w = WHR({"w2": 300})
    for day in range(1, 20):
        w.create_game("a", "b", "B", day, 2)
        w.create_game("a", "b", "D", day, 2)
        w.create_game("a", "b", "D", day, 2)
        w.create_game("b", "a", "B", day, 0)
        w.create_game("a", "b", "B", day, 0)
    w.iterate(400)
    assert w.nu > 0.0
    assert w.handicap_gamma[2] != 1.0
    assert _advantage_gradient(w, w.handicap_gamma, 2) == pytest.approx(0.0, abs=1e-5)


def test_all_draw_data_is_already_stationary_at_no_advantage():
    """Equal strengths and only draws: the Davidson split is symmetric in S and
    O, so the advantage gradient is exactly zero and nothing should move."""
    w = WHR({"w2": 300, "estimate_handicap_zero": True})
    for day in range(1, 10):
        w.create_game("a", "b", "D", day, 0)
        w.create_game("b", "a", "D", day, 0)
    import warnings

    from whr import HandicapBaselineWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HandicapBaselineWarning)
        w.iterate(200)
    assert w.handicap_gamma[0] == pytest.approx(1.0, abs=1e-9)


def test_draw_free_advantage_estimation_is_unchanged_by_the_davidson_terms():
    """The Davidson corrections all carry a factor T = nu*sqrt(S*O), so at
    nu == 0 they are exactly zero and the accumulation must be bit-identical to
    the Bradley-Terry-only arithmetic."""
    w = _base(handicap=True, komi=True, iters=300, seed=11)
    assert w.nu == 0.0
    h_grad, h_hess, k_grad, k_hess, _hg, _hw, _kg, _kw = w._accumulate_handicap_komi()
    # rebuild the pure Bradley-Terry terms by hand and require exact equality
    bt_h_grad: dict = {}
    bt_h_hess: dict = {}
    for g in w.games:
        s = g.bpd.gamma() * w.handicap_gamma[g.handicap]
        o = g.wpd.gamma() * w.komi_gamma.get(g.extras.get("komi"), 1.0)
        div = 1.0 / (s + o)
        key = g.handicap
        bt_h_grad[key] = bt_h_grad.get(key, 0.0) + g.bpd.gamma() * div
        bt_h_hess[key] = bt_h_hess.get(key, 0.0) + g.bpd.gamma() * o * div**2
    for key in bt_h_grad:
        assert h_grad[key] == pytest.approx(bt_h_grad[key], rel=1e-12)
        assert h_hess[key] == pytest.approx(bt_h_hess[key], rel=1e-12)


# --------------------------------------------------------------------------- #
# the two evaluation paths must agree
# --------------------------------------------------------------------------- #
def _day_with_n_games(n):
    """One player-day carrying ``n`` decisive games, half won half lost."""
    w = WHR({"w2": 300})
    for i in range(n):
        opponent = f"opp{i}"
        w.create_game("a", opponent, "B" if i % 2 else "W", 1, 0)
    w.iterate(5)
    return w.player_by_name("a").days[0]


@pytest.mark.parametrize("n", [1, 2, 5, 63, 64, 65, 200])
def test_python_and_numpy_paths_agree_across_the_threshold(n):
    """Small days take a Python loop (a numpy call costs ~2.5us regardless of
    size, so it loses badly at 1-5 games); large days keep numpy. Both must
    compute the same thing, so the threshold cannot change a result."""
    from whr import playerday as pd_module

    day = _day_with_n_games(n)
    original = pd_module._NUMPY_THRESHOLD
    try:
        pd_module._NUMPY_THRESHOLD = 0  # force numpy
        numpy_vals = (
            day.log_likelihood_derivative(),
            day.log_likelihood_second_derivative(),
            day.log_likelihood(),
        )
        pd_module._NUMPY_THRESHOLD = 10**9  # force the Python loop
        python_vals = (
            day.log_likelihood_derivative(),
            day.log_likelihood_second_derivative(),
            day.log_likelihood(),
        )
    finally:
        pd_module._NUMPY_THRESHOLD = original
    assert python_vals == pytest.approx(numpy_vals, rel=1e-12, abs=1e-12)


def test_the_legacy_term_rows_still_expose_the_opponent_gamma():
    """``won_game_terms``/``lost_game_terms`` keep their pre-vectorization
    ``[a, b, c, d]`` shape for backward compatibility, now built from the flat
    gamma cache the hot paths use."""
    day = _day_with_n_games(4)
    assert [row[3] for row in day.won_game_terms()] == day.won_gammas()
    assert [row[3] for row in day.lost_game_terms()] == day.lost_gammas()
    assert all(row[:3] == [1.0, 0.0, 1.0] for row in day.won_game_terms())
    assert all(
        row[0] == 0.0 and row[1] == row[3] and row[2] == 1.0
        for row in day.lost_game_terms()
    )


def test_clearing_the_cache_drops_the_flat_gammas_too():
    day = _day_with_n_games(3)
    day.won_gammas()
    day.won_game_terms()
    day.clear_game_terms_cache()
    assert day._won_gammas is None
    assert day._won_game_terms is None
    assert day._lost_gammas is None
    assert day._lost_game_terms is None


def test_the_advantage_layout_cache_is_invalidated_by_a_new_game():
    w = WHR({"w2": 300})
    for day in range(1, 5):
        w.create_game("a", "b", "B", day, 2)
    w.iterate(10)
    first = w._advantage_layout()
    assert w._advantage_layout() is first  # cached
    w.create_game("a", "b", "W", 5, 3)
    second = w._advantage_layout()
    assert second is not first
    assert 3 in second[3]  # the new handicap key made it into the index map
