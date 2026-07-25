import math

import pytest

from whr import utils, whole_history_rating
from whr.player import Player


def test_new_config_keys_defaults_and_copy():
    src = {"w2": 300}
    w = whole_history_rating.WHR(config=src)
    assert w.config["initial_prior_wins"] == 0.5
    assert w.config["hessian_damping"] == 1.0
    assert "initial_prior_wins" not in src  # caller dict not mutated
    w.create_game("a", "b", "B", 1, 0)
    player = w.player_by_name("a")
    assert player.initial_prior_wins == 0.5
    assert player.hessian_damping == 1.0


def test_anchor_terms_match_coulom_formula():
    w = whole_history_rating.WHR()  # initial_prior_wins == 0.5
    w.create_game("a", "b", "B", 1, 0)
    day = w.player_by_name("a").days[0]
    day.set_gamma(2.0)
    k, g = 0.5, 2.0
    assert day.anchor_gradient() == pytest.approx(k * (1 - 2 * g / (1 + g)))
    assert day.anchor_hessian() == pytest.approx(-2 * k * g / ((1 + g) ** 2))
    assert day.anchor_log_likelihood() == pytest.approx(
        k * (math.log(g) - 2 * math.log(1 + g))
    )


def test_anchor_strength_scales_with_config():
    w = whole_history_rating.WHR(config={"initial_prior_wins": 1.0})
    w.create_game("a", "b", "B", 1, 0)
    day = w.player_by_name("a").days[0]
    day.set_gamma(2.0)
    assert day.anchor_gradient() == pytest.approx(1.0 * (1 - 2 * 2.0 / 3.0))


def test_lower_prior_reduces_compression():
    def spread(k):
        w = whole_history_rating.WHR(config={"initial_prior_wins": k})
        for d in range(1, 21):
            w.create_game("strong", "weak", "B", d, 0)  # strong (black) always wins
        w.iterate(300)
        elos = dict(w.get_ordered_ratings(current=True))
        return abs(elos["strong"] - elos["weak"])

    assert spread(0.5) > spread(1.0)


def test_gradient_anchor_applies_only_to_day_zero():
    # Locks the `idx == 0` guard in Player.gradient: the first-day anchor
    # term must only be added to g[0], never to later days.
    def make_player(initial_prior_wins):
        w = whole_history_rating.WHR(config={"initial_prior_wins": initial_prior_wins})
        w.create_game("a", "b", "B", 1, 0)
        w.create_game("a", "b", "W", 2, 0)
        player = w.player_by_name("a")
        # Move ratings away from the gamma == 1 fixed point, where the anchor
        # gradient is zero regardless of its strength (1 - 2*1/(1+1) == 0),
        # so that differing initial_prior_wins actually produce differing
        # anchor terms.
        player.days[0].set_gamma(2.0)
        player.days[1].set_gamma(3.0)
        return player

    p_low = make_player(0.5)
    p_high = make_player(5.0)

    r_low = [d.r for d in p_low.days]
    r_high = [d.r for d in p_high.days]
    assert r_low == r_high  # identical setup; only the anchor strength differs

    g_low = p_low.gradient(r_low, p_low.days, p_low.compute_sigma2())
    g_high = p_high.gradient(r_high, p_high.days, p_high.compute_sigma2())

    assert g_low[0] != pytest.approx(g_high[0])
    assert g_low[1] == pytest.approx(g_high[1])


def test_hessian_damping_configurable_and_stable():
    for damping in (0.1, 1.0, 10.0):
        w = whole_history_rating.WHR(config={"hessian_damping": damping})
        for d in range(1, 6):
            w.create_game("a", "b", "B", d, 0)
            w.create_game("a", "b", "W", d, 0)
        w.iterate(50)
        elo, _ = w.ratings_for_player("a", current=True)
        assert math.isfinite(elo)


def test_hessian_uses_damping_param():
    w = whole_history_rating.WHR(config={"hessian_damping": 5.0})
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    p = w.player_by_name("a")
    sigma2 = p.compute_sigma2()
    diag_small, _ = Player.hessian(p.days, sigma2, 0.0)
    diag_big, _ = Player.hessian(p.days, sigma2, 5.0)
    assert diag_big[1] == pytest.approx(diag_small[1] - 5.0)


def test_player_log_likelihood_closed_form():
    w = whole_history_rating.WHR()  # initial_prior_wins == 0.5
    w.create_game("a", "b", "B", 1, 0)  # a (black) beats b on day 1
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    a.days[0].set_gamma(3.0)
    b.days[0].set_gamma(1.0)
    ga, k = 3.0, 0.5
    expected_game = math.log(ga / (ga + 1.0))  # one day, opponent gamma == 1
    expected_anchor = k * (math.log(ga) - 2 * math.log(1 + ga))
    assert a.log_likelihood() == pytest.approx(expected_game + expected_anchor)


def test_total_log_likelihood_finite_and_improves():
    w = whole_history_rating.WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    w.create_game("a", "b", "B", 3, 0)
    w.iterate(1)
    start = w.log_likelihood()
    w.iterate(30)
    end = w.log_likelihood()
    assert math.isfinite(start) and math.isfinite(end)
    assert end >= start - 1e-6


def test_undefeated_player_is_finite_no_exception():
    w = whole_history_rating.WHR()
    for d in range(1, 11):
        w.create_game("winner", "loser", "B", d, 0)  # winner (black) always wins
    w.iterate(100)  # must NOT raise
    elo, unc = w.ratings_for_player("winner", current=True)
    assert math.isfinite(elo) and math.isfinite(unc)


def test_non_finite_step_raises(monkeypatch):
    from whr import playerday

    w = whole_history_rating.WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    monkeypatch.setattr(
        playerday.PlayerDay, "log_likelihood_derivative", lambda self: float("nan")
    )
    with pytest.raises(utils.UnstableRatingException):
        w.iterate(1)


def test_advantage_newton_step_does_not_overflow():
    # Regression: a degenerate / ill-conditioned handicap or komi key can be
    # driven to an extreme gamma by a Newton step; the *next* step,
    # ``gamma * math.exp(-grad / hess)``, was unclamped and overflowed with
    # ``OverflowError: math range error`` (seen on a large real dataset). The
    # log-space step is now trust-region clamped, so the update stays finite.
    w = whole_history_rating.WHR()
    n = 2000
    for i in range(n):
        # ~40% of games won by the komi (white) side, so the komi key has
        # hundreds of wins -> an unclamped correction step ~ that count.
        winner = "B" if (i % 5) < 3 else "W"
        w.create_game("black", "white", winner, i, 0, extras={"komi": 9.5})
    w.iterate(3)
    w.komi_gamma[9.5] = 1e-306  # simulate an overshoot from a prior Newton step
    w._newton_handicap_komi()  # must NOT raise OverflowError
    assert math.isfinite(w.komi_gamma[9.5])
    assert w.komi_gamma[9.5] > 0.0


def test_clamped_log_step_is_bounded_and_finite():
    step = whole_history_rating.WHR._clamped_log_step
    cap = whole_history_rating._MAX_ADVANTAGE_LOG_STEP
    # A huge Newton step (-grad/hess) is clamped to the trust region (both
    # signs)...
    assert step(1e6, -1.0) == pytest.approx(cap)
    assert step(-1e6, -1.0) == pytest.approx(-cap)
    # ...so exponentiating it can never overflow.
    assert math.isfinite(math.exp(step(1e6, -1.0)))
    # A small, well-conditioned step is returned unchanged.
    assert step(-0.5, -2.0) == pytest.approx(-0.25)
    # Non-finite / degenerate inputs mean "no update".
    assert step(1.0, 0.0) == 0.0
    assert step(float("nan"), -1.0) == 0.0


def test_auto_iterate_converges_on_gradient_norm():
    w = whole_history_rating.WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
        w.create_game("a", "b", "W", d, 0)
    iterations, converged = w.auto_iterate(precision=1e-2, time_limit=10)
    assert converged is True
    assert iterations > 0
    assert w.max_gradient_norm() < 1e-2


def test_uncertainty_integrated_three_outcome_does_not_overflow():
    # Regression: the uncertainty-integrated win/draw/loss path exponentiates
    # the half-gap ``h = d/2``, so an extreme rating gap (or a modest gap plus a
    # huge sigma) could push ``math.exp(h)`` past the double range and raise
    # ``OverflowError: math range error`` -- for a matchup the *point* path
    # returns fine. An opt-in flag must not introduce a new failure mode, so the
    # half-gap is clamped to a range where the split is already saturated to
    # full double precision.
    w = whole_history_rating.WHR()
    w.create_game("a", "b", "D", 1, 0)
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(20)
    day_a = w.player_by_name("a").days[-1]
    day_b = w.player_by_name("b").days[-1]
    day_a.set_gamma(math.exp(709))  # largest representable gamma
    day_b.set_gamma(5e-324)  # smallest positive subnormal
    day_a.uncertainty = 0.4
    day_b.uncertainty = 0.4

    point = w.win_draw_loss_probabilities("a", "b")
    integrated = w.win_draw_loss_probabilities("a", "b", account_for_uncertainty=True)

    assert all(math.isfinite(p) and p >= 0.0 for p in integrated)
    assert sum(integrated) == pytest.approx(1.0)
    # a ~1453-nat gap is saturated: both paths agree that "a" wins, near-surely
    assert point[0] == 1.0
    assert integrated[0] == 1.0

    # a plausible gap with an implausibly large sigma must also stay finite
    day_a.set_gamma(math.exp(600.0))
    day_b.set_gamma(math.exp(-600.0))
    day_a.uncertainty = 5000.0
    day_b.uncertainty = 5000.0
    wide = w.win_draw_loss_probabilities("a", "b", account_for_uncertainty=True)
    assert all(math.isfinite(p) and p >= 0.0 for p in wide)
    assert sum(wide) == pytest.approx(1.0)
