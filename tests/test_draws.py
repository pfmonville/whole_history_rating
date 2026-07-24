import math
import random

import pytest

from whr.player import Player
from whr.whole_history_rating import WHR


def test_draw_result_accepted_and_routed():
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)
    w.load_games(["a b D 2"])
    assert w._has_draws is True
    a = w.player_by_name("a")
    drawn_days = [d for d in a.days if d.drawn_games]
    assert len(drawn_days) == 2  # one draw on each of days 1 and 2


def test_no_draws_leaves_nu_zero():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    assert w._has_draws is False
    assert w.nu == 0.0
    assert w.draw_tendency == 0.0


def test_effective_gammas_fold_advantages():
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    g = w.create_game("black", "white", "B", 1, 2)
    w.player_by_name("black").days[0].set_gamma(3.0)
    w.player_by_name("white").days[0].set_gamma(2.0)
    s_black, o_white = g.effective_gammas(g.black_player)
    assert s_black == pytest.approx(
        3.0 * 10 ** (200.0 / 400.0)
    )  # handicap boosts black
    assert o_white == pytest.approx(2.0)  # komi default gamma 1
    # symmetry: querying as white swaps S/O
    s_white, o_black = g.effective_gammas(g.white_player)
    assert (s_white, o_black) == pytest.approx((o_white, s_black))


def test_pinned_draw_config_default_none():
    assert WHR().config["pinned_draw"] is None
    assert WHR(config={"pinned_draw": 1.5}).config["pinned_draw"] == 1.5


def test_draws_do_not_inflate_komi_gamma():
    """Regression test for a cross-function bug: `_accumulate_handicap_komi`
    used to fall through to its `else` branch for draws (winner != "B"),
    mis-crediting them as WHITE/komi wins with a Bradley-Terry denominator
    that doesn't apply to a draw.

    This base is built to be exactly colour-swap symmetric: for every
    decisive game where black wins, there is a mirror game (colours and
    players swapped) where white wins, so the two players stay equal
    strength and the Newton gradient on komi_gamma[6.5] is exactly zero at
    gamma=1 -- there is no real white/komi advantage in this data. A single
    draw game is then added. Before the fix, that draw is silently counted
    as an extra komi/white win, pushing komi_gamma[6.5] well above 1.0
    (observed ~1.5 here, ~2.43 in a similar denser base). After the fix,
    draws are skipped by the handicap/komi accumulator and komi_gamma[6.5]
    stays at its symmetric-equilibrium value of ~1.0.
    """
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("b", "a", "B", 1, 0)
    w.create_game("a", "b", "W", 1, 0)
    w.create_game("b", "a", "W", 1, 0)
    w.create_game("a", "b", "D", 1, 0)
    w.iterate(50)
    assert abs(w.komi_gamma[6.5] - 1.0) < 0.1


def test_davidson_derivatives_match_closed_form():
    w = WHR()
    g = w.create_game("a", "b", "D", 1, 0)  # a draw on day 1
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    nu = 1.5
    s, o = g.effective_gammas(a_day.player)  # S=2, O=1 (no advantages)
    t = nu * math.sqrt(s * o)
    z = s + o + t
    n = s + t / 2.0
    n_prime = s + t / 4.0
    w_weight = 0.5  # draw
    exp_grad = w_weight - n / z
    exp_hess = (n / z) ** 2 - n_prime / z
    grad, hess = a_day.davidson_derivatives(nu)
    assert grad == pytest.approx(exp_grad)
    assert hess == pytest.approx(exp_hess)


def test_davidson_reduces_to_bt_at_nu_zero():
    # Same win/loss data, compute the day's game-part gradient both ways at nu=0.
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)  # a (black) won
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    davidson_grad, davidson_hess = a_day.davidson_derivatives(0.0)
    assert davidson_grad == pytest.approx(a_day.log_likelihood_derivative())
    assert davidson_hess == pytest.approx(a_day.log_likelihood_second_derivative())


def test_multiday_gradient_hessian_route_to_davidson_when_draw_tendency_positive():
    """Task 2 coverage gap: with ``draw_tendency > 0``, ``Player.gradient``
    and the static ``Player.hessian`` must take each day's Davidson game-part
    (``day.davidson_derivatives``) rather than the plain win/loss derivatives.

    Nothing wires ``draw_tendency`` above zero until Task 3, so this test sets
    it directly and reconstructs the expected gradient/hessian from
    ``davidson_derivatives`` plus the (independently expressed) temporal-prior
    / anchor terms -- this fails if the ``> 0`` routing branch is removed.
    """
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)  # day 1: a draws with b
    w.create_game("a", "b", "B", 5, 0)  # day 5: a (black) wins outright
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    # Distinct, non-symmetric gammas so the Davidson vs. win/loss derivatives
    # can't accidentally coincide (e.g. S == O would zero out some terms).
    a.days[0].set_gamma(2.0)
    b.days[0].set_gamma(1.0)
    a.days[1].set_gamma(3.0)
    b.days[1].set_gamma(1.5)
    a.draw_tendency = 0.3

    assert len(a.days) == 2
    r = [d.r for d in a.days]
    sigma2 = a.compute_sigma2()

    grad = a.gradient(r, a.days, sigma2)
    diag, _sub_diag = Player.hessian(a.days, sigma2, a.hessian_damping)

    for idx, day in enumerate(a.days):
        davidson_grad, davidson_hess = day.davidson_derivatives(0.3)
        prior_grad = 0.0
        prior_hess = 0.0
        if idx < len(a.days) - 1:
            prior_grad += -(r[idx] - r[idx + 1]) / sigma2[idx]
            prior_hess += -1 / sigma2[idx]
        if idx > 0:
            prior_grad += -(r[idx] - r[idx - 1]) / sigma2[idx - 1]
            prior_hess += -1 / sigma2[idx - 1]
        expected_grad = davidson_grad + prior_grad
        expected_hess = davidson_hess + prior_hess - a.hessian_damping
        if idx == 0:
            expected_grad += day.anchor_gradient()
            expected_hess += day.anchor_hessian()
        assert grad[idx] == pytest.approx(expected_grad)
        assert diag[idx] == pytest.approx(expected_hess)

    # Sanity: falling back to the win/loss derivatives would give a
    # different game-part on BOTH days here -- the draw day because a drawn
    # game never appears in won_games/lost_games, and the decisive day
    # because the Davidson draw-mass term (`t`) still perturbs the ratio for
    # nu > 0 even without an actual draw that day. This confirms the
    # reconstruction above is tight enough to distinguish the two paths.
    for day in a.days:
        bt_grad = day.log_likelihood_derivative()
        bt_hess = day.log_likelihood_second_derivative()
        davidson_grad, davidson_hess = day.davidson_derivatives(0.3)
        assert davidson_grad != pytest.approx(bt_grad)
        assert davidson_hess != pytest.approx(bt_hess)


def test_single_day_update_by_1d_newton_routes_to_davidson():
    """Task 2 coverage gap: ``PlayerDay.update_by_1d_newtons_method`` must
    take the Davidson game-part when ``draw_tendency > 0``, for a
    single-day player.
    """
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)  # a's only day: a single draw vs b
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    a.days[0].set_gamma(2.0)
    b.days[0].set_gamma(1.0)
    a.draw_tendency = 0.5

    day = a.days[0]
    r_before = day.r

    davidson_grad, davidson_hess = day.davidson_derivatives(0.5)
    bt_grad = day.log_likelihood_derivative()
    bt_hess = day.log_likelihood_second_derivative()
    anchor_grad = day.anchor_gradient()
    anchor_hess = day.anchor_hessian()
    damping = a.hessian_damping

    expected_davidson_r = r_before - (davidson_grad + anchor_grad) / (
        davidson_hess + anchor_hess - damping
    )
    counterfactual_bt_r = r_before - (bt_grad + anchor_grad) / (
        bt_hess + anchor_hess - damping
    )
    # Sanity: the two paths must actually diverge for this data, else the
    # test couldn't tell which branch ran (a drawn game is never counted in
    # won_games/lost_games, so the BT path sees no game at all here).
    assert expected_davidson_r != pytest.approx(counterfactual_bt_r)

    day.update_by_1d_newtons_method()

    assert day.r == pytest.approx(expected_davidson_r)
    assert day.r != pytest.approx(counterfactual_bt_r)


def _davidson_balanced_history(rng, nu_true, n_pairs=40, n_games=60):
    """Equal players, colour-swapped, single day; outcomes sampled from Davidson
    with a known nu_true (equal gammas => S=O=1 => P(draw)=nu/(2+nu))."""
    w = WHR()
    p_draw = nu_true / (2.0 + nu_true)
    p_win = 1.0 / (2.0 + nu_true)
    for k in range(n_pairs):
        a, b = f"a{k}", f"b{k}"
        for _ in range(n_games):
            r = rng.random()
            outcome = "D" if r < p_draw else ("B" if r < p_draw + p_win else "W")
            w.create_game(a, b, outcome, 1, 0)
            w.create_game(
                b,
                a,
                outcome if outcome == "D" else ("W" if outcome == "B" else "B"),
                1,
                0,
            )
    return w


def test_recovers_known_draw_tendency():
    rng = random.Random(7)
    nu_true = 1.5
    w = _davidson_balanced_history(rng, nu_true)
    w.iterate(80)
    assert w.draw_tendency == pytest.approx(nu_true, abs=0.3)


def test_pinned_draw_is_not_moved():
    w = WHR(config={"pinned_draw": 0.8})
    for d in range(1, 11):
        w.create_game("a", "b", "D", d, 0)
        w.create_game("a", "b", "B", d, 0)
    w.iterate(30)
    assert w.draw_tendency == pytest.approx(0.8)


def test_no_draw_iteration_still_matches_baseline():
    # A draw-free scenario iterates with the win/loss path untouched.
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
        w.create_game("a", "b", "W", d, 0)
    w.iterate(30)
    assert w.nu == 0.0
    elo, _ = w.ratings_for_player("a", current=True)
    assert math.isfinite(elo)
