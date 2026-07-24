import math

import pytest

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
