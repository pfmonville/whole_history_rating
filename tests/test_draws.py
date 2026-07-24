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
