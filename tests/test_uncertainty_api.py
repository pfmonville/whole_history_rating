import math

import pytest

from whr.whole_history_rating import WHR

_K = 400.0 / math.log(10)


def _rated(games, iters=50):
    w = WHR()
    w.load_games(games)
    w.iterate(iters)
    return w


def test_rating_difference_formula_and_ci():
    w = _rated(["a b B 1", "a b W 2", "a b B 3", "c b B 1", "c b W 2"])
    a = w.player_by_name("a").days[-1]
    b = w.player_by_name("b").days[-1]
    res = w.rating_difference("a", "b")
    assert res["difference"] == pytest.approx(a.elo - b.elo)
    expected_se = math.sqrt(a.uncertainty + b.uncertainty) * _K
    assert res["std_error"] == pytest.approx(expected_se)
    lo, hi = res["confidence_interval_95"]
    assert lo == pytest.approx(res["difference"] - 1.96 * expected_se)
    assert hi == pytest.approx(res["difference"] + 1.96 * expected_se)


def test_rating_difference_specific_days():
    w = _rated(["a b B 1", "a b W 5", "a b B 9"])
    res = w.rating_difference("a", "b", day_a=1, day_b=5)
    a1 = next(d for d in w.player_by_name("a").days if d.day == 1)
    b5 = next(d for d in w.player_by_name("b").days if d.day == 5)
    assert res["difference"] == pytest.approx(a1.elo - b5.elo)


def test_rating_difference_unknown_player_raises():
    w = _rated(["a b B 1", "a b W 2"])
    with pytest.raises(ValueError):
        w.rating_difference("a", "ghost")
    with pytest.raises(ValueError):
        w.rating_difference("a", "b", day_a=999)
