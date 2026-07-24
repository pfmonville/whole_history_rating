import math

import numpy as np
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


def test_rating_covariance_diagonal_matches_uncertainty():
    w = _rated(["a b B 1", "a b W 5", "a b B 9", "a b W 13"])
    days, cov = w.rating_covariance("a")
    assert days == [1, 5, 9, 13]
    p = w.player_by_name("a")
    # diagonal, converted back to r-space, matches stored per-day uncertainty
    for i, d in enumerate(p.days):
        assert cov[i, i] / (_K**2) == pytest.approx(d.uncertainty, rel=1e-6, abs=1e-9)


def test_rating_covariance_symmetric_and_psd():
    w = _rated(["a b B 1", "a b W 5", "a b B 9"])
    _, cov = w.rating_covariance("a")
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > -1e-9).all()  # positive semi-definite


def test_rating_change_uses_joint_covariance_not_marginals():
    w = _rated(["a b B 1", "a b W 5", "a b B 9", "a b W 13"], iters=60)
    p = w.player_by_name("a")
    res = w.rating_change("a", 1, 13)
    d_from = next(d for d in p.days if d.day == 1)
    d_to = next(d for d in p.days if d.day == 13)
    assert res["change"] == pytest.approx(d_to.elo - d_from.elo)
    naive_se = math.sqrt(d_from.uncertainty + d_to.uncertainty) * _K
    # consecutive days positively correlated -> joint SE strictly smaller than naive
    assert res["std_error"] < naive_se


def test_rating_covariance_and_change_errors():
    w = _rated(["a b B 1", "a b W 2"])
    with pytest.raises(ValueError):
        w.rating_covariance("ghost")
    with pytest.raises(ValueError):
        w.rating_change("a", 1, 999)
