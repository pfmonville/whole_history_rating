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
