import math

import pytest

from whr import whole_history_rating


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
