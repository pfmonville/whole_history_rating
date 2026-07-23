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
