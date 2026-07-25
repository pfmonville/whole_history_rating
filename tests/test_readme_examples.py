"""Locks the numbers printed in README.md.

Every worked example in the README shows real output. Those numbers silently
rotted twice (once when 3.0.0 changed the anchor/damping, once when 3.1.0 made
komi opt-in), leaving figures a reader could not reproduce. These tests run the
documented snippets verbatim and assert the documented values, so a change that
moves them fails here instead of in the docs.

If a test here fails after an intentional model change, re-run the snippet, put
the new value in README.md, and update the expectation in the same commit.
"""

import math

import pytest

from whr import WHR


def _running_example() -> WHR:
    """The README's three-game example: shusaku vs shusai, B / W / W."""
    whr = WHR()
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 2, 0)
    whr.create_game("shusaku", "shusai", "W", 3, 0)
    whr.iterate(50)
    return whr


def test_readme_viewing_ratings():
    whr = _running_example()
    assert whr.ratings_for_player("shusaku") == [
        (1, -50, 0.26),
        (2, -51, 0.26),
        (3, -52, 0.26),
    ]
    assert whr.ratings_for_player("shusai") == [
        (1, 50, 0.26),
        (2, 51, 0.26),
        (3, 52, 0.26),
    ]


def test_readme_player_by_name_elos():
    whr = _running_example()
    player = whr.player_by_name("shusaku")
    assert [(d.day, round(d.elo, 1)) for d in player.days] == [
        (1, -49.8),
        (2, -51.1),
        (3, -51.7),
    ]


def test_readme_inspecting_the_fit():
    whr = _running_example()
    # A log *density*: legitimately positive, which the README now says.
    assert whr.log_likelihood() == pytest.approx(0.3301006161791349, rel=1e-9)
    assert whr.log_likelihood() > 0.0
    assert whr.max_gradient_norm() == pytest.approx(9.54e-05, abs=1e-7)


def test_readme_prediction_and_its_elo_gap():
    whr = _running_example()
    p_shusaku, p_shusai = whr.probability_future_match("shusaku", "shusai", 0)
    assert round(p_shusaku * 100, 2) == 35.50
    assert round(p_shusai * 100, 2) == 64.50

    ordered = dict(whr.get_ordered_ratings(current=True))
    assert round(ordered["shusaku"], 2) == -51.69
    assert round(ordered["shusai"], 2) == 52.05
    gap = ordered["shusai"] - ordered["shusaku"]
    assert round(gap, 2) == 103.74
    # the README claims the prediction is exactly Bradley-Terry on that gap
    assert 1.0 / (1.0 + 10 ** (gap / 400.0)) == pytest.approx(p_shusaku, rel=1e-12)


def test_readme_draws_example():
    whr = _running_example()
    whr.create_game("shusaku", "shusai", "D", 4, 0)
    whr.load_games(["shusaku shusai D 5"])
    whr.auto_iterate()
    assert whr.draw_tendency == pytest.approx(1.39, abs=0.01)
    wdl = whr.win_draw_loss_probabilities("shusaku", "shusai")
    assert tuple(round(x, 2) for x in wdl) == (0.21, 0.40, 0.39)
    assert sum(wdl) == pytest.approx(1.0)
    # the README says a draw is the single most likely outcome here
    assert max(range(3), key=lambda i: wdl[i]) == 1


def test_readme_uncertainty_aware_three_outcome_example():
    """The README's uncertainty-integrated win/draw/loss block, including its
    claim that the hedge compresses the win/loss ODDS while the draw
    probability falls and both decisive outcomes rise."""
    whr = _running_example()
    whr.create_game("shusaku", "shusai", "D", 4, 0)
    whr.load_games(["shusaku shusai D 5"])
    whr.auto_iterate()

    point = whr.win_draw_loss_probabilities("shusaku", "shusai")
    hedged = whr.win_draw_loss_probabilities(
        "shusaku", "shusai", account_for_uncertainty=True
    )
    assert tuple(round(x, 4) for x in point) == (0.2146, 0.3999, 0.3855)
    assert tuple(round(x, 4) for x in hedged) == (0.2209, 0.3918, 0.3872)
    assert sum(hedged) == pytest.approx(1.0)

    # the odds quoted in the README, and their compression toward 1.0
    assert round(point[0] / point[2], 3) == 0.557
    assert round(hedged[0] / hedged[2], 3) == 0.571
    assert point[0] / point[2] < hedged[0] / hedged[2] < 1.0

    # "the draw probability goes down, and both decisive outcomes go up"
    assert hedged[1] < point[1]
    assert hedged[0] > point[0] and hedged[2] > point[2]


def test_readme_rating_difference():
    whr = WHR()
    for day in range(1, 11):
        whr.create_game("north", "referee", "B", day, 0)
    for day in range(1, 11):
        whr.create_game("south", "referee", "W", day, 0)
    whr.auto_iterate()
    res = whr.rating_difference("north", "south")
    assert round(res["difference"], 2) == 1054.66
    assert round(res["std_error"], 2) == 85.73
    lo, hi = res["confidence_interval_95"]
    assert (round(lo, 2), round(hi, 2)) == (886.63, 1222.69)


def test_readme_rating_change_and_naive_comparison():
    whr = WHR()
    whr.load_games(
        ["casey dana B 1", "casey dana W 5", "casey dana B 9", "casey dana W 13"]
    )
    whr.iterate(60)

    days, cov = whr.rating_covariance("casey")
    assert days == [1, 5, 9, 13]

    res = whr.rating_change("casey", day_from=1, day_to=13)
    assert round(res["change"], 2) == -6.67
    assert round(res["std_error"], 2) == 57.48
    lo, hi = res["confidence_interval_95"]
    assert (round(lo, 2), round(hi, 2)) == (-119.32, 105.99)

    # the three covariance entries the README quotes, and both derived numbers
    i, j = days.index(1), days.index(13)
    v_from, v_to, c = cov[i][i], cov[j][j], cov[i][j]
    assert round(float(v_from), 2) == 6628.33
    assert round(float(v_to), 2) == 6789.29
    assert round(float(c), 2) == 5057.12
    assert round(math.sqrt(v_to + v_from - 2 * c), 2) == 57.48  # reported
    assert round(math.sqrt(v_to + v_from), 2) == 115.83  # naive, independent
    # README: ignoring the correlation overstates the error by about 2x
    assert math.sqrt(v_to + v_from) / math.sqrt(v_to + v_from - 2 * c) == pytest.approx(
        2.02, abs=0.01
    )


def test_readme_uncertainty_aware_prediction():
    whr = WHR()
    whr.load_games(["rookie champ B 1", "rookie champ B 2"])
    whr.iterate(50)
    point = whr.probability_future_match("rookie", "champ")
    hedged = whr.probability_future_match(
        "rookie", "champ", account_for_uncertainty=True
    )
    assert tuple(round(x, 3) for x in point) == (0.883, 0.117)
    assert tuple(round(x, 3) for x in hedged) == (0.856, 0.144)
    # README: hedged is pulled toward 0.5
    assert 0.5 < hedged[0] < point[0]


def test_readme_fit_w2_example():
    whr = WHR()
    whr.load_games(
        [f"riser anchor {'B' if day > 15 else 'W'} {day}" for day in range(1, 41)]
        + [f"other anchor {'B' if day % 3 else 'W'} {day}" for day in range(1, 41)]
    )
    res = whr.fit_w2(
        candidates=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
        n_splits=5,
        iterations=50,
    )
    assert res["best_w2"] == 3000.0
    assert res["n_splits"] == 5
    assert res["n_test_scored"] == 66
    assert res["n_test_skipped"] == 0
    expected = {
        10.0: 0.8488,
        30.0: 0.8404,
        100.0: 0.8164,
        300.0: 0.7733,
        1000.0: 0.712,
        3000.0: 0.6687,
    }
    for w2, want in expected.items():
        assert res["log_loss"][w2] == pytest.approx(want, abs=5e-4)
    # README: the curve falls monotonically, which is why best_w2 is the largest
    losses = [res["log_loss"][w2] for w2 in sorted(res["log_loss"])]
    assert losses == sorted(losses, reverse=True)


def test_readme_documented_config_defaults():
    config = WHR().config
    assert config["w2"] == 300.0
    assert config["uncased"] is False
    assert config["initial_prior_wins"] == 0.5
    assert config["hessian_damping"] == 1.0
    assert config["drift_kernel_radius"] == 100
    assert config["pinned_handicap"] == {}
    assert config["pinned_komi"] == {}
    assert config["estimate_handicap_zero"] is False
    assert config["pinned_draw"] is None


def test_readme_display_offset_leaves_predictions_unchanged():
    """The README's answer to "can I make ratings look like real elo?" —
    adding a constant to every rating changes no prediction."""
    whr = _running_example()
    before = whr.probability_future_match("shusaku", "shusai", 0)
    for player in whr.players.values():
        for day in player.days:
            day.elo = day.elo + 1500
    after = whr.probability_future_match("shusaku", "shusai", 0)
    assert after == pytest.approx(before, rel=1e-12)
