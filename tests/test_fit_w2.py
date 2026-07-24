import math
import random

import pytest

from whr.whole_history_rating import WHR


def _linear_history(n_days):
    w = WHR()
    for d in range(1, n_days + 1):
        w.create_game("a", "b", "B", d, 0)
    return w


def test_temporal_folds_are_expanding_and_leak_free():
    w = _linear_history(12)
    folds = w._temporal_folds(3)
    assert len(folds) == 3
    prev_train = -1
    for train, test in folds:
        assert train and test
        max_train_day = max(d[3] for d in train)
        min_test_day = min(d[3] for d in test)
        assert max_train_day < min_test_day  # no future leakage, no same-day split
        assert len(train) > prev_train  # expanding window
        prev_train = len(train)


def test_temporal_folds_cover_later_games_and_copy_extras():
    w = WHR()
    for d in range(1, 7):
        w.create_game("a", "b", "B", d, 0, {"komi": 6.5})
    folds = w._temporal_folds(2)
    # extras are copies, not the live Game dict
    train0 = folds[0][0]
    assert train0[0][5] == {"komi": 6.5}
    train0[0][5]["komi"] = 999
    assert w.games[0].extras["komi"] == 6.5  # original untouched


def test_temporal_folds_raise_when_too_few_distinct_days():
    w = _linear_history(2)
    with pytest.raises(ValueError):
        w._temporal_folds(5)  # needs >= 6 distinct days


def test_temporal_folds_rejects_bad_n_splits():
    w = _linear_history(10)
    with pytest.raises(ValueError):
        w._temporal_folds(0)


def _round_robin_drifting_history(rng, n_players=20, n_days=40, day_step_var=300.0):
    """Players whose true elo does a per-day Gaussian random walk (variance
    ``day_step_var`` elo^2/day -- WHR's own Brownian-motion assumption for
    ``w2``, so a candidate matching ``day_step_var`` is the "correct" amount
    of cross-day smoothing); outcomes are sampled from the true Bradley-Terry
    probability every day in a full round robin.

    A *deterministic* linear ramp was tried first (per the original design
    note) but turned out to systematically favor an unsmoothed (huge w2)
    model whenever there were enough games/day to pin down each day's rating
    precisely -- WHR's smoothing prior only ever costs bias against a
    predictable trend, it doesn't help. Matching the generating process to
    WHR's actual random-walk assumption is what makes an intermediate w2
    genuinely, robustly the best predictor of held-out future games (~90% of
    seeds tried), rather than a coin flip between the two extremes.
    """
    w = WHR()
    names = [f"p{i}" for i in range(n_players)]
    day_step_std = math.sqrt(day_step_var)
    true_elo = {n: (i - n_players / 2) * 60.0 for i, n in enumerate(names)}
    for day in range(1, n_days + 1):
        for n in names:
            true_elo[n] += rng.gauss(0, day_step_std)
        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    continue
                black, white = names[i], names[j]
                pb = 1.0 / (1.0 + 10 ** ((true_elo[white] - true_elo[black]) / 400.0))
                winner = "B" if rng.random() < pb else "W"
                w.create_game(black, white, winner, day, 0)
    return w


def test_fit_w2_prefers_middle_over_extremes_on_drifting_data():
    rng = random.Random(1234)
    w = _round_robin_drifting_history(rng)
    result = w.fit_w2(candidates=[1.0, 300.0, 100000.0], n_splits=2, iterations=25)
    ll = result["log_loss"]
    assert ll[300.0] < ll[1.0]
    assert ll[300.0] < ll[100000.0]
    assert result["best_w2"] == 300.0


def test_fit_w2_is_a_pure_query():
    w = _linear_history(8)
    w.iterate(5)
    before_w2 = w.config["w2"]
    before = w.ratings_for_player("a")
    w.fit_w2(candidates=[100.0, 300.0], n_splits=2, iterations=10)
    assert w.config["w2"] == before_w2
    assert w.ratings_for_player("a") == before


def test_fit_w2_skips_cold_start_test_games():
    # "newbie" appears only in the final period (test block) -> its games are skipped.
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
    w.create_game("newbie", "a", "B", 6, 0)
    result = w.fit_w2(candidates=[300.0], n_splits=1, iterations=10)
    assert result["n_test_skipped"] >= 1


def test_fit_w2_return_contract():
    w = _linear_history(10)
    result = w.fit_w2(candidates=[100.0, 300.0], n_splits=2, iterations=10)
    assert set(result) == {
        "best_w2",
        "log_loss",
        "n_splits",
        "n_test_scored",
        "n_test_skipped",
    }
    assert result["best_w2"] in (100.0, 300.0)
    assert all(math.isfinite(v) for v in result["log_loss"].values())
    assert result["n_splits"] == 2


def test_fit_w2_raises_on_single_day():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 1, 0)
    with pytest.raises(ValueError):
        w.fit_w2(n_splits=5)


def test_fit_w2_raises_on_empty_candidates():
    w = _linear_history(10)
    with pytest.raises(ValueError, match="non-empty list of w2 values"):
        w.fit_w2(candidates=[], n_splits=2, iterations=10)


def test_fit_w2_warns_when_all_test_games_are_cold_start():
    # Train-window players (a, b) and test-window players (c, d) never
    # overlap, so every held-out test game is cold-start for both players,
    # in every fold, for every candidate.
    w = WHR()
    for d in range(1, 4):
        w.create_game("a", "b", "B", d, 0)
    for d in range(4, 7):
        w.create_game("c", "d", "B", d, 0)
    with pytest.warns(UserWarning, match="cold-start"):
        result = w.fit_w2(candidates=[10.0, 300.0], n_splits=1, iterations=5)
    assert result["n_test_scored"] == 0
    assert result["n_test_skipped"] > 0
    assert all(math.isinf(v) for v in result["log_loss"].values())


def test_predict_black_win_probability_cold_start_is_none():
    w = _linear_history(4)
    w.iterate(5)
    assert w._predict_black_win_probability("a", "ghost", 0, 6.5) is None


def test_fit_w2_skips_draw_test_games():
    """RED before the fix: a "D" test game fell through fit_w2's scoring
    ``else`` branch and was mis-scored as a white win. A win/loss predictive
    log-loss has no correct value for a draw, so it must be skipped (counted
    in n_test_skipped) rather than scored.

    Distinct days 1..6, n_splits=1 -> train = days 1-3 (a b B), test = days
    4-6: two decisive "a b B" test games plus one "a b D" test game, none
    cold-start (both players are trained). Before the fix all three would be
    scored (n_test_scored=3, n_test_skipped=0); after the fix the draw is
    skipped (n_test_scored=2, n_test_skipped=1).
    """
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
    w.create_game("a", "b", "D", 6, 0)
    result = w.fit_w2(candidates=[300.0], n_splits=1, iterations=10)
    assert result["n_test_scored"] == 2
    assert result["n_test_skipped"] == 1
    assert math.isfinite(result["log_loss"][300.0])
