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
