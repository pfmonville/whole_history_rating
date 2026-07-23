import math

import pytest

from whr.whole_history_rating import WHR


def test_drift_kernel_radius_default_and_configurable():
    assert WHR().config["drift_kernel_radius"] == 100
    assert WHR(config={"drift_kernel_radius": 30}).config["drift_kernel_radius"] == 30


def test_remove_drift_empty_base_returns_empty():
    assert WHR().remove_drift() == {}


def test_remove_drift_single_day_is_finite_and_no_raise():
    w = WHR()
    w.load_games(["a b B 5"])
    w.iterate(10)
    corrections = w.remove_drift()
    assert all(math.isfinite(c) for c in corrections.values())


def test_remove_drift_return_contract():
    w = WHR()
    for d in range(1, 11):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(20)
    corrections = w.remove_drift()
    day_set = {pd.day for p in w.players.values() for pd in p.days}
    assert set(corrections) == day_set
    assert all(
        isinstance(k, int) and isinstance(v, float) and math.isfinite(v)
        for k, v in corrections.items()
    )


def test_remove_drift_preserves_same_day_win_probability():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "c b B 2", "a b B 3"])
    w.iterate(50)
    game = w.games[2]  # c vs b on day 2 — both players share that day
    before = game.white_win_probability()
    w.remove_drift()
    after = game.white_win_probability()
    assert after == pytest.approx(before, abs=1e-9)


def test_compute_drift_zero_support_day_is_zero_not_error():
    # Games at day 1 and day 100 with a small radius: the middle of the gap
    # has no games within `drift_kernel_radius`, so filtered_count == 0 there.
    # The guard must return 0.0 for that day rather than raising ZeroDivisionError.
    w = WHR(config={"drift_kernel_radius": 5})
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "B", 100, 0)
    w.iterate(10)
    drift = w._compute_drift()  # private, exercised directly to hit the guard
    assert drift[50] == 0.0
    assert all(math.isfinite(v) for v in drift.values())


def test_remove_drift_cancels_linear_drift():
    # One fresh, independent matchup per day for 300 days; inject a linear
    # drift by setting every player-day's elo equal to its day number, so the
    # mean strength on day d is exactly d. Symmetric Gaussian smoothing of a
    # linear field returns the centre value, so fully-interior days must be
    # recentred to ~0 after remove_drift.
    w = WHR()
    for d in range(1, 301):
        w.create_game(f"b{d}", f"w{d}", "B", d, 0)
    for game in w.games:
        assert game.bpd is not None and game.wpd is not None
        game.bpd.elo = float(game.day)
        game.wpd.elo = float(game.day)
    corrections = w.remove_drift()
    assert all(math.isfinite(c) for c in corrections.values())
    for d in (120, 150, 180):  # fully interior (>100 from both ends)
        elos = [pd.elo for p in w.players.values() for pd in p.days if pd.day == d]
        assert elos
        for elo in elos:
            assert abs(elo) < 1.0  # was ~d, now recentred to ~0
