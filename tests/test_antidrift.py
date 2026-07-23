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


def test_remove_drift_rejects_huge_day_span():
    # `time_step` values that are epoch timestamps (or otherwise not a compact
    # day index) blow up the O(day-span) arrays allocated by _compute_drift.
    # A span > 1_000_000 must raise a clear ValueError instead of silently
    # hanging or exhausting memory.
    w = WHR()
    w.create_game("a", "b", "B", 0, 0)
    w.create_game("a", "b", "B", 1_500_000, 0)
    w.iterate(5)
    with pytest.raises(ValueError, match="day span"):
        w.remove_drift()


def test_remove_drift_rejects_invalid_drift_kernel_radius():
    # radius=0 currently produced an opaque IndexError (zero-length kernel
    # array) rather than a clear, actionable error.
    w = WHR(config={"drift_kernel_radius": 0})
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(5)
    with pytest.raises(ValueError, match="drift_kernel_radius must be an int >= 1"):
        w.remove_drift()


def test_remove_drift_rejects_non_int_drift_kernel_radius():
    w = WHR(config={"drift_kernel_radius": 30.5})
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(5)
    with pytest.raises(ValueError, match="drift_kernel_radius must be an int >= 1"):
        w.remove_drift()


def test_remove_drift_is_idempotent_on_interior_days():
    # Recreate a history with a strong linear drift (as in
    # test_remove_drift_cancels_linear_drift, so the first remove_drift()
    # call applies a large, non-trivial correction), then call remove_drift()
    # a second time: on already-de-drifted ratings, a further pass should
    # apply ~0 additional correction, since the Gaussian smoothing exactly
    # reconstructs (and thus fully cancels) a linear field away from the
    # domain's edges. Near the two edges (within 2*radius of the boundary)
    # residual boundary-truncation effects from the *first* pass are, in
    # turn, picked up by the second pass, so we only assert idempotence on
    # days safely in the interior (more than 2*radius away from both ends).
    radius = 100
    n_days = 1000
    w = WHR(config={"drift_kernel_radius": radius})
    for d in range(1, n_days + 1):
        w.create_game(f"b{d}", f"w{d}", "B", d, 0)
    for game in w.games:
        assert game.bpd is not None and game.wpd is not None
        game.bpd.elo = float(game.day)
        game.wpd.elo = float(game.day)

    first = w.remove_drift()
    assert max(abs(v) for v in first.values()) > 1.0  # non-trivial correction

    second = w.remove_drift()
    margin = 2 * radius
    interior_days = [d for d in second if margin < d < n_days - margin + 1]
    assert interior_days
    assert max(abs(second[d]) for d in interior_days) < 1e-6


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
