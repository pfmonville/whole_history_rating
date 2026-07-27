"""``time_step`` handling, and the uncertainty sentinel.

``time_step`` is a day index counted from an origin of the caller's choosing.
Fractional values are meaningful -- the Wiener prior uses ``|delta| * w2`` -- but
they used to be accepted by ``create_game`` while ``load_games`` could not parse
them and ``remove_drift`` raised a bare ``TypeError`` on them, including for whole
floats like ``1.0``. All three paths now agree.
"""

import math
import warnings

import pytest

from whr import UncomputedUncertaintyWarning
from whr.whole_history_rating import WHR


def _rate(days, w2=30):
    """`a` strictly stronger than `b`, rated across the given day values."""
    w = WHR({"w2": w2})
    for day in days:
        for _ in range(3):
            w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
    w.iterate(30)
    return w


# --------------------------------------------------------------------------- #
# accepted values
# --------------------------------------------------------------------------- #
def test_integer_days_are_the_normal_case():
    assert _rate([1, 2, 3]).player_by_name("a").days[-1].elo > 0


def test_fractional_days_are_accepted_and_rated():
    w = _rate([1.5, 2.5, 3.5])
    assert [d.day for d in w.player_by_name("a").days] == [1.5, 2.5, 3.5]
    assert w.player_by_name("a").days[-1].elo > 0


def test_negative_days_are_accepted():
    assert _rate([-5, -4, -3]).player_by_name("a").days[-1].elo > 0


def test_an_integral_float_day_is_narrowed_to_int():
    """So a float and an int spelling of the same day are the SAME playing day
    rather than two, and everything downstream that indexes by day agrees."""
    w = WHR({"w2": 30})
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 1.0, 0)
    days = [d.day for d in w.player_by_name("a").days]
    assert days == [1]
    assert all(isinstance(d, int) for d in days)


# --------------------------------------------------------------------------- #
# rejected values, at the point of entry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["3", None, [1], {"day": 1}])
def test_non_numeric_time_step_is_a_clear_typeerror(bad):
    with pytest.raises(TypeError, match="time_step"):
        WHR().create_game("a", "b", "B", bad, 0)


@pytest.mark.parametrize("bad", [True, False])
def test_bool_time_step_is_rejected(bad):
    """A bool is an int subclass, so it used to sail through as day 0 or 1."""
    with pytest.raises(TypeError, match="bool"):
        WHR().create_game("a", "b", "B", bad, 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_time_step_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        WHR().create_game("a", "b", "B", bad, 0)


# --------------------------------------------------------------------------- #
# remove_drift used to crash on any float day
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "days",
    [
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [1, 2.5, 3],
        [-3, -2, -1],
        [0.25, 0.75, 1.25],
    ],
)
def test_remove_drift_handles_every_accepted_day_shape(days):
    w = _rate(days)
    applied = w.remove_drift()
    assert set(applied) == {d if not float(d).is_integer() else int(d) for d in days}
    assert all(math.isfinite(v) for v in applied.values())


def test_remove_drift_bins_fractional_days_into_whole_days():
    """Days inside one whole day share a correction: the kernel already averages
    over +/- radius days, so sub-day resolution cannot survive it anyway."""
    w = _rate([5.1, 5.9])
    applied = w.remove_drift()
    assert applied[5.1] == pytest.approx(applied[5.9])


def test_remove_drift_still_preserves_same_day_predictions():
    """The invariant that makes remove_drift safe: a uniform per-day shift cannot
    change a within-day comparison."""
    w = _rate([1.5, 2.5, 3.5])
    before = w.probability_future_match("a", "b", 0)
    w.remove_drift()
    after = w.probability_future_match("a", "b", 0)
    assert after == pytest.approx(before, rel=1e-12)


def test_the_epoch_span_guard_still_fires_and_explains_itself():
    w = WHR({"w2": 30})
    for day in (0, 10**7):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
    w.iterate(5)
    with pytest.raises(ValueError, match="epoch timestamp"):
        w.remove_drift()


# --------------------------------------------------------------------------- #
# load_games must accept what create_game accepts
# --------------------------------------------------------------------------- #
def test_load_games_accepts_a_fractional_day():
    w = WHR()
    w.load_games(["a b B 1.5"])
    assert w.games[0].day == 1.5


def test_load_games_narrows_an_integral_float_day():
    w = WHR()
    w.load_games(["a b B 2.0"])
    assert w.games[0].day == 2
    assert isinstance(w.games[0].day, int)


def test_load_games_rejects_a_non_numeric_day_with_a_useful_message():
    with pytest.raises(ValueError, match="time_step"):
        WHR().load_games(["a b B notaday"])


def test_load_games_tolerates_surrounding_whitespace():
    w = WHR()
    w.load_games(["  a b B 1  "])
    assert (w.games[0].black_player.name, w.games[0].day) == ("a", 1)


def test_load_games_names_the_repeated_separator_instead_of_failing_on_int():
    """A doubled separator shifts every later field; the old error was
    "invalid literal for int() with base 10: 'B'"."""
    with pytest.raises(ValueError, match="Empty field"):
        WHR().load_games(["a  b B 1"])


def test_load_games_field_count_error_says_what_was_expected():
    with pytest.raises(ValueError, match="expected 4 to 6"):
        WHR().load_games(["a b B"])


# --------------------------------------------------------------------------- #
# the -1 uncertainty sentinel
# --------------------------------------------------------------------------- #
def test_ratings_for_player_warns_while_uncertainties_are_uncomputed():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    with pytest.warns(UncomputedUncertaintyWarning, match="sentinel"):
        rows = w.ratings_for_player("a")
    assert rows[0][2] == -1  # still returned, so an un-rated base stays readable


def test_the_sentinel_warning_fires_only_once_per_instance():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    with pytest.warns(UncomputedUncertaintyWarning) as record:
        for _ in range(10):
            w.ratings_for_player("a")
            w.ratings_for_player("b")
    assert len(record) == 1


def test_no_sentinel_warning_after_iterate():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UncomputedUncertaintyWarning)
        rows = w.ratings_for_player("a")
    assert rows[0][2] >= 0


def test_the_siblings_still_raise_in_that_same_state():
    """Documents the asymmetry the warning exists to bridge."""
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    with pytest.raises(ValueError, match="uncertainties not computed"):
        w.rating_difference("a", "b")


def test_warning_subclasses_userwarning():
    assert issubclass(UncomputedUncertaintyWarning, UserWarning)
