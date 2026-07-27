"""Freeing the handicap-0 baseline is only sound if the data can identify it.

With the ``handicap`` key ``0`` pinned (the default) the advantage scale has a
fixed zero. ``estimate_handicap_zero=True`` frees it, which adds a global
black-advantage parameter -- identifiable only if colour assignment varies
independently of who is playing. When a competitor sits on one side of the board,
that parameter trades off against their strength: differences between handicap
keys stay right while the overall level leaks into the ratings.
"""

import math
import warnings

import pytest

from whr import HandicapBaselineWarning
from whr.whole_history_rating import WHR


def _one_sided(estimate_zero, iterations=200):
    """`a` is ALWAYS black, `b` always white, and they are exactly equal: the
    handicap-0 games split 1-1 and the handicap-2 games run 4-1 to black."""
    w = WHR({"w2": 300, "estimate_handicap_zero": estimate_zero})
    for day in range(1, 30):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
        for _ in range(4):
            w.create_game("a", "b", "B", day, 2)
        w.create_game("a", "b", "W", day, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HandicapBaselineWarning)
        w.iterate(iterations)
    return w


def _colour_balanced(estimate_zero, iterations=200):
    """The same strengths and the same handicap-2 edge, but colours alternate,
    which pins the level and makes the freed baseline identifiable."""
    w = WHR({"w2": 300, "estimate_handicap_zero": estimate_zero})
    for day in range(1, 30):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
        for _ in range(2):
            w.create_game("a", "b", "B", day, 2)
            w.create_game("b", "a", "B", day, 2)
        w.create_game("a", "b", "W", day, 2)
        w.create_game("b", "a", "W", day, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HandicapBaselineWarning)
        w.iterate(iterations)
    return w


def _gap(w):
    return w.player_by_name("a").days[-1].elo - w.player_by_name("b").days[-1].elo


# --------------------------------------------------------------------------- #
# the hazard itself
# --------------------------------------------------------------------------- #
def test_pinned_baseline_keeps_equal_players_equal():
    w = _one_sided(False)
    assert _gap(w) == pytest.approx(0.0, abs=1.0)
    assert w.probability_future_match("a", "b", 0)[0] == pytest.approx(0.5, abs=1e-3)
    assert w.handicap_gamma[0] == 1.0


def test_freed_baseline_on_one_sided_colours_fabricates_a_rating_gap():
    """The regression this module documents: two players who are equal by
    construction come out tens of elo apart, and the no-handicap_key prediction
    path is wrong by the same amount."""
    w = _one_sided(True)
    assert abs(_gap(w)) > 30.0
    assert w.probability_future_match("a", "b", 0)[0] > 0.55
    assert w.handicap_gamma[0] != 1.0


def test_the_key_differences_survive_even_when_the_level_leaks():
    """Only the level is unidentified, so what the model is actually asked
    (key 2 versus key 0) stays correct -- which is why this is easy to miss."""

    def key_gap(w):
        return 400 * (math.log10(w.handicap_gamma[2]) - math.log10(w.handicap_gamma[0]))

    assert key_gap(_one_sided(False)) == pytest.approx(
        key_gap(_one_sided(True)), abs=1.0
    )


def test_colour_balanced_data_makes_the_freed_baseline_harmless():
    balanced_freed = _colour_balanced(True)
    balanced_pinned = _colour_balanced(False)
    assert _gap(balanced_freed) == pytest.approx(0.0, abs=1.0)
    assert balanced_freed.probability_future_match("a", "b", 0)[0] == pytest.approx(
        0.5, abs=1e-3
    )
    assert balanced_freed.handicap_gamma[0] == pytest.approx(1.0, abs=1e-6)
    assert _gap(balanced_pinned) == pytest.approx(_gap(balanced_freed), abs=1.0)


# --------------------------------------------------------------------------- #
# the statistic and the warning
# --------------------------------------------------------------------------- #
def test_one_sided_game_share_is_1_when_nobody_changes_colour():
    w = WHR()
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
    assert w.one_sided_game_share() == 1.0


def test_one_sided_game_share_is_0_when_everyone_alternates():
    w = WHR()
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("b", "a", "B", day, 0)
    assert w.one_sided_game_share() == 0.0


def test_one_sided_game_share_is_0_on_an_empty_base():
    assert WHR().one_sided_game_share() == 0.0


def test_home_and_away_league_is_not_flagged():
    """The sports case: "black" is the home side and every team plays both, so a
    league schedule must not trip the check."""
    w = WHR()
    teams = ["t1", "t2", "t3", "t4"]
    for day, home in enumerate(teams, start=1):
        for away in teams:
            if home != away:
                w.create_game(home, away, "B", day, 0)
                w.create_game(away, home, "B", day + 10, 0)
    assert w.one_sided_game_share() == 0.0


def test_iterate_warns_when_the_baseline_cannot_be_identified():
    w = WHR({"w2": 300, "estimate_handicap_zero": True})
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
    with pytest.warns(HandicapBaselineWarning) as record:
        w.iterate(5)
    message = str(record[0].message)
    assert "estimate_handicap_zero" in message
    assert "pinned_handicap" in message
    assert "one_sided_game_share" in message


def test_the_warning_fires_only_once_per_instance():
    w = WHR({"w2": 300, "estimate_handicap_zero": True})
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
    with pytest.warns(HandicapBaselineWarning) as record:
        for _ in range(5):
            w.iterate(2)
    assert len(record) == 1


def test_no_warning_when_the_baseline_stays_pinned():
    """The default must be silent even on perfectly one-sided data."""
    w = WHR({"w2": 300})
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HandicapBaselineWarning)
        w.iterate(5)


def test_no_warning_when_colours_are_balanced():
    w = WHR({"w2": 300, "estimate_handicap_zero": True})
    for day in range(1, 10):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("b", "a", "B", day, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HandicapBaselineWarning)
        w.iterate(5)


def test_warning_subclasses_userwarning():
    assert issubclass(HandicapBaselineWarning, UserWarning)
