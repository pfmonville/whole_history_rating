"""Two input shapes that used to break: float days that differ only by
round-off, and ``load_games`` extras written the way Python prints a dict."""

import pytest

from whr import UnstableRatingException
from whr.whole_history_rating import WHR


# --------------------------------------------------------------------------- #
# float days
# --------------------------------------------------------------------------- #
def test_days_that_differ_only_by_float_round_off_are_the_same_day():
    """0.1 + 0.2 is 0.30000000000000004: it used to become a second day
    5.6e-17 from 0.3, and iterate() died on a bare ZeroDivisionError."""
    w = WHR({"w2": 30})
    w.create_game("a", "b", "B", 0.1 + 0.2, 0)
    w.create_game("a", "b", "W", 0.3, 0)
    w.create_game("a", "b", "B", 1.3, 0)
    assert [d.day for d in w.players["a"].days] == [0.3, 1.3]
    w.iterate(20)


def test_a_float_that_is_an_integer_up_to_round_off_is_that_integer():
    w = WHR()
    game = w.create_game("a", "b", "B", 0.1 * 3 * 10, 0)  # 3.0000000000000004
    assert game.day == 3 and type(game.day) is int


def test_genuinely_distinct_fractional_days_stay_distinct():
    w = WHR({"w2": 30})
    for day in (0.3, 0.3001, 0.30000001):
        w.create_game("a", "b", "B", day, 0)
    assert [d.day for d in w.players["a"].days] == [0.3, 0.30000001, 0.3001]


def test_a_drift_prior_too_tight_to_solve_raises_the_librarys_exception():
    """A tiny w2 makes the prior precision swamp the game terms until the
    tridiagonal pivot cancels to 0: that is an unstable rating, not a bare
    ZeroDivisionError."""
    w = WHR({"w2": 1e-12})
    for day in range(1, 50):
        for _ in range(3):
            w.create_game("a", "b", "B" if day % 2 else "W", day, 0)
    with pytest.raises(UnstableRatingException, match="w2"):
        w.iterate(50)


# --------------------------------------------------------------------------- #
# load_games extras
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line, separator",
    [
        ("a b B 1 0 {'komi': 6.5}", " "),
        ("a b B 1 {'komi': 6.5}", " "),
        ("a,b,B,1,0,{'komi': 6.5, 'venue': 'x'}", ","),
        ("a ; b ; B ; 1 ; 0 ; {'komi': 6.5, 'venue': 'x y'}", ";"),
    ],
    ids=["space", "space_no_handicap", "comma_two_keys", "semicolon_spaces"],
)
def test_an_extras_dict_may_contain_the_separator(line, separator):
    """The line used to be split before the dict was parsed, so the default
    ' ' separator could not carry {'komi': 6.5} at all."""
    w = WHR()
    w.load_games([line], separator=separator)
    (game,) = w.games
    assert game.extras["komi"] == 6.5
    assert (game.black_player.name, game.white_player.name, game.day) == ("a", "b", 1)


def test_a_player_name_with_braces_is_not_mistaken_for_extras():
    w = WHR()
    w.load_games(["team{1} team{2} B 1", "team{1} team{2} W 2 0 {'komi': 6.5}"])
    assert sorted(w.players) == ["team{1}", "team{2}"]
    assert w.games[1].extras == {"komi": 6.5}


def test_a_line_whose_extras_are_not_a_dict_is_still_rejected():
    with pytest.raises(ValueError, match="extras"):
        WHR().load_games(["a b B 1 0 {'komi': 6.5"])
