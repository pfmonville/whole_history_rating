"""Inputs are checked where they enter, with an error that names them.

Each case here used to be accepted and then fail far away (a bare
``ZeroDivisionError`` inside ``iterate``, a hang), be silently ignored, or
silently change the model.
"""

import math
import pickle

import numpy as np
import pytest

from whr.whole_history_rating import WHR


# --------------------------------------------------------------------------- #
# time steps from numpy / pandas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "day, stored",
    [
        (np.int64(3), 3),
        (np.int32(3), 3),
        (np.uint8(3), 3),
        (np.float64(3.0), 3),
        (np.float32(3.5), 3.5),
    ],
    ids=["int64", "int32", "uint8", "float64", "float32"],
)
def test_a_numpy_day_is_accepted_as_a_plain_number(day, stored):
    w = WHR()
    game = w.create_game("a", "b", "B", day, 0)
    assert game.day == stored
    assert type(game.day) is type(stored)


def test_a_numpy_bool_day_is_rejected_like_a_bool():
    with pytest.raises(TypeError, match="bool"):
        WHR().create_game("a", "b", "B", np.bool_(True), 0)


def test_a_save_holding_numpy_days_loads(tmp_path):
    """Before 3.4.0 days were not validated, so a base built from a numpy
    column saved np.int64 days; those files must still load."""
    w = WHR()
    for day in (1, 2, 3):
        w.create_game("a", "b", "B" if day % 2 else "W", day, 0)
    w.iterate(20)
    path = tmp_path / "old.pkl"
    w.save_base(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    data["games"] = [(*g[:3], np.int64(g[3]), *g[4:]) for g in data["games"]]
    data["ratings"] = {
        name: [(np.int64(day), r, u) for day, r, u in days]
        for name, days in data["ratings"].items()
    }
    with path.open("wb") as f:
        pickle.dump(data, f)
    assert WHR.load_base(path).ratings_for_player("a") == w.ratings_for_player("a")


# --------------------------------------------------------------------------- #
# w2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w2", [0, -10, math.nan, math.inf, "300"])
def test_an_invalid_w2_is_rejected_by_name(w2):
    with pytest.raises((ValueError, TypeError), match="w2"):
        WHR({"w2": w2})


@pytest.mark.parametrize("candidate", [0.0, -10.0, math.nan, math.inf])
def test_an_invalid_fit_w2_candidate_is_rejected_by_name(candidate):
    w = WHR()
    for day in range(10):
        w.create_game("a", "b", "BW"[day % 2], day, 0)
    with pytest.raises(ValueError, match="w2"):
        w.fit_w2(candidates=[candidate, 100.0], n_splits=2)


# --------------------------------------------------------------------------- #
# iterate / auto_iterate
# --------------------------------------------------------------------------- #
def _base():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "b c B 2"])
    return w


@pytest.mark.parametrize("count", [-5, 2.5, True, "3"])
def test_iterate_rejects_a_count_that_is_not_a_natural_number(count):
    with pytest.raises((ValueError, TypeError), match="count"):
        _base().iterate(count)


def test_iterate_zero_does_not_pretend_to_have_fitted():
    w = _base()
    w.iterate(10)
    w.create_game("a", "c", "W", 3, 0)
    w.iterate(0)
    assert w.games_since_last_fit == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -7},
        {"precision": 0.0},
        {"precision": -1e-3},
        {"precision": math.nan},
        {"time_limit": -1},
    ],
    ids=[
        "batch_zero",
        "batch_negative",
        "precision_zero",
        "precision_negative",
        "precision_nan",
        "time_limit_negative",
    ],
)
def test_auto_iterate_rejects_arguments_it_cannot_honour(kwargs):
    """precision=0 used to loop forever; batch_size=0 spun without iterating."""
    with pytest.raises((ValueError, TypeError)):
        _base().auto_iterate(**kwargs)


# --------------------------------------------------------------------------- #
# handicap / komi keys
# --------------------------------------------------------------------------- #
def test_a_string_handicap_that_collides_with_a_number_is_rejected():
    """'0' read from a CSV is not the pinned baseline 0: it used to be
    estimated freely, silently shifting every rating."""
    w = WHR()
    with pytest.raises(ValueError, match=r"'0'.*0"):
        w.create_game("a", "b", "B", 1, "0")
    assert w.players == {}


def test_a_number_that_collides_with_an_existing_string_key_is_rejected():
    w = WHR()
    w.create_game("a", "b", "B", 1, "2")
    with pytest.raises(ValueError, match="2"):
        w.create_game("a", "b", "B", 2, 2)


def test_a_string_komi_that_collides_with_a_number_is_rejected():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0, komi=6.5)
    with pytest.raises(ValueError, match="6.5"):
        w.create_game("a", "b", "B", 2, 0, komi="6.5")


def test_string_keys_that_do_not_collide_are_fine():
    w = WHR()
    w.create_game("a", "b", "B", 1, "even")
    w.create_game("a", "b", "B", 2, 0, komi="half")
    w.create_game("a", "b", "B", 3, ("stones", 2))
    assert "even" in w.handicap_gamma and "half" in w.komi_gamma
    assert ("stones", 2) in w.handicap_gamma


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_a_misspelled_config_key_warns_with_a_suggestion():
    with pytest.warns(UserWarning, match=r"'draw_rates'.*'draw_rate'"):
        WHR({"draw_rates": 0.25})


def test_a_known_key_in_the_wrong_case_warns():
    with pytest.warns(UserWarning, match=r"'W2'.*'w2'"):
        WHR({"W2": 14})


def test_the_removed_debug_key_of_old_saves_is_ignored_quietly():
    WHR({"debug": False})  # filterwarnings=error: any warning fails this


def test_nested_config_dicts_are_not_shared_with_the_caller():
    pins = {1: 100.0}
    w = WHR({"pinned_handicap": pins})
    pins[1] = 300.0
    assert w.config["pinned_handicap"] == {1: 100.0}
    assert WHR(w.config).handicap_gamma[1] == w.handicap_gamma[1]


def test_display_uncertainty_is_checked_where_it_is_read():
    """The guide says it may be changed on a live instance; a typo there used
    to silently switch to elo."""
    w = _base()
    w.iterate(10)
    w.config["display_uncertainty"] = "Variance"
    with pytest.raises(ValueError, match="display_uncertainty"):
        w.ratings_for_player("a")
