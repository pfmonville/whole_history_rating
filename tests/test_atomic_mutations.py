"""A call that fails changes nothing.

``save_base`` used to truncate the target before pickling, so a failed save
destroyed the previous good file. ``load_games`` committed each line as it went,
so a bad line left the lines before it loaded and a retry counted them twice.
``create_game`` created both players before checking the advantage keys, so an
unhashable handicap left two players with no games behind.
"""

import os
import pickle
import stat

import pytest

from whr.whole_history_rating import WHR


def _state(w):
    """What a failed call must not have touched."""
    return (
        sorted(w.players),
        [str(g) for g in w.games],
        w.games_since_last_fit,
        dict(w.handicap_gamma),
        dict(w.komi_gamma),
    )


def _fitted():
    w = WHR({"w2": 30})
    w.load_games(["a b B 1", "a b W 2", "b c B 2", "a c W 3"])
    w.iterate(20)
    return w


# --------------------------------------------------------------------------- #
# save_base
# --------------------------------------------------------------------------- #
def test_a_failed_save_keeps_the_previous_file(tmp_path):
    path = tmp_path / "base.pkl"
    w = _fitted()
    w.save_base(path)
    before = path.read_bytes()

    w.create_game("a", "b", "B", 4, 0, extras={"note": lambda: None})
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        w.save_base(path)

    assert path.read_bytes() == before
    assert WHR.load_base(path).ratings_for_player("a") == _fitted().ratings_for_player(
        "a"
    )


def test_a_failed_save_leaves_no_temporary_file_behind(tmp_path):
    w = _fitted()
    w.create_game("a", "b", "B", 4, 0, extras={"note": lambda: None})
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        w.save_base(tmp_path / "base.pkl")
    assert os.listdir(tmp_path) == []


def test_saving_over_a_file_keeps_its_permissions(tmp_path):
    path = tmp_path / "base.pkl"
    w = _fitted()
    w.save_base(path)
    path.chmod(0o640)
    w.save_base(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_saving_through_a_symlink_updates_its_target(tmp_path):
    target = tmp_path / "real.pkl"
    link = tmp_path / "link.pkl"
    w = _fitted()
    w.save_base(target)
    link.symlink_to(target)

    w.create_game("a", "b", "B", 4, 0)
    w.save_base(link)

    assert link.is_symlink()
    assert len(WHR.load_base(target).games) == 5


# --------------------------------------------------------------------------- #
# create_game
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        {"handicap": [1]},
        {"komi": {"x": 1}},
        {"winner": "tie"},
        {"time_step": float("nan")},
        {"white": "a"},
    ],
    ids=["unhashable_handicap", "unhashable_komi", "winner", "time_step", "self_play"],
)
def test_a_rejected_game_changes_nothing(bad):
    w = _fitted()
    before = _state(w)
    game = {"black": "a", "white": "newcomer", "winner": "B", "time_step": 5}
    game.update(bad)
    game.setdefault("handicap", 0)
    with pytest.raises((TypeError, ValueError, AttributeError)):
        w.create_game(**game)
    assert _state(w) == before


# --------------------------------------------------------------------------- #
# load_games
# --------------------------------------------------------------------------- #
def test_a_bad_line_loads_nothing():
    w = _fitted()
    before = _state(w)
    with pytest.raises(ValueError):
        w.load_games(["d e B 5", "d f W 6", "d e tie 7", "e f B 8"])
    assert _state(w) == before


def test_a_retry_after_fixing_the_line_loads_each_game_once():
    w = WHR()
    lines = ["d e B 5", "d f W 6", "d e tie 7"]
    with pytest.raises(ValueError):
        w.load_games(lines)
    lines[2] = "d e D 7"
    w.load_games(lines)
    assert len(w.games) == 3


def test_the_error_names_the_offending_line():
    w = WHR()
    with pytest.raises(ValueError) as excinfo:
        w.load_games(["d e B 5", "d f W 6", "d e tie 7"])
    assert "line 3" in "\n".join(excinfo.value.__notes__)


def test_blank_lines_are_skipped():
    """``text.split("\\n")`` on a file that ends in a newline yields an empty
    last line; so does a blank line between records."""
    w = WHR()
    w.load_games("d e B 5\n\n  \nd f W 6\n".split("\n"))
    assert len(w.games) == 2


@pytest.mark.parametrize("keys", [(6.5, "6.5"), ("6.5", 6.5)])
def test_lookalike_keys_within_a_batch_are_rejected_atomically(keys):
    w = _fitted()
    before = _state(w)
    lines = [f"d e B 5 {{'komi': {key!r}}}" for key in keys]
    with pytest.raises(ValueError, match="komi") as excinfo:
        w.load_games(lines)
    assert "line 2" in "\n".join(excinfo.value.__notes__)
    assert _state(w) == before


def test_consistent_new_categories_in_a_batch_roundtrip(tmp_path):
    w = WHR()
    w.load_games(["a b B 1 {'komi': 6.5}", "a b W 2 {'komi': 6.5}"])
    w.iterate(20)
    path = tmp_path / "base.pkl"
    w.save_base(path)
    loaded = WHR.load_base(path)
    assert loaded.ratings_for_player("a") == w.ratings_for_player("a")
    assert loaded.komi_gamma == w.komi_gamma
