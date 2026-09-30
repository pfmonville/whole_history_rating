"""save_base / load_base must be invisible: a reloaded base behaves exactly like
the one that was saved.

Every piece of fitted state added so far (players without games, handicap/komi
advantages, the draw tendency, the fit status) was at first forgotten by the
save format and fixed one at a time. Instead of one test per attribute, these
tests compare everything a caller can observe, before and after a round trip,
over bases that exercise each feature. A new piece of state that the format
forgets shows up here without anyone having to remember to test it.
"""

import pickle
import random
import warnings

import numpy as np
import pytest

from whr.whole_history_rating import WHR


def _games(w, rng, n_games, players, days, outcomes="BW"):
    for _ in range(n_games):
        a, b = rng.sample(players, 2)
        w.create_game(a, b, rng.choice(outcomes), rng.randrange(days), 0)


def _plain():
    w = WHR({"w2": 30})
    _games(w, random.Random(1), 120, list("abcde"), 15)
    w.iterate(50)
    return w


def _with_draws():
    w = WHR({"w2": 30})
    _games(w, random.Random(2), 150, list("abcde"), 15, outcomes="BWD")
    w.iterate(50)
    return w


def _pinned_draws():
    w = WHR({"w2": 30, "pinned_draw": 0.4})
    _games(w, random.Random(3), 100, list("abcd"), 10, outcomes="BWD")
    w.iterate(50)
    return w


def _handicap_komi():
    w = WHR({"w2": 30, "pinned_handicap": {2: 60.0}})
    rng = random.Random(4)
    for _ in range(150):
        a, b = rng.sample(list("abcde"), 2)
        w.create_game(
            a,
            b,
            rng.choice("BW"),
            rng.randrange(15),
            rng.choice([0, 1, 2]),
            komi=rng.choice([6.5, 0.5]),
        )
    w.iterate(50)
    return w


def _stale():
    w = _with_draws()
    w.create_game("a", "b", "W", 3, 0)
    w.create_game("c", "d", "D", 14, 0)
    return w


def _unfitted():
    w = WHR({"w2": 30})
    _games(w, random.Random(5), 40, list("abcd"), 8)
    return w


def _uncased_with_idle_player():
    w = WHR({"w2": 30, "uncased": True})
    _games(w, random.Random(6), 80, ["Ann", "BOB", "cy", "Dee"], 10)
    w.iterate(50)
    # queried before ever playing: kept by the base, with no games
    w.player_by_name("Newcomer")
    return w


def _disconnected():
    w = WHR({"w2": 30})
    rng = random.Random(7)
    _games(w, rng, 60, list("abc"), 10)
    _games(w, rng, 60, list("xyz"), 10)
    w.iterate(50)
    return w


BASES = {
    "plain": _plain,
    "draws": _with_draws,
    "pinned_draws": _pinned_draws,
    "handicap_komi": _handicap_komi,
    "stale": _stale,
    "unfitted": _unfitted,
    "uncased_idle_player": _uncased_with_idle_player,
    "disconnected": _disconnected,
}


def _observe(w):
    """Everything a caller can read from ``w``, plus the warnings each read
    raises. Reads only: ``w`` is left as a continuation can use it."""
    seen = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        names = sorted(w.players)
        rated = [n for n in names if w.players[n].days]
        seen["config"] = w.config
        seen["games_since_last_fit"] = w.games_since_last_fit
        seen["draw_tendency"] = w.draw_tendency
        seen["handicap_gamma"] = dict(w.handicap_gamma)
        seen["komi_gamma"] = dict(w.komi_gamma)
        seen["components"] = w.connected_components()
        seen["ratings"] = {n: w.ratings_for_player(n) for n in rated}
        seen["ordered"] = w.get_ordered_ratings()
        seen["log_likelihood"] = w.log_likelihood()
        seen["max_gradient_norm"] = w.max_gradient_norm()
        seen["covariance"] = {
            n: (lambda d, c: (d, np.asarray(c).tolist()))(*w.rating_covariance(n))
            for n in rated
        }
        seen["predictions"] = {
            (a, b): (
                w.probability_future_match(a, b),
                w.win_draw_loss_probabilities(a, b),
                w.probability_future_match(a, b, account_for_uncertainty=True),
            )
            for a in names
            for b in names
            if a < b
        }
    seen["warnings"] = sorted({type(c.message).__name__ for c in caught})
    return seen


def _roundtrip(w, tmp_path):
    path = tmp_path / "base.pkl"
    w.save_base(path)
    return WHR.load_base(path)


@pytest.mark.parametrize("make", BASES.values(), ids=BASES.keys())
def test_a_reloaded_base_reads_exactly_like_the_saved_one(make, tmp_path):
    original = make()
    loaded = _roundtrip(make(), tmp_path)
    assert _observe(loaded) == _observe(original)


@pytest.mark.parametrize("make", BASES.values(), ids=BASES.keys())
def test_a_reloaded_base_continues_exactly_like_the_saved_one(make, tmp_path):
    """New games and further iterations after a reload land where they would
    have landed without the save/load in between."""
    original = make()
    loaded = _roundtrip(make(), tmp_path)
    for w in (original, loaded):
        w.create_game("a" if "a" in w.players else "ann", "Z", "W", 20, 0)
    assert _observe(loaded) == _observe(original)
    for w in (original, loaded):
        w.iterate(5)
    assert _observe(loaded) == _observe(original)


def test_the_saved_file_records_its_format_version(tmp_path):
    path = tmp_path / "base.pkl"
    _plain().save_base(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    assert data["format_version"] == WHR.SAVE_FORMAT_VERSION


def test_a_file_from_a_newer_format_is_refused_rather_than_half_read(tmp_path):
    """Loading it would silently drop whatever state the newer format added."""
    path = tmp_path / "base.pkl"
    _plain().save_base(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    data["format_version"] = WHR.SAVE_FORMAT_VERSION + 1
    with path.open("wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="newer version of whole-history-rating"):
        WHR.load_base(path)


def test_a_file_without_a_format_version_still_loads(tmp_path):
    """Bases saved by 2.0.0 - 3.6.1 carry no version: read them as before."""
    path = tmp_path / "base.pkl"
    original = _with_draws()
    original.save_base(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    for key in ("format_version", "ever_fitted", "games_since_fit"):
        data.pop(key)
    with path.open("wb") as f:
        pickle.dump(data, f)
    loaded = WHR.load_base(path)
    assert loaded.ratings_for_player("a") == original.ratings_for_player("a")
    assert loaded.log_likelihood() == original.log_likelihood()


def _save_pre_rounding_days(path, version, collision=False):
    """Flat save schema of 2.0.0--3.6.2, before dates were rounded on input.

    Write the old wire format directly so this remains a migration test even
    when today's create_game/save_base normalizes all its dates.
    """
    days = [0.1 + 0.2, 1.3]
    if collision:
        days.insert(0, 0.3)
    state = {
        "config": {"w2": 300},
        "games": [
            ("a", "b", "B" if i % 2 else "W", d, 0, {}) for i, d in enumerate(days)
        ],
        "ratings": {
            "a": [(d, 0.2 + i * 0.1, 0.4) for i, d in enumerate(days)],
            "b": [(d, -0.2 - i * 0.1, 0.5) for i, d in enumerate(days)],
            "idle": [],
        },
        "nu": 0.0,
        "handicap_gamma": {0: 1.0},
        "komi_gamma": {},
    }
    if version == 2:
        state.update(format_version=2, ever_fitted=True, games_since_fit=0)
    path.write_bytes(pickle.dumps(state))
    return state


@pytest.mark.parametrize("version", [1, 2])
def test_pre_rounding_save_preserves_ratings_after_date_migration(tmp_path, version):
    path = tmp_path / "old.pkl"
    state = _save_pre_rounding_days(path, version)
    w = WHR.load_base(path)
    for name in ("a", "b"):
        assert [d.day for d in w.players[name].days] == [0.3, 1.3]
        assert [(d.r, d.uncertainty) for d in w.players[name].days] == [
            (r, u) for _, r, u in state["ratings"][name]
        ]
    assert "idle" in w.players
    if version == 2:
        assert w.games_since_last_fit == 0
    loaded = _roundtrip(w, tmp_path)
    w.iterate(5)
    loaded.iterate(5)
    assert _observe(w) == _observe(loaded)


@pytest.mark.parametrize("version", [1, 2])
def test_colliding_saved_days_keep_games_but_require_a_new_fit(tmp_path, version):
    path = tmp_path / "old.pkl"
    _save_pre_rounding_days(path, version, collision=True)
    with pytest.warns(UserWarning, match="merged.*refit"):
        w = WHR.load_base(path)
    assert len(w.games) == 3
    assert [d.day for d in w.players["a"].days] == [0.3, 1.3]
    assert "idle" in w.players
    assert w.games_since_last_fit == 3
    assert all(d.uncertainty == -1 for p in w.players.values() for d in p.days)
    with pytest.raises(ValueError, match="uncertainties not computed"):
        w.rating_difference("a", "b")
    w.iterate(30)
    assert w.games_since_last_fit == 0
    assert _observe(_roundtrip(w, tmp_path)) == _observe(w)
