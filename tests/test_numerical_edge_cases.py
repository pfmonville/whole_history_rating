"""Edge cases of the model: data the draw model cannot fit, extreme handicaps,
settings changed on a live base, and the starting point of an inserted day."""

import math

import pytest

from whr import DrawModelWarning
from whr.whole_history_rating import WHR


# --------------------------------------------------------------------------- #
# draw data the Davidson model cannot fit
# --------------------------------------------------------------------------- #
def _all_draws(**config):
    w = WHR({"w2": 30, **config})
    for day in range(1, 11):
        w.create_game("a", "b", "D", day, 0)
        w.create_game("b", "c", "D", day, 0)
    return w


def test_when_every_game_is_a_draw_nu_does_not_run_away():
    """Its maximum-likelihood value is infinite: nu used to grow with every
    iteration (4e4 after 1000, 2e5 after 5000) while auto_iterate reported
    convergence."""
    short, long = _all_draws(), _all_draws()
    with pytest.warns(DrawModelWarning, match="every game is a draw"):
        short.iterate(10)
    with pytest.warns(DrawModelWarning):
        long.iterate(1000)
    assert long.nu == short.nu
    assert math.isfinite(long.nu)


def test_when_every_game_is_a_draw_auto_iterate_converges_on_the_ratings():
    w = _all_draws()
    with pytest.warns(DrawModelWarning):
        iterations, converged = w.auto_iterate(precision=1e-6)
    assert converged and iterations <= 200


def test_a_declared_draw_rate_fits_all_draw_data_quietly():
    w = _all_draws(draw_rate=0.9)
    w.iterate(10)  # filterwarnings=error: no DrawModelWarning


def _with_handicap(draws, **config):
    w = WHR({"w2": 30, **config})
    for day in range(1, 21):
        w.create_game("a", "b", "B", day, 1)
        w.create_game("b", "a", "W", day, 1)
        w.create_game("a", "b", "W", day, 0)
        w.create_game("b", "a", "B", day, 0)
        if draws:
            w.create_game("a", "b", "D", day, 1)
            w.create_game("b", "a", "D", day, 1)
    return w


def test_draws_in_data_declared_drawless_are_left_out_of_every_update():
    """With pinned_draw=0 the player updates ignored drawn games while the
    handicap update counted them as half-wins: 30 draws moved handicap 1
    from 0.33 to 1.08."""
    without = _with_handicap(draws=False, pinned_draw=0.0)
    without.iterate(100)
    drawn = _with_handicap(draws=True, pinned_draw=0.0)
    with pytest.warns(DrawModelWarning, match="40 drawn game"):
        drawn.iterate(100)
    assert drawn.handicap_gamma[1] == pytest.approx(without.handicap_gamma[1])
    assert drawn.ratings_for_player("a") == without.ratings_for_player("a")


# --------------------------------------------------------------------------- #
# extreme raw handicaps in the predictors
# --------------------------------------------------------------------------- #
def _fitted():
    w = WHR({"w2": 30, "draw_rate": 0.2})
    w.load_games(["a b B 1", "a b W 2", "a b D 3"])
    w.iterate(30)
    return w


@pytest.mark.parametrize("handicap", [1e6, -1e6, 1e300, -1e300])
def test_an_extreme_handicap_saturates_instead_of_overflowing(handicap):
    w = _fitted()
    black_wins = handicap > 0
    p1, p2 = w.probability_future_match("a", "b", handicap)
    assert (p1, p2) == pytest.approx((1.0, 0.0) if black_wins else (0.0, 1.0))
    win, draw, loss = w.win_draw_loss_probabilities("a", "b", handicap)
    assert (win, draw, loss) == pytest.approx(
        (1.0, 0.0, 0.0) if black_wins else (0.0, 0.0, 1.0), abs=1e-12
    )
    p1k, _ = w.probability_future_match("a", "b", handicap, handicap_key=0)
    assert p1k == pytest.approx(1.0 if black_wins else 0.0)


def test_an_ordinary_handicap_gives_the_same_numbers_as_before():
    w = _fitted()
    a, b = w.players["a"].days[-1], w.players["b"].days[-1]
    expected = a.gamma() / (a.gamma() + 10 ** ((b.elo - 150) / 400.0))
    assert w.probability_future_match("a", "b", 150)[0] == expected


@pytest.mark.parametrize(
    "method", ["probability_future_match", "win_draw_loss_probabilities"]
)
@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("key", [None, 0])
@pytest.mark.parametrize("handicap", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_handicaps_are_rejected(method, integrated, key, handicap):
    w = _fitted()
    with pytest.raises(ValueError, match="handicap.*finite"):
        getattr(w, method)(
            "a",
            "b",
            handicap,
            handicap_key=key,
            account_for_uncertainty=integrated,
        )


# --------------------------------------------------------------------------- #
# settings changed on a live base
# --------------------------------------------------------------------------- #
# "d" first plays after the change: the old code gave it the new value and
# kept the old one for a, b and c
GAMES = ["a b B 1", "a b W 5", "b c B 5", "a c W 9", "c a B 12", "d b B 12", "a d W 13"]


@pytest.mark.parametrize(
    "setting, value",
    [("w2", 10.0), ("initial_prior_wins", 2.0), ("hessian_damping", 0.5)],
)
def test_a_setting_changed_on_a_live_base_applies_to_every_player(setting, value):
    """Players copied these settings when created: after a change, old players
    kept the old value and new ones took the new, and save/load then rebuilt
    everyone with the new one -- a different model."""
    changed = WHR({"w2": 300.0})
    changed.load_games(GAMES[:3])
    changed.config[setting] = value
    changed.load_games(GAMES[3:])
    changed.iterate(100)

    fresh = WHR({"w2": 300.0, setting: value})
    fresh.load_games(GAMES)
    fresh.iterate(100)

    for name in "abcd":
        assert changed.ratings_for_player(name) == fresh.ratings_for_player(name)


# --------------------------------------------------------------------------- #
# an inserted day's starting point
# --------------------------------------------------------------------------- #
def test_a_day_inserted_before_a_players_first_day_starts_from_that_day():
    """It used to start from the player's *last* day (index -1 wrapped around)."""
    w = WHR({"w2": 3000})
    for day, winner in [(10, "B")] * 6 + [(20, "W")] * 6:
        w.create_game("a", "b", winner, day, 0)
    w.iterate(50)
    first, last = w.players["a"].days[0], w.players["a"].days[-1]
    assert abs(first.r - last.r) > 0.1  # the test needs two distinct days

    w.create_game("a", "b", "B", 1, 0)
    inserted = w.players["a"].days[0]
    assert inserted.day == 1
    assert inserted.r == first.r


# --------------------------------------------------------------------------- #
# Game-level probabilities and draws
# --------------------------------------------------------------------------- #
def test_game_win_probabilities_are_documented_as_given_a_decisive_result():
    from whr import Game

    for method in (Game.white_win_probability, Game.black_win_probability):
        assert "decisive" in method.__doc__


def test_a_drawn_game_scores_as_neither_side_predicted():
    w = _fitted()
    drawn = next(g for g in w.games if g.winner == "D")
    assert drawn.prediction_score() == 0.5
