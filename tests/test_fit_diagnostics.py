"""The fit's diagnostics describe the model actually being fitted.

Two properties, each checked directly rather than against frozen numbers:

* reads do not depend on what was called before them. Each player-day caches
  its opponents' gammas; a read that trusted a cache filled before games were
  added (or before ``remove_drift`` moved the ratings) reported the old state,
  until some unrelated call happened to clear it;
* ``log_likelihood()`` is the log-posterior the fit maximizes, so at a
  converged fit its derivative in every fitted direction is zero.
"""

import math
import random

import pytest

from whr.whole_history_rating import WHR


def _fitted(outcomes="BW", handicap=False, **config):
    w = WHR({"w2": 30, **config})
    rng = random.Random(11)
    for _ in range(160):
        a, b = rng.sample(list("abcde"), 2)
        w.create_game(
            a,
            b,
            rng.choice(outcomes),
            rng.randrange(12),
            rng.choice([0, 1]) if handicap else 0,
            komi=rng.choice([6.5, 0.5]) if handicap else None,
        )
    w.iterate(200)
    return w


def _add_games_to_an_existing_day(w):
    day = w.players["a"].days[2].day
    for _ in range(15):
        w.create_game("a", "b", "B", day, 0)


def _remove_drift(w):
    w.remove_drift()


CHANGES = {"games_added": _add_games_to_an_existing_day, "drift_removed": _remove_drift}
READS = {
    "log_likelihood": lambda w: w.log_likelihood(),
    "rating_covariance": lambda w: w.rating_covariance("a")[1].tolist(),
    "rating_change": lambda w: w.rating_change(
        "a", w.players["a"].days[0].day, w.players["a"].days[-1].day
    ),
}


@pytest.mark.filterwarnings("ignore::whr.StaleFitWarning")
@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_a_read_does_not_depend_on_what_ran_before_it(read, change):
    w = _fitted()
    read(w)  # fill the caches with the state before the change
    change(w)
    first = read(w)
    w.max_gradient_norm()  # an unrelated read that happens to clear them
    assert read(w) == first


def test_log_likelihood_of_a_single_game_counts_it_once():
    w = WHR({"w2": 30})
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(30)
    a, b = w.players["a"].days[0], w.players["b"].days[0]
    game = math.log(a.gamma() / (a.gamma() + b.gamma()))
    priors = a.anchor_log_likelihood() + b.anchor_log_likelihood()
    assert w.log_likelihood() == pytest.approx(game + priors, rel=1e-12)


def _derivative(w, get, put, h=1e-6):
    """Central finite difference of log_likelihood along one fitted value."""
    x = get()
    put(x + h)
    up = w.log_likelihood()
    put(x - h)
    down = w.log_likelihood()
    put(x)
    return (up - down) / (2 * h)


BASES = {
    "decisive": lambda: _fitted(),
    "draws": lambda: _fitted("BWD"),
    "handicap_komi": lambda: _fitted("BW", handicap=True),
    # draws in the data, but "no draws" declared: nu stays 0 and the fit
    # carries no draw term, so neither may the log-posterior
    "draws_declared_absent": lambda: _fitted("BWD", pinned_draw=0.0),
}


@pytest.mark.parametrize("make", BASES.values(), ids=BASES.keys())
def test_log_likelihood_is_maximal_at_the_fit(make):
    w = make()
    w.auto_iterate(precision=1e-10)
    directions = []
    for player in w.players.values():
        for day in player.days:
            directions.append((lambda d=day: d.r, lambda v, d=day: setattr(d, "r", v)))
    for table, pinned in (
        (w.handicap_gamma, w._pinned_handicap_keys),
        (w.komi_gamma, w._pinned_komi_keys),
    ):
        for key in set(table) - pinned:
            directions.append(
                (
                    lambda t=table, k=key: math.log(t[k]),
                    lambda v, t=table, k=key: t.__setitem__(k, math.exp(v)),
                )
            )
    if w.nu > 0:
        directions.append(
            (lambda: math.log(w.nu), lambda v: setattr(w, "nu", math.exp(v)))
        )
    slopes = [_derivative(w, get, put) for get, put in directions]
    assert max(abs(s) for s in slopes) < 1e-5
