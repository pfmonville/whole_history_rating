"""Declaring whether a domain has draws at all.

Absence of draws in the data is ambiguous: it can mean "this sport cannot draw"
(tennis) or "no draw has happened yet" (two weeks into a football season). The
library cannot tell those apart, and the difference matters -- the first makes
``P(draw) = 0`` correct, the second makes it a confident false claim that also
sends any log-loss to infinity the moment a draw occurs.

So the intent is the caller's to declare, through ``pinned_draw`` /
``draw_rate``, and the ambiguous case warns rather than guessing.
"""

import math
import warnings

import pytest

from whr import NoDrawsWarning
from whr.whole_history_rating import WHR


def _fit(w, *, draws=0, day_of_draw=5):
    """A > B > C from results alone, optionally with some drawn A-B games."""
    for day in range(1, 40):
        w.create_game("a", "b", "B", day, 0)
        w.create_game("b", "c", "B", day, 0)
    for i in range(draws):
        w.create_game("a", "b", "D", day_of_draw + i, 0)
    w.auto_iterate(time_limit=10, precision=1e-3)
    return w


def _wdl(w, *args, **kwargs):
    """Predict without letting the ambiguity warning fail the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDrawsWarning)
        return w.win_draw_loss_probabilities(*args, **kwargs)


# --------------------------------------------------------------------------- #
# pinned_draw must apply whether or not a draw was ever recorded
# --------------------------------------------------------------------------- #
def test_pinned_draw_applies_with_no_draws_in_the_data():
    """The regression this module exists for: ``pinned_draw`` used to be applied
    only inside ``create_game``'s draw branch, so it was silently ignored in the
    one situation where a caller needs it -- knowing draws are possible while
    having observed none yet."""
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.79}))
    assert w.nu == pytest.approx(0.79)
    win, draw, loss = _wdl(w, "a", "b", 0)
    assert draw > 0.0
    assert win + draw + loss == pytest.approx(1.0)


def test_pinned_draw_is_honoured_before_any_game_is_created():
    assert WHR({"pinned_draw": 0.5}).nu == pytest.approx(0.5)
    assert WHR({"pinned_draw": 0.0}).nu == 0.0
    assert WHR().nu == 0.0


def test_pinned_draw_still_wins_over_the_data_when_draws_exist():
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.25}), draws=6)
    assert w.nu == pytest.approx(0.25)  # not re-fitted


def test_pinned_draw_zero_disables_draws_even_with_draws_present():
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.0}), draws=6)
    assert w.nu == 0.0
    assert _wdl(w, "a", "b", 0)[1] == 0.0


def test_unpinned_draws_are_still_estimated_from_the_data():
    """The default path must be untouched."""
    w = _fit(WHR({"w2": 30}), draws=6)
    assert w.nu > 0.0
    assert _wdl(w, "a", "b", 0)[1] > 0.0


# --------------------------------------------------------------------------- #
# draw_rate: state the intent in the unit the caller actually has
# --------------------------------------------------------------------------- #
def test_nu_from_draw_rate_matches_the_even_matchup_identity():
    """At equal strengths ``T = nu*s`` and ``Z = (2 + nu)*s``, so
    ``P(draw) = nu / (2 + nu)`` and a target rate ``p`` needs
    ``nu = 2p / (1 - p)``."""
    for p in (0.0, 0.05, 0.1, 0.25, 0.252, 0.5):
        nu = WHR.nu_from_draw_rate(p)
        assert nu == pytest.approx(2 * p / (1 - p))
        assert WHR.draw_rate_from_nu(nu) == pytest.approx(p)


def test_draw_rate_delivers_the_requested_rate_on_an_even_matchup():
    for p in (0.1, 0.25, 0.33):
        w = WHR({"w2": 30, "draw_rate": p})
        for day in range(1, 30):  # perfectly symmetric pair
            w.create_game("x", "y", "B", day, 0)
            w.create_game("x", "y", "W", day, 0)
        w.auto_iterate(time_limit=10, precision=1e-3)
        assert _wdl(w, "x", "y", 0)[1] == pytest.approx(p, abs=1e-3)


def test_draw_rate_zero_is_a_deliberate_no_draws_declaration():
    w = _fit(WHR({"w2": 30, "draw_rate": 0.0}))
    assert w.nu == 0.0
    assert _wdl(w, "a", "b", 0)[1] == 0.0


def test_draw_rate_and_pinned_draw_together_is_an_error():
    with pytest.raises(
        ValueError, match="draw_rate.*pinned_draw|pinned_draw.*draw_rate"
    ):
        WHR({"draw_rate": 0.25, "pinned_draw": 0.79})


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, float("nan")])
def test_draw_rate_outside_zero_one_is_rejected(bad):
    with pytest.raises(ValueError):
        WHR({"draw_rate": bad})


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, float("nan")])
def test_nu_from_draw_rate_rejects_the_same_range(bad):
    with pytest.raises(ValueError):
        WHR.nu_from_draw_rate(bad)


def test_negative_pinned_draw_is_rejected():
    """A negative draw tendency has no meaning in Davidson's model."""
    with pytest.raises(ValueError):
        WHR({"pinned_draw": -0.5})


# --------------------------------------------------------------------------- #
# the ambiguous case warns, once, and names both resolutions
# --------------------------------------------------------------------------- #
def test_three_outcome_prediction_warns_when_draws_were_never_observed():
    w = _fit(WHR({"w2": 30}))
    with pytest.warns(NoDrawsWarning) as record:
        win, draw, loss = w.win_draw_loss_probabilities("a", "b", 0)
    assert draw == 0.0
    assert win + draw + loss == pytest.approx(1.0)
    message = str(record[0].message)
    assert "pinned_draw" in message and "draw_rate" in message


def test_the_warning_fires_only_once_per_instance():
    """A scoring loop over a season must not emit thousands of warnings."""
    w = _fit(WHR({"w2": 30}))
    with pytest.warns(NoDrawsWarning) as record:
        for _ in range(50):
            w.win_draw_loss_probabilities("a", "b", 0)
    assert len(record) == 1


def test_no_warning_once_a_draw_has_been_observed():
    w = _fit(WHR({"w2": 30}), draws=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoDrawsWarning)
        w.win_draw_loss_probabilities("a", "b", 0)


@pytest.mark.parametrize(
    "cfg",
    [
        {"pinned_draw": 0.0},
        {"pinned_draw": 0.7},
        {"draw_rate": 0.0},
        {"draw_rate": 0.2},
    ],
)
def test_declaring_the_intent_silences_the_warning(cfg):
    """Including ``pinned_draw=0.0`` -- 'this domain has no draws' is an answer,
    not the absence of one."""
    w = _fit(WHR({"w2": 30, **cfg}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoDrawsWarning)
        w.win_draw_loss_probabilities("a", "b", 0)


def test_two_outcome_prediction_never_warns():
    """``probability_future_match`` makes no claim about draws."""
    w = _fit(WHR({"w2": 30}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoDrawsWarning)
        w.probability_future_match("a", "b", 0)


def test_warning_subclasses_userwarning_so_existing_filters_catch_it():
    assert issubclass(NoDrawsWarning, UserWarning)


# --------------------------------------------------------------------------- #
# the reduction itself stays exact
# --------------------------------------------------------------------------- #
def test_nu_zero_reduces_exactly_to_the_two_outcome_prediction():
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.0}))
    win, draw, loss = _wdl(w, "a", "b", 0)
    p1, p2 = w.probability_future_match("a", "b", 0)
    assert draw == 0.0
    assert win == pytest.approx(p1, abs=1e-15)
    assert loss == pytest.approx(p2, abs=1e-15)


def test_nu_zero_reduces_exactly_under_the_uncertainty_integration_too():
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.0}))
    win, draw, loss = _wdl(w, "a", "b", 0, account_for_uncertainty=True)
    p1, p2 = w.probability_future_match("a", "b", 0, account_for_uncertainty=True)
    assert draw == 0.0
    assert win == pytest.approx(p1, abs=1e-15)
    assert loss == pytest.approx(p2, abs=1e-15)


def test_zero_draw_probability_is_what_makes_log_loss_undefined():
    """Documents *why* the ambiguous case is worth a warning rather than being
    left to the caller to notice."""
    w = _fit(WHR({"w2": 30, "pinned_draw": 0.0}))
    p_draw = _wdl(w, "a", "b", 0)[1]
    assert p_draw == 0.0
    with pytest.raises(ValueError):
        math.log(p_draw)


def test_declared_draw_rate_survives_a_save_load_round_trip(tmp_path):
    """A reloaded base must still know the intent, or ``nu`` would silently be
    re-fitted (or warned about) on further iteration."""
    path = str(tmp_path / "base.pkl")
    w = _fit(WHR({"w2": 30, "draw_rate": 0.25}))
    w.save_base(path)
    back = WHR.load_base(path)
    assert back.draws_declared is True
    assert back.nu == pytest.approx(w.nu)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoDrawsWarning)
        back.win_draw_loss_probabilities("a", "b", 0)


def test_draw_rate_is_kept_by_the_unpicklable_config_fallback(tmp_path):
    """``save_base`` falls back to a whitelist of config keys when the config
    cannot be pickled; dropping ``draw_rate`` there would lose the declaration."""
    path = str(tmp_path / "base.pkl")
    w = _fit(WHR({"w2": 30, "draw_rate": 0.25, "unpicklable": lambda x: x}))
    with pytest.warns(UserWarning, match="cannot be pickled"):
        w.save_base(path)
    back = WHR.load_base(path)
    assert back.config["draw_rate"] == 0.25
    assert back.nu == pytest.approx(WHR.nu_from_draw_rate(0.25))
