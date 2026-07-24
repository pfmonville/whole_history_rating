"""Equivalence-regression safety net for Phase-7 (numpy vectorization).

Tasks 2-3 will vectorize the per-game Newton-update loops. Vectorizing sums
over games/days reorders float summation, which can shift results by
~1e-12. This module freezes the CURRENT (pre-vectorization) outputs of
three representative scenarios — draw-free ratings, handicap/komi, and
draws — as hard-coded ground truth, and asserts the (eventually vectorized)
code reproduces them within rel=1e-9 / abs=1e-9. This is a much tighter
tolerance than the drift vectorization is expected to introduce, so any
regression beyond ordinary float-reordering noise should trip these tests.

The `expected_*` constants below are not placeholders: they were captured by
running each scenario against the pre-vectorization code (this same
codebase, before Phase-7 tasks 2-3 land) and reading the actual outputs.
"""

import pytest

from whr.whole_history_rating import WHR


def _scenario_draw_free():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "c b B 2", "a c B 3", "a b B 4"])
    w.iterate(50)
    return w


def _scenario_handicap_komi():
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    for d in range(1, 8):
        w.create_game("x", "y", "B", d, 2)
        w.create_game("y", "x", "W", d, 2)
    w.iterate(50)
    return w


def _scenario_draws():
    w = WHR()
    for d in range(1, 8):
        w.create_game("p", "q", "D", d, 0)
        w.create_game("p", "q", "B", d, 0)
    w.iterate(50)
    return w


def test_equivalence_draw_free_ratings():
    w = _scenario_draw_free()
    got = dict(w.get_ordered_ratings(current=True))
    # Frozen pre-vectorization ground truth (Phase-7 task 1).
    expected = {
        "a": -88.82430411827175,
        "c": 1.8381608951720134,
        "b": 95.38534493183062,
    }
    assert got.keys() == expected.keys()
    for name, elo in expected.items():
        assert got[name] == pytest.approx(elo, rel=1e-9, abs=1e-9)


def test_equivalence_handicap_komi_gamma_and_ratings():
    w = _scenario_handicap_komi()
    # Frozen pre-vectorization ground truth (Phase-7 task 1).
    expected_handicap_gamma_2 = 3.1622776601683795
    expected_ratings = {
        "x": 303.4581914690472,
        "y": -329.10626468760404,
    }

    assert w.handicap_gamma[2] == pytest.approx(
        expected_handicap_gamma_2, rel=1e-9, abs=1e-9
    )
    got = dict(w.get_ordered_ratings(current=True))
    assert got.keys() == expected_ratings.keys()
    for name, elo in expected_ratings.items():
        assert got[name] == pytest.approx(elo, rel=1e-9, abs=1e-9)


def test_equivalence_draws_tendency_wdl_and_log_likelihood():
    w = _scenario_draws()
    # Frozen pre-vectorization ground truth (Phase-7 task 1).
    expected_draw_tendency = 4.76010000035184
    expected_wdl_pq = (
        0.47784280615549235,
        0.49914605655153316,
        0.02301113729297449,
    )
    expected_log_likelihood = -5.341776751124541

    assert w.draw_tendency == pytest.approx(expected_draw_tendency, rel=1e-9, abs=1e-9)
    got_wdl = w.win_draw_loss_probabilities("p", "q")
    assert got_wdl == pytest.approx(expected_wdl_pq, rel=1e-9, abs=1e-9)
    assert w.log_likelihood() == pytest.approx(
        expected_log_likelihood, rel=1e-9, abs=1e-9
    )
