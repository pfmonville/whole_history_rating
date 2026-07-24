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

import math

import pytest

from whr import playerday as PD
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


def _reference_accumulate_handicap_komi(w: WHR):
    """Pre-vectorization Python-loop reference for
    ``WHR._accumulate_handicap_komi`` (Task 2), kept here as an independent
    ground truth the vectorized implementation must reproduce."""
    h_grad: dict = {}
    h_hess: dict = {}
    k_grad: dict = {}
    k_hess: dict = {}
    h_games: dict = {}
    h_wins: dict = {}
    k_games: dict = {}
    k_wins: dict = {}
    for g in w.games:
        if g.bpd is None or g.wpd is None:
            continue
        if g.winner == "D":
            continue
        h = g.handicap
        k = g.extras["komi"]
        gh = w.handicap_gamma[h]
        gk = w.komi_gamma[k]
        gb = g.bpd.gamma()
        gw = g.wpd.gamma()
        c_komi = gw
        d_komi = gb * gh
        c_handicap = gb
        d_handicap = gw * gk
        div = 1.0 / (d_komi + d_handicap)
        h_grad[h] = h_grad.get(h, 0.0) + c_handicap * div
        h_hess[h] = h_hess.get(h, 0.0) + c_handicap * d_handicap * div * div
        k_grad[k] = k_grad.get(k, 0.0) + c_komi * div
        k_hess[k] = k_hess.get(k, 0.0) + c_komi * d_komi * div * div
        h_games[h] = h_games.get(h, 0) + 1
        k_games[k] = k_games.get(k, 0) + 1
        if g.winner == "B":
            h_wins[h] = h_wins.get(h, 0) + 1
        else:
            k_wins[k] = k_wins.get(k, 0) + 1
    return h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins


def _reference_nu_gradient_hessian(w: WHR):
    """Pre-vectorization Python-loop reference for
    ``WHR._nu_gradient_hessian`` (Task 2)."""
    gradient = 0.0
    hessian = 0.0
    for game in w.games:
        if game.bpd is None or game.wpd is None:
            continue
        s, o = game.effective_gammas(game.black_player)
        t = w.nu * math.sqrt(s * o)
        z = s + o + t
        ratio = t / z
        gradient += (1.0 if game.winner == "D" else 0.0) - ratio
        hessian += -ratio * (1.0 - ratio)
    return gradient, hessian


def _assert_dicts_close(got: dict, expected: dict) -> None:
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-9)


def test_unit_equivalence_accumulate_handicap_komi_and_nu_gradient():
    """Task 2: the vectorized ``_accumulate_handicap_komi``/
    ``_nu_gradient_hessian`` must match an independent reference Python-loop
    implementation of the same (pre-vectorization) algorithm, within
    rel=1e-9, on a base mixing handicap, komi, AND draws."""
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    for d in range(1, 6):
        w.create_game("x", "y", "B", d, 2, {"komi": 6.5})
        w.create_game("y", "x", "W", d, 2, {"komi": 6.5})
        w.create_game("x", "y", "D", d, 2, {"komi": 6.5})
        w.create_game("x", "y", "B", d, 0, {"komi": 0.5})
        w.create_game("y", "x", "W", d, 0, {"komi": 0.5})
    w.iterate(10)

    (
        h_grad_ref,
        h_hess_ref,
        k_grad_ref,
        k_hess_ref,
        h_games_ref,
        h_wins_ref,
        k_games_ref,
        k_wins_ref,
    ) = _reference_accumulate_handicap_komi(w)
    (
        h_grad,
        h_hess,
        k_grad,
        k_hess,
        h_games,
        h_wins,
        k_games,
        k_wins,
    ) = w._accumulate_handicap_komi()

    _assert_dicts_close(h_grad, h_grad_ref)
    _assert_dicts_close(h_hess, h_hess_ref)
    _assert_dicts_close(k_grad, k_grad_ref)
    _assert_dicts_close(k_hess, k_hess_ref)
    assert h_games == h_games_ref
    assert h_wins == h_wins_ref
    assert k_games == k_games_ref
    assert k_wins == k_wins_ref

    gradient_ref, hessian_ref = _reference_nu_gradient_hessian(w)
    gradient, hessian = w._nu_gradient_hessian()
    assert gradient == pytest.approx(gradient_ref, rel=1e-9, abs=1e-9)
    assert hessian == pytest.approx(hessian_ref, rel=1e-9, abs=1e-9)


def test_accumulate_handicap_komi_and_nu_gradient_empty_and_degenerate_cases():
    """Task 2 gotcha: empty games / no-decisive-games / no-draws must not
    raise numpy errors, and must return zero accumulation."""
    # No games at all.
    empty = WHR()
    h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins = (
        empty._accumulate_handicap_komi()
    )
    assert (h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins) == (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    gradient, hessian = empty._nu_gradient_hessian()
    assert (gradient, hessian) == (0.0, 0.0)

    # Only draws: no decisive games for handicap/komi accumulation, but
    # _nu_gradient_hessian still has games to accumulate over.
    only_draws = WHR()
    for d in range(1, 4):
        only_draws.create_game("p", "q", "D", d, 0)
    h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins = (
        only_draws._accumulate_handicap_komi()
    )
    assert (h_grad, h_hess, k_grad, k_hess, h_games, h_wins, k_games, k_wins) == (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    gradient_ref, hessian_ref = _reference_nu_gradient_hessian(only_draws)
    gradient, hessian = only_draws._nu_gradient_hessian()
    assert gradient == pytest.approx(gradient_ref, rel=1e-9, abs=1e-9)
    assert hessian == pytest.approx(hessian_ref, rel=1e-9, abs=1e-9)

    # No draws at all: _nu_gradient_hessian still must not error (no games
    # have bpd/wpd set here since none have been iterated, but the games
    # list is non-empty).
    no_draws = WHR()
    for d in range(1, 4):
        no_draws.create_game("p", "q", "B", d, 0)
    gradient_ref, hessian_ref = _reference_nu_gradient_hessian(no_draws)
    gradient, hessian = no_draws._nu_gradient_hessian()
    assert gradient == pytest.approx(gradient_ref, rel=1e-9, abs=1e-9)
    assert hessian == pytest.approx(hessian_ref, rel=1e-9, abs=1e-9)


def _reference_bt_terms(day: PD.PlayerDay) -> tuple[float, float, float]:
    """Pre-vectorization Python-loop reference for
    ``PlayerDay.log_likelihood_derivative``/``_second_derivative``/
    ``log_likelihood`` (Task 3), kept here as an independent ground truth
    the vectorized implementation must reproduce."""
    gamma = day.gamma()
    tally_deriv = 0.0
    tally_second = 0.0
    ll = 0.0
    for g in day.won_games:
        d = g.opponents_adjusted_gamma(day.player)
        tally_deriv += 1.0 / (gamma + d)
        tally_second += d / (gamma + d) ** 2.0
        ll += math.log(gamma) - math.log(gamma + d)
    for g in day.lost_games:
        d = g.opponents_adjusted_gamma(day.player)
        tally_deriv += 1.0 / (gamma + d)
        tally_second += d / (gamma + d) ** 2.0
        ll += math.log(d) - math.log(gamma + d)
    derivative = len(day.won_games) - gamma * tally_deriv
    second = -gamma * tally_second
    return derivative, second, ll


def _reference_davidson_derivatives(
    day: PD.PlayerDay, nu: float
) -> tuple[float, float]:
    """Pre-vectorization Python-loop reference for
    ``PlayerDay.davidson_derivatives`` (Task 3)."""
    gradient = 0.0
    hessian = 0.0
    for game, weight in day._weighted_games():
        s, o = game.effective_gammas(day.player)
        t = nu * math.sqrt(s * o)
        z = s + o + t
        n = s + t / 2.0
        n_prime = s + t / 4.0
        ratio = n / z
        gradient += weight - ratio
        hessian += ratio * ratio - n_prime / z
    return gradient, hessian


def _reference_davidson_log_likelihood(day: PD.PlayerDay, nu: float) -> float:
    """Pre-vectorization Python-loop reference for
    ``PlayerDay.davidson_log_likelihood`` (Task 3)."""
    ll = 0.0
    for game, weight in day._weighted_games():
        s, o = game.effective_gammas(day.player)
        t = nu * math.sqrt(s * o)
        z = s + o + t
        if weight == 1.0:
            num = s
        elif weight == 0.0:
            num = o
        else:
            num = t
        ll += math.log(num) - math.log(z)
    return ll


def test_unit_equivalence_bt_terms():
    """Task 3: the vectorized ``log_likelihood_derivative``/
    ``_second_derivative``/``log_likelihood`` must match an independent
    reference Python-loop implementation of the same (pre-vectorization)
    algorithm, on a single day mixing several won AND lost games against
    distinct opponents (so the opponents' adjusted gammas actually vary)."""
    w = WHR()
    w.create_game("a", "b1", "B", 1, 0)  # a (black) won
    w.create_game("a", "b2", "W", 1, 0)  # a (black) lost to white b2
    w.create_game("b3", "a", "W", 1, 0)  # a (white) won
    w.create_game("b4", "a", "B", 1, 0)  # a (white) lost to black b4

    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b1").days[0].set_gamma(1.0)
    w.player_by_name("b2").days[0].set_gamma(1.5)
    w.player_by_name("b3").days[0].set_gamma(0.7)
    w.player_by_name("b4").days[0].set_gamma(3.0)

    assert len(a_day.won_games) == 2
    assert len(a_day.lost_games) == 2

    ref_deriv, ref_second, ref_ll = _reference_bt_terms(a_day)
    assert a_day.log_likelihood_derivative() == pytest.approx(ref_deriv, rel=1e-9)
    assert a_day.log_likelihood_second_derivative() == pytest.approx(
        ref_second, rel=1e-9
    )
    assert a_day.log_likelihood() == pytest.approx(ref_ll, rel=1e-9)


def test_unit_equivalence_davidson_terms():
    """Task 3: the vectorized ``davidson_derivatives``/
    ``davidson_log_likelihood`` must match an independent reference
    Python-loop implementation of the same (pre-vectorization) algorithm,
    on a single day mixing won, lost, AND drawn games against distinct
    opponents."""
    w = WHR()
    w.create_game("a", "b1", "B", 1, 0)  # a (black) won
    w.create_game("a", "b2", "W", 1, 0)  # a (black) lost to white b2
    w.create_game("b3", "a", "D", 1, 0)  # a draws with b3
    w.create_game("a", "b4", "D", 1, 0)  # a draws with b4
    w.create_game("b5", "a", "W", 1, 0)  # a (white) won

    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b1").days[0].set_gamma(1.0)
    w.player_by_name("b2").days[0].set_gamma(1.5)
    w.player_by_name("b3").days[0].set_gamma(0.6)
    w.player_by_name("b4").days[0].set_gamma(2.5)
    w.player_by_name("b5").days[0].set_gamma(0.9)

    assert len(a_day.won_games) == 2
    assert len(a_day.lost_games) == 1
    assert len(a_day.drawn_games) == 2

    # nu=0.0 is included for davidson_derivatives (no log() involved: the
    # draw-mass term t=nu*sqrt(s*o) is just 0, z=s+o stays positive). It is
    # excluded from the log_likelihood comparison below: with actual drawn
    # games present, nu=0 assigns them zero probability (t=0 => log(num=t)
    # is log(0)), which is undefined for BOTH the reference and vectorized
    # implementations -- not a vectorization artifact.
    for nu in (0.0, 0.3, 1.3):
        ref_grad, ref_hess = _reference_davidson_derivatives(a_day, nu)
        grad, hess = a_day.davidson_derivatives(nu)
        assert grad == pytest.approx(ref_grad, rel=1e-9)
        assert hess == pytest.approx(ref_hess, rel=1e-9)

    for nu in (0.3, 1.3):
        ref_ll = _reference_davidson_log_likelihood(a_day, nu)
        assert a_day.davidson_log_likelihood(nu) == pytest.approx(ref_ll, rel=1e-9)


def test_bt_and_davidson_terms_empty_day():
    """Task 3 gotcha: a day with no games must return 0.0 (not raise) for
    all five vectorized methods."""
    w = WHR()
    player = w.player_by_name("solo")
    day = PD.PlayerDay(player, 1)
    day.set_gamma(2.0)

    assert day.log_likelihood_derivative() == 0.0
    assert day.log_likelihood_second_derivative() == 0.0
    assert day.log_likelihood() == 0.0
    assert day.davidson_derivatives(1.5) == (0.0, 0.0)
    assert day.davidson_log_likelihood(1.5) == 0.0
