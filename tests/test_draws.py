import math
import random

import pytest

from whr.player import Player
from whr.whole_history_rating import WHR


def test_draw_result_accepted_and_routed():
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)
    w.load_games(["a b D 2"])
    assert w._has_draws is True
    a = w.player_by_name("a")
    drawn_days = [d for d in a.days if d.drawn_games]
    assert len(drawn_days) == 2  # one draw on each of days 1 and 2


def test_no_draws_leaves_nu_zero():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    assert w._has_draws is False
    assert w.nu == 0.0
    assert w.draw_tendency == 0.0


def test_effective_gammas_fold_advantages():
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    g = w.create_game("black", "white", "B", 1, 2)
    w.player_by_name("black").days[0].set_gamma(3.0)
    w.player_by_name("white").days[0].set_gamma(2.0)
    s_black, o_white = g.effective_gammas(g.black_player)
    assert s_black == pytest.approx(
        3.0 * 10 ** (200.0 / 400.0)
    )  # handicap boosts black
    assert o_white == pytest.approx(2.0)  # komi default gamma 1
    # symmetry: querying as white swaps S/O
    s_white, o_black = g.effective_gammas(g.white_player)
    assert (s_white, o_black) == pytest.approx((o_white, s_black))


def test_pinned_draw_config_default_none():
    assert WHR().config["pinned_draw"] is None
    assert WHR(config={"pinned_draw": 1.5}).config["pinned_draw"] == 1.5


def test_draws_do_not_inflate_komi_gamma():
    """Regression test for a cross-function bug: `_accumulate_handicap_komi`
    used to fall through to its `else` branch for draws (winner != "B"),
    mis-crediting them as WHITE/komi wins with a Bradley-Terry denominator
    that doesn't apply to a draw.

    This base is built to be exactly colour-swap symmetric: for every
    decisive game where black wins, there is a mirror game (colours and
    players swapped) where white wins, so the two players stay equal
    strength and the Newton gradient on komi_gamma[6.5] is exactly zero at
    gamma=1 -- there is no real white/komi advantage in this data. A single
    draw game is then added. Before the fix, that draw is silently counted
    as an extra komi/white win, pushing komi_gamma[6.5] well above 1.0
    (observed ~1.5 here, ~2.43 in a similar denser base). After the fix,
    draws are skipped by the handicap/komi accumulator and komi_gamma[6.5]
    stays at its symmetric-equilibrium value of ~1.0.
    """
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("b", "a", "B", 1, 0)
    w.create_game("a", "b", "W", 1, 0)
    w.create_game("b", "a", "W", 1, 0)
    w.create_game("a", "b", "D", 1, 0)
    w.iterate(50)
    assert abs(w.komi_gamma[6.5] - 1.0) < 0.1


def test_davidson_derivatives_match_closed_form():
    w = WHR()
    g = w.create_game("a", "b", "D", 1, 0)  # a draw on day 1
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    nu = 1.5
    s, o = g.effective_gammas(a_day.player)  # S=2, O=1 (no advantages)
    t = nu * math.sqrt(s * o)
    z = s + o + t
    n = s + t / 2.0
    n_prime = s + t / 4.0
    w_weight = 0.5  # draw
    exp_grad = w_weight - n / z
    exp_hess = (n / z) ** 2 - n_prime / z
    grad, hess = a_day.davidson_derivatives(nu)
    assert grad == pytest.approx(exp_grad)
    assert hess == pytest.approx(exp_hess)


def test_davidson_reduces_to_bt_at_nu_zero():
    # Same win/loss data, compute the day's game-part gradient both ways at nu=0.
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)  # a (black) won
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    davidson_grad, davidson_hess = a_day.davidson_derivatives(0.0)
    assert davidson_grad == pytest.approx(a_day.log_likelihood_derivative())
    assert davidson_hess == pytest.approx(a_day.log_likelihood_second_derivative())


def test_multiday_gradient_hessian_route_to_davidson_when_draw_tendency_positive():
    """Task 2 coverage gap: with ``draw_tendency > 0``, ``Player.gradient``
    and the static ``Player.hessian`` must take each day's Davidson game-part
    (``day.davidson_derivatives``) rather than the plain win/loss derivatives.

    Nothing wires ``draw_tendency`` above zero until Task 3, so this test sets
    it directly and reconstructs the expected gradient/hessian from
    ``davidson_derivatives`` plus the (independently expressed) temporal-prior
    / anchor terms -- this fails if the ``> 0`` routing branch is removed.
    """
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)  # day 1: a draws with b
    w.create_game("a", "b", "B", 5, 0)  # day 5: a (black) wins outright
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    # Distinct, non-symmetric gammas so the Davidson vs. win/loss derivatives
    # can't accidentally coincide (e.g. S == O would zero out some terms).
    a.days[0].set_gamma(2.0)
    b.days[0].set_gamma(1.0)
    a.days[1].set_gamma(3.0)
    b.days[1].set_gamma(1.5)
    a.draw_tendency = 0.3

    assert len(a.days) == 2
    r = [d.r for d in a.days]
    sigma2 = a.compute_sigma2()

    grad = a.gradient(r, a.days, sigma2)
    diag, _sub_diag = Player.hessian(a.days, sigma2, a.hessian_damping)

    for idx, day in enumerate(a.days):
        davidson_grad, davidson_hess = day.davidson_derivatives(0.3)
        prior_grad = 0.0
        prior_hess = 0.0
        if idx < len(a.days) - 1:
            prior_grad += -(r[idx] - r[idx + 1]) / sigma2[idx]
            prior_hess += -1 / sigma2[idx]
        if idx > 0:
            prior_grad += -(r[idx] - r[idx - 1]) / sigma2[idx - 1]
            prior_hess += -1 / sigma2[idx - 1]
        expected_grad = davidson_grad + prior_grad
        expected_hess = davidson_hess + prior_hess - a.hessian_damping
        if idx == 0:
            expected_grad += day.anchor_gradient()
            expected_hess += day.anchor_hessian()
        assert grad[idx] == pytest.approx(expected_grad)
        assert diag[idx] == pytest.approx(expected_hess)

    # Sanity: falling back to the win/loss derivatives would give a
    # different game-part on BOTH days here -- the draw day because a drawn
    # game never appears in won_games/lost_games, and the decisive day
    # because the Davidson draw-mass term (`t`) still perturbs the ratio for
    # nu > 0 even without an actual draw that day. This confirms the
    # reconstruction above is tight enough to distinguish the two paths.
    for day in a.days:
        bt_grad = day.log_likelihood_derivative()
        bt_hess = day.log_likelihood_second_derivative()
        davidson_grad, davidson_hess = day.davidson_derivatives(0.3)
        assert davidson_grad != pytest.approx(bt_grad)
        assert davidson_hess != pytest.approx(bt_hess)


def test_single_day_update_by_1d_newton_routes_to_davidson():
    """Task 2 coverage gap: ``PlayerDay.update_by_1d_newtons_method`` must
    take the Davidson game-part when ``draw_tendency > 0``, for a
    single-day player.
    """
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)  # a's only day: a single draw vs b
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    a.days[0].set_gamma(2.0)
    b.days[0].set_gamma(1.0)
    a.draw_tendency = 0.5

    day = a.days[0]
    r_before = day.r

    davidson_grad, davidson_hess = day.davidson_derivatives(0.5)
    bt_grad = day.log_likelihood_derivative()
    bt_hess = day.log_likelihood_second_derivative()
    anchor_grad = day.anchor_gradient()
    anchor_hess = day.anchor_hessian()
    damping = a.hessian_damping

    expected_davidson_r = r_before - (davidson_grad + anchor_grad) / (
        davidson_hess + anchor_hess - damping
    )
    counterfactual_bt_r = r_before - (bt_grad + anchor_grad) / (
        bt_hess + anchor_hess - damping
    )
    # Sanity: the two paths must actually diverge for this data, else the
    # test couldn't tell which branch ran (a drawn game is never counted in
    # won_games/lost_games, so the BT path sees no game at all here).
    assert expected_davidson_r != pytest.approx(counterfactual_bt_r)

    day.update_by_1d_newtons_method()

    assert day.r == pytest.approx(expected_davidson_r)
    assert day.r != pytest.approx(counterfactual_bt_r)


def _davidson_balanced_history(rng, nu_true, n_pairs=40, n_games=60):
    """Equal players, colour-swapped, single day; outcomes sampled from Davidson
    with a known nu_true (equal gammas => S=O=1 => P(draw)=nu/(2+nu))."""
    w = WHR()
    p_draw = nu_true / (2.0 + nu_true)
    p_win = 1.0 / (2.0 + nu_true)
    for k in range(n_pairs):
        a, b = f"a{k}", f"b{k}"
        for _ in range(n_games):
            r = rng.random()
            outcome = "D" if r < p_draw else ("B" if r < p_draw + p_win else "W")
            w.create_game(a, b, outcome, 1, 0)
            w.create_game(
                b,
                a,
                outcome if outcome == "D" else ("W" if outcome == "B" else "B"),
                1,
                0,
            )
    return w


def test_recovers_known_draw_tendency():
    rng = random.Random(7)
    nu_true = 1.5
    w = _davidson_balanced_history(rng, nu_true)
    w.iterate(80)
    assert w.draw_tendency == pytest.approx(nu_true, abs=0.3)


def test_pinned_draw_is_not_moved():
    w = WHR(config={"pinned_draw": 0.8})
    for d in range(1, 11):
        w.create_game("a", "b", "D", d, 0)
        w.create_game("a", "b", "B", d, 0)
    w.iterate(30)
    assert w.draw_tendency == pytest.approx(0.8)


def test_no_draw_iteration_still_matches_baseline():
    # A draw-free scenario iterates with the win/loss path untouched.
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
        w.create_game("a", "b", "W", d, 0)
    w.iterate(30)
    assert w.nu == 0.0
    elo, _ = w.ratings_for_player("a", current=True)
    assert math.isfinite(elo)


def test_win_draw_loss_sums_to_one_and_reflects_nu():
    rng = random.Random(3)
    w = _davidson_balanced_history(rng, 1.5, n_pairs=20, n_games=40)
    w.iterate(60)
    p1, pd, p2 = w.win_draw_loss_probabilities("a0", "b0")
    assert p1 + pd + p2 == pytest.approx(1.0)
    assert all(p >= 0.0 for p in (p1, pd, p2))
    assert pd > 0.05  # meaningful draw mass with nu ~ 1.5


def test_win_draw_loss_no_draws_gives_zero_draw():
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(20)
    p1, pd, p2 = w.win_draw_loss_probabilities("a", "b")
    assert pd == pytest.approx(0.0)
    assert p1 + p2 == pytest.approx(1.0)


def test_no_draw_regression_locks_compatibility_invariant():
    """Final compatibility check for the whole draws feature: a base that
    never sees a "D" outcome must be completely untouched by phase 6.

    ``_has_draws`` stays False, the global draw tendency ``nu`` (aka
    ``draw_tendency``) never leaves its 0.0 initial value (so every Davidson
    formula reduces to plain Bradley-Terry, per
    ``test_davidson_reduces_to_bt_at_nu_zero`` above), and normal iteration
    still converges to finite ratings for every player -- exactly the
    pre-phase-6 behaviour.
    """
    w = WHR()
    for d in range(1, 11):
        w.create_game("alice", "bob", "B", d, 0)
        w.create_game("alice", "carol", "W", d, 0)
        w.create_game("bob", "carol", "B", d, 0)
    w.auto_iterate()

    assert w._has_draws is False
    assert w.nu == 0.0
    assert w.draw_tendency == 0.0

    for name in ("alice", "bob", "carol"):
        for _day, elo, uncertainty in w.ratings_for_player(name):
            assert math.isfinite(elo)
            assert math.isfinite(uncertainty)


# --- Fix 1: save_base/load_base must persist/restore the fitted nu ---------


def test_save_load_round_trip_preserves_nu(tmp_path):
    """RED before the fix: save_base does not dump ``self.nu``, so
    load_base's game replay re-seeds nu=1.0 on the first "D" game (via
    ``_add_game``'s seeding logic), silently discarding any fitted draw
    tendency that moved away from that seed.
    """
    rng = random.Random(3)
    w = _davidson_balanced_history(rng, 1.5, n_pairs=20, n_games=40)
    w.iterate(60)
    pre_nu = w.draw_tendency
    assert pre_nu != pytest.approx(1.0)  # actually fitted away from the seed
    pre_probs = w.win_draw_loss_probabilities("a0", "b0")

    path = tmp_path / "draws_base.pkl"
    w.save_base(str(path))
    loaded = WHR.load_base(str(path))

    assert loaded.draw_tendency == pytest.approx(pre_nu, abs=1e-9)
    assert loaded.win_draw_loss_probabilities("a0", "b0") == pytest.approx(
        pre_probs, abs=1e-9
    )


def test_save_base_unpicklable_config_fallback_preserves_pinned_draw(tmp_path):
    """`pinned_draw` must survive the unpicklable-config fallback allowlist
    (else a pinned nu is silently dropped on that path)."""
    w = WHR(config={"pinned_draw": 0.8, "bad": lambda x: x})
    w.create_game("a", "b", "D", 1, 0)
    w.create_game("a", "b", "B", 1, 0)
    path = tmp_path / "pinned_draw_base.pkl"
    with pytest.warns(UserWarning):
        w.save_base(str(path))
    loaded = WHR.load_base(str(path))
    assert loaded.config["pinned_draw"] == 0.8


# --- Fix 2: log_likelihood must include the Davidson draw contribution -----


def test_davidson_log_likelihood_matches_bt_at_nu_zero():
    """At nu=0, davidson_log_likelihood must equal the plain BT
    log_likelihood() for every game -- if this doesn't match, the formula is
    wrong (per the design spec's own nu=0 sanity check)."""
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)  # a (black) won
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    assert a_day.davidson_log_likelihood(0.0) == pytest.approx(a_day.log_likelihood())


def test_davidson_log_likelihood_matches_bt_at_nu_zero_for_a_loss():
    w = WHR()
    w.create_game("a", "b", "W", 1, 0)  # a (black) lost
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(1.0)
    w.player_by_name("b").days[0].set_gamma(2.0)
    assert a_day.davidson_log_likelihood(0.0) == pytest.approx(a_day.log_likelihood())


def test_whr_log_likelihood_includes_draw_contribution():
    """RED before the fix: Player.log_likelihood summed only
    won_game_terms()+lost_game_terms(), which are both empty on an all-draw
    day, so the day's real game contribution was silently dropped (treated
    as 0) instead of using the Davidson formula.
    """
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)
    w.iterate(20)
    score = w.log_likelihood()
    assert math.isfinite(score)

    # Recompute what the score would be under the pre-fix behaviour: the
    # game part always comes from day.log_likelihood() (BT win/loss terms
    # only), regardless of draw_tendency.
    old_style = 0.0
    for p in w.players.values():
        for day in p.days:
            old_style += day.log_likelihood()
        if p.days:
            old_style += p.days[0].anchor_log_likelihood()
        sigma2 = p.compute_sigma2()
        for i, s2 in enumerate(sigma2):
            rd = p.days[i + 1].r - p.days[i].r
            old_style += -(rd**2) / (2 * s2) - 0.5 * math.log(2 * math.pi * s2)
    assert score != pytest.approx(old_style)


# --- Fix 4: max_gradient_norm must fold in the nu gradient ------------------


def test_max_gradient_norm_includes_nu_gradient():
    """RED before the fix: max_gradient_norm ignored the nu gradient
    entirely, so auto_iterate could report convergence while nu's own
    Newton gradient was still far from zero.
    """
    w = WHR()
    for d in range(1, 4):
        w.create_game("a", "b", "D", d, 0)
        w.create_game("a", "b", "B", d, 0)
    w.iterate(1)  # one player Newton step: nu itself has barely moved yet
    nu_gradient, _nu_hessian = w._nu_gradient_hessian()
    assert abs(nu_gradient) > 1e-3  # nu hasn't converged yet
    assert w.max_gradient_norm() >= abs(nu_gradient) - 1e-12


def test_auto_iterate_converges_with_nu_gradient_included():
    rng = random.Random(11)
    w = _davidson_balanced_history(rng, 1.2, n_pairs=10, n_games=30)
    iterations, converged = w.auto_iterate(precision=1e-3, time_limit=30)
    assert converged
    assert w.max_gradient_norm() < 1e-3
    assert iterations >= 1


# --- Fix 5: reject an invalid winner ----------------------------------------


def test_invalid_winner_raises_value_error_on_create_game():
    w = WHR()
    with pytest.raises(ValueError):
        w.create_game("a", "b", "tie", 1, 0)


def test_invalid_winner_raises_value_error_via_load_games():
    w = WHR()
    with pytest.raises(ValueError):
        w.load_games(["a b tie 1"])


def test_valid_winners_still_accepted_case_insensitively():
    w = WHR()
    w.create_game("a", "b", "b", 1, 0)
    w.create_game("a", "b", "w", 2, 0)
    w.create_game("a", "b", "d", 3, 0)
    assert [g.winner for g in w.games] == ["B", "W", "D"]
