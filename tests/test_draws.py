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
    w.create_game("a", "b", "B", 1, 0, komi=6.5)
    w.create_game("b", "a", "B", 1, 0, komi=6.5)
    w.create_game("a", "b", "W", 1, 0, komi=6.5)
    w.create_game("b", "a", "W", 1, 0, komi=6.5)
    w.create_game("a", "b", "D", 1, 0, komi=6.5)
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


# --- uncertainty-integrated three-outcome prediction ------------------------
#
# Fixtures below are chosen to sit in specific (rating gap d, nu, sigma)
# regimes, because the hedging direction genuinely differs between them; the
# comment on each records the regime it pins.


def _lopsided_with_draws():
    """Clear favourite plus a healthy draw rate: d ~ +3.5 nats, nu ~ 3.0."""
    w = WHR()
    for day in range(1, 7):
        w.create_game("fav", "dog", "B", day, 0)
        w.create_game("fav", "dog", "B", day, 0)
        w.create_game("fav", "dog", "D", day, 0)
    w.iterate(60)
    return w


def _even_with_draws():
    """Exactly even pair (colour-swapped wins), so d == 0: nu ~ 1.0."""
    w = WHR()
    for day in range(1, 5):
        w.create_game("x", "y", "D", day, 0)
        w.create_game("x", "y", "B", day, 0)
        w.create_game("y", "x", "B", day, 0)
    w.iterate(60)
    return w


def _marginal_favourite_with_draws():
    """Barely-favoured player: d ~ +0.22 nats, nu ~ 0.9."""
    w = WHR()
    for day in range(1, 4):
        for _ in range(5):
            w.create_game("m", "n", "B", day, 0)
        for _ in range(4):
            w.create_game("m", "n", "W", day, 0)
        for _ in range(4):
            w.create_game("m", "n", "D", day, 0)
    w.iterate(60)
    return w


def test_win_draw_loss_uncertainty_default_unchanged():
    w = _lopsided_with_draws()
    point = w.win_draw_loss_probabilities("fav", "dog")
    assert (
        w.win_draw_loss_probabilities("fav", "dog", account_for_uncertainty=False)
        == point
    )
    # uncertainty_steps is inert while the flag is off, including invalid values
    assert w.win_draw_loss_probabilities("fav", "dog", uncertainty_steps=0) == point


def test_win_draw_loss_uncertainty_bad_steps_raises():
    w = _lopsided_with_draws()
    with pytest.raises(ValueError):
        w.win_draw_loss_probabilities(
            "fav", "dog", account_for_uncertainty=True, uncertainty_steps=0
        )
    with pytest.raises(ValueError):
        w.win_draw_loss_probabilities(
            "fav", "dog", account_for_uncertainty=True, uncertainty_steps=-3
        )


def test_win_draw_loss_uncertainty_sigma_zero_fallback():
    # Unknown players: both variances are 0 (iterate() never ran), exercising
    # the sigma == 0 short-circuit back to the point estimate.
    w = WHR()
    point = w.win_draw_loss_probabilities("ghost1", "ghost2")
    assert (
        w.win_draw_loss_probabilities("ghost1", "ghost2", account_for_uncertainty=True)
        == point
    )


def test_win_draw_loss_uncertainty_still_sums_to_one():
    # Normalisation is a property of the quadrature (each node contributes a
    # triple summing to 1), so it must hold at every grid size.
    w = _lopsided_with_draws()
    for steps in (1, 2, 4, 9):
        probs = w.win_draw_loss_probabilities(
            "fav", "dog", account_for_uncertainty=True, uncertainty_steps=steps
        )
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in probs)


def test_win_draw_loss_uncertainty_hedges_clear_favourite():
    w = _lopsided_with_draws()
    p1, pd, p2 = w.win_draw_loss_probabilities("fav", "dog")
    q1, qd, q2 = w.win_draw_loss_probabilities(
        "fav", "dog", account_for_uncertainty=True
    )
    assert p1 > 0.5  # genuinely favoured to start with
    assert q1 < p1  # favourite gives up mass
    assert qd + q2 > pd + p2  # ...to the draw/underdog side
    assert q2 > p2  # and the underdog specifically gains


def test_win_draw_loss_uncertainty_compresses_win_loss_odds():
    """Odds compression is the invariant that always holds, and the mechanism
    that fixes log-loss: whatever the draw mass does, P(win)/P(loss) moves
    toward 1."""
    w = _lopsided_with_draws()
    p1, _, p2 = w.win_draw_loss_probabilities("fav", "dog")
    q1, _, q2 = w.win_draw_loss_probabilities(
        "fav", "dog", account_for_uncertainty=True
    )
    assert 1.0 < q1 / q2 < p1 / p2

    m = _marginal_favourite_with_draws()
    r1, _, r2 = m.win_draw_loss_probabilities("m", "n")
    s1, _, s2 = m.win_draw_loss_probabilities("m", "n", account_for_uncertainty=True)
    assert 1.0 < s1 / s2 < r1 / r2


def test_win_draw_loss_uncertainty_lowers_draw_for_even_matchup():
    """Counterintuitive but correct: uncertainty does NOT push mass toward the
    draw. Davidson's draw curve nu/(2*cosh(d/2)+nu) is concave near d == 0, so
    spreading the rating gap *drains* the draw for an even matchup. Pinned so a
    future change to the quadrature has to notice it broke this."""
    w = _even_with_draws()
    p1, pd, p2 = w.win_draw_loss_probabilities("x", "y")
    q1, qd, q2 = w.win_draw_loss_probabilities("x", "y", account_for_uncertainty=True)
    assert p1 == pytest.approx(p2)  # even pair: d == 0
    assert qd < pd  # draw mass falls
    assert q1 > p1 and q2 > p2  # it splits evenly to both sides
    assert q1 == pytest.approx(q2)  # symmetry preserved by the symmetric grid


def test_win_draw_loss_uncertainty_raises_win_for_marginal_favourite():
    """The other counterintuitive case: for a *barely* favoured player the win
    probability goes UP, because the draw mass leaking outward (see the even
    matchup above) splits to both sides and outweighs the odds compression.
    Only a clear favourite loses win probability."""
    w = _marginal_favourite_with_draws()
    p1, pd, p2 = w.win_draw_loss_probabilities("m", "n")
    q1, qd, q2 = w.win_draw_loss_probabilities("m", "n", account_for_uncertainty=True)
    assert 0.5 > p1 > p2  # favoured, but only just, with heavy draw mass
    assert q1 > p1  # win probability RISES
    assert qd < pd  # because the draw drains
    assert q2 > p2  # underdog still gains more (odds compress)
    assert q1 - p1 < q2 - p2


def test_win_draw_loss_uncertainty_matches_two_outcome_path_at_nu_zero():
    """With no draws fitted (nu == 0) Davidson collapses to Bradley-Terry, and
    the rating gap d integrated here is exactly the logit integrated by
    ``probability_future_match``, so the two uncertainty-aware paths agree.

    The win probability agrees bit-for-bit. The loss probability agrees only to
    float precision, and that gap is meaningful rather than sloppy:
    ``probability_future_match`` returns the forced complement ``1.0 - p1``,
    whereas this method integrates the loss on its own. Normalisation here is a
    property of the quadrature, never imposed -- which is what keeps the
    three-way split unbiased.
    """
    w = WHR()
    for day in range(1, 5):
        w.create_game("p", "q", "B", day, 0)
        w.create_game("p", "q", "B", day, 0)
        w.create_game("p", "q", "W", day, 0)
    w.iterate(50)
    assert w.nu == 0.0

    q1, qd, q2 = w.win_draw_loss_probabilities("p", "q", account_for_uncertainty=True)
    two_win, two_loss = w.probability_future_match(
        "p", "q", account_for_uncertainty=True
    )
    assert qd == 0.0
    assert q1 == two_win  # exact: same d, same grid, same weights
    assert q2 == pytest.approx(two_loss, abs=1e-15)  # independent, not complemented
    assert q1 + qd + q2 == pytest.approx(1.0)


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
