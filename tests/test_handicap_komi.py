import math
import pickle

import pytest

from whr import playerday
from whr.game import Game
from whr.player import Player
from whr.whole_history_rating import WHR


def test_advantage_config_defaults():
    w = WHR()
    assert w.config["pinned_handicap"] == {}
    assert w.config["pinned_komi"] == {}
    assert w.config["estimate_handicap_zero"] is False


def test_handicap_zero_baseline_pinned_by_default():
    w = WHR()
    assert w.handicap_gamma[0] == 1.0
    assert 0 in w._pinned_handicap_keys


def test_estimate_handicap_zero_unpins_baseline():
    w = WHR(config={"estimate_handicap_zero": True})
    assert 0 not in w._pinned_handicap_keys


def test_pinned_values_converted_to_gamma_and_marked():
    w = WHR(config={"pinned_handicap": {2: 200.0}, "pinned_komi": {6.5: -30.0}})
    assert w.handicap_gamma[2] == pytest.approx(10 ** (200.0 / 400.0))
    assert w.komi_gamma[6.5] == pytest.approx(10 ** (-30.0 / 400.0))
    assert 2 in w._pinned_handicap_keys and 6.5 in w._pinned_komi_keys


def test_config_not_mutated():
    src = {"pinned_handicap": {2: 200.0}}
    WHR(config=src)
    assert src == {"pinned_handicap": {2: 200.0}}


def test_ensure_advantage_keys_grows_tables_to_one():
    w = WHR()
    w._ensure_advantage_keys(3, 7.5)
    assert w.handicap_gamma[3] == 1.0
    assert w.komi_gamma[7.5] == 1.0
    # does not overwrite an existing (pinned) key
    w2 = WHR(config={"pinned_handicap": {3: 100.0}})
    before = w2.handicap_gamma[3]
    w2._ensure_advantage_keys(3, 6.5)
    assert w2.handicap_gamma[3] == before


def test_pinned_handicap_reproduces_legacy_elo_behaviour():
    # A handicap pinned to E elo must behave exactly like the old fixed-elo
    # handicap: black's win prob equals the equal-komi Bradley-Terry value with
    # black boosted by E elo.
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    w.create_game("black", "white", "B", 1, 2)  # handicap key 2, pinned to +200
    w.player_by_name("black").days[0].elo = 0.0
    w.player_by_name("white").days[0].elo = 0.0
    game = w.games[0]
    # black gamma boosted by 200 elo vs white (komi default gamma == 1)
    gb = 10 ** (200.0 / 400.0)
    assert game.white_win_probability() == pytest.approx(1.0 / (1.0 + gb))


def test_handicap_zero_default_komi_game_has_no_adjustment():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    game = w.games[0]
    w.player_by_name("a").days[0].elo = 0.0
    w.player_by_name("b").days[0].elo = 0.0
    # γ_h[0]=1 (baseline), γ_k[6.5]=1 (init) -> even game
    assert game.white_win_probability() == pytest.approx(0.5)


def test_direct_game_without_tables_treats_advantages_as_one(monkeypatch):
    cfg = {
        "debug": False,
        "w2": 300.0,
        "uncased": False,
        "initial_prior_wins": 0.5,
        "hessian_damping": 1.0,
    }
    b = Player("b", {**cfg})
    wp = Player("w", {**cfg})
    game = Game(b, wp, "B", 1, 5)
    game.bpd = playerday.PlayerDay(b, 1)
    game.wpd = playerday.PlayerDay(wp, 1)
    game.bpd.elo = 0.0
    game.wpd.elo = 0.0
    # No tables -> handicap/komi treated as gamma 1 -> even
    assert game.white_win_probability() == pytest.approx(0.5)


def _elo(gamma):
    return math.log10(gamma) * 400.0


def test_recovers_known_handicap_advantage():
    # Pin komi so only handicap is free; swap colours so a and b are equal by
    # symmetry, and the systematic 76% black win-rate is attributed to
    # handicap 2 (~200 elo). Single day => no temporal drift.
    w = WHR(config={"pinned_komi": {6.5: 0.0}})
    for _ in range(76):
        w.create_game("a", "b", "B", 1, 2)
    for _ in range(24):
        w.create_game("a", "b", "W", 1, 2)
    for _ in range(76):
        w.create_game("b", "a", "B", 1, 2)
    for _ in range(24):
        w.create_game("b", "a", "W", 1, 2)
    w.iterate(200)
    assert _elo(w.handicap_gamma[2]) == pytest.approx(200.0, abs=40.0)


def test_recovers_white_side_advantage_via_komi():
    # handicap 0 (pinned baseline gamma=1) so only komi is free; swap colours;
    # white wins 64% -> ~100 elo attributed to komi 6.5. Single day.
    w = WHR()
    for _ in range(64):
        w.create_game("a", "b", "W", 1, 0, komi=6.5)
    for _ in range(36):
        w.create_game("a", "b", "B", 1, 0, komi=6.5)
    for _ in range(64):
        w.create_game("b", "a", "W", 1, 0, komi=6.5)
    for _ in range(36):
        w.create_game("b", "a", "B", 1, 0, komi=6.5)
    w.iterate(200)
    assert _elo(w.komi_gamma[6.5]) == pytest.approx(100.0, abs=40.0)


def test_pinned_key_is_not_moved_by_estimation():
    w = WHR(config={"pinned_handicap": {2: 300.0}})
    for d in range(1, 21):
        w.create_game("a", "b", "B", d, 2)
    w.iterate(50)
    assert w.handicap_gamma[2] == pytest.approx(10 ** (300.0 / 400.0))


def test_baseline_handicap_zero_not_moved():
    w = WHR()
    for d in range(1, 21):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(50)
    assert w.handicap_gamma[0] == 1.0


def test_estimate_handicap_zero_lets_it_move():
    w = WHR(config={"estimate_handicap_zero": True})
    for d in range(1, 41):
        w.create_game("a", "b", "B" if d % 3 else "W", d, 0)
    w.iterate(50)
    # unpinned and mixed results -> generally moves off 1.0
    assert w.handicap_gamma[0] != 1.0


def test_all_win_category_guard_leaves_gamma_untouched():
    # A handicap key with only black wins has no finite estimate -> not updated.
    w = WHR()
    for d in range(1, 11):
        w.create_game("a", "b", "B", d, 3)  # handicap 3, always black win
    w.iterate(50)
    assert w.handicap_gamma[3] == 1.0


def _build_base_with_learned_handicap_advantage() -> WHR:
    # Same balanced colour-swap + pinned-komi design as
    # test_recovers_known_handicap_advantage, so handicap_gamma[2] estimates
    # to a clearly-non-1 value.
    w = WHR(config={"pinned_komi": {6.5: 0.0}})
    for _ in range(76):
        w.create_game("a", "b", "B", 1, 2)
    for _ in range(24):
        w.create_game("a", "b", "W", 1, 2)
    for _ in range(76):
        w.create_game("b", "a", "B", 1, 2)
    for _ in range(24):
        w.create_game("b", "a", "W", 1, 2)
    w.iterate(200)
    return w


def test_save_load_round_trip_preserves_learned_advantage(tmp_path):
    w = _build_base_with_learned_handicap_advantage()
    pre_gamma = w.handicap_gamma[2]
    pre_prob = w.games[0].black_win_probability()
    assert pre_gamma != pytest.approx(1.0)

    path = tmp_path / "base.pkl"
    w.save_base(str(path))
    loaded = WHR.load_base(str(path))

    assert loaded.handicap_gamma[2] == pytest.approx(pre_gamma, abs=1e-9)
    assert loaded.games[0].black_win_probability() == pytest.approx(pre_prob, abs=1e-9)


def test_legacy_load_preserves_learned_advantage(tmp_path):
    w = _build_base_with_learned_handicap_advantage()
    pre_gamma = w.handicap_gamma[2]
    assert pre_gamma != pytest.approx(1.0)

    path = tmp_path / "legacy_base.pkl"
    with open(path, "wb") as f:
        pickle.dump([w.players, w.games, w.config], f)
    loaded = WHR.load_base(str(path))

    assert loaded.handicap_gamma[2] == pytest.approx(pre_gamma, abs=1e-9)


def test_underflowed_pinned_gamma_raises_attribute_error_not_zero_division():
    # -130000 elo underflows gamma = 10 ** (-130000/400) to 0.0 in float64.
    # opponents_adjusted_gamma divides by the komi gamma, so an unvalidated
    # zero divisor must not surface as a raw ZeroDivisionError; the guard is
    # supposed to catch it and raise the intended AttributeError instead.
    w = WHR(config={"pinned_komi": {6.5: -130000.0}})
    w.create_game("a", "b", "B", 1, 0, komi=6.5)
    game = w.games[0]
    with pytest.raises(AttributeError, match="bad adjusted gamma"):
        game.white_win_probability()


def test_auto_iterate_waits_for_komi_advantage_to_stabilize():
    # Perfectly colour-swap-symmetric setup: player ratings sit exactly at
    # their fixed point (r=0) from the very first iteration, so the player
    # gradient alone is already ~0, while the (pinned-handicap-free) komi
    # advantage gamma still needs several more Newton steps to settle.
    # Before the fix, max_gradient_norm() only looked at player gradients, so
    # auto_iterate declared convergence after a single iteration even though
    # komi_gamma[6.5] kept moving substantially afterwards.
    def build() -> WHR:
        w = WHR(config={"pinned_handicap": {0: 0.0}})
        for _ in range(64):
            w.create_game("a", "b", "W", 1, 0, komi=6.5)
        for _ in range(36):
            w.create_game("a", "b", "B", 1, 0, komi=6.5)
        for _ in range(64):
            w.create_game("b", "a", "W", 1, 0, komi=6.5)
        for _ in range(36):
            w.create_game("b", "a", "B", 1, 0, komi=6.5)
        return w

    w = build()
    iterations, converged = w.auto_iterate(precision=1e-3, batch_size=1, time_limit=5)
    assert converged is True
    assert w.max_gradient_norm() < 1e-3

    before_elo = _elo(w.komi_gamma[6.5])
    w.iterate(5)
    after_elo = _elo(w.komi_gamma[6.5])
    assert abs(after_elo - before_elo) < 0.01, (
        f"komi advantage still moved {after_elo - before_elo:.4f} elo after "
        f"auto_iterate declared convergence at iteration {iterations}"
    )


def test_legacy_load_without_advantage_attrs_still_predicts(tmp_path):
    # Simulate a genuinely pre-phase-3 legacy pickle: games/players predate
    # the handicap_gamma/komi_gamma and initial_prior_wins/hessian_damping
    # attributes entirely.
    w = _build_base_with_learned_handicap_advantage()
    for game in w.games:
        del game.handicap_gamma
        del game.komi_gamma
    for player in w.players.values():
        del player.initial_prior_wins
        del player.hessian_damping

    path = tmp_path / "old_legacy_base.pkl"
    with open(path, "wb") as f:
        pickle.dump([w.players, w.games, w.config], f)
    loaded = WHR.load_base(str(path))

    assert loaded.handicap_gamma[2] == pytest.approx(1.0)
    for game in loaded.games:
        assert math.isfinite(game.black_win_probability())
        assert math.isfinite(game.white_win_probability())


# --- komi opt-in (3.1.0) ---------------------------------------------------


def test_komi_is_opt_in_no_default_key():
    # No komi passed -> no komi key registered and no komi advantage modelled;
    # the game's komi gamma is neutral (1.0), and nothing crashes.
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.iterate(10)
    assert w.komi_gamma == {}
    assert "komi" not in w.games[0].extras
    assert math.isfinite(w.games[0].white_win_probability())


def test_handicap_estimated_with_no_komi():
    # Colour-swap-balanced so only the handicap 'h' (on black) is free; no komi
    # is passed. The handicap must still be estimated (~+100 elo for a 64% edge)
    # and no komi key is ever created.
    w = WHR()
    for _ in range(64):
        w.create_game("a", "b", "B", 1, "h")
    for _ in range(36):
        w.create_game("a", "b", "W", 1, "h")
    for _ in range(64):
        w.create_game("b", "a", "B", 1, "h")
    for _ in range(36):
        w.create_game("b", "a", "W", 1, "h")
    w.iterate(200)
    assert _elo(w.handicap_gamma["h"]) == pytest.approx(100.0, abs=40.0)
    assert w.komi_gamma == {}


def test_komi_arg_and_extras_are_equivalent():
    a = WHR()
    a.create_game("x", "y", "W", 1, 0, komi=7.5)
    b = WHR()
    b.create_game("x", "y", "W", 1, 0, extras={"komi": 7.5})
    assert a.games[0].extras.get("komi") == 7.5
    assert b.games[0].extras.get("komi") == 7.5
    assert 7.5 in a.komi_gamma and 7.5 in b.komi_gamma


def test_save_load_roundtrips_komi_and_no_komi(tmp_path):
    w = WHR()
    w.create_game("a", "b", "B", 1, 0, komi=7.5)  # a komi game
    w.create_game("a", "b", "W", 2, 0)  # a no-komi game
    w.iterate(20)
    path = str(tmp_path / "state.pkl")
    WHR.save_base(w, path)
    w2 = WHR.load_base(path)
    assert w2.games[0].extras.get("komi") == 7.5
    assert w2.games[1].extras.get("komi") is None
    assert 7.5 in w2.komi_gamma
