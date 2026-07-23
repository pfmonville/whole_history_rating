import math

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
        w.create_game("a", "b", "W", 1, 0)
    for _ in range(36):
        w.create_game("a", "b", "B", 1, 0)
    for _ in range(64):
        w.create_game("b", "a", "W", 1, 0)
    for _ in range(36):
        w.create_game("b", "a", "B", 1, 0)
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
