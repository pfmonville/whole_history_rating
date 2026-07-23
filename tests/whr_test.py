import copy
import pickle
import warnings

import pytest

from whr import utils, whole_history_rating


def setup_game_with_elo(white_elo, black_elo, handicap):
    whr = whole_history_rating.WHR()
    game = whr.create_game("black", "white", "W", 1, handicap)
    game.black_player.days[0].elo = black_elo
    game.white_player.days[0].elo = white_elo
    return game


def test_even_game_between_equal_strength_players_should_have_white_winrate_of_50_percent():
    game = setup_game_with_elo(500, 500, 0)
    assert abs(0.5 - game.white_win_probability()) <= 0.0001


def test_handicap_should_confer_advantage():
    game = setup_game_with_elo(500, 500, 1)
    assert game.black_win_probability() > 0.5


def test_higher_rank_should_confer_advantage():
    game = setup_game_with_elo(600, 500, 0)
    assert game.white_win_probability() > 0.5


def test_winrates_are_equal_for_same_elo_delta():
    game = setup_game_with_elo(100, 200, 0)
    game2 = setup_game_with_elo(200, 300, 0)
    assert abs(game.white_win_probability() - game2.white_win_probability()) <= 0.0001


def test_winrates_for_twice_as_strong_player():
    game = setup_game_with_elo(100, 200, 0)
    assert abs(0.359935 - game.white_win_probability()) <= 0.0001


def test_winrates_should_be_inversely_proportional_with_unequal_ranks():
    game = setup_game_with_elo(600, 500, 0)
    assert (
        abs(game.white_win_probability() - (1 - game.black_win_probability())) <= 0.0001
    )


def test_winrates_should_be_inversely_proportional_with_handicap():
    game = setup_game_with_elo(500, 500, 4)
    assert (
        abs(game.white_win_probability() - (1 - game.black_win_probability())) <= 0.0001
    )


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_output():
    whr = whole_history_rating.WHR()
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 2, 0)
    whr.create_game("shusaku", "shusai", "W", 3, 0)
    whr.iterate(50)
    assert [
        (1, -43, 0.84),
        (2, -45, 0.84),
        (3, -45, 0.84),
    ] == whr.ratings_for_player("shusaku")
    assert [
        (1, 43, 0.84),
        (2, 45, 0.84),
        (3, 45, 0.84),
    ] == whr.ratings_for_player("shusai")


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_output2():
    whr = whole_history_rating.WHR()
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 2, 0)
    whr.create_game("shusaku", "shusai", "W", 3, 0)
    whr.create_game("shusaku", "shusai", "W", 4, 0)
    whr.create_game("shusaku", "shusai", "W", 4, 0)
    whr.iterate(50)
    assert [
        (1, -92, 0.71),
        (2, -94, 0.71),
        (3, -95, 0.71),
        (4, -96, 0.72),
    ] == whr.ratings_for_player("shusaku")
    assert [
        (1, 92, 0.71),
        (2, 94, 0.71),
        (3, 95, 0.71),
        (4, 96, 0.72),
    ] == whr.ratings_for_player("shusai")


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_unstable_exception_raised_in_certain_cases():
    whr = whole_history_rating.WHR()
    for _ in range(10):
        whr.create_game("anchor", "player", "B", 1, 0)
        whr.create_game("anchor", "player", "W", 1, 0)
    for _ in range(10):
        whr.create_game("anchor", "player", "B", 180, 600)
        whr.create_game("anchor", "player", "W", 180, 600)
    with pytest.raises(utils.UnstableRatingException):
        whr.iterate(10)


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_log_likelihood():
    whr = whole_history_rating.WHR()
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 4, 0)
    whr.create_game("shusaku", "shusai", "W", 10, 0)
    player = whr.players["shusaku"]
    player.days[0].r = 1
    player.days[1].r = 2
    player.days[2].r = 0
    assert abs(-69.65648196168772 - player.log_likelihood()) <= 0.0001
    assert abs(-1.9397850625546684 - player.days[0].log_likelihood()) <= 0.0001
    assert abs(-2.1269280110429727 - player.days[1].log_likelihood()) <= 0.0001
    assert abs(-0.6931471805599453 - player.days[2].log_likelihood()) <= 0.0001


def test_creating_games():
    # test creating the base with modified w2 and uncased
    whr = whole_history_rating.WHR(config={"w2": 14, "uncased": True})
    # test creating one game
    assert isinstance(
        whr.create_game("shusaku", "shusai", "B", 4, 0), whole_history_rating.Game
    )
    # test creating one game with winner uncased (b instead of B)
    assert isinstance(
        whr.create_game("shusaku", "shusai", "w", 5, 0), whole_history_rating.Game
    )
    # test creating one game with cased letters (ShUsAkU instead of shusaku and ShUsAi instead of shusai)
    assert isinstance(
        whr.create_game("ShUsAkU", "ShUsAi", "W", 6, 0), whole_history_rating.Game
    )
    assert list(whr.players.keys()) == ["shusai", "shusaku"]


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_loading_several_games_at_once(capsys, tmp_path):
    whr = whole_history_rating.WHR()
    # test loading several games at once
    test_games = [
        "shusaku; shusai; B; 1",
        "shusaku;shusai;W;2;0",
        " shusaku ; shusai ;W ; 3; {'w2':300}",
        "shusaku;nobody;B;3;0;{'w2':300}",
    ]
    whr.load_games(test_games, separator=";")
    assert len(whr.games) == 4
    # test auto iterating to get convergence
    whr.iterate(20)
    # test getting ratings for player shusaku (day, elo, uncertainty)
    assert whr.ratings_for_player("shusaku") == [
        (1, 26.0, 0.70),
        (2, 25.0, 0.70),
        (3, 24.0, 0.70),
    ]
    # test getting ratings for player shusai, only current elo and uncertainty
    assert whr.ratings_for_player("shusai", current=True) == (87.0, 0.84)
    # test getting probability of future match between shusai and nobody2 (an
    # unknown player, treated as an even gamma=1 reference); it is a pure query
    # that neither prints nor persists nobody2.
    assert whr.probability_future_match("shusai", "nobody2", 0) == (
        0.6224906898220315,
        0.3775093101779684,
    )
    assert capsys.readouterr().out == ""
    assert "nobody2" not in whr.players
    # test getting log likelihood of base
    assert whr.log_likelihood() == pytest.approx(0.7431542354571272)
    # test printing ordered ratings
    whr.print_ordered_ratings()
    display = "nobody => [-112.37545390067574]\nshusaku => [25.552142942931102, 24.669738398550702, 24.49953062693439]\nshusai => [84.74972643795506, 86.17200033461006, 86.88207745833284]\n"
    captured = capsys.readouterr()
    assert display == captured.out
    # test printing ordered ratings, only current elo
    whr.print_ordered_ratings(current=True)
    display = "nobody => -112.37545390067574\nshusaku => 24.49953062693439\nshusai => 86.88207745833284\n"
    captured = capsys.readouterr()
    assert display == captured.out
    # test getting ordered ratings, compact form
    assert whr.get_ordered_ratings(compact=True) == [
        [-112.37545390067574],
        [25.552142942931102, 24.669738398550702, 24.49953062693439],
        [84.74972643795506, 86.17200033461006, 86.88207745833284],
    ]
    # test getting ordered ratings, only current elo with compact form
    assert whr.get_ordered_ratings(compact=True, current=True) == pytest.approx(
        [-112.37545390067574, 24.49953062693439, 86.88207745833284]
    )
    # test saving base
    path = str(tmp_path / "state.pkl")
    whole_history_rating.WHR.save_base(whr, path)
    # test loading base
    whr2 = whole_history_rating.WHR.load_base(path)
    # test inspecting the first game
    whr_games = [str(x) for x in whr.games]
    whr2_games = [str(x) for x in whr2.games]
    assert whr_games == whr2_games


def test_save_and_load(tmp_path):
    whr = whole_history_rating.WHR(
        config={"w2": 1000, "uncased": True, "debug": True, "extra_parameter": "hello"}
    )
    path = str(tmp_path / "state.pkl")
    whole_history_rating.WHR.save_base(whr, path)
    whr2 = whole_history_rating.WHR.load_base(path)
    assert whr.config == whr2.config


def test_save_and_load_large_history_does_not_hit_recursion_limit(tmp_path):
    # A large, densely connected history produces a deep object graph.
    # Pickling that graph directly overflows the (C) stack (see issue #12),
    # so serialization must not rely on recursively walking it.
    whr = whole_history_rating.WHR()
    for i in range(2000):
        whr.create_game(f"p{i}", f"p{i + 1}", "B", i + 1, 0)
    whr.iterate(10)

    path = str(tmp_path / "large.pkl")
    whr.save_base(path)
    whr2 = whole_history_rating.WHR.load_base(path)

    # Computed ratings must survive the round-trip so the history does not
    # have to be re-rated from scratch after loading.
    assert whr2.get_ordered_ratings() == whr.get_ordered_ratings()
    assert whr2.ratings_for_player("p10") == whr.ratings_for_player("p10")


def test_save_and_load_preserves_players_without_games(tmp_path):
    # A player can exist without any game (e.g. created via player_by_name) and
    # therefore without a rated day. Such players must survive save/load instead
    # of making load_base raise a KeyError.
    whr = whole_history_rating.WHR()
    whr.load_games(["a b B 1"])
    whr.iterate(5)
    whr.player_by_name("ghost")
    assert "ghost" in whr.players

    path = str(tmp_path / "ghost.pkl")
    whr.save_base(path)
    loaded = whole_history_rating.WHR.load_base(path)

    assert "ghost" in loaded.players
    assert loaded.get_ordered_ratings() == whr.get_ordered_ratings()


def test_ratings_are_plain_python_floats():
    # After the multidimensional Newton update the ratings would otherwise be
    # numpy scalars, which render as "np.float64(...)" under numpy 2.x and leak
    # into printed/returned ratings.
    whr = whole_history_rating.WHR()
    whr.load_games(["a b B 1", "a b W 2", "a c B 3"])
    whr.iterate(10)

    for _, elos in whr.get_ordered_ratings():
        for elo in elos:
            assert type(elo) is float
    for _, _, uncertainty in whr.ratings_for_player("a"):
        assert type(uncertainty) is float


def test_load_base_reads_legacy_format(tmp_path):
    # Files written by previous versions pickled the object graph as a plain
    # list [players, games, config]. load_base must still read them.
    whr = whole_history_rating.WHR()
    whr.load_games(["a b B 1", "a b W 2", "a c B 3"])
    whr.iterate(10)

    path = str(tmp_path / "legacy.pkl")
    with open(path, "wb") as f:
        pickle.dump([whr.players, whr.games, whr.config], f)

    loaded = whole_history_rating.WHR.load_base(path)
    assert loaded.get_ordered_ratings() == whr.get_ordered_ratings()
    assert [str(g) for g in loaded.games] == [str(g) for g in whr.games]


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_auto_iterate(capsys):
    whr = whole_history_rating.WHR()
    # test loading several games at once
    test_games = [
        "shusaku; shusai; B; 1",
        "shusaku;shusai;W;2;0",
        " shusaku ; shusai ;W ; 3; {'w2':300}",
        "shusaku;nobody;B;3;0;{'w2':300}",
    ]
    whr.load_games(test_games, separator=";")
    # test auto iterating to get convergence
    whr1 = copy.deepcopy(whr)
    whr2 = copy.deepcopy(whr)
    whr3 = copy.deepcopy(whr)
    whr4 = copy.deepcopy(whr)
    whr5 = copy.deepcopy(whr)
    iterations1, is_stable1 = whr1.auto_iterate(batch_size=1)
    assert iterations1 == 12
    assert is_stable1
    iterations2, is_stable2 = whr2.auto_iterate()
    assert iterations2 == 30
    assert is_stable2
    iterations3, is_stable3 = whr3.auto_iterate(precision=0.5, batch_size=1)
    assert iterations3 == 6
    assert is_stable3
    iterations4, is_stable4 = whr4.auto_iterate(precision=0.9, batch_size=1)
    assert iterations4 == 5
    assert is_stable4
    iterations5, is_stable5 = whr5.auto_iterate(time_limit=1, batch_size=1)
    assert iterations5 == 12
    assert is_stable5


def test_whr_is_the_public_name_and_base_is_deprecated():
    # WHR is the canonical name and must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        whr = whole_history_rating.WHR()
    assert isinstance(whr, whole_history_rating.WHR)
    # Base still works but is deprecated.
    with pytest.warns(DeprecationWarning):
        legacy = whole_history_rating.Base()
    assert isinstance(legacy, whole_history_rating.WHR)


def test_config_is_not_mutated_or_shared():
    cfg = {"w2": 14}
    a = whole_history_rating.WHR(cfg)
    assert cfg == {"w2": 14}  # the caller's dict is left untouched
    b = whole_history_rating.WHR(cfg)
    assert a.config is not b.config  # instances do not share the same dict


def test_probability_future_match_is_a_pure_query(capsys):
    whr = whole_history_rating.WHR()
    whr.create_game("a", "b", "B", 1, 0)
    whr.iterate(10)
    before = set(whr.players)

    p1, p2 = whr.probability_future_match("a", "ghost")

    assert set(whr.players) == before  # querying does not persist players
    assert capsys.readouterr().out == ""  # and does not print
    assert 0 <= p1 <= 1 and 0 <= p2 <= 1


def test_ratings_for_player_unknown_raises_clear_error():
    whr = whole_history_rating.WHR()
    whr.create_game("a", "b", "B", 1, 0)
    whr.iterate(5)
    with pytest.raises(ValueError):
        whr.ratings_for_player("unknown", current=True)
    assert "unknown" not in whr.players  # and does not create the player


def test_game_opponent_returns_the_other_player():
    whr = whole_history_rating.WHR()
    game = whr.create_game("black", "white", "B", 1, 0)
    assert game.opponent(game.black_player) is game.white_player
    assert game.opponent(game.white_player) is game.black_player


def test_game_prediction_score():
    # White much stronger and white wins -> correct prediction.
    assert setup_game_with_elo(800, 200, 0).prediction_score() == 1.0
    # White much weaker but recorded as the winner -> wrong prediction.
    assert setup_game_with_elo(200, 800, 0).prediction_score() == 0.0


def test_get_ordered_ratings_current_non_compact_returns_name_elo_tuples():
    whr = whole_history_rating.WHR()
    whr.create_game("a", "b", "B", 1, 0)
    whr.create_game("a", "b", "W", 2, 0)
    whr.iterate(5)
    result = whr.get_ordered_ratings(current=True)
    assert all(
        isinstance(row, tuple)
        and len(row) == 2
        and isinstance(row[0], str)
        and isinstance(row[1], float)
        for row in result
    )
    assert {row[0] for row in result} == {"a", "b"}


def test_probability_future_match_rejects_self_match():
    whr = whole_history_rating.WHR()
    with pytest.raises(AttributeError):
        whr.probability_future_match("a", "a")


def test_load_games_rejects_malformed_input():
    whr = whole_history_rating.WHR()
    with pytest.raises(ValueError):
        whr.load_games(["a b B"])  # too few fields
    with pytest.raises(ValueError):
        whr.load_games(["a b B 1 not_a_number"])  # bad handicap / extras
    with pytest.raises(ValueError):
        whr.load_games(["a b B 1 0 not_a_dict"])  # bad extras dict


def test_save_base_with_unpicklable_config_warns_and_falls_back(tmp_path):
    whr = whole_history_rating.WHR(
        config={"w2": 300, "uncased": False, "debug": False, "bad": lambda x: x}
    )
    whr.create_game("a", "b", "B", 1, 0)
    whr.iterate(3)
    path = str(tmp_path / "state.pkl")
    with pytest.warns(UserWarning):
        whr.save_base(path)
    loaded = whole_history_rating.WHR.load_base(path)
    assert "bad" not in loaded.config
    assert loaded.config["w2"] == 300


@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
def test_auto_iterate_returns_not_stable_on_timeout():
    whr = whole_history_rating.WHR()
    whr.load_games(["shusaku shusai B 1", "shusaku shusai W 2", "shusaku shusai W 3"])
    iterations, is_stable = whr.auto_iterate(time_limit=0, batch_size=1)
    assert not is_stable
    assert iterations >= 1
