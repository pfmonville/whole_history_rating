import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from whr import whole_history_rating
from whr import utils


def setup_game_with_elo(white_elo, black_elo, handicap):
    whr = whole_history_rating.Base()
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


def test_output():
    whr = whole_history_rating.Base()
    whr.create_game("shusaku", "shusai", "B", 1, 0)
    whr.create_game("shusaku", "shusai", "W", 2, 0)
    whr.create_game("shusaku", "shusai", "W", 3, 0)
    whr.create_game("shusaku", "shusai", "W", 4, 0)
    whr.create_game("shusaku", "shusai", "W", 4, 0)
    whr.iterate(50)
    assert [
        [1, -92, 71],
        [2, -94, 71],
        [3, -95, 71],
        [4, -96, 72],
    ] == whr.ratings_for_player("shusaku")
    assert [
        [1, 92, 71],
        [2, 94, 71],
        [3, 95, 71],
        [4, 96, 72],
    ] == whr.ratings_for_player("shusai")


def test_unstable_exception_raised_in_certain_cases():
    whr = whole_history_rating.Base()
    for _ in range(10):
        whr.create_game("anchor", "player", "B", 1, 0)
        whr.create_game("anchor", "player", "W", 1, 0)
    for _ in range(10):
        whr.create_game("anchor", "player", "B", 180, 600)
        whr.create_game("anchor", "player", "W", 180, 600)
    with pytest.raises(utils.UnstableRatingException):
        whr.iterate(10)


def test_log_likelihood():
    whr = whole_history_rating.Base()
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
    whr = whole_history_rating.Base(config={"w2": 14, "uncased": True})
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


def test_loading_several_games_at_once(capsys):
    whr = whole_history_rating.Base()
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
    whr.auto_iterate()
    # test getting ratings for player shusaku (day, elo, uncertainty)
    assert whr.ratings_for_player("shusaku") == [
        [1, 26.0, 70.0],
        [2, 25.0, 70.0],
        [3, 24.0, 70.0],
    ]
    # test getting ratings for player shusai, only current elo and uncertainty
    assert whr.ratings_for_player("shusai", current=True) == (87.0, 84.0)
    # test getting probability of future match between shusaku and nobody2 (which default to 1 win 1 loss)
    assert whr.probability_future_match("shusai", "nobody2", 0) == (
        0.6224906898220315,
        0.3775093101779684,
    )
    # test getting log likelihood of base
    assert whr.log_likelihood() == 0.7431542354571272
    # test printing ordered ratings
    whr.print_ordered_ratings()
    display = "win probability: shusai:0.62%; nobody2:0.38%\nnobody => [-112.37545390067574]\nshusaku => [25.552142942931102, 24.669738398550702, 24.49953062693439]\nshusai => [84.74972643795506, 86.17200033461006, 86.88207745833284]\n"
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
    assert whr.get_ordered_ratings(compact=True, current=True) == [
        -112.37545390067574,
        24.49953062693439,
        86.88207745833284,
    ]
    # test saving base
    whole_history_rating.Base.save_base(whr, "test_whr.pkl")
    # test loading base
    whr2 = whole_history_rating.Base.load_base("test_whr.pkl")
    # test inspecting the first game
    whr_games = [x.inspect() for x in whr.games]
    whr2_games = [x.inspect() for x in whr2.games]
    assert whr_games == whr2_games
