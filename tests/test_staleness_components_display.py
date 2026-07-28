"""Three things the library used to leave to the caller to notice.

* a fit going stale after games are added, with no signal at all;
* comparisons across groups of players that never met, answered confidently;
* the elo display scale, which callers had to shift by hand -- and whose obvious
  in-place spelling silently erodes.
"""

import math
import warnings

import pytest

from whr import DisconnectedPlayersWarning, StaleFitWarning
from whr.whole_history_rating import WHR


def out_lines(capsys):
    """Drain and split captured stdout."""
    return capsys.readouterr().out.splitlines()


def _fitted(w2=300, iters=200):
    w = WHR({"w2": w2})
    for day in range(1, 20):
        for _ in range(3):
            w.create_game("a", "b", "B", day, 0)
        w.create_game("a", "b", "W", day, 0)
        w.create_game("b", "c", "B", day, 0)
    w.iterate(iters)
    return w


# --------------------------------------------------------------------------- #
# A. the fit going stale
# --------------------------------------------------------------------------- #
def test_a_fresh_fit_reports_no_pending_games():
    w = _fitted()
    assert w.games_since_last_fit == 0


def test_adding_games_marks_the_fit_stale():
    w = _fitted()
    w.create_game("a", "b", "W", 20, 0)
    w.create_game("a", "b", "W", 20, 0)
    assert w.games_since_last_fit == 2


def test_reading_ratings_while_stale_warns():
    """A brand-new day also has no computed uncertainty, so both warnings fire --
    two correct, independent signals about the same read."""
    w = _fitted()
    w.create_game("a", "b", "W", 20, 0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.ratings_for_player("a")
    stale = [r for r in rec if r.category is StaleFitWarning]
    assert len(stale) == 1
    assert "iterate" in str(stale[0].message)


@pytest.mark.parametrize(
    "read",
    [
        lambda w: w.ratings_for_player("a"),
        lambda w: w.get_ordered_ratings(),
        lambda w: w.print_ordered_ratings(),
        lambda w: w.probability_future_match("a", "b", 0),
        lambda w: w.win_draw_loss_probabilities("a", "b", 0),
    ],
)
def test_every_rating_read_surface_warns_while_stale(read):
    w = _fitted()
    w.create_game("a", "b", "W", 20, 0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        read(w)
    assert any(r.category is StaleFitWarning for r in rec)


def test_the_warning_fires_once_per_stale_episode():
    """Quiet in a read loop, but speaks again after the next batch of games."""
    w = _fitted()
    w.create_game("a", "b", "W", 20, 0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(10):
            w.ratings_for_player("a")
    assert sum(1 for r in rec if r.category is StaleFitWarning) == 1
    w.create_game("a", "b", "W", 21, 0)  # a new episode
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.ratings_for_player("a")
    assert sum(1 for r in rec if r.category is StaleFitWarning) == 1


def test_re_iterating_clears_the_staleness():
    w = _fitted()
    w.create_game("a", "b", "W", 20, 0)
    w.iterate(50)
    assert w.games_since_last_fit == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleFitWarning)
        w.ratings_for_player("a")


def test_a_never_fitted_base_does_not_warn_about_staleness():
    """Nothing is out of date if nothing was ever fitted; the caller is plainly
    driving the state by hand, and the uncertainty sentinel covers that."""
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleFitWarning)
        w.probability_future_match("a", "b", 0)


def test_the_stale_read_really_is_wrong_not_merely_out_of_date():
    """Why this warrants a warning: adding results to a day a player already had
    moves the rating far, while the stale read looks perfectly plausible."""
    w = _fitted()
    before = w.ratings_for_player("a", current=True)
    for _ in range(15):
        w.create_game("a", "b", "W", 19, 0)  # an EXISTING day, contradictory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StaleFitWarning)
        stale = w.ratings_for_player("a", current=True)
    w.iterate(300)
    after = w.ratings_for_player("a", current=True)
    assert stale == before  # indistinguishable from the fitted value
    assert abs(stale[0] - after[0]) > 100  # and hundreds of elo out


# --------------------------------------------------------------------------- #
# B. disconnected groups
# --------------------------------------------------------------------------- #
def _two_pools():
    w = WHR({"w2": 300})
    for day in range(1, 15):
        w.create_game("even1", "even2", "B", day, 0)
        w.create_game("even2", "even1", "B", day, 0)
        for k in range(3):
            w.create_game("king", f"weak{k}", "B", day, 0)
    w.iterate(200)
    return w


def test_connected_components_finds_the_groups_largest_first():
    w = _two_pools()
    groups = w.connected_components()
    assert len(groups) == 2
    assert groups[0] == frozenset({"king", "weak0", "weak1", "weak2"})
    assert groups[1] == frozenset({"even1", "even2"})


def test_connected_components_is_empty_without_games():
    assert WHR().connected_components() == []


def test_a_fully_linked_base_is_one_component():
    w = _fitted()
    assert w.connected_components() == [frozenset({"a", "b", "c"})]


def test_components_are_recomputed_after_a_game_links_two_pools():
    w = _two_pools()
    assert len(w.connected_components()) == 2
    w.create_game("king", "even1", "B", 20, 0)
    assert len(w.connected_components()) == 1


def test_predicting_across_components_warns():
    w = _two_pools()
    with pytest.warns(DisconnectedPlayersWarning, match="connected_components"):
        w.probability_future_match("king", "even1", 0)


def test_predicting_within_a_component_does_not_warn():
    w = _two_pools()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DisconnectedPlayersWarning)
        w.probability_future_match("king", "weak0", 0)
        w.probability_future_match("even1", "even2", 0)


def test_an_unknown_player_does_not_trigger_the_component_warning():
    """A cold start is already documented as an even reference; warning on it
    would fire on every unseen name."""
    w = _fitted()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DisconnectedPlayersWarning)
        w.probability_future_match("a", "never-seen", 0)


def test_the_cross_pool_answer_is_confident_and_unfounded():
    """The measurement that motivates the warning."""
    w = _two_pools()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DisconnectedPlayersWarning)
        p = w.probability_future_match("king", "even1", 0)[0]
    assert p > 0.9  # on no shared game whatsoever


def test_component_warning_respects_uncased():
    w = WHR({"w2": 300, "uncased": True})
    for day in range(1, 15):
        w.create_game("even1", "even2", "B", day, 0)
        w.create_game("king", "weak0", "B", day, 0)
    w.iterate(100)
    with pytest.warns(DisconnectedPlayersWarning):
        w.probability_future_match("KING", "Even1", 0)


# --------------------------------------------------------------------------- #
# 1 + 2. display scale
# --------------------------------------------------------------------------- #
def test_display_offset_defaults_to_zero_and_changes_nothing():
    w = _fitted()
    assert w.display_offset == 0.0
    assert w.config["display_offset"] == 0.0


def test_display_offset_shifts_every_display_surface():
    plain = _fitted()
    shifted = _fitted()
    shifted.config["display_offset"] = 1500.0
    assert (
        shifted.ratings_for_player("a", current=True)[0]
        == plain.ratings_for_player("a", current=True)[0] + 1500
    )
    assert [e for _d, e, _u in shifted.ratings_for_player("a")] == [
        e + 1500 for _d, e, _u in plain.ratings_for_player("a")
    ]
    assert shifted.get_ordered_ratings(current=True, compact=True) == pytest.approx(
        [e + 1500 for e in plain.get_ordered_ratings(current=True, compact=True)]
    )
    assert shifted.get_ordered_ratings(compact=True)[0] == pytest.approx(
        [e + 1500 for e in plain.get_ordered_ratings(compact=True)[0]]
    )
    names_shifted = [n for n, _ in shifted.get_ordered_ratings()]
    assert names_shifted == [n for n, _ in plain.get_ordered_ratings()]  # order intact


def test_display_offset_is_printed(capsys):
    w = _fitted()
    w.config["display_offset"] = 1500.0
    w.print_ordered_ratings(current=True)
    shifted = [float(line.split(" => ")[1]) for line in out_lines(capsys)]
    plain = _fitted()
    plain.print_ordered_ratings(current=True)
    base = [float(line.split(" => ")[1]) for line in out_lines(capsys)]
    assert shifted == pytest.approx([e + 1500.0 for e in base])


def test_display_offset_never_touches_a_prediction_or_a_difference():
    """The whole point: the offset is presentation, so anything consuming
    *differences* must be bit-identical with and without it."""
    plain = _fitted()
    shifted = _fitted()
    shifted.config["display_offset"] = 1500.0
    assert shifted.probability_future_match("a", "b", 0) == pytest.approx(
        plain.probability_future_match("a", "b", 0), rel=1e-15
    )
    assert shifted.rating_difference("a", "b") == pytest.approx(
        plain.rating_difference("a", "b"), rel=1e-15
    )
    assert shifted.rating_change("a", 1, 5) == pytest.approx(
        plain.rating_change("a", 1, 5), rel=1e-15
    )


def test_display_offset_does_not_erode_under_further_iteration():
    """Unlike writing the offset into day.elo by hand."""
    w = _fitted()
    w.config["display_offset"] = 1500.0
    first = w.ratings_for_player("a", current=True)[0]
    w.iterate(500)
    after = w.ratings_for_player("a", current=True)[0]
    # the underlying rating still drifts a few elo as the fit converges further;
    # what must NOT happen is the ~1400-elo decay an in-place offset suffers
    assert after == pytest.approx(first, abs=15)
    assert after > 1000


def test_display_offset_for_anchors_the_field_mean():
    w = _fitted()
    w.config["display_offset"] = w.display_offset_for(target=1500.0)
    last_day = max(d.day for p in w.players.values() for d in p.days)
    on_day = [
        d.elo + w.display_offset
        for p in w.players.values()
        for d in p.days
        if d.day == last_day
    ]
    assert sum(on_day) / len(on_day) == pytest.approx(1500.0)


def test_display_offset_for_anchors_a_named_player():
    w = _fitted()
    w.config["display_offset"] = w.display_offset_for(target=2000.0, player="a")
    assert w.ratings_for_player("a", current=True)[0] == 2000


def test_display_offset_for_anchors_a_named_player_on_a_given_day():
    w = _fitted()
    w.config["display_offset"] = w.display_offset_for(target=2000.0, player="a", day=3)
    got = dict((d, e) for d, e, _u in w.ratings_for_player("a"))
    assert got[3] == 2000


def test_display_offset_for_does_not_apply_itself():
    w = _fitted()
    assert w.display_offset_for(target=1500.0) != 0.0
    assert w.display_offset == 0.0


def test_display_offset_for_rejects_nothing_to_anchor_on():
    with pytest.raises(ValueError, match="no rated player-days"):
        WHR().display_offset_for(target=1500.0)


def test_display_offset_for_rejects_an_unknown_player():
    w = _fitted()
    with pytest.raises(ValueError, match="No ratings available"):
        w.display_offset_for(target=1500.0, player="ghost")


def test_display_offset_for_rejects_an_unrated_day():
    w = _fitted()
    with pytest.raises(ValueError):
        w.display_offset_for(target=1500.0, player="a", day=9999)


# --- uncertainty units ----------------------------------------------------- #
def test_uncertainty_defaults_to_the_stored_variance():
    w = _fitted()
    _day, _elo, unc = w.ratings_for_player("a")[0]
    assert unc == pytest.approx(round(w.player_by_name("a").days[0].uncertainty, 2))


def test_display_uncertainty_elo_reports_a_standard_error():
    """The trap this closes: the default 0.26 is a variance in nat^2, whose elo
    standard error is sqrt(0.26)*400/ln(10) = 88.6 -- a factor of ~340 apart, in a
    column of elo values."""
    w = _fitted()
    variance = w.player_by_name("a").days[0].uncertainty
    w.config["display_uncertainty"] = "elo"
    _day, _elo, unc = w.ratings_for_player("a")[0]
    expected = math.sqrt(variance) * 400.0 / math.log(10)
    assert unc == pytest.approx(round(expected, 2))
    assert unc > 10 * variance  # emphatically not the same number


def test_display_uncertainty_elo_matches_rating_difference_units():
    """`rating_difference` already reports elo standard errors; the elo display
    must agree with it for a single player."""
    w = _fitted()
    w.config["display_uncertainty"] = "elo"
    _day, _elo, unc = w.ratings_for_player("a", current=False)[-1]
    day = w.player_by_name("a").days[-1]
    assert unc == pytest.approx(
        round(math.sqrt(day.uncertainty) * 400.0 / math.log(10), 2)
    )


def test_display_uncertainty_elo_passes_the_uncomputed_sentinel_through():
    from whr import UncomputedUncertaintyWarning

    w = WHR({"display_uncertainty": "elo"})
    w.create_game("a", "b", "B", 1, 0)
    with pytest.warns(UncomputedUncertaintyWarning):
        assert w.ratings_for_player("a")[0][2] == -1


def test_display_uncertainty_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="display_uncertainty"):
        WHR({"display_uncertainty": "nats"})


def test_display_settings_survive_a_save_load_round_trip(tmp_path):
    path = str(tmp_path / "base.pkl")
    w = _fitted()
    w.config["display_offset"] = 1500.0
    w.config["display_uncertainty"] = "elo"
    w.save_base(path)
    back = WHR.load_base(path)
    assert back.display_offset == 1500.0
    assert back.config["display_uncertainty"] == "elo"
    assert back.ratings_for_player("a", current=True) == w.ratings_for_player(
        "a", current=True
    )
