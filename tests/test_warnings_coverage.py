"""Every read of the fit warns the same way, and the docs say what the code does.

StaleFitWarning was added read by read, and four reads were missed. Instead of
one test per method, every public name of ``WHR`` is classified here: a read of
the fitted ratings must warn when the fit is stale; anything else must say why
it is exempt. A new public method fails ``test_every_public_name_is_classified``
until someone decides which it is.
"""

import re
from pathlib import Path

import pytest

from whr import (
    DisconnectedPlayersWarning,
    StaleFitWarning,
    UncomputedUncertaintyWarning,
)
from whr.whole_history_rating import WHR

ROOT = Path(__file__).resolve().parents[1]

# Reads of the fitted ratings: stale after games are added, so they must warn.
READS = {
    "ratings_for_player": lambda w: w.ratings_for_player("a"),
    "get_ordered_ratings": lambda w: w.get_ordered_ratings(),
    "print_ordered_ratings": lambda w: w.print_ordered_ratings(),
    "probability_future_match": lambda w: w.probability_future_match("a", "b"),
    "win_draw_loss_probabilities": lambda w: w.win_draw_loss_probabilities("a", "b"),
    "rating_difference": lambda w: w.rating_difference("a", "b"),
    "rating_covariance": lambda w: w.rating_covariance("a"),
    "rating_change": lambda w: w.rating_change("a", 1, 3),
    "display_offset_for": lambda w: w.display_offset_for(1500, "a"),
}

# Everything else, with the reason it does not warn.
EXEMPT = {
    # writes, fits, persistence
    "create_game": "adds a game: this is what makes the fit stale",
    "load_games": "adds games",
    "iterate": "fits",
    "auto_iterate": "fits",
    "fit_w2": "fits its own fresh models, never reads this one's ratings",
    "remove_drift": "rewrites the ratings it is given",
    "save_base": "persists state, stale or not; load_base restores the status",
    "load_base": "persists state",
    "player_by_name": "low-level accessor (creates the player if missing)",
    # gauges of the current state
    "games_since_last_fit": "the staleness gauge itself",
    "max_gradient_norm": "the convergence gauge; describes the current state",
    "log_likelihood": "a diagnostic of the current state, stale or not",
    "connected_components": "computed from the games, always current",
    "one_sided_game_share": "computed from the games, always current",
    # configuration and constants
    "draw_tendency": "the fitted nu; a property read constantly, even internally",
    "nu": "as draw_tendency",
    "draws_declared": "reads the config",
    "display_offset": "reads the config",
    "nu_from_draw_rate": "static conversion",
    "draw_rate_from_nu": "static conversion",
    "SAVE_FORMAT_VERSION": "constant",
}


def test_every_public_name_is_classified():
    public = {name for name in dir(WHR) if not name.startswith("_")}
    assert public == set(READS) | set(EXEMPT)
    assert not set(READS) & set(EXEMPT)


def _stale():
    w = WHR({"w2": 30, "draw_rate": 0.2})
    w.load_games(["a b B 1", "a b W 2", "b c B 2", "a c W 3", "a b D 3"])
    w.iterate(30)
    w.create_game("a", "b", "B", 3, 0)
    return w


@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_a_read_of_a_stale_fit_warns(read, capsys):
    w = _stale()
    with pytest.warns(StaleFitWarning):
        read(w)


def test_rating_difference_across_pools_that_never_met_warns():
    w = WHR({"w2": 30})
    w.load_games(["a b B 1", "a b W 2", "x y B 1", "x y W 2"])
    w.iterate(30)
    with pytest.warns(DisconnectedPlayersWarning):
        w.rating_difference("a", "x")


# --------------------------------------------------------------------------- #
# reads before any fit
# --------------------------------------------------------------------------- #
def _unfitted():
    w = WHR({"w2": 30})
    w.load_games(["a b B 1", "a b W 2", "a c W 3"])
    return w


@pytest.mark.parametrize(
    "read",
    [lambda w: w.rating_covariance("a"), lambda w: w.rating_change("a", 1, 3)],
    ids=["rating_covariance", "rating_change"],
)
def test_a_covariance_read_before_any_fit_warns_that_it_means_nothing(read):
    """They return a value, as their docstrings promise, but not silently: the
    numbers describe an un-fitted state."""
    with pytest.warns(UncomputedUncertaintyWarning, match="not been computed"):
        read(_unfitted())


def test_the_uncomputed_uncertainty_warning_describes_the_other_reads_truly():
    with pytest.warns(UncomputedUncertaintyWarning) as record:
        _unfitted().ratings_for_player("a")
    message = str(record[0].message)
    assert "rating_difference" in message
    assert "rating_covariance/rating_change raise" not in message
    assert "rating_difference/rating_covariance raise" not in message


# --------------------------------------------------------------------------- #
# displayed values
# --------------------------------------------------------------------------- #
def test_a_small_variance_is_not_displayed_as_zero():
    """A well-measured player (variance ~0.0036, i.e. ~10 elo) used to show
    0.0, rounded to 2 decimals."""
    w = WHR({"w2": 30})
    for day in range(1, 4):
        for i in range(400):
            w.create_game("a", "b", "BW"[i % 2], day, 0)
    w.iterate(50)
    shown = [u for _, _, u in w.ratings_for_player("a")]
    stored = [d.uncertainty for d in w.players["a"].days]
    for s, u in zip(stored, shown, strict=True):
        assert u > 0
        assert u == pytest.approx(s, rel=0.05)


def test_a_variance_of_ordinary_size_still_shows_two_decimals():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "a b W 3"])
    w.iterate(50)
    for _, _, u in w.ratings_for_player("a"):
        assert u == round(u, 2)


def test_connected_components_cannot_be_edited_through_its_result():
    w = WHR()
    w.load_games(["a b B 1", "x y B 1"])
    w.connected_components().pop()
    assert len(w.connected_components()) == 2


# --------------------------------------------------------------------------- #
# the docs
# --------------------------------------------------------------------------- #
def _docs():
    return "\n".join(
        (ROOT / p).read_text(encoding="utf-8")
        for p in ("README.md", "docs/api.md", "docs/user-guide.md")
    )


def test_the_docs_do_not_call_properties_like_methods():
    docs = _docs()
    for name in ("draws_declared", "draw_tendency", "games_since_last_fit"):
        assert not re.search(rf"\b{name}\(\)", docs), name


def test_the_docs_do_not_claim_the_covariance_reads_raise_before_a_fit():
    assert "`rating_covariance` and `rating_change` raise" not in _docs()


def test_the_draw_declarations_the_docs_pair_up_are_the_same_rate():
    """The guide offers ``draw_rate=R`` "or" ``pinned_draw=NU`` as two
    spellings of one declaration: they must describe the same draw rate."""
    pairs = re.findall(
        r"draw_rate['\"]?[=:]\s*([0-9.]+)[^\n]*?\bor\b[^\n]*?pinned_draw['\"]?[=:]\s*([0-9.]+)",
        _docs(),
    )
    assert pairs
    for rate, nu in pairs:
        assert WHR.nu_from_draw_rate(float(rate)) == pytest.approx(float(nu), abs=5e-3)


def test_one_sided_share_matches_its_docstring_at_the_boundary():
    """A player on one side in exactly 5% of games: the docstring and the code
    must agree on whether that is one-sided."""
    w = WHR({"estimate_handicap_zero": True})
    for day in range(20):
        if day == 0:
            w.create_game("other", "fixed", "B", day, 0)
        else:
            w.create_game("fixed", "other", "B", day, 0)
    doc = WHR.one_sided_game_share.__doc__
    assert "fewer than ``5%``" in doc
    assert w.one_sided_game_share() == 0.0  # exactly 5%: not one-sided
