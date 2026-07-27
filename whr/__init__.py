"""Whole-History Rating (WHR).

A Python implementation of Rémi Coulom's Whole-History Rating algorithm, a
dynamic Bayesian rating system that estimates players' skills continuously
over time.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whr.game import Game
from whr.player import Player
from whr.playerday import PlayerDay
from whr.utils import (
    HandicapBaselineWarning,
    NoDrawsWarning,
    UncomputedUncertaintyWarning,
    UnstableRatingException,
)
from whr.whole_history_rating import WHR, Base

try:
    __version__ = version("whole-history-rating")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "WHR",
    "Base",
    "Game",
    "HandicapBaselineWarning",
    "NoDrawsWarning",
    "Player",
    "PlayerDay",
    "UncomputedUncertaintyWarning",
    "UnstableRatingException",
    "__version__",
]
