from __future__ import annotations


class UnstableRatingException(Exception):
    pass


class NoDrawsWarning(UserWarning):
    """A three-outcome prediction was asked for on data containing no draws.

    Absence of draws is ambiguous and the library cannot resolve it: it may mean
    the domain cannot draw (tennis, basketball), in which case ``P(draw) = 0`` is
    correct, or that no draw has occurred *yet* (an early-season league), in
    which case ``P(draw) = 0`` is a confident false claim -- and one that sends
    log-loss to infinity as soon as a draw happens.

    Only the caller knows which. Declaring it with ``pinned_draw`` or
    ``draw_rate`` silences this; ``pinned_draw=0.0`` (equivalently
    ``draw_rate=0.0``) is a valid answer, meaning "no draws, deliberately".

    A ``UserWarning`` subclass so existing broad filters still catch it, while
    ``warnings.simplefilter("ignore", NoDrawsWarning)`` can target just this.
    """
