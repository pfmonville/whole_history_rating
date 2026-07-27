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


class HandicapBaselineWarning(UserWarning):
    """``estimate_handicap_zero`` is on, but the data cannot identify the baseline.

    With the ``handicap`` key ``0`` pinned (the default) the advantage scale has
    a fixed zero. Freeing it adds a global black-advantage parameter, and that
    parameter is only identifiable if colour assignment varies independently of
    who is playing. When players sit on one side of the board -- one competitor
    always "black" -- the free baseline trades off against their strength: the
    *differences* between handicap keys stay correct while the overall level
    leaks into the ratings, so genuinely equal players can be reported tens of
    elo apart and ``probability_future_match`` without a ``handicap_key`` is
    wrong by the same amount.

    The fix is usually to leave ``estimate_handicap_zero`` at ``False``. If the
    baseline really must be estimated, anchor the scale another way -- pin a
    handicap value you know via ``pinned_handicap``.

    The check is a heuristic on observed colour shares, so it can miss subtler
    confounds; a quiet run is not a proof of identifiability.
    """


class UncomputedUncertaintyWarning(UserWarning):
    """Uncertainties were read before ``iterate()`` computed any.

    ``PlayerDay.uncertainty`` starts at the sentinel ``-1.0``, which is not a
    real standard deviation -- it means "not computed yet". ``ratings_for_player``
    returns it as-is rather than raising, so an un-rated base can still be
    inspected, but a ``-1`` reaching a calculation is a bug. Its siblings
    (``rating_difference``, ``rating_covariance``, ``rating_change``) raise a
    ``ValueError`` in the same state.

    Call ``iterate()`` or ``auto_iterate()`` first.
    """
