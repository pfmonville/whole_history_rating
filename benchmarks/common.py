"""Shared benchmarking utilities for the WHR real-data benchmarks.

The three benchmark scripts (nba.py, tennis.py, football.py) all follow the same
recipe, implemented here once:

1. Parse the source data into a chronological list of ``Match`` records.
2. Map calendar dates to integer WHR ``time_step`` values (days since the first
   match), so that ``w2`` is expressed per real day.
3. Split the timeline into a *training* prefix and a held-out *test* suffix
   (forward validation: never train on a match played after a test match).
4. Fit a single WHR model on the training prefix.
5. Predict every test match from the players' latest training-day ratings and
   score the predictions with log-loss / accuracy / calibration.

Frozen-rating holdout (step 5) is the standard, cheap holdout: one fit, no leak.
We keep the test window short (typically the last season) so freezing ratings
over the test period is a mild approximation. ``nba.py`` additionally runs a
walk-forward *online* evaluation that folds each revealed result back in and
re-iterates, the fair analogue of a per-game-updated rating system.

None of this is a bit-exact reproduction of the reference papers' pipelines; it
is an honest, clearly-specified re-run of comparable metrics on the same data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

# Two-way outcomes are encoded from player1's perspective.
WIN = "W"  # player1 won
LOSS = "L"  # player1 lost
DRAW = "D"  # draw (football only)

EPS = 1e-15  # log-loss probability clip


@dataclass(frozen=True)
class Match:
    """A single result, oriented from ``p1``'s perspective."""

    day: int  # WHR time_step: integer days since the dataset's first match
    when: date
    p1: str
    p2: str
    outcome: str  # WIN / LOSS / DRAW (p1's perspective)
    # Optional handicap category key applied to p1 (e.g. "home" for home-court
    # advantage). 0 means no advantage. WHR estimates each key's strength.
    handicap: object = 0


def to_day_index(when: date, origin: date) -> int:
    """Calendar date -> integer WHR time_step (days since ``origin``)."""
    return (when - origin).days


def temporal_split(
    matches: list[Match], test_fraction: float | None = None, cutoff: date | None = None
) -> tuple[list[Match], list[Match]]:
    """Split chronologically. Either hold out the last ``test_fraction`` of
    matches, or everything on/after ``cutoff``. Matches must be pre-sorted by
    day. Returns ``(train, test)``."""
    if (test_fraction is None) == (cutoff is None):
        raise ValueError("pass exactly one of test_fraction / cutoff")
    if cutoff is not None:
        train = [m for m in matches if m.when < cutoff]
        test = [m for m in matches if m.when >= cutoff]
    else:
        assert test_fraction is not None
        k = int(round(len(matches) * (1.0 - test_fraction)))
        train, test = matches[:k], matches[k:]
    return train, test


# --------------------------------------------------------------------------- #
# WHR wiring
# --------------------------------------------------------------------------- #
def build_and_fit(
    train: list[Match],
    w2: float,
    *,
    with_draws: bool = False,
    pinned_draw: float | None = None,
    time_limit: float = 120.0,
    precision: float = 1e-3,
    verbose: bool = True,
):
    """Create a WHR instance, load the training matches, and iterate to
    convergence (or ``time_limit`` seconds). Returns the fitted instance."""
    from whr import WHR

    # Komi is a Go concept; none of these sports use it, so no `komi=` is passed
    # to create_game below and no komi advantage is modelled. (Before WHR 3.1.0
    # made komi opt-in, every game silently carried an *estimated* komi key of
    # 6.5, and these benchmarks had to neutralise it with
    # `pinned_komi={6.5: 0.0}` — no longer necessary.)
    config: dict = {"w2": w2}
    if pinned_draw is not None:
        config["pinned_draw"] = pinned_draw
    whr = WHR(config)

    for m in train:
        if m.outcome == DRAW:
            winner = "D"
        elif m.outcome == WIN:
            winner = "B"  # p1 is "black"
        else:
            winner = "W"  # p1 lost -> white (p2) won
        whr.create_game(m.p1, m.p2, winner, m.day, m.handicap)

    iters = whr.auto_iterate(time_limit=time_limit, precision=precision, batch_size=10)
    if verbose:
        gn = whr.max_gradient_norm()
        extra = f", nu={whr.draw_tendency:.4f}" if with_draws else ""
        print(f"    fit: {len(train)} games, grad_norm={gn:.2e}, iters~{iters}{extra}")
    return whr


def predict_two_way(whr, p1: str, p2: str) -> float:
    """P(p1 beats p2) from current ratings. Unknown players -> even (0.5)."""
    p1_win, _ = whr.probability_future_match(p1, p2, 0)
    return float(p1_win)


def predict_three_way(whr, p1: str, p2: str) -> tuple[float, float, float]:
    """(P(p1 win), P(draw), P(p2 win)) under the fitted Davidson model."""
    win, draw, loss = whr.win_draw_loss_probabilities(p1, p2, 0)
    return float(win), float(draw), float(loss)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, p))


def binary_log_loss(pairs: list[tuple[float, int]]) -> float:
    """Mean negative log-likelihood. ``pairs`` = (predicted P(win), actual 0/1)."""
    if not pairs:
        return float("nan")
    s = 0.0
    for p, y in pairs:
        p = _clip(p)
        s += -(math.log(p) if y == 1 else math.log(1.0 - p))
    return s / len(pairs)


def binary_accuracy(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    correct = sum(1 for p, y in pairs if (p > 0.5) == bool(y))
    # ties (p == 0.5) count as half
    half = sum(0.5 for p, _ in pairs if p == 0.5)
    return (correct + 0.0 * half) / len(pairs)


def three_way_log_loss(rows: list[tuple[tuple[float, float, float], int]]) -> float:
    """``rows`` = ((P_win, P_draw, P_loss), class in {0:win,1:draw,2:loss})."""
    if not rows:
        return float("nan")
    s = 0.0
    for probs, cls in rows:
        s += -math.log(_clip(probs[cls]))
    return s / len(rows)


def three_way_accuracy(
    rows: list[tuple[tuple[float, float, float], int]],
) -> float:
    if not rows:
        return float("nan")
    correct = 0
    for probs, cls in rows:
        pred = max(range(3), key=lambda i: probs[i])
        correct += pred == cls
    return correct / len(rows)


def calibration_bins(
    pairs: list[tuple[float, int]], n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Reliability curve. Returns per-bin (mean predicted, empirical win rate,
    count) for non-empty bins."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, int(p * n_bins))
        buckets[idx].append((p, y))
    out = []
    for b in buckets:
        if not b:
            continue
        mean_p = sum(p for p, _ in b) / len(b)
        emp = sum(y for _, y in b) / len(b)
        out.append((mean_p, emp, len(b)))
    return out


def base_rate_log_loss_binary(train: list[Match], test: list[Match]) -> float:
    """Naive baseline: predict the training P(p1/home win) for every match."""
    wins = sum(1 for m in train if m.outcome == WIN)
    rate = wins / len(train) if train else 0.5
    return binary_log_loss([(rate, 1 if m.outcome == WIN else 0) for m in test])


def base_rate_log_loss_three_way(train: list[Match], test: list[Match]) -> float:
    n = len(train) or 1
    pw = sum(1 for m in train if m.outcome == WIN) / n
    pd = sum(1 for m in train if m.outcome == DRAW) / n
    pl = sum(1 for m in train if m.outcome == LOSS) / n
    cls = {WIN: 0, DRAW: 1, LOSS: 2}
    return three_way_log_loss([((pw, pd, pl), cls[m.outcome]) for m in test])
