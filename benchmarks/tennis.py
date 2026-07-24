"""ATP tennis benchmark — WHR in the style of TrueSkill Through Time.

Data: Jeff Sackmann's ATP match files (`atp_matches_YYYY.csv`), the canonical
source TrueSkill Through Time analyses. We use the seasons available from a
reachable mirror (2000-2015): ~48k main-tour singles matches, ~1950 players,
covering the Federer / Nadal / Djokovic / Roddick / Hewitt / Murray era — the
window of TTT's famous "history of tennis" figure.

Protocol (forward validation, no leakage):
  * time_step = day index from ``tourney_date`` (already tournament-granular,
    since a player's matches within one tournament share the date).
  * w2 chosen on a validation season, frozen, then reported on a later season.
  * TRAINING orients each match winner->loser (the Bradley-Terry datum).
    TEST orientation is by player id (independent of the result) so the label
    is never leaked into the prediction.

Run:  uv run --with pandas --with matplotlib python benchmarks/tennis.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# Tennis skill is stable week to week, so small w2 (rigid ratings) fits best;
# the grid reaches down to 1 elo^2/day.
W2_GRID = [1.0, 3.0, 10.0, 30.0]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tennis")
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _d(yyyymmdd: int) -> date:
    s = str(int(yyyymmdd))
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "atp_matches_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    origin = _d(df["tourney_date"].min())
    recs = []
    for r in df.itertuples(index=False):
        when = _d(r.tourney_date)
        day = (when - origin).days
        recs.append(
            {
                "day": day,
                "when": when,
                "year": when.year,
                "winner": str(r.winner_name),
                "loser": str(r.loser_name),
                "wid": str(r.winner_id),
                "lid": str(r.loser_id),
            }
        )
    return recs


def train_matches(recs) -> list[C.Match]:
    """Winner -> p1 (black) for fitting. Outcome is always WIN by construction."""
    return [
        C.Match(
            day=r["day"], when=r["when"], p1=r["winner"], p2=r["loser"], outcome=C.WIN
        )
        for r in recs
    ]


def test_pairs(whr, recs) -> list[tuple[float, int]]:
    """Neutral orientation by player id: p1 = smaller id. y = 1 iff p1 won.
    Predict P(p1 wins) from frozen ratings."""
    pairs = []
    for r in recs:
        if r["wid"] <= r["lid"]:
            p1, p2, y = r["winner"], r["loser"], 1
        else:
            p1, p2, y = r["loser"], r["winner"], 0
        pairs.append((C.predict_two_way(whr, p1, p2), y))
    return pairs


def main():
    os.makedirs(RESULTS, exist_ok=True)
    recs = load()
    years = sorted({r["year"] for r in recs})
    print(
        f"loaded {len(recs)} matches, {years[0]}-{years[-1]}, "
        f"{len({r['winner'] for r in recs} | {r['loser'] for r in recs})} players",
        flush=True,
    )

    tr_v = [r for r in recs if r["year"] <= 2012]
    va = [r for r in recs if r["year"] == 2013]
    tr_t = [r for r in recs if r["year"] <= 2013]
    te = [r for r in recs if r["year"] == 2014]
    print(
        f"train_v={len(tr_v)} valid={len(va)} train_t={len(tr_t)} test={len(te)}",
        flush=True,
    )

    # ---- select w2 on validation ----
    best_w2, best_ll = None, float("inf")
    for w2 in W2_GRID:
        whr = C.build_and_fit(
            train_matches(tr_v), w2, time_limit=240, precision=5e-3, verbose=True
        )
        ll = C.binary_log_loss(test_pairs(whr, va))
        print(f"  w2={w2:>6}: valid log-loss = {ll:.4f}", flush=True)
        if ll < best_ll:
            best_ll, best_w2 = ll, w2
    print(f"selected w2={best_w2} (valid log-loss {best_ll:.4f})", flush=True)

    # ---- test season ----
    whr = C.build_and_fit(
        train_matches(tr_t), best_w2, time_limit=240, precision=5e-3, verbose=True
    )
    pairs = test_pairs(whr, te)
    results = {
        "dataset": "atp_tennis",
        "years": [years[0], years[-1]],
        "w2": best_w2,
        "test_season": 2014,
        "n_test": len(te),
        "models": {
            "base_rate_0.5": {
                "log_loss": C.binary_log_loss([(0.5, y) for _, y in pairs]),
                "accuracy": 0.5,
                "n": len(pairs),
            },
            "whr": {
                "log_loss": C.binary_log_loss(pairs),
                "accuracy": C.binary_accuracy(pairs),
                "n": len(pairs),
            },
        },
        "calibration_whr": C.calibration_bins(pairs),
    }
    for name, m in results["models"].items():
        print(
            f"  {name:<14} log-loss={m['log_loss']:.4f} acc={m['accuracy']:.4f}",
            flush=True,
        )

    with open(os.path.join(RESULTS, "tennis_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _dump_history_curves(recs, best_w2)
    print(
        "wrote results/tennis_results.json and results/tennis_curves.json", flush=True
    )


def _dump_history_curves(recs, w2):
    """Fit the FULL history once and dump each player's rating curve as
    ``(year, display_elo)`` pairs. Rendering lives in make_figures.py."""
    whr = C.build_and_fit(
        train_matches(recs), w2, time_limit=240, precision=5e-3, verbose=True
    )
    stars = [
        "Roger Federer",
        "Rafael Nadal",
        "Novak Djokovic",
        "Andy Murray",
        "Lleyton Hewitt",
        "Andy Roddick",
    ]
    origin = recs[0]["when"]
    curves = {}
    for name in stars:
        try:
            r = whr.ratings_for_player(name)
        except ValueError:
            continue
        # +1500 is a goratings-style display shift (only rating *differences*
        # are meaningful in WHR).
        curves[name] = [[origin.year + day / 365.25, elo + 1500] for (day, elo, _) in r]
    with open(os.path.join(RESULTS, "tennis_curves.json"), "w") as f:
        json.dump({"display_shift": 1500, "curves": curves}, f)


if __name__ == "__main__":
    main()
