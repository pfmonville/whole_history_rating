"""Football benchmark — WHR's Davidson draw model on real league data.

Data: openfootball/football.json, the big-five European leagues (England,
Spain, Germany, Italy, France), seasons 2014-15 .. 2023-24: ~18k matches with a
~25% draw rate. This is the natural stress test for WHR 3.0.0's Davidson draw
model (``win_draw_loss_probabilities``), the feature a win/loss-only rating
system cannot provide.

We compare three 3-way (home-win / draw / away-win) predictors on a held-out
season:
  * base_rate         — predict the training H/D/A frequencies for every match.
  * whr_bt_constdraw  — a draw-blind Bradley-Terry WHR (draws dropped from
                        training) whose 2-way P(win) is split into 3 classes
                        using a *constant* empirical draw rate. Isolates "just
                        assume a fixed draw rate".
  * whr_davidson      — the full Davidson model: the global draw tendency nu is
                        estimated, so draw probability rises for evenly-matched
                        teams. This is what the constant-draw baseline can't do.

Home advantage is folded in via WHR's handicap machinery (handicap key "home"
on the home team). Forward validation: w2 is picked on a validation season and
reported on a later, unseen season.

Run:  uv run --with pandas python benchmarks/football.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import provenance as P  # noqa: E402

W2_GRID = [30.0, 100.0, 300.0, 1000.0]
BIN_DAYS = 7
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "football")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
DATA_SOURCE = "https://github.com/openfootball/football.json"

CLS = {C.WIN: 0, C.DRAW: 1, C.LOSS: 2}


def load():
    """Return records with a season label (from the filename) and a global
    weekly time index (across all seasons)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        season = os.path.basename(f).split("_")[0]  # e.g. "2022-23"
        d = json.load(open(f))
        for m in d["matches"]:
            ft = (m.get("score") or {}).get("ft")
            if not ft or len(ft) != 2 or not m.get("date"):
                continue
            when = date.fromisoformat(m["date"])
            hs, as_ = int(ft[0]), int(ft[1])
            outcome = C.WIN if hs > as_ else (C.DRAW if hs == as_ else C.LOSS)
            recs.append(
                {
                    "season": season,
                    "when": when,
                    "home": m["team1"],
                    "away": m["team2"],
                    "outcome": outcome,
                }
            )
    recs.sort(key=lambda r: r["when"])
    origin = recs[0]["when"]
    for r in recs:
        r["day"] = (r["when"] - origin).days // BIN_DAYS
    return recs


def matches_of(recs, with_home=True) -> list[C.Match]:
    return [
        C.Match(
            day=r["day"],
            when=r["when"],
            p1=r["home"],
            p2=r["away"],
            outcome=r["outcome"],
            handicap="home" if with_home else 0,
        )
        for r in recs
    ]


def davidson_rows(whr, recs):
    rows = []
    for r in recs:
        w, d, ls = whr.win_draw_loss_probabilities(
            r["home"], r["away"], 0, handicap_key="home"
        )
        rows.append(((float(w), float(d), float(ls)), CLS[r["outcome"]]))
    return rows


def bt_constdraw_rows(whr, recs, draw_rate):
    """Draw-blind: 2-way P(home win) split with a constant draw rate."""
    rows = []
    for r in recs:
        p_home, _ = whr.probability_future_match(
            r["home"], r["away"], 0, handicap_key="home"
        )
        w = (1.0 - draw_rate) * float(p_home)
        ls = (1.0 - draw_rate) * (1.0 - float(p_home))
        rows.append(((w, draw_rate, ls), CLS[r["outcome"]]))
    return rows


def base_rate_rows(train, recs):
    n = len(train)
    pw = sum(1 for m in train if m["outcome"] == C.WIN) / n
    pd_ = sum(1 for m in train if m["outcome"] == C.DRAW) / n
    pl = sum(1 for m in train if m["outcome"] == C.LOSS) / n
    return [((pw, pd_, pl), CLS[r["outcome"]]) for r in recs]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    recs = load()
    seasons = sorted({r["season"] for r in recs})
    dr_all = sum(1 for r in recs if r["outcome"] == C.DRAW) / len(recs)
    print(
        f"loaded {len(recs)} matches, seasons {seasons[0]}..{seasons[-1]}, "
        f"{len({r['home'] for r in recs} | {r['away'] for r in recs})} teams, "
        f"draw rate {dr_all:.3f}",
        flush=True,
    )

    VALID, TEST = "2021-22", "2022-23"
    tr_v = [r for r in recs if r["season"] < VALID]
    va = [r for r in recs if r["season"] == VALID]
    tr_t = [r for r in recs if r["season"] < TEST]
    te = [r for r in recs if r["season"] == TEST]
    print(
        f"train_v={len(tr_v)} valid={len(va)} train_t={len(tr_t)} test={len(te)}",
        flush=True,
    )

    # ---- select w2 on validation (Davidson + home) ----
    best_w2, best_ll = None, float("inf")
    for w2 in W2_GRID:
        whr = C.build_and_fit(
            matches_of(tr_v),
            w2,
            with_draws=True,
            time_limit=90,
            precision=3e-3,
            verbose=True,
        )
        ll = C.three_way_log_loss(davidson_rows(whr, va))
        print(f"  w2={w2:>6}: valid 3-way log-loss = {ll:.4f}", flush=True)
        if ll < best_ll:
            best_ll, best_w2 = ll, w2
    print(f"selected w2={best_w2} (valid 3-way log-loss {best_ll:.4f})", flush=True)

    # ---- test season ----
    train_draw_rate = sum(1 for r in tr_t if r["outcome"] == C.DRAW) / len(tr_t)

    whr_dav = C.build_and_fit(
        matches_of(tr_t),
        best_w2,
        with_draws=True,
        time_limit=150,
        precision=3e-3,
        verbose=True,
    )
    # draw-blind BT: drop draws from training
    tr_t_nodraw = [r for r in tr_t if r["outcome"] != C.DRAW]
    whr_bt = C.build_and_fit(
        matches_of(tr_t_nodraw),
        best_w2,
        with_draws=False,
        time_limit=150,
        precision=3e-3,
        verbose=True,
    )

    models = {
        "base_rate": base_rate_rows(tr_t, te),
        "whr_bt_constdraw": bt_constdraw_rows(whr_bt, te, train_draw_rate),
        "whr_davidson": davidson_rows(whr_dav, te),
    }
    results = {
        "dataset": "football_big5",
        "seasons": [seasons[0], seasons[-1]],
        "w2": best_w2,
        "test_season": TEST,
        "n_test": len(te),
        "draw_rate_overall": dr_all,
        "draw_tendency_nu": whr_dav.draw_tendency,
        "home_advantage_elo": _elo(whr_dav.handicap_gamma.get("home", 1.0)),
        "models": {},
    }
    print(
        f"TEST SEASON {TEST}:  (nu={whr_dav.draw_tendency:.4f}, "
        f"home={results['home_advantage_elo']:.1f} elo)",
        flush=True,
    )
    for name, rows in models.items():
        results["models"][name] = {
            "log_loss": C.three_way_log_loss(rows),
            "accuracy": C.three_way_accuracy(rows),
            "n": len(rows),
        }
        m = results["models"][name]
        print(
            f"  {name:<20} 3way-log-loss={m['log_loss']:.4f} acc={m['accuracy']:.4f}",
            flush=True,
        )

    P.attach_provenance(
        results,
        dataset_source=DATA_SOURCE,
        dataset_files=sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))),
        package_names=[],
    )
    with open(os.path.join(RESULTS, "football_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/football_results.json", flush=True)


def _elo(gamma: float) -> float:
    return math.log(gamma) * 400.0 / math.log(10)


if __name__ == "__main__":
    main()
