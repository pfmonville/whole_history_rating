"""NBA benchmark — WHR vs FiveThirtyEight Elo/RAPTOR (KickScore's dataset).

Data: FiveThirtyEight's complete NBA history (`nba_elo.csv`), the exact file the
KickScore "history of the NBA" example downloads. Each row is one game with the
home team as ``team1`` (except neutral-site games), final scores, and — usefully
— FiveThirtyEight's own pre-game win probability (`elo_prob1`) and, for recent
seasons, RAPTOR (`raptor_prob1`). Those give us strong *published* baselines to
score against WHR on identical games.

Protocol (forward validation, no leakage):
  * time_step = 14-day bins since the first game (WHR's time unit is arbitrary;
    coarsening keeps the 73-year history tractable and w2 is re-tuned to match).
  * w2 is selected on a validation season, then frozen; the reported numbers are
    on a later, unseen test season.
  * WHR is fit once on the pre-test games; test games are predicted from each
    team's latest pre-test rating (frozen holdout). FiveThirtyEight's baselines
    are *online* (updated every game), so they have a structural edge here — a
    point we make explicitly in the report.

Run:  uv run --with pandas --with matplotlib python benchmarks/nba.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BIN_DAYS = 14
# w2 is elo^2 of random-walk variance PER time-step (here per 14-day bin);
# player.py converts it to natural-rating units. A season is ~13 bins, so these
# span a per-season rating std of roughly 40-220 elo.
W2_GRID = [100.0, 300.0, 1000.0, 3000.0]
# Train the *predictor* on the modern era only (keeps fits fast; ancient
# franchises add little to a 2018-19 prediction). The history plot below still
# uses the full 1947- dataset.
TRAIN_FROM_SEASON = 1985
DATA = os.path.join(os.path.dirname(__file__), "data", "nba_elo.csv")
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def load() -> tuple[list, dict, date]:
    df = pd.read_csv(DATA).dropna(subset=["score1", "score2"])
    d0 = date.fromisoformat(df["date"].min())
    matches: list[C.Match] = []
    fte: dict[int, tuple[float, float | None]] = {}  # idx -> (elo_prob1, raptor_prob1)
    for i, r in enumerate(df.itertuples(index=False)):
        when = date.fromisoformat(r.date)
        day = (when - d0).days // BIN_DAYS
        outcome = C.WIN if r.score1 > r.score2 else C.LOSS
        # team1 is the home team unless the game is at a neutral site.
        neutral = int(getattr(r, "neutral", 0) or 0) == 1
        hcap = 0 if neutral else "home"
        m = C.Match(
            day=day, when=when, p1=r.team1, p2=r.team2, outcome=outcome, handicap=hcap
        )
        matches.append(m)
        raptor = getattr(r, "raptor_prob1", None)
        fte[i] = (
            float(r.elo_prob1) if pd.notna(r.elo_prob1) else None,
            float(raptor) if raptor is not None and pd.notna(raptor) else None,
        )
        # stash the season + row index on the match via a parallel list
    seasons = df["season"].tolist()
    return matches, {"fte": fte, "seasons": seasons}, d0


def strip_handicap(matches: list[C.Match]) -> list[C.Match]:
    return [C.Match(m.day, m.when, m.p1, m.p2, m.outcome, 0) for m in matches]


def eval_whr(train, test, w2, use_home):
    whr = C.build_and_fit(
        train if use_home else strip_handicap(train),
        w2,
        time_limit=90,
        precision=5e-3,
        verbose=True,
    )
    pairs = []
    for m in test:
        if use_home and m.handicap == "home":
            p1_win, _ = whr.probability_future_match(m.p1, m.p2, 0, handicap_key="home")
        else:
            p1_win, _ = whr.probability_future_match(m.p1, m.p2, 0)
        pairs.append((float(p1_win), 1 if m.outcome == C.WIN else 0))
    return whr, pairs


def eval_whr_online(train, test, w2):
    """Walk-forward online eval (warm-started), the fair analogue of
    FiveThirtyEight's per-game-updated Elo: fit on the training prefix, then
    step through the test season bin by bin — predict a bin's games from the
    current ratings BEFORE revealing them, then fold them in and re-iterate a
    few sweeps (warm start). No leakage: every prediction uses only earlier
    results."""
    whr = C.build_and_fit(train, w2, time_limit=90, precision=5e-3, verbose=False)
    from collections import defaultdict

    bins = defaultdict(list)
    for m in test:
        bins[m.day].append(m)
    pairs = []
    for day in sorted(bins):
        for m in bins[day]:  # predict first
            if m.handicap == "home":
                p1, _ = whr.probability_future_match(m.p1, m.p2, 0, handicap_key="home")
            else:
                p1, _ = whr.probability_future_match(m.p1, m.p2, 0)
            pairs.append((float(p1), 1 if m.outcome == C.WIN else 0))
        for m in bins[day]:  # then reveal + fold in
            whr.create_game(
                m.p1, m.p2, "B" if m.outcome == C.WIN else "W", m.day, m.handicap
            )
        whr.iterate(6)  # warm-started incremental update
    return whr, pairs


def main():
    os.makedirs(RESULTS, exist_ok=True)
    matches, meta, d0 = load()
    seasons = meta["seasons"]
    fte = meta["fte"]
    # attach season + original index for splitting / baselines
    idx_season = list(enumerate(seasons))
    print(
        f"loaded {len(matches)} games, {len(set(seasons))} seasons "
        f"({min(seasons)}-{max(seasons)})",
        flush=True,
    )

    # home win rate sanity check (confirms team1 == home)
    non_neutral = [m for m in matches if m.handicap == "home"]
    hw = sum(1 for m in non_neutral if m.outcome == C.WIN) / len(non_neutral)
    print(f"home (team1) win rate on non-neutral games: {hw:.3f}", flush=True)

    # Split by season: train <=2017, validation 2018, test 2019.
    def subset(pred):
        return [m for m, (_i, s) in zip(matches, idx_season, strict=True) if pred(s)]

    train_v = subset(lambda s: TRAIN_FROM_SEASON <= s <= 2017)
    valid = subset(lambda s: s == 2018)
    train_t = subset(lambda s: TRAIN_FROM_SEASON <= s <= 2018)
    test = subset(lambda s: s == 2019)
    print(
        f"train_v={len(train_v)} valid={len(valid)} "
        f"train_t={len(train_t)} test={len(test)}",
        flush=True,
    )

    # ---- select w2 on validation (train on <=2017), WHR+home ----
    best_w2, best_ll = None, float("inf")
    for w2 in W2_GRID:
        _, pairs = eval_whr(train_v, valid, w2, use_home=True)
        ll = C.binary_log_loss(pairs)
        print(f"  w2={w2:>6}: valid log-loss (WHR+home) = {ll:.4f}", flush=True)
        if ll < best_ll:
            best_ll, best_w2 = ll, w2
    print(f"selected w2={best_w2} (valid log-loss {best_ll:.4f})", flush=True)

    # ---- test-season metrics ----
    results = {
        "dataset": "nba",
        "bin_days": BIN_DAYS,
        "w2": best_w2,
        "n_test": len(test),
        "test_season": 2019,
        "home_win_rate": hw,
        "models": {},
    }

    # WHR plain and WHR+home, refit on <=2018
    whr_home, pairs_home = eval_whr(train_t, test, best_w2, use_home=True)
    _, pairs_plain = eval_whr(train_t, test, best_w2, use_home=False)

    # FiveThirtyEight baselines on the SAME test games
    test_indices = [i for (i, s) in idx_season if s == 2019]
    elo_pairs, raptor_pairs = [], []
    for i in test_indices:
        y = 1 if matches[i].outcome == C.WIN else 0
        ep, rp = fte[i]
        if ep is not None:
            elo_pairs.append((ep, y))
        if rp is not None:
            raptor_pairs.append((rp, y))

    def record(name, pairs):
        results["models"][name] = {
            "log_loss": C.binary_log_loss(pairs),
            "accuracy": C.binary_accuracy(pairs),
            "n": len(pairs),
        }
        print(
            f"  {name:<16} log-loss={results['models'][name]['log_loss']:.4f} "
            f"acc={results['models'][name]['accuracy']:.4f} "
            f"(n={len(pairs)})",
            flush=True,
        )

    # WHR + home, updated online through the test season (fair vs 538's online Elo)
    _, pairs_online = eval_whr_online(train_t, test, best_w2)

    print("TEST SEASON 2018-19 (season=2019):", flush=True)
    record("base_rate", [(hw, y) for (_, y) in pairs_plain])
    record("whr_plain", pairs_plain)
    record("whr_home_frozen", pairs_home)
    record("whr_home_online", pairs_online)
    record("fte_elo", elo_pairs)
    if raptor_pairs:
        record("fte_raptor", raptor_pairs)

    results["calibration_whr_home"] = C.calibration_bins(pairs_home)
    results["home_advantage_elo"] = _home_elo(whr_home)
    print(
        f"estimated home advantage: {results['home_advantage_elo']:.1f} elo", flush=True
    )

    with open(os.path.join(RESULTS, "nba_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_history(matches, best_w2)
    print("wrote results/nba_results.json and results/nba_history.png", flush=True)


def _home_elo(whr) -> float:
    """Convert the estimated 'home' handicap gamma to elo."""
    import math

    g = whr.handicap_gamma.get("home", 1.0)
    return math.log(g) * 400.0 / math.log(10)


def _plot_history(matches, w2):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping plot: {e})", flush=True)
        return
    whr = C.build_and_fit(
        strip_handicap(matches), w2, time_limit=120, precision=5e-3, verbose=True
    )
    franchises = ["BOS", "LAL", "CHI", "GSW", "SAS"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for fr in franchises:
        try:
            r = whr.ratings_for_player(fr)
        except ValueError:
            continue
        # convert bin index back to approximate year for the x-axis
        d0 = matches[0].when
        xs = [d0.year + (day * BIN_DAYS) / 365.25 for (day, _, _) in r]
        ys = [elo + 1500 for (_, elo, _) in r]  # goratings-style display shift
        ax.plot(xs, ys, label=fr, lw=1.2)
    ax.set_title("WHR team strength over NBA history (display elo = WHR + 1500)")
    ax.set_xlabel("year")
    ax.set_ylabel("rating")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "nba_history.png"), dpi=110)


if __name__ == "__main__":
    main()
