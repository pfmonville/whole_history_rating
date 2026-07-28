"""Head-to-head: WHR vs KickScore vs TrueSkill Through Time, same data & metric.

The other benchmark scripts score WHR against whatever comparator the *dataset*
already contains (FiveThirtyEight's published probabilities for the NBA) or
against baselines. That leaves an obvious gap: KickScore and TTT supplied the
datasets and the metrics, but were never actually run. This script runs them.

Running the reference implementations locally — rather than quoting their papers
— is what makes the comparison valid: every system sees exactly the same
training games, the same held-out season, and is scored with the same metric, so
no difference in data vintage, train/test split or time discretisation can
explain the gap.

Fairness rules applied to all three:
  * identical training prefix, validation season and test season;
  * every system's hyper-parameters are swept on the validation season and the
    best carried to the test season — nobody is left at a default the others were
    tuned away from. Crucially the competitors get their *scale* knob swept too,
    not just their dynamics knob: WHR's rating scale is fixed by the elo
    convention so it has only ``w2`` (dynamics), whereas KickScore also has a
    prior variance and TTT a performance noise ``beta`` that set how sharp their
    probabilities are. Leaving those at defaults would have handicapped them on
    a sport whose noise level differs from the default's implied one;
  * identical time unit (integer day index) and identical frozen-rating protocol
    (fit on the training prefix, then predict the test season);
  * cold starts (a competitor never seen in the training prefix -- ~4.5% of the
    tennis test matches) are answered from each library's *own* prior, never
    from a hard-coded 0.5. KickScore needs its items declared up front for that,
    so the test-set names are registered without observations; no test result is
    ever observed, only the identity of who is playing;
  * home advantage, where the sport has one, is modelled in each library's own
    idiom (WHR: a handicap category; KickScore/TTT: an extra always-home item on
    the home team).

Run:  uv run --with pandas --with kickscore --with TrueSkillThroughTime \
          python benchmarks/versus.py [tennis|nba|football]
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import date
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


EDGE_TOL = 1e-4  # the precision every log-loss in this file is reported at


def _grid_edges(best, grid, losses=None):
    """Which of the selected hyper-parameters sit on a *binding* end of their axis.

    An optimum on a grid end means the true optimum may lie outside the grid, so
    the number reported for that system is a lower bound on its quality: widen
    the axis and re-run.

    But an end is only binding if the objective actually moves along that axis.
    KickScore's NBA ``prior_var`` scored 0.6547 at all six values, so the argmin
    landed on the smallest purely by tie-break -- flagging that would have sent
    us chasing an axis the data is indifferent to. When ``losses`` is supplied
    (mapping the frozen kwargs of each grid point to its validation loss), an
    axis whose loss spread at the optimum is within ``EDGE_TOL`` is reported as
    ``False`` and appears in the companion ``flat_axes`` set instead.
    """
    edges, flat = {}, []
    for k, v in best.items():
        vals = sorted({g[k] for g in grid})
        if len(vals) <= 1:
            continue
        if all(isinstance(x, bool) for x in vals):
            continue  # a boolean switch has no outside to extend into
        at_end = v in (vals[0], vals[-1])
        if at_end and losses is not None:
            profile = [
                loss
                for params, loss in losses
                if all(params[o] == best[o] for o in best if o != k)
            ]
            if profile and max(profile) - min(profile) <= EDGE_TOL:
                flat.append(k)
                at_end = False
        edges[k] = at_end
    return edges, flat


def _grid(**axes):
    """Cartesian product of named hyper-parameter axes -> list of kwarg dicts.

    The FIRST axis varies slowest, so consecutive grid points share it. That is
    what makes ``_cached_fit`` below effective: ``w2`` is always declared first
    and ``predict_uncertainty`` last, so the two predict_uncertainty variants of
    one ``w2`` are adjacent and the fit between them is reused.
    """
    keys = list(axes)
    combos = [{}]
    for key in keys:
        combos = [{**c, key: v} for c in combos for v in axes[key]]
    return combos


# One-entry fit cache. WHR's ``predict_uncertainty`` axis changes only how a
# fitted model is *queried* -- ``auto_iterate`` never sees it -- so half of every
# WHR grid was refitting identical models. Holding just the last fit is enough
# (see ``_grid`` on ordering) and keeps memory to a single model.
_LAST_FIT: dict[str, Any] = {}


def _cached_fit(kind: str, train: list, w2: float, build):
    """The fitted model for ``(kind, train, w2)``, refitting only when it changes."""
    key = (kind, id(train), w2)
    if _LAST_FIT.get("key") != key:
        # keep `train` referenced so its id cannot be reused by another object
        _LAST_FIT.clear()
        _LAST_FIT.update(key=key, train=train, model=build())
    return _LAST_FIT["model"]


# Tennis grids. WHR has one knob (its scale is pinned by the elo convention);
# the competitors get their scale knob swept as well as their dynamics knob.
#
# Every axis below was widened after a first pass reported ``on_grid_edge``:
# KickScore's optimum sat on the low end of BOTH its axes and TTT's on the high
# end of ``beta``, so an un-widened grid would have understated both of them.
# The rule applied throughout is direction-blind -- an axis is extended whenever
# the selected value is a grid end, whichever system it belongs to.
# ``predict_uncertainty`` switches WHR to Coulom's ``Predict``, integrating the
# point probability over the Gaussian implied by the players' rating variances.
# It belongs in the sweep because both competitors fold their posterior variance
# into every prediction (KickScore through ``probabilities``, TTT through the
# ``s1^2+s2^2+2*beta^2`` denominator) -- scoring WHR's bare point estimate
# against two uncertainty-integrated ones was the mirror image of the KickScore
# cold-start bug, this time against WHR. Swept rather than switched on, so the
# validation season decides.
TENNIS_WHR_GRID = _grid(
    w2=[0.3, 1.0, 3.0, 10.0, 30.0], predict_uncertainty=[False, True]
)
TENNIS_KS_GRID = _grid(
    wiener_var=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    prior_var=[0.05, 0.125, 0.25, 0.5, 1.0, 3.0],
)
# ``beta`` reached 16.0 -- the previous ceiling -- with ``gamma`` at its own
# ceiling too, so both axes are pushed out again.
TENNIS_TTT_GRID = _grid(
    gamma=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    beta=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
)


# --------------------------------------------------------------------------- #
# Metrics (shared with the other scripts)
# --------------------------------------------------------------------------- #
def _score(pairs):
    return {
        "log_loss": C.binary_log_loss(pairs),
        "accuracy": C.binary_accuracy(pairs),
        "n": len(pairs),
    }


# --------------------------------------------------------------------------- #
# Tennis data (same loader/orientation as tennis.py, so scores are comparable)
# --------------------------------------------------------------------------- #
def _d(yyyymmdd) -> date:
    s = str(int(yyyymmdd))
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def load_tennis():
    import pandas as pd

    files = sorted(
        glob.glob(os.path.join(RESULTS, "..", "data", "tennis", "atp_matches_*.csv"))
    )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    origin = _d(df["tourney_date"].min())
    recs = []
    for r in df.itertuples(index=False):
        when = _d(r.tourney_date)
        recs.append(
            {
                "day": (when - origin).days,
                "year": when.year,
                "winner": str(r.winner_name),
                "loser": str(r.loser_name),
                "wid": str(r.winner_id),
                "lid": str(r.loser_id),
            }
        )
    return recs


def oriented(recs):
    """Neutral orientation by player id (never by the result), so every system
    is asked the same question: P(p1 beats p2)."""
    out = []
    for r in recs:
        if r["wid"] <= r["lid"]:
            out.append((r["winner"], r["loser"], 1, r["day"]))
        else:
            out.append((r["loser"], r["winner"], 0, r["day"]))
    return out


# --------------------------------------------------------------------------- #
# The three systems, each fitted on `train` and asked to predict `test`
# --------------------------------------------------------------------------- #
def run_whr(train, test, *, w2, predict_uncertainty=False):
    def build():
        from whr import WHR

        whr = WHR({"w2": w2})
        for r in train:
            whr.create_game(r["winner"], r["loser"], "B", r["day"], 0)
        whr.auto_iterate(time_limit=400, precision=1e-3)
        return whr

    whr = _cached_fit("tennis", train, w2, build)
    pairs = []
    for p1, p2, y, _day in oriented(test):
        p, _ = whr.probability_future_match(
            p1, p2, 0, account_for_uncertainty=predict_uncertainty
        )
        pairs.append((float(p), y))
    return pairs


def _tennis_items(recs):
    return {r["winner"] for r in recs} | {r["loser"] for r in recs}


def run_kickscore(train, test, *, wiener_var, prior_var=1.0):
    import kickscore

    model = kickscore.BinaryModel()
    t0 = min(r["day"] for r in train) - 1
    kernel = kickscore.kernel.Constant(var=prior_var) + kickscore.kernel.Wiener(
        var=wiener_var, t0=t0
    )
    # Register the test-set players too, with no observations, so KickScore
    # answers cold starts from its own prior instead of a hard-coded 0.5 --
    # WHR and TTT both fall back to their priors, so anything else would
    # penalise KickScore on the ~4.5% of test matches with a debutant. Only
    # the identity of who plays is used here; no test result is observed.
    for name in _tennis_items(train) | _tennis_items(test):
        model.add_item(name, kernel=kernel)
    for r in train:
        model.observe(winners=[r["winner"]], losers=[r["loser"]], t=r["day"])
    model.fit(max_iter=100)

    last_t = max(r["day"] for r in train)
    pairs = []
    for p1, p2, y, _day in oriented(test):
        p, _ = model.probabilities(team1=[p1], team2=[p2], t=last_t)
        pairs.append((float(p), y))
    return pairs


def run_ttt(train, test, *, gamma, sigma=6.0, beta=1.0):
    import trueskillthroughtime as ttt

    composition = [[[r["winner"]], [r["loser"]]] for r in train]
    times = [r["day"] for r in train]
    history = ttt.History(
        composition=composition, times=times, sigma=sigma, beta=beta, gamma=gamma
    )
    history.convergence(epsilon=1e-3, iterations=10, verbose=False)
    curves = history.learning_curves()
    # each player's LAST posterior in the training window
    last = {name: pts[-1][1] for name, pts in curves.items()}

    prior = ttt.Gaussian(0.0, sigma)
    denom_extra = 2.0 * beta * beta
    pairs = []
    for p1, p2, y, _day in oriented(test):
        g1 = last.get(p1, prior)
        g2 = last.get(p2, prior)
        # TTT's own prediction: verified identical to
        # ttt.Game([[Player(g1)], [Player(g2)]]).evidence
        den = math.sqrt(g1.sigma**2 + g2.sigma**2 + denom_extra)
        p = ttt.cdf((g1.mu - g2.mu) / den, 0.0, 1.0)
        pairs.append((float(p), y))
    return pairs


# --------------------------------------------------------------------------- #
def sweep(name, runner, grid, train_v, valid, train_t, test):
    """Tune every hyper-parameter on the validation season, then report on the
    test season. ``grid`` is a list of kwarg dicts (see ``_grid``)."""
    best, best_ll, losses = None, float("inf"), []
    for params in grid:
        ll = C.binary_log_loss(runner(train_v, valid, **params))
        losses.append((params, ll))
        flag = ""
        if ll < best_ll:
            best_ll, best, flag = ll, params, "  <- best so far"
        print(f"    {name} {params} valid log-loss={ll:.4f}{flag}", flush=True)
    print(f"  {name}: selected {best} (valid {best_ll:.4f})", flush=True)
    scored = _score(runner(train_t, test, **best))
    scored["params"] = best
    scored["valid_log_loss"] = best_ll
    scored["grid_size"] = len(grid)
    scored["on_grid_edge"], scored["flat_axes"] = _grid_edges(best, grid, losses)
    print(
        f"  {name}: TEST log-loss={scored['log_loss']:.4f} "
        f"acc={scored['accuracy'] * 100:.1f}%  edge={scored['on_grid_edge']}"
        f" flat={scored['flat_axes']}",
        flush=True,
    )
    return scored


def main_tennis():
    recs = load_tennis()
    years = sorted({r["year"] for r in recs})
    train_v = [r for r in recs if r["year"] <= 2012]
    valid = [r for r in recs if r["year"] == 2013]
    train_t = [r for r in recs if r["year"] <= 2013]
    test = [r for r in recs if r["year"] == 2014]
    print(
        f"tennis: {len(recs)} matches {years[0]}-{years[-1]} | "
        f"train_v={len(train_v)} valid={len(valid)} "
        f"train_t={len(train_t)} test={len(test)}",
        flush=True,
    )

    out = {
        "dataset": "atp_tennis",
        "test_season": 2014,
        "n_test": len(test),
        "models": {},
    }
    out["models"]["whr"] = sweep(
        "WHR", run_whr, TENNIS_WHR_GRID, train_v, valid, train_t, test
    )
    out["models"]["ttt"] = sweep(
        "TTT", run_ttt, TENNIS_TTT_GRID, train_v, valid, train_t, test
    )
    out["models"]["kickscore"] = sweep(
        "KickScore", run_kickscore, TENNIS_KS_GRID, train_v, valid, train_t, test
    )
    out["models"]["coin_flip"] = _score(
        [(0.5, y) for _p1, _p2, y, _d in oriented(test)]
    )

    path = os.path.join(RESULTS, "versus_tennis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.relpath(path)}", flush=True)


# --------------------------------------------------------------------------- #
# NBA — 2-way with home advantage, and 538's published probabilities in-data
# --------------------------------------------------------------------------- #
NBA_BIN_DAYS = 14
NBA_TRAIN_FROM = 1985
HOME = "__HOME__"  # pseudo-item carrying the home advantage

NBA_WHR_GRID = _grid(
    w2=[100.0, 300.0, 1000.0, 3000.0], predict_uncertainty=[False, True]
)
NBA_KS_GRID = _grid(
    wiener_var=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    prior_var=[0.05, 0.125, 0.25, 0.5, 1.0, 3.0],
)
NBA_TTT_GRID = _grid(
    gamma=[0.003, 0.01, 0.03, 0.1, 0.3], beta=[0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
)


def load_nba():
    import pandas as pd

    path = os.path.join(RESULTS, "..", "data", "nba_elo.csv")
    df = pd.read_csv(path).dropna(subset=["score1", "score2"])
    origin = date.fromisoformat(df["date"].min())
    recs = []
    for r in df.itertuples(index=False):
        when = date.fromisoformat(r.date)
        home_won = r.score1 > r.score2
        neutral = int(getattr(r, "neutral", 0) or 0) == 1
        raptor = getattr(r, "raptor_prob1", None)
        recs.append(
            {
                "day": (when - origin).days // NBA_BIN_DAYS,
                "season": int(r.season),
                "home": r.team1,
                "away": r.team2,
                "home_won": bool(home_won),
                "neutral": neutral,
                "elo_prob1": float(r.elo_prob1) if pd.notna(r.elo_prob1) else None,
                "raptor_prob1": (
                    float(raptor) if raptor is not None and pd.notna(raptor) else None
                ),
            }
        )
    return recs


def nba_pairs_whr(train, test, *, w2, predict_uncertainty=False):
    def build():
        from whr import WHR

        whr = WHR({"w2": w2})
        for r in train:
            hcap = 0 if r["neutral"] else "home"
            whr.create_game(
                r["home"], r["away"], "B" if r["home_won"] else "W", r["day"], hcap
            )
        whr.auto_iterate(time_limit=180, precision=1e-3)
        return whr

    whr = _cached_fit("nba", train, w2, build)
    pairs = []
    for r in test:
        key = None if r["neutral"] else "home"
        p, _ = whr.probability_future_match(
            r["home"],
            r["away"],
            0,
            handicap_key=key,
            account_for_uncertainty=predict_uncertainty,
        )
        pairs.append((float(p), 1 if r["home_won"] else 0))
    return pairs


def _team_items(recs):
    return {r["home"] for r in recs} | {r["away"] for r in recs}


def nba_pairs_kickscore(train, test, *, wiener_var, prior_var=1.0):
    import kickscore

    model = kickscore.BinaryModel()
    t0 = min(r["day"] for r in train) - 1
    kernel = kickscore.kernel.Constant(var=prior_var) + kickscore.kernel.Wiener(
        var=wiener_var, t0=t0
    )
    # test-set teams registered without observations -- see run_kickscore
    for name in _team_items(train) | _team_items(test):
        model.add_item(name, kernel=kernel)
    # home advantage as a constant-in-time item, KickScore's own idiom
    model.add_item(HOME, kernel=kickscore.kernel.Constant(var=prior_var))
    for r in train:
        home_side = [r["home"]] if r["neutral"] else [r["home"], HOME]
        if r["home_won"]:
            model.observe(winners=home_side, losers=[r["away"]], t=r["day"])
        else:
            model.observe(winners=[r["away"]], losers=home_side, t=r["day"])
    model.fit(max_iter=100)

    last_t = max(r["day"] for r in train)
    pairs = []
    for r in test:
        home_side = [r["home"]] if r["neutral"] else [r["home"], HOME]
        p, _ = model.probabilities(team1=home_side, team2=[r["away"]], t=last_t)
        pairs.append((float(p), 1 if r["home_won"] else 0))
    return pairs


def nba_pairs_ttt(train, test, *, gamma, sigma=6.0, beta=1.0):
    import trueskillthroughtime as ttt

    composition, times = [], []
    for r in train:
        home_side = [r["home"]] if r["neutral"] else [r["home"], HOME]
        away_side = [r["away"]]
        # composition is ranked winner-first when `results` is omitted
        composition.append(
            [home_side, away_side] if r["home_won"] else [away_side, home_side]
        )
        times.append(r["day"])
    history = ttt.History(
        composition=composition, times=times, sigma=sigma, beta=beta, gamma=gamma
    )
    history.convergence(epsilon=1e-3, iterations=10, verbose=False)
    curves = history.learning_curves()
    last = {name: pts[-1][1] for name, pts in curves.items()}
    prior = ttt.Gaussian(0.0, sigma)

    pairs = []
    for r in test:
        g_home = last.get(r["home"], prior)
        g_away = last.get(r["away"], prior)
        mu = g_home.mu - g_away.mu
        var = g_home.sigma**2 + g_away.sigma**2 + 2.0 * beta * beta
        if not r["neutral"]:
            g_h = last.get(HOME, ttt.Gaussian(0.0, 0.0))
            mu += g_h.mu
            var += g_h.sigma**2
        p = ttt.cdf(mu / math.sqrt(var), 0.0, 1.0)
        pairs.append((float(p), 1 if r["home_won"] else 0))
    return pairs


def main_nba():
    recs = load_nba()
    train_v = [r for r in recs if NBA_TRAIN_FROM <= r["season"] <= 2017]
    valid = [r for r in recs if r["season"] == 2018]
    train_t = [r for r in recs if NBA_TRAIN_FROM <= r["season"] <= 2018]
    test = [r for r in recs if r["season"] == 2019]
    print(
        f"nba: {len(recs)} games | train_v={len(train_v)} valid={len(valid)} "
        f"train_t={len(train_t)} test={len(test)}",
        flush=True,
    )

    out = {
        "dataset": "nba",
        "test_season": 2019,
        "n_test": len(test),
        "bin_days": NBA_BIN_DAYS,
        "models": {},
    }
    out["models"]["whr"] = sweep(
        "WHR", nba_pairs_whr, NBA_WHR_GRID, train_v, valid, train_t, test
    )
    out["models"]["ttt"] = sweep(
        "TTT", nba_pairs_ttt, NBA_TTT_GRID, train_v, valid, train_t, test
    )
    out["models"]["kickscore"] = sweep(
        "KickScore", nba_pairs_kickscore, NBA_KS_GRID, train_v, valid, train_t, test
    )
    # 538's own published pre-game probabilities on the identical test games
    for key, label in (("elo_prob1", "fte_elo"), ("raptor_prob1", "fte_raptor")):
        pairs = [
            (r[key], 1 if r["home_won"] else 0) for r in test if r[key] is not None
        ]
        if pairs:
            out["models"][label] = _score(pairs)
            print(
                f"  {label}: TEST log-loss={out['models'][label]['log_loss']:.4f} "
                f"acc={out['models'][label]['accuracy'] * 100:.1f}%",
                flush=True,
            )
    rate = sum(1 for r in train_t if r["home_won"]) / len(train_t)
    out["models"]["base_rate"] = _score(
        [(rate, 1 if r["home_won"] else 0) for r in test]
    )

    path = os.path.join(RESULTS, "versus_nba.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.relpath(path)}", flush=True)


# --------------------------------------------------------------------------- #
# Football — 3-way (home win / draw / away win), the draw-model comparison
# --------------------------------------------------------------------------- #
FB_BIN_DAYS = 7
CLS = {"W": 0, "D": 1, "L": 2}  # from the home team's perspective

# WHR estimates its draw tendency (nu) from the data, so it needs no draw knob.
# The competitors take theirs as a hyper-parameter, so they get it swept -- and
# now so does WHR. ``win_draw_loss_probabilities`` originally had no
# ``account_for_uncertainty`` parameter, which left WHR's football number scored
# on bare point estimates while both competitors folded in their posterior
# variance; the parameter exists as of the entry above this benchmark in the
# CHANGELOG, so the axis is swept here on the same footing as the two-outcome
# sports. Worth measuring rather than assuming: the three-outcome hedge
# compresses the win/loss odds instead of shifting mass toward the draw (the
# Davidson draw curve is concave near an even gap and convex in the tails), so
# its effect on a ternary log-loss is not the obvious one.
FB_WHR_GRID = _grid(
    w2=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    predict_uncertainty=[False, True],
)
FB_KS_GRID = _grid(
    wiener_var=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], margin=[0.1, 0.3, 0.6]
)
FB_TTT_GRID = _grid(gamma=[0.01, 0.03, 0.1, 0.3], p_draw=[0.20, 0.25, 0.30, 0.35, 0.40])


def load_football():
    recs = []
    pattern = os.path.join(RESULTS, "..", "data", "football", "*.json")
    for path in sorted(glob.glob(pattern)):
        season = os.path.basename(path).split("_")[0]
        with open(path) as f:
            payload = json.load(f)
        for match in payload["matches"]:
            ft = (match.get("score") or {}).get("ft")
            if not ft or len(ft) != 2 or not match.get("date"):
                continue
            when = date.fromisoformat(match["date"])
            hs, as_ = int(ft[0]), int(ft[1])
            recs.append(
                {
                    "season": season,
                    "when": when,
                    "home": match["team1"],
                    "away": match["team2"],
                    "outcome": "W" if hs > as_ else ("D" if hs == as_ else "L"),
                }
            )
    recs.sort(key=lambda r: r["when"])
    origin = recs[0]["when"]
    for r in recs:
        r["day"] = (r["when"] - origin).days // FB_BIN_DAYS
    return recs


def _three_way_score(rows):
    return {
        "log_loss": C.three_way_log_loss(rows),
        "accuracy": C.three_way_accuracy(rows),
        "n": len(rows),
    }


def fb_rows_whr(train, test, *, w2, predict_uncertainty=False):
    def build():
        from whr import WHR

        whr = WHR({"w2": w2})
        for r in train:
            winner = {"W": "B", "D": "D", "L": "W"}[r["outcome"]]
            whr.create_game(r["home"], r["away"], winner, r["day"], "home")
        whr.auto_iterate(time_limit=300, precision=1e-3)
        return whr

    whr = _cached_fit("football", train, w2, build)
    rows = []
    for r in test:
        w, d, ls = whr.win_draw_loss_probabilities(
            r["home"],
            r["away"],
            0,
            handicap_key="home",
            account_for_uncertainty=predict_uncertainty,
        )
        rows.append(((float(w), float(d), float(ls)), CLS[r["outcome"]]))
    return rows


def fb_rows_kickscore(train, test, *, wiener_var, margin, prior_var=1.0):
    import kickscore

    model = kickscore.TernaryModel(margin=margin)
    t0 = min(r["day"] for r in train) - 1
    kernel = kickscore.kernel.Constant(var=prior_var) + kickscore.kernel.Wiener(
        var=wiener_var, t0=t0
    )
    # test-set teams registered without observations -- see run_kickscore
    for name in _team_items(train) | _team_items(test):
        model.add_item(name, kernel=kernel)
    model.add_item(HOME, kernel=kickscore.kernel.Constant(var=prior_var))
    for r in train:
        home_side = [r["home"], HOME]
        if r["outcome"] == "W":
            model.observe(winners=home_side, losers=[r["away"]], t=r["day"])
        elif r["outcome"] == "L":
            model.observe(winners=[r["away"]], losers=home_side, t=r["day"])
        else:
            model.observe(winners=home_side, losers=[r["away"]], t=r["day"], tie=True)
    model.fit(max_iter=100)

    last_t = max(r["day"] for r in train)
    rows = []
    for r in test:
        probs = model.probabilities(
            team1=[r["home"], HOME], team2=[r["away"]], t=last_t
        )
        rows.append((tuple(float(x) for x in probs), CLS[r["outcome"]]))
    return rows


def fb_rows_ttt(train, test, *, gamma, p_draw, sigma=6.0, beta=1.0):
    import trueskillthroughtime as ttt

    composition, results, times = [], [], []
    for r in train:
        home_side = [r["home"], HOME]
        away_side = [r["away"]]
        composition.append([home_side, away_side])
        # equal results encode a draw; otherwise the winner scores higher
        if r["outcome"] == "W":
            results.append([1.0, 0.0])
        elif r["outcome"] == "L":
            results.append([0.0, 1.0])
        else:
            results.append([0.0, 0.0])
        times.append(r["day"])
    history = ttt.History(
        composition=composition,
        results=results,
        times=times,
        sigma=sigma,
        beta=beta,
        gamma=gamma,
        p_draw=p_draw,
    )
    history.convergence(epsilon=1e-3, iterations=10, verbose=False)
    curves = history.learning_curves()
    last = {name: pts[-1][1] for name, pts in curves.items()}
    prior = ttt.Gaussian(0.0, sigma)
    zero = ttt.Gaussian(0.0, 0.0)

    rows = []
    for r in test:
        g_home = last.get(r["home"], prior)
        g_away = last.get(r["away"], prior)
        g_h = last.get(HOME, zero)
        delta = g_home.mu + g_h.mu - g_away.mu
        c = math.sqrt(
            g_home.sigma**2 + g_away.sigma**2 + g_h.sigma**2 + 2.0 * beta * beta
        )
        # TTT's own draw band: verified that at equal skills this reproduces
        # p_draw exactly, and that the three probabilities sum to 1.
        m = ttt.compute_margin(p_draw, c)
        hi = ttt.cdf((m - delta) / c, 0.0, 1.0)
        lo = ttt.cdf((-m - delta) / c, 0.0, 1.0)
        rows.append(((1.0 - hi, hi - lo, lo), CLS[r["outcome"]]))
    return rows


def sweep3(name, runner, grid, train_v, valid, train_t, test):
    best, best_ll, losses = None, float("inf"), []
    for params in grid:
        ll = C.three_way_log_loss(runner(train_v, valid, **params))
        losses.append((params, ll))
        flag = ""
        if ll < best_ll:
            best_ll, best, flag = ll, params, "  <- best so far"
        print(f"    {name} {params} valid 3way-log-loss={ll:.4f}{flag}", flush=True)
    print(f"  {name}: selected {best} (valid {best_ll:.4f})", flush=True)
    scored = _three_way_score(runner(train_t, test, **best))
    scored["params"] = best
    scored["valid_log_loss"] = best_ll
    scored["grid_size"] = len(grid)
    scored["on_grid_edge"], scored["flat_axes"] = _grid_edges(best, grid, losses)
    print(
        f"  {name}: TEST 3way-log-loss={scored['log_loss']:.4f} "
        f"acc={scored['accuracy'] * 100:.1f}%  edge={scored['on_grid_edge']}"
        f" flat={scored['flat_axes']}",
        flush=True,
    )
    return scored


def main_football():
    recs = load_football()
    valid_season, test_season = "2021-22", "2022-23"
    train_v = [r for r in recs if r["season"] < valid_season]
    valid = [r for r in recs if r["season"] == valid_season]
    train_t = [r for r in recs if r["season"] < test_season]
    test = [r for r in recs if r["season"] == test_season]
    draw_rate = sum(1 for r in recs if r["outcome"] == "D") / len(recs)
    print(
        f"football: {len(recs)} matches, draw rate {draw_rate:.3f} | "
        f"train_v={len(train_v)} valid={len(valid)} "
        f"train_t={len(train_t)} test={len(test)}",
        flush=True,
    )

    out = {
        "dataset": "football_big5",
        "test_season": test_season,
        "n_test": len(test),
        "draw_rate_overall": draw_rate,
        "models": {},
    }
    out["models"]["whr"] = sweep3(
        "WHR", fb_rows_whr, FB_WHR_GRID, train_v, valid, train_t, test
    )
    out["models"]["ttt"] = sweep3(
        "TTT", fb_rows_ttt, FB_TTT_GRID, train_v, valid, train_t, test
    )
    out["models"]["kickscore"] = sweep3(
        "KickScore", fb_rows_kickscore, FB_KS_GRID, train_v, valid, train_t, test
    )
    n = len(train_t)
    base = (
        sum(1 for r in train_t if r["outcome"] == "W") / n,
        sum(1 for r in train_t if r["outcome"] == "D") / n,
        sum(1 for r in train_t if r["outcome"] == "L") / n,
    )
    out["models"]["base_rate"] = _three_way_score(
        [(base, CLS[r["outcome"]]) for r in test]
    )

    path = os.path.join(RESULTS, "versus_football.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.relpath(path)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tennis"
    if which == "tennis":
        main_tennis()
    elif which == "nba":
        main_nba()
    elif which == "football":
        main_football()
    else:
        raise SystemExit(f"unknown dataset {which!r} (tennis, nba, football)")
