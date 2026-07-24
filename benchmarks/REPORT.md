# WHR on real competition data — benchmark report

This report puts `whole-history-rating` (WHR) head-to-head with the real
datasets and metrics used by two well-known dynamic rating systems. (Numbers were
first produced on 3.0.0 and re-verified bit-identical on 3.1.0, which made komi
opt-in — see §4.)

- **KickScore** (Maystre et al., *Pairwise Comparisons with Flexible
  Time-Dynamics*, KDD 2019) — NBA and football.
- **TrueSkill Through Time** (Landfried et al.) — the ATP tennis history.

WHR belongs to the same family as both: a Bayesian Bradley–Terry model whose
player strengths follow a Gaussian random walk in time. So the question is not
"is WHR a different idea" but "does this implementation actually perform on the
same data as the published systems". Short answer: **yes** — within a couple of
percent of purpose-built models, and its new 3.0.0 features (estimated
handicap/home advantage, and the Davidson draw model) behave exactly as
intended on real data.

> **Reproducibility & honesty.** Every number below is produced by the scripts
> in this directory from data downloaded at runtime; see
> [`README.md`](README.md) for exact sources and commands. These are
> *comparable re-runs*, not bit-exact reproductions of the papers: train/test
> splits, data vintages and time discretisation differ, so published numbers
> are **reference points**, not a controlled head-to-head. Where a reference
> protocol is known (KickScore's NBA data/cutoff) we follow it closely.

---

## Headline

| Benchmark | Metric (held-out) | Best baseline in data | **WHR** | Verdict |
|---|---|---|---|---|
| **NBA** 2018-19 | log-loss / acc | 538 RAPTOR 0.615 / 65.6% | **0.634 / 64.3%** (online) | within ~0.02 nats of a bespoke, feature-rich NBA model |
| **Tennis (ATP)** 2014 | log-loss / acc | coin-flip 0.693 / 50% | **0.616 / 67.0%** | in the accuracy band Elo/TTT-style models reach on ATP |
| **Football** big-5 2022-23 | 3-way log-loss / acc | base rate 1.063 / 45.7% | **1.009 / 51.5%** (Davidson) | draws modelled; beats draw-blind ablation on log-loss |

All three fits converge (gradient ∞-norm ≈ 3–5·10⁻³) and recover historically
correct strength curves (see the NBA figure).

---

## 1. NBA — vs FiveThirtyEight Elo / RAPTOR (KickScore's dataset)

**Data.** FiveThirtyEight's complete NBA/BAA history — the exact `nba_elo.csv`
KickScore's NBA example uses — 69,377 games, 1947–2020, `team1` = home team.
Handily, the file carries FiveThirtyEight's *own* pre-game win probabilities
(Elo and RAPTOR), which we score as published baselines on the identical test
games.

**Setup.** 14-day time bins; home-court advantage modelled with a WHR handicap
key on the home team; `w2` chosen on 2017-18 (→ `w2 = 1000` elo²/bin) and
reported on the unseen **2018-19** season (n = 1312). "frozen" = fit at season
start; "online" = walk-forward, folding each game in as it is revealed (the fair
analogue of 538's per-game-updated Elo).

| Model | log-loss | accuracy |
|---|---|---|
| base rate (61.9% home) | 0.6783 | 59.1% |
| WHR, no home advantage | 0.6839 | 60.3% |
| WHR + home (frozen at season start) | 0.6699 | 63.6% |
| **WHR + home (online)** | **0.6343** | **64.3%** |
| FiveThirtyEight Elo (online) | 0.6192 | 65.3% |
| FiveThirtyEight RAPTOR (online) | 0.6152 | 65.6% |

**Findings.**
- WHR **estimated the home-court advantage at +98 elo** purely from results —
  bang on the well-known NBA home edge (~+3 points ≈ +100 elo). The no-home
  ablation is *worse than the base rate* on log-loss, confirming how much of NBA
  predictability is home court, and that WHR's handicap machinery captures it.
- Once evaluated **fairly online**, WHR lands at **0.634 / 64.3%**, within
  ~0.015–0.02 nats and ~1–1.3 accuracy points of 538's Elo and RAPTOR — models
  purpose-built for the NBA with margin-of-victory, rest, travel and roster
  features. A general-purpose rating system with a *single* global home constant
  getting this close is a strong result.
- The frozen→online jump (0.670 → 0.634) is itself the lesson: WHR is designed
  to be re-fit as data arrives.

**History recovered.** Fitting the full 1947–2020 history reproduces known NBA
eras — Celtics dominance in the late 50s/60s, the Bulls' 1996 peak, the Spurs'
Duncan-era plateau, and the Warriors' 2015-17 spike:

![NBA team strength over history](results/nba_history_light.png)

---

## 2. Tennis (ATP) — in the style of TrueSkill Through Time

**Data.** Jeff Sackmann's ATP match files, seasons 2000–2015 from a reachable
mirror: 48,335 main-tour singles matches, 1,948 players — the Federer / Nadal /
Djokovic / Roddick / Hewitt / Murray era. Time step = day index from
`tourney_date` (already tournament-granular). `w2` chosen on 2013 (→ `w2 = 3`
elo²/day; tennis skill is stable, so ratings should drift slowly) and reported
on the unseen **2014** season (n = 2816).

**No-leak orientation.** Training records each match winner→loser (the
Bradley–Terry datum). For the test set, matches are oriented by player id
(independent of the result) and we predict P(player-1 wins), so the outcome is
never leaked into the prediction.

| Model | log-loss | accuracy |
|---|---|---|
| coin flip (0.5) | 0.6931 | 50.0% |
| **WHR** | **0.6164** | **67.0%** |

**Findings.**
- WHR predicts **67.0%** of held-out 2014 ATP matches correctly with a log-loss
  of **0.616** — squarely in the band that Elo- and TrueSkill-Through-Time-style
  models reach on ATP data. (This is a reference-point comparison, not a matched
  reproduction: the window here is 2000–2015, not TTT's full open era.)
- The fit spans ~1,950 players over 15 years and converges cleanly
  (gradient ∞-norm ≈ 5·10⁻³), demonstrating WHR at a realistic scale.

**History recovered.** The skill-over-time curves reproduce the era's story:
Hewitt and Roddick on top in the early 2000s, Federer's mid-decade peak, Nadal's
2005 breakout, and Djokovic climbing past the field to the top by 2011–2015:

![WHR skill over time, ATP 2000-2015](results/tennis_history_light.png)

---

## 3. Football — the Davidson draw model on real league data

**Data.** `openfootball/football.json`, big-five European leagues (England,
Spain, Germany, Italy, France) 2014-15…2023-24: 18,085 matches, 202 teams,
**25.2% draws** — the natural test for WHR 3.0.0's Davidson draw model. Weekly
bins; home advantage via a handicap key; `w2` chosen on 2021-22 (→ `w2 = 30`)
and reported on the unseen **2022-23** season (n = 1826).

Three 3-way (home-win / draw / away-win) predictors:

| Model | 3-way log-loss | accuracy |
|---|---|---|
| base rate (H/D/A frequencies) | 1.0630 | 45.7% |
| WHR Bradley–Terry + constant draw rate | 1.0132 | 51.7% |
| **WHR Davidson (draws modelled) + home** | **1.0089** | 51.5% |

**Findings.**
- WHR **estimated a global draw tendency ν ≈ 0.79** and a **home advantage of
  +80 elo** — both realistic for European football (home edge ≈ 0.3–0.4 goals).
- Both WHR variants beat the base rate decisively (log-loss 1.063 → 1.01,
  accuracy 45.7% → ~52%).
- **The Davidson model beats the "assume a constant draw rate" ablation on
  log-loss** (1.0089 vs 1.0132). The gain is real but modest, and it is a
  *calibration* gain, not an accuracy one: Davidson raises draw probability for
  evenly-matched sides, which sharpens the predicted distribution, but a draw is
  rarely the single most likely outcome (≈27% < home ≈45%), so top-1 accuracy
  barely moves. This is exactly the behaviour the model is supposed to have, and
  it sits in the same ballpark as KickScore's published football log-losses
  (~1.0).

---

## 4. Cross-cutting findings

**WHR is competitive with bespoke systems.** On NBA it is within ~2% log-loss of
538's tuned models; on football it matches the published dynamic-model range;
the recovered history curves are correct. Nothing here suggests an
implementation defect — the model performs as the literature predicts.

**Two things the exercise surfaced about the library:**

1. **`w2` is in elo² per time-step and the results are very sensitive to it.**
   The optimum differs by two orders of magnitude across sports (tennis ≈ 1,
   NBA ≈ 1000 per bin) because it must match how fast real skill drifts *and* the
   chosen time unit. This is correct behaviour, but it means **`w2` should be
   tuned per dataset** — `WHR.fit_w2()` exists for exactly this. A first run with
   a too-small `w2` produced predictions no better than the base rate.

2. **Two library issues these benchmarks surfaced — both now fixed upstream.**
   Originally every `create_game` carried WHR's default Go komi key `6.5`, which
   was *estimated* unless pinned (unlike the handicap-`0` key, which is
   auto-pinned). On these sports (no komi) that free global parameter is
   degenerate: it absorbed real skill signal, and the unclamped
   `math.exp(-grad / hess)` in `_newton_handicap_komi` raised
   `OverflowError: math range error`. These benchmarks originally worked around
   it with `pinned_komi={6.5: 0.0}`. Both root causes are now fixed in the
   library — the Newton step is trust-region clamped (**3.0.1**) and komi is
   **opt-in** with no silent default (**3.1.0**) — so the workaround is gone and
   the benchmarks simply pass no komi. Re-running them across that change
   produced **bit-identical** results, as expected: a komi pinned to 0 elo and
   no komi at all are mathematically the same thing.

---

## 5. Limitations

- Not bit-exact reproductions (see the note at the top). In particular the
  tennis window is 2000–2015 (a reachable mirror), not the full open era TTT
  used, and time is discretised into bins.
- `w2` is chosen on a single validation season, not full cross-validation.
- Frozen-holdout under-sells WHR on long test windows; only NBA adds the online
  walk-forward. Tennis/football use frozen holdout (short, one-season windows).
- Multi-league football pools disconnected rating pools in one instance
  (leagues don't play each other); predictions are within-league, which is all
  the test evaluates.

## 6. Reproduce

```bash
uv run --with pandas --with matplotlib python benchmarks/nba.py
uv run --with pandas --with matplotlib python benchmarks/tennis.py
uv run --with pandas python benchmarks/football.py
```

Raw metrics are written to `results/*_results.json`; run logs to `results/*.log`.
