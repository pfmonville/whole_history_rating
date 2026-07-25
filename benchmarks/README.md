# WHR real-data benchmarks

Empirical benchmarks of `whole-history-rating` on real competition data, in two
layers:

**`versus.py` — the head-to-head.** Fits WHR,
[KickScore](https://github.com/lucasmaystre/kickscore) and
[TrueSkill Through Time](https://github.com/glandfried/TrueSkillThroughTime)
*locally*, on the same training prefix, tuned on the same validation season,
scored on the same test season with the same metric. This is the file to read if
the question is "how does WHR compare".

**`nba.py` / `tennis.py` / `football.py` — the WHR-only deep dives.** These add
what a head-to-head cannot: online walk-forward updating, calibration curves,
ablations, and the rating-history figures.

| Script | Dataset | What it exercises |
|---|---|---|
| `versus.py` | all three, one per invocation | WHR vs KickScore vs TTT under one protocol |
| `nba.py` | FiveThirtyEight NBA history (1947–2020) | core WHR + estimated home-court advantage (`handicap`) + online updating |
| `tennis.py` | ATP singles 2000–2015 (Jeff Sackmann) | WHR over ~1950 players / many days; skill-vs-time curves |
| `football.py` | Big-five European leagues 2014–2024 (openfootball) | the **Davidson draw model** (`win_draw_loss_probabilities`) on ~25%-draw data |

## Running

```bash
uv run --with pandas --with kickscore --with TrueSkillThroughTime python benchmarks/versus.py tennis
```

Swap `tennis` for `nba` or `football`. Then the WHR-only runs:

```bash
uv run --with pandas --with matplotlib python benchmarks/nba.py
uv run --with pandas --with matplotlib python benchmarks/tennis.py
uv run --with pandas python benchmarks/football.py
```

Each script downloads its data into `benchmarks/data/` (git-ignored) on first
run, fits WHR, writes a `*_results.json` (and, for NBA/tennis, a `*_curves.json`
of per-day rating curves) into `benchmarks/results/`, and prints a summary.

Figures are rendered separately, so the expensive fits are not repeated when a
chart's design changes:

```bash
uv run --with matplotlib python benchmarks/make_figures.py
```

That writes each README figure twice — `*_light.png` and `*_dark.png`, stepped
from the same palette — which the README serves via
`<picture media="(prefers-color-scheme: dark)">`. See `REPORT.md` for the
write-up.

## Data sources (all fetched at runtime)

- **NBA** — `nba_elo.csv`, FiveThirtyEight's complete NBA/BAA history, the exact
  file KickScore's NBA notebook uses (S3 mirror
  `lum-public.s3.eu-west-1.amazonaws.com/nba_elo.csv`). Carries FiveThirtyEight's
  own Elo/RAPTOR pre-game probabilities, used here as published baselines.
- **Tennis** — Jeff Sackmann's `atp_matches_YYYY.csv` (2000–2015 available from a
  reachable GitHub mirror). Standard columns: `tourney_date`, `winner_name`,
  `loser_name`, `score`, `round`.
- **Football** — `openfootball/football.json`, big-five leagues 2014-15…2023-24
  (`team1` = home, `score.ft` = `[home, away]`).

## Method (common to all)

- **Time.** `time_step` is an integer day/week/fortnight index (WHR's time unit
  is arbitrary; coarser bins keep long histories tractable and `w2` is retuned
  to match). The unit per benchmark is noted in its script.
- **Hyper-parameter selection.** Chosen on a *validation* season to minimise
  held-out log-loss, then frozen; reported numbers are on a **later, unseen test
  season**. No test peeking. In `versus.py` this applies to every system, and
  crucially to the competitors' probability-*scale* knobs (KickScore's
  `prior_var`, TTT's `beta`) as well as their dynamics knobs — leaving those at
  defaults understated both of them badly on the first pass.
- **Grid honesty.** Each result records `on_grid_edge` and `flat_axes`. An
  optimum on a grid end means the true optimum may lie outside it, so grids were
  widened and re-run until no flag remained. An axis along which the loss does
  not move at all is reported as flat rather than chased.
- **Forward validation.** Fit only on games before the test window; test games
  are scored against ratings that never saw them. `nba.py` additionally runs an
  *online* walk-forward (predict a bin, then fold it in and re-iterate), the fair
  analogue of FiveThirtyEight's per-game-updated Elo. `versus.py` is frozen-only,
  for comparability.
- **Metrics.** Predictive log-loss (KickScore's primary metric) and accuracy
  (TrueSkill Through Time's), plus a calibration curve.

## Honest caveats

The head-to-head runs the reference implementations locally, so differences in
data vintage or train/test split cannot explain its gaps — but it is **not** a
reproduction of the reference papers' published numbers, which use different
splits and preprocessing. Three protocol decisions are judgement calls: cold
starts are answered from each library's own prior (not a hard-coded 0.5), home
advantage is expressed in each library's own idiom, and convergence budgets are
per-library rather than matched on wall-clock. Komi (a Go concept) is simply not
passed — it is opt-in as of WHR 3.1.0. Test sets are one season each, so
differences of a few thousandths of a nat should be read as "close", not
"settled". Full detail in `REPORT.md` §5.
