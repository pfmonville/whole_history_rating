# WHR real-data benchmarks

Empirical benchmarks of `whole-history-rating` on the real datasets used by two
well-known dynamic rating systems, to sanity-check that WHR is competitive with
the published state of the art on the same data:

| Script | Dataset | Reference system | What it exercises |
|---|---|---|---|
| `nba.py` | FiveThirtyEight NBA history (1947–2020) | [KickScore](https://github.com/lucasmaystre/kickscore) NBA example | core WHR + estimated home-court advantage (`handicap`) + online updating |
| `tennis.py` | ATP singles 2000–2015 (Jeff Sackmann) | [TrueSkill Through Time](https://github.com/glandfried/TrueSkillThroughTime) | WHR over ~1950 players / many days; skill-vs-time curves |
| `football.py` | Big-five European leagues 2014–2024 (openfootball) | KickScore (football) | the **Davidson draw model** (`win_draw_loss_probabilities`) on ~25%-draw data |

## Running

```bash
uv run --with pandas --with matplotlib python benchmarks/nba.py
uv run --with pandas --with matplotlib python benchmarks/tennis.py
uv run --with pandas python benchmarks/football.py
```

Each script downloads its data into `benchmarks/data/` (git-ignored) on first
run, fits WHR, writes a `*_results.json` and (NBA/tennis) a `*_history.png` into
`benchmarks/results/`, and prints a summary. See `REPORT.md` for the write-up.

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

## Method (common to all three)

- **Time.** `time_step` is an integer day/week/fortnight index (WHR's time unit
  is arbitrary; coarser bins keep long histories tractable and `w2` is retuned
  to match). The unit per benchmark is noted in its script.
- **`w2` selection.** `w2` (elo² of random-walk variance per time-step) is chosen
  on a *validation* season to minimise held-out log-loss, then frozen; the
  reported numbers are on a **later, unseen test season**. No test peeking.
- **Forward validation.** WHR is fit only on games before the test window. Test
  games are scored against ratings that never saw them. `nba.py` additionally
  runs an *online* walk-forward (predict a bin, then fold it in and re-iterate),
  the fair analogue of FiveThirtyEight's per-game-updated Elo.
- **Metrics.** Predictive log-loss (KickScore's primary metric) and accuracy
  (TrueSkill Through Time's), plus a calibration curve.

## Honest caveats

These are *comparable* re-runs, **not** bit-exact reproductions of the reference
papers. Train/test splits, data vintages and time discretisation differ from the
originals, so the published numbers are reference points rather than a controlled
head-to-head. Komi (a Go concept) is pinned off; home advantage is modelled with
WHR's handicap keys. Details and numbers are in `REPORT.md`.
