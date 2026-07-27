# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2026-07-25

A usability audit of the whole public surface, looking for the same class of
problem the 3.3.0 draw work fixed: config that does not apply, API asymmetries,
and degenerate cases that return a misleading value instead of saying so.

### Added
- **`HandicapBaselineWarning`** and **`one_sided_game_share()`** —
  `estimate_handicap_zero=True` frees the `handicap` key `0` baseline, which adds
  a global black-advantage parameter. That parameter is only identifiable if
  colour assignment varies independently of who is playing. When a competitor
  sits on one side of the board, the free baseline trades off against their
  strength: differences between handicap keys stay correct while the overall
  level **leaks into the ratings**. Measured on a base built so two players are
  exactly equal, the option reported them **90 elo apart** and made
  `probability_future_match` without a `handicap_key` return `0.63` instead of
  `0.50`; with colours alternating the same data is unaffected.

  The first `iterate()` now warns once when the option is on and more than half
  the games involve a player who plays ≥95% of them on one side.
  `one_sided_game_share()` exposes the statistic. It is a heuristic — a quiet run
  is not a proof of identifiability — and it is deliberately silent on a
  home-and-away league schedule, the common sports shape.
- **`UncomputedUncertaintyWarning`** — `ratings_for_player` returns the raw `-1`
  uncertainty sentinel before `iterate()` has computed anything, while
  `rating_difference` / `rating_covariance` / `rating_change` raise a `ValueError`
  in that same state. The sentinel still comes back, so an un-rated base stays
  inspectable, but it now warns once per instance so a `-1` cannot be mistaken
  for a standard deviation.

### Fixed
- **`remove_drift()` raised a bare `TypeError` on any non-integer day**,
  including whole floats like `1.0`
  (`can't multiply sequence by non-int of type 'float'`). Fractional days are
  meaningful to the model and were accepted by `create_game`, so this was a
  crash on valid input. The drift smoothing now runs on whole-day bins — the
  kernel already averages over ±`drift_kernel_radius` days, so sub-day
  resolution could never survive it anyway.
- **`load_games` rejected fractional days** that `create_game` accepted, with
  `invalid literal for int() with base 10`. It now parses them, and reports a
  non-numeric day by naming `time_step` instead.
- **`time_step` is validated at `create_game`**, where the mistake is, rather
  than as a cryptic failure from inside the maths. Non-numbers and NaN/infinity
  are rejected; a `bool` is rejected too, having silently meant day 0 or 1. An
  integral float is narrowed to `int`, so `1.0` and `1` are the *same* playing
  day rather than two.
- `load_games` error messages: a repeated separator now says so instead of
  failing with `invalid literal for int() with base 10: 'B'` (the blank field
  shifts every field after it), the field-count error states what was expected,
  and a line with surrounding whitespace is accepted rather than rejected.

### Documentation
- The README's "make ratings look like real elo" recipe shifts a *copy* and is
  safe, but it did not warn against the in-place variant. Assigning
  `day.elo += OFFSET` also leaves predictions unchanged, yet the offset is not a
  fixed point of the fit: the first-day anchor pulls it back, so a later
  `iterate()` erodes it silently (1500 → ~100 over 500 iterations). Now
  documented, with the erosion pinned by a test.
- `create_game` documents what a day value is and what is accepted;
  `ratings_for_player` documents the `-1` sentinel and its divergence from the
  raising siblings; `estimate_handicap_zero` carries the identifiability hazard
  rather than only the words "identifiability confound".

### Verified unchanged
Audited and found sound, for the record: `save_base`/`load_base` preserves `nu`,
draw declarations, handicap/komi gammas, ratings, uncertainties and predictions;
self-play is rejected on both the create and predict paths; `uncased` normalises
across `create_game`, `load_games`, `player_by_name`, predictions and orderings;
empty and single-game bases return sensible values and unknown players raise
clearly; `fit_w2` does not mutate its instance; `pinned_handicap` / `pinned_komi`
pin exactly; `drift_kernel_radius` takes effect; and the `remove_drift` day-span
guard already named the value, the threshold and the fix. The three sports
benchmarks are bit-identical.

## [3.3.0] - 2026-07-25

### Added
- **`draw_rate` config key** — declare the draw tendency as a draw *percentage*
  rather than as Davidson's `nu`, which is the unit callers actually have:
  `WHR({"draw_rate": 0.25})`. Setting both `draw_rate` and `pinned_draw` raises
  `ValueError`; they are two spellings of one decision. Two static helpers
  expose the conversion in both directions, `WHR.nu_from_draw_rate(p)` and
  `WHR.draw_rate_from_nu(nu)`, implementing `nu = 2p / (1 - p)`.

  The identity is exact **between players of equal strength**, where
  `P(draw) = nu / (2 + nu)`. Draws are likeliest between equals, so over a real
  fixture list the observed rate lands below the requested one — big-five
  football fits `nu = 0.79`, i.e. 28.3% between equals against 25.2% observed
  overall. Documented as a prior to run on until real draws are available to
  fit, not as a substitute for fitting.
- **`NoDrawsWarning`** (exported from `whr`) — `win_draw_loss_probabilities`
  warns **once per instance** when no draw was ever recorded *and* no draw
  tendency was declared. In that case `P(draw)` is exactly `0.0`, which is
  correct for a domain that cannot draw and a confident false claim for one that
  merely has not drawn yet; the library cannot tell those apart, and a `P(draw)`
  of 0 makes log-loss infinite the moment a draw occurs. The message names both
  resolutions. A `UserWarning` subclass, so broad filters still catch it while
  `simplefilter("ignore", NoDrawsWarning)` targets just this one. Warning rather
  than raising keeps existing callers working; once per instance keeps a
  season-long scoring loop quiet.
- **`draws_declared`** property — whether an intent was stated either way,
  including `pinned_draw=0.0` / `draw_rate=0.0`, which declare "no draws" and
  are answers rather than the absence of one.

### Fixed
- **`pinned_draw` was silently ignored on data containing no draws** — it was
  applied inside `create_game`'s draw branch, so it did nothing in exactly the
  situation a caller reaches for it: knowing draws are possible while having
  observed none yet. It is now resolved once in `__init__` and honoured before
  any game exists. `pinned_draw` and `draw_rate` are also now validated
  (finite, non-negative; `draw_rate` in `[0, 1)`) instead of being accepted and
  producing nonsense.
- `draw_rate` is included in the config keys preserved by `save_base`'s
  unpicklable-config fallback, so a reloaded base does not lose the declaration
  and silently start re-fitting `nu`.

## [3.2.0] - 2026-07-25

### Added
- `win_draw_loss_probabilities(..., account_for_uncertainty=False,
  uncertainty_steps=4)` — the three-outcome predictor now integrates over the
  players' rating uncertainty, matching the flag `probability_future_match`
  already had. Previously only the two-outcome path could hedge, which showed up
  when benchmarking against KickScore and TrueSkill Through Time: both fold
  their posterior variance into every prediction, and WHR's point estimates were
  measurably overconfident (equal or better accuracy, worse log-loss). The
  default `False` is byte-identical to previous releases.

  Dividing Davidson's split by `sqrt(s1 * s2)` leaves all three outcomes
  depending on the ratings only through the scalar gap `d = ln(s1) - ln(s2)`, so
  this is Coulom's `Predict` on the same grid with the same `sigma`, applied to a
  three-way split. Normalisation is therefore a property of the quadrature
  rather than an imposed step (every node contributes a triple summing to 1), and
  at `nu == 0` the integrated win/loss pair is exactly
  `probability_future_match(..., account_for_uncertainty=True)`.

  **The hedge compresses the win/loss odds; it does not move mass toward the
  draw.** Davidson's draw curve is concave near an even gap and convex in the
  tails, so integrating *drains* the draw for a close matchup and feeds it for a
  lopsided one — and a barely-favoured player's win probability can therefore
  rise. The underdog never loses probability and the odds never move away from
  even. Documented in the README and pinned by tests.

  The integrated path clamps the exponentiated half-gap, so that an extreme
  rating gap (or a modest gap with a very large sigma) cannot raise
  `OverflowError: math range error` where the point-estimate path returns
  normally — an opt-in flag should not add a failure mode. The clamp sits far
  beyond where the split has already saturated to `(1, 0, 0)` in double
  precision, so it changes no useful prediction.

  This closes the gap the head-to-head benchmark below identified, and that
  benchmark has been re-run to use it: `benchmarks/versus.py football` now sweeps
  a `predict_uncertainty` axis like the two-outcome sports, and its validation
  season selects it. The effect is small — **1.0089 → 1.0085** three-way
  log-loss, against 0.0023 on tennis and 0.0039 on the NBA — which is consistent
  with the compression behaviour described above: the hedge tightens the
  win/loss odds instead of feeding the draw, and the concave/convex split means
  close and lopsided fixtures pull in opposite directions. WHR's football lead
  over KickScore moves from 0.45% to 0.49%, so it never depended on the missing
  feature.
- `benchmarks/versus.py`: a real head-to-head that **runs** KickScore and
  TrueSkill Through Time locally instead of quoting their papers. All three
  systems share a training prefix, validation season, test season, time unit and
  metric, and every system's hyper-parameters — including the competitors'
  probability-*scale* knobs — are tuned on the validation season. Each result
  records `on_grid_edge` / `flat_axes` so a grid that constrained an optimum is
  visible rather than silent.

### Changed
- The README benchmark section now reports the head-to-head instead of comparing
  WHR against baselines and its own ablations. The honest summary: **WHR leads
  the three-outcome football benchmark and trails KickScore on the NBA (0.6%)
  and TrueSkill Through Time on tennis (1.6%)**, with FiveThirtyEight's
  domain-specific NBA probabilities ahead of all three. Earlier framing implied
  a comparison against KickScore and TTT that had not actually been run.
- The comparison figure is a dot plot with a cropped axis; as zero-baseline bars,
  a 4% quality gap rendered as six visually identical bars.

### Fixed
- Three protocol biases in the benchmark harness, two of which had understated
  WHR's competitors and one WHR itself: KickScore cold starts returned a
  hard-coded `0.5` instead of its own prior (~4.5% of tennis test matches);
  hyper-parameter grids were too narrow, capping TrueSkill Through Time's `beta`
  at 2.0 when its optimum is 32; and WHR was scored on bare point estimates
  while both competitors integrated their posterior variance, so
  `account_for_uncertainty` is now swept as a hyper-parameter.

  The benchmark also surfaced a genuine gap — `win_draw_loss_probabilities`
  could not hedge for rating uncertainty at all — which the first entry above
  now closes.

## [3.1.0] - 2026-07-24

### Changed
- **Komi is now opt-in, and this changes ratings.** Prior versions silently
  assumed a Go komi of `6.5` for *every* game and estimated its advantage,
  even for non-Go data — a category key that quietly absorbed part of the real
  skill signal (it could even distort the ranking order). Now `create_game`
  takes an explicit `komi=` argument that defaults to `None` = **no komi
  modelled**; pass a value (e.g. `komi=6.5`) to estimate that komi category as
  before. **Migration:** to reproduce the pre-3.1.0 result, pass `komi=6.5`
  (or your real komi) on every game. An `extras={"komi": …}` dict is still
  honoured for backward compatibility.

### Added
- `create_game(..., handicap, komi=None, extras=None)` — a first-class `komi`
  argument (placed before `extras`). A game with no komi contributes a neutral
  komi gamma and registers no komi key, so nothing is estimated for it.
- A "How well does it actually work?" section in the README: WHR benchmarked on
  the real datasets used by KickScore (NBA, football) and TrueSkill Through Time
  (ATP tennis), with figures, a summary table and a reproducible `benchmarks/`
  suite.

### Fixed
- **Stale worked examples in the README.** Several documented outputs still
  showed pre-3.0.0 values (e.g. `ratings_for_player` reported `-43 … 0.84` where
  the current fit gives `-50 … 0.26`), `log_likelihood` was described as
  approaching 0 from below when it is a log *density* that can be positive, and
  the `rating_change` example's "naive `115.83`" was quoted without saying it is
  `sqrt(Var(from) + Var(to))`. Every number in the README is now the real current
  output of the snippet above it, with its derivation shown, and
  `tests/test_readme_examples.py` asserts them so they cannot silently rot again.

## [3.0.1] - 2026-07-24

### Fixed
- `OverflowError: math range error` during iteration when a handicap, komi or
  draw-tendency (`nu`) Newton step became extreme (e.g. a degenerate or
  ill-conditioned advantage key on a large dataset). The scalar log-space
  advantage/`nu` updates (`value *= exp(-grad / hess)`) are now trust-region
  clamped, so a pathological key can no longer overflow the step. The cap is far
  larger than any step a well-conditioned fit takes, so normal results are
  unchanged.

## [3.0.0] - 2026-07-24

Major release. Contains a breaking behaviour change — `handicap` is now an
estimated Bradley-Terry category rather than a fixed elo constant (see the
first "Changed" entry for the one-line migration) — plus broad rating-value
changes as the model gained anti-drift, estimated handicap/komi, draws, an
uncertainty API, data-driven `w2` selection, and numpy vectorization.

### Changed
- **Ratings values change.** `handicap` is now an estimated Bradley-Terry
  category (was a fixed elo constant added to black's elo), and komi is now
  modelled (was silently ignored). Both advantages are co-estimated with
  player ratings each iteration, so games carrying a handicap or a shared komi
  value shift ratings versus earlier releases. To reproduce the old fixed-elo
  handicap behaviour, pin it: `WHR(config={'pinned_handicap': {h: elo}})`.
- The first-day anchor now uses Coulom's
  `initial_prior_wins` strength (default 0.5 instead of an implicit 1.0),
  reducing the compression of weakly-connected players toward 0 elo.
- Newton stability now comes from a configurable `hessian_damping`
  (default 1.0, Coulom's `HessianEpsilon`) instead of a fixed `-0.001` nudge.
  Single-day players now apply this damping in their 1-D Newton step as well,
  consistent with multi-day players and `covariance()`; their ratings differ
  slightly as a result.
- `auto_iterate(precision=...)` now converges on the gradient infinity-norm
  (natural-rating units) rather than the change in ratings between batches.
- `WHR.log_likelihood()` is now a correct log-posterior (game likelihood +
  first-day prior + Gaussian Wiener prior).
- Reported per-day uncertainties (from `ratings_for_player`) also change,
  since the larger default `hessian_damping` flows through `covariance()`.
- The per-game hot loops (global handicap/komi/draw-tendency accumulation,
  and the per-player per-day Bradley-Terry/Davidson term computations) are
  now numpy-vectorized for large histories. Results are unchanged up to
  floating-point reordering; no new dependency.

### Added
- Config keys `initial_prior_wins` (default 0.5) and `hessian_damping`
  (default 1.0).
- `WHR.remove_drift()`: opt-in anti-inflation step (faithful port of Coulom's
  `RemoveDrift`). Call it after convergence to cancel global rating drift over
  time so ratings are comparable across eras; it mutates ratings in place,
  returns the per-day elo corrections, and preserves same-day win
  probabilities. New config key `drift_kernel_radius` (default 100).
- Config keys `pinned_handicap` and `pinned_komi` (each a `{key: elo}` dict of
  known advantages to pin instead of estimate) and `estimate_handicap_zero`
  (default `False`). The handicap and komi advantages are co-estimated with
  player ratings each iteration and readable via `WHR.handicap_gamma` /
  `WHR.komi_gamma` (dicts of key → gamma). `save_base`/`load_base` now persist
  these estimated advantage tables.
- `WHR.fit_w2()`: selects `w2` by temporal expanding-window cross-validated
  predictive log-loss (train on past games, predict future). It is a pure
  query — it returns `{best_w2, log_loss, ...}` without mutating the
  instance; apply the chosen value yourself.
- `WHR.rating_difference(name_a, name_b, day_a=None, day_b=None)`: the elo gap
  between two players (each player's given day, else their last day), with a
  standard error and 95% CI. Uses the INDEPENDENCE APPROXIMATION
  `Var(a-b) ~= Var(a)+Var(b)`, since WHR computes no cross-player covariance.
- `WHR.rating_covariance(name)` / `WHR.rating_change(name, day_from, day_to)`:
  the exact within-player joint covariance of one player's day ratings (in
  elo²), and the elo change between two of that player's days with a standard
  error derived from the joint covariance rather than the naive sum of
  marginals — enabling trajectory confidence bands and a correct "did this
  player change significantly?" test.
- `probability_future_match(..., account_for_uncertainty=False,
  uncertainty_steps=4)`: an opt-in Coulom-style Gaussian-quadrature
  integration of the win probability over both players' rating uncertainty,
  hedging the prediction toward 0.5 when ratings are uncertain. Default
  `False` preserves the existing point-prediction behaviour exactly.
- Draws, via the Davidson model: pass `"D"` as the `winner` to `create_game`/
  `load_games`. A global draw tendency, `WHR.draw_tendency` (`nu`), is
  estimated alongside player ratings whenever draws are present, or can be
  pinned to a known value via the `pinned_draw` config key.
  `WHR.win_draw_loss_probabilities(name1, name2, ...)` gives the 3-way
  `(P(win), P(draw), P(loss))` prediction under the fitted model. Draw-free
  data is completely unaffected: `nu` stays `0.0`, which makes the Davidson
  formulas reduce exactly to the existing Bradley-Terry ones.

### Removed
- The `> 650` elo guard and the `sys.maxsize` log-likelihood guard.
  `UnstableRatingException` now fires only on a genuinely non-finite result;
  undefeated/isolated players converge to a finite rating via the prior.
- The undocumented `debug` config key, which was accepted but had no effect.

## [2.0.0] - 2026-07-22

This is a substantial rework of the packaging and public API. It contains
breaking changes; see the migration notes below.

### Added

- `WHR` is now the public class name, exported directly from the package
  (`from whr import WHR`), along with `Game`, `Player`, `PlayerDay`,
  `UnstableRatingException` and `__version__`.
- A proper build backend (hatchling), package metadata (MIT license, authors,
  classifiers, project URLs) and a `whr/__init__.py`, so `pip install` ships a
  working package.
- `save_base`/`load_base` now serialize a flat description (config, games and
  computed ratings), so saving and loading works for a history of any size and
  preserves the computed ratings on reload (#12).
- Loading of legacy pickle files written by older versions is still supported.
- Developer tooling: ruff (lint + format), mypy, pytest configuration, and a
  GitHub Actions CI matrix (Python 3.11, 3.12, 3.13).

### Changed

- **Breaking:** the `Base` class has been renamed to `WHR`. `Base` remains as a
  deprecated alias that emits a `DeprecationWarning`; it will be removed in a
  future release.
- **Breaking:** `probability_future_match` is now a pure query — it no longer
  prints to stdout, and unknown players are treated as an even (gamma = 1)
  reference without being added to the base.
- **Breaking:** `ratings_for_player` raises `ValueError` for an unknown player
  instead of silently creating it and then failing with `IndexError`.
- **Breaking:** the save/load file format changed (old files remain readable).
- **Breaking:** the minimum supported Python is now 3.11 and numpy `>= 2.0`.
- Ratings (`elo`, uncertainty) are returned as plain Python floats.

### Fixed

- `RecursionError` when saving or loading large, densely connected histories
  (#12).
- The library no longer calls `sys.exit()` on a numerical overflow; it raises
  `UnstableRatingException` instead.
- The `config` dict passed to the constructor is no longer mutated, and two
  instances no longer share the same config object.

### Removed

- The dead `Game.handicap_proc` attribute.
- The placeholder `main.py`.

## Migration from 1.x

- Replace `whole_history_rating.Base(...)` with `whole_history_rating.WHR(...)`
  (or `from whr import WHR`). `Base` still works but is deprecated.
- `probability_future_match` no longer prints; use its return value.
- Wrap `ratings_for_player` in a `try/except ValueError` if you may query an
  unknown player.
- Re-save any state you need in the new format; old files still load.
