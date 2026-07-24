# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - unreleased

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
