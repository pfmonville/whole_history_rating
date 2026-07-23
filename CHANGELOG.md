# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - unreleased

### Changed
- **Ratings values change.** The first-day anchor now uses Coulom's
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

### Added
- Config keys `initial_prior_wins` (default 0.5) and `hessian_damping`
  (default 1.0).

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
