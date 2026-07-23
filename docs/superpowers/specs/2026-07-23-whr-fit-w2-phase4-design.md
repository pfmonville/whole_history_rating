# WHR — Phase 4: fit `w2` by temporal cross-validation (review point #7)

Date: 2026-07-23
Status: design, awaiting approval

## Context

Phase 4 of the WHR roadmap. Phases 1, cleanup, 2, 3 are merged to local
`master`. This phase adds review point #7: choose the `w2` hyperparameter (the
Wiener-prior variance rate that governs how fast ratings may change over time)
from the data instead of leaving it hard-coded at 300, by maximising
**out-of-sample predictive log-likelihood**.

Coulom's `Extract` (`~/Documents/git/WHR/src/CWHR.cpp:795-833`) partitions games
into strength-stratified slices. Per the user's decision we instead use a
**temporal** hold-out: `w2` controls temporal volatility, so the CV that tunes
it correctly must predict *future* games from *past* ones (no future leakage).

The library stays on the WHR model (#11 excluded).

## Goal

Expose `WHR.fit_w2(...)` that reports the `w2` with the best temporal
out-of-sample predictive log-loss over a set of candidates, so users can pick a
data-driven `w2` instead of the default 300.

## Design decisions (settled + proposed)

- **Temporal expanding-window k-folds** (settled with the user — the rigorous
  choice; `w2` governs temporal volatility, so training always precedes testing,
  and averaging over several cutoffs lowers the variance of the estimate).
  Games are ordered by day and split into `n_splits + 1` contiguous, roughly
  equal-sized blocks `B0 … B_n`. Fold `i` (`1 … n_splits`) trains on
  `B0 … B_{i-1}` and tests on `B_i` (sklearn `TimeSeriesSplit` style). Default
  `n_splits = 5`; `n_splits = 1` reduces to a single first-half/second-half
  temporal hold-out for users who want speed. Random k-fold is intentionally
  NOT offered — it leaks temporal information.
- **Metric:** predictive log-loss POOLED over all folds' test games,
  `-(1/N)·Σ log P(actual outcome)` where the sum runs over every scored test
  game across every fold. Lower is better; `fit_w2` returns the `w2` minimising
  it. (Pooling weights by game count and gives one stable number per candidate.)
- **Prediction of a test game:** train a fresh `WHR` (same config, candidate
  `w2`) on the train games, iterate to convergence, then for each test game
  compute `P(black wins)` from the two players' latest trained day rating
  (last day ≤ the test game's day) and the game's handicap/komi via the trained
  `handicap_gamma`/`komi_gamma`. Games whose black or white player never appears
  in train (cold-start) are **skipped** — they contribute the prior 0.5
  independent of `w2`, so they cannot affect the argmin; skipping keeps the
  metric clean. (Log this count.)
- **Search:** grid over `candidates` (a user-supplied list, else a default
  log-spaced grid `[10, 30, 100, 300, 1000, 3000]`). Evaluate each, return the
  best plus the full `{w2: log_loss}` map. (No optimiser in v1; a golden-section
  refine is a possible extension.)
- **Fixed iteration budget** per candidate fit (`iterations`, default 50) for
  reproducibility, rather than `auto_iterate` (whose stopping varies with time).
- **Pure query — no hidden mutation.** `fit_w2` does not change the current
  instance's ratings or config; it returns the result. The user chooses a `w2`
  and builds/iterates their model with it. (This avoids surprising in-place
  state changes on what reads like a diagnostic call.)

## API

```python
def fit_w2(
    self,
    candidates: list[float] | None = None,
    n_splits: int = 5,
    iterations: int = 50,
) -> dict:
    """Pick w2 by temporal expanding-window cross-validated predictive log-loss.

    For each candidate w2 and each temporal fold, trains a fresh model (this
    instance's config but the candidate w2) on the earlier games and scores
    predictive log-loss on the fold's held-out later games; the loss is pooled
    across folds. Does NOT mutate this instance. Returns:
      {"best_w2": float,
       "log_loss": {w2: pooled_log_loss, ...},
       "n_splits": int, "n_test_scored": int, "n_test_skipped": int}
    Raises ValueError if the games span fewer than `n_splits + 1` distinct days
    (no temporal split possible) or there are too few games.
    """
```

Helper (internal): `_temporal_folds(n_splits) -> list[tuple[train_games, test_games]]`
— orders the raw game descriptions by day and yields the `n_splits`
expanding-window (train, test) splits.

## Implementation notes

- A candidate fit builds a new `WHR({**self.config, "w2": candidate})`, replays
  the train games via `create_game`, and iterates `iterations` times. This
  reuses all existing machinery (anchor, damping, handicap/komi estimation,
  Wiener prior). Handicap/komi pins and other config carry over unchanged.
- Predicting a test game reuses the same Bradley-Terry math as
  `Game.white_win_probability`, evaluated at the players' last trained day and
  the game's handicap/komi keys against the trained advantage tables. Clamp the
  probability into `[eps, 1-eps]` before `log` to avoid `-inf` on a 0/1
  prediction.
- Cost is `len(candidates) × n_splits × iterations × O(train games)` —
  expensive on large histories until #10 vectorisation lands (the user has
  accepted this, since #10 is coming). Documented; tests use small data and a
  small `n_splits`.

## Compatibility

- Purely additive: a new method + no change to defaults (`w2` stays 300 unless
  the user acts on `fit_w2`'s result). Non-breaking.

## Testing plan (TDD, property-based)

1. **Recovers a planted volatility.** Generate a history from a known Wiener
   volatility (players whose true strength drifts at a known rate); `fit_w2`
   over a grid returns a `best_w2` closer to the generating scale than to the
   extremes of the grid. (Property: the argmin is interior / matches the
   planted regime, not that it hits an exact number.)
2. **A too-small w2 and a too-large w2 both score worse** than a middle one on
   the returned `log_loss` map (unimodal-ish in w2 for drifting data).
3. **Temporal folds correctness.** `_temporal_folds(n_splits)` returns
   `n_splits` (train, test) pairs where every test game's day is `>=` every
   train game's day in that fold (no future leakage), the train set grows across
   folds (expanding window), and the union of test blocks covers the later
   games. `n_splits=1` gives one first-half/second-half split.
4. **Cold-start games skipped.** A test game with a player absent from train is
   excluded from the score and counted in `n_test_skipped`.
5. **Pure query.** `fit_w2` does not change `self.config["w2"]`, the players, or
   their ratings (assert identical before/after).
6. **Degenerate inputs raise clearly.** All games on one day → `ValueError`;
   too few games → `ValueError`.
7. **Return contract.** Keys present and typed; `best_w2 ∈ candidates`;
   `log_loss` finite for every candidate.

Coverage stays at the locked 95% floor.

## Out of scope (this phase)

- k-fold / strength-stratified CV (Coulom's `Extract` scheme) — temporal chosen.
- Tuning other hyperparameters (`initial_prior_wins`, `hessian_damping`) — could
  reuse the same harness later.
- An optimiser over `w2` (grid only in v1).
- Auto-applying the chosen `w2` / re-fitting the instance (pure query).
- Later roadmap points #9, #8, #10.

## Open review points (mostly settled)

1. ~~Single hold-out vs k-folds~~ → **temporal expanding-window k-folds**,
   `n_splits` default 5 (settled with the user).
2. Default candidate grid `[10, 30, 100, 300, 1000, 3000]`.
3. Pure query (returns result, no mutation) vs an `apply=True` convenience.
4. Cold-start test games skipped (vs scored at 0.5).
