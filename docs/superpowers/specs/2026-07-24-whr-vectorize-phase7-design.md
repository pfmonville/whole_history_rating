# WHR — Phase 7: vectorization (review point #10)

Date: 2026-07-24
Status: design, awaiting approval

## Context

Final phase of the WHR roadmap. Phases 1, cleanup, 2, 3, 4, 5, 6 are merged to
local `master`. This phase adds review point #10: **vectorize the per-game hot
loops with numpy** so the algorithm scales to large histories (~170k games),
without changing results. Chosen with the user: **numpy** (no numba/new
dependency), **algorithm-preserving** (results identical up to floating-point
reordering), with the small number of exact-equality tests loosened to
`pytest.approx`.

## Non-goals / invariants

- **Do NOT change the algorithm.** Keep the per-player Gauss-Seidel loop
  (`for player in self.players.values(): player.run_one_newton_iteration()`) —
  each player still sees earlier players' same-iteration updates. Do NOT batch
  the Newton update ACROSS players (that would be Jacobi → a different
  convergence trajectory → materially different results, not just float noise).
- Keep the tridiagonal (Thomas) solve in `update_by_ndim_newton`/`covariance`
  as-is (already `O(n)`).
- Results must match the pre-vectorization values to within ~`1e-9` (only
  float summation order changes). A regression test asserts this.

## Targets (hot per-game loops → numpy)

### 1. Global all-games loops (largest win, lowest risk)

- `WHR._accumulate_handicap_komi` (`whole_history_rating.py:67-`): currently a
  Python `for g in self.games` accumulating per-key gradients/Hessians/counts.
  Vectorize: build numpy arrays over games (black/white effective gammas,
  handicap keys, komi keys, results), compute the per-game terms with array ops,
  and accumulate per key with `numpy.add.at` (or `bincount` on integer-mapped
  keys). Draws still skipped.
- `WHR._nu_gradient_hessian` (`whole_history_rating.py:1053-`): same shape —
  vectorize the `T/Z` accumulation over all games.

### 2. Per-player per-day game terms

- `PlayerDay.log_likelihood_derivative`/`log_likelihood_second_derivative`
  (BT path) reduce, for a day, to: `derivative = n_wins − gamma·Σ 1/(gamma+d_i)`,
  `second = −gamma·Σ d_i/(gamma+d_i)²`, where `d_i` are the day's opponents'
  adjusted gammas (won ∪ lost). Represent `d_i` as a numpy array (rebuilt each
  iteration, like the current per-day term cache) and compute the sums with
  numpy. `log_likelihood` similarly.
- `PlayerDay.davidson_derivatives(nu)` (draw path): arrays of own/opp effective
  gammas `S,O` and weights `w` over the day's games; `T=nu·sqrt(S·O)`,
  `Z=S+O+T`, `N=S+T/2`, `N'=S+T/4`; `gradient=Σ(w−N/Z)`, `hessian=Σ((N/Z)²−N'/Z)`
  vectorized. `davidson_log_likelihood` likewise.

The per-day arrays are rebuilt each iteration from the current opponent gammas
(preserving Gauss-Seidel); the vectorization is purely how each day's per-game
arithmetic is summed. Small per-day arrays give a modest win; the global loops
(over all games) give the large win at scale.

## Floating-point / test policy

Vectorized summation reorders float additions, so values differ from the
Python-loop path by ~`1e-12`. Consequences and handling:

- Numeric exact-equality assertions on computed ratings / log-likelihood /
  probabilities (e.g. `test_output`, `test_output2`,
  `test_loading_several_games_at_once`) → change `==`/exact to
  `pytest.approx(..., rel=1e-9, abs=1e-9)`. The VALUES are unchanged; only the
  comparison tolerance is added.
- `print_ordered_ratings` display-string assertions print full-precision elos;
  a `1e-12` change alters trailing digits → the exact string compare breaks.
  Change those tests to PARSE the printed numbers and compare with
  `pytest.approx`, rather than string-equality. The public print format itself
  is unchanged (no code change to the print methods).
- `fit_w2`'s recovery margins (~`4e-3`) are far above float-reorder noise
  (~`1e-12`), so strict algorithm-preservation keeps that test stable — but
  re-confirm it after the change (the earlier watch-item).

## Testing plan (TDD)

1. **Equivalence regression (the core guard).** Before touching the loops,
   capture the ratings / `log_likelihood` / a prediction for a couple of fixed
   scenarios (draw-free, handicap, komi, and a draw scenario). After
   vectorizing, assert the vectorized results match those captured values within
   `rel=1e-9` (or compute both paths and compare). This is the primary proof
   that vectorization is behaviour-preserving.
2. **Existing suite green with loosened tolerances** — the exact-equality
   golden tests updated to `approx`/parse-and-approx as above; everything else
   unchanged (draw-free, handicap, komi, uncertainty, fit_w2 all still pass).
3. **`_accumulate_handicap_komi` / `_nu_gradient_hessian` unit-equivalence:**
   for a fixed base, the vectorized accumulation returns the same per-key
   gradients/Hessians/counts (within `1e-9`) as a reference Python-loop
   computation kept in the test.
4. **Benchmark smoke test:** a test that builds a moderately large history
   (e.g. a few thousand games) and runs `iterate` — asserting it completes (and
   optionally logging a timing). NOT a hard wall-clock assertion (machine
   dependent); just guards that the vectorized path runs at scale without error.
   A manual before/after timing on ~50k–170k games is reported in the PR, not
   asserted in CI.

Coverage stays at the locked 95% floor.

## Compatibility

- No public API change; no behaviour change beyond float-level reordering.
- Version-wise this is not breaking; it rides in the same release as the rest.

## Out of scope

- numba / JIT (rejected — dependency weight).
- Cross-player (Jacobi) batching (rejected — changes results).
- Restructuring the tridiagonal solve or the Gauss-Seidel scheme.
- Any further roadmap points (this is the last; #11 excluded).

## Open review points

1. numpy, algorithm-preserving, exact tests → `approx` (settled with the user).
2. Vectorize both the global all-games loops AND the per-player per-day terms
   (vs global-only). Global-only is simpler/lower-risk but leaves the
   per-player path scalar; doing both is the real scale win. Confirm doing both.
3. Tolerance `rel=1e-9` for the loosened assertions; parse-and-approx for the
   print-string tests.
4. Benchmark as a smoke test only (no hard wall-clock assertion in CI).
