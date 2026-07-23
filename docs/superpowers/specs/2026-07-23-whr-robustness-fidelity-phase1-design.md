# WHR — Phase 1: Robustness & fidelity (points 1, 4, 5, 6)

Date: 2026-07-23
Status: design, awaiting approval

## Context

`whole_history_rating` is a Python port of Rémi Coulom's Whole-History Rating.
A review proposed 10 improvements. This spec covers **phase 1 only**: numerical
robustness and fidelity to Coulom's reference (points 1, 4, 5, 6). Points 2, 3,
7, 8, 9, 10 are deferred to later phases; point 11 (Kickscore/GP) is explicitly
out of scope — the library is WHR and stays on the WHR model.

The reference C++ implementation was read directly
(`/Users/pf.monville/Documents/git/WHR`, `src/CWHR.cpp`) and every decision
below is grounded in it.

### Preserved (do not regress)

The Python temporal prior is the **exact Gaussian Wiener prior**
`sigma2 = |Δdays| · w2` (`player.py:130`). This is *more* rigorous than
Coulom's "virtual wins" approximation (`CWHR.cpp:252,470-485`). It is kept
unchanged.

## Goal

- Remove the three ad-hoc numerical hacks (`-0.001` Hessian nudge, `if
  candidate > 650: raise`, `sys.maxsize` checks).
- Stop the excessive compression of weakly-connected players toward 0 elo.
- Guarantee a finite, stable solution (including for undefeated/isolated
  players) with a principled convergence criterion.
- Match Coulom's reference where the current code diverges from it.

New behavior is the **default** (decision taken with the user). New knobs are
exposed in `config` with sensible defaults. Ratings produced will change; this
is documented in the CHANGELOG.

## Design decisions (settled)

### #1 — Anchor: faithful to Coulom (Approach A)

Coulom anchors the **first player-day of each player** with a Bradley-Terry
prior of strength `InitialPriorWins = 0.5` (`CWHR.cpp:36,372-380`):

```
g_anchor = k · (1 − 2γ/(1+γ))
h_anchor = −2k · γ/(1+γ)²        with k = initial_prior_wins
```

The current Python does the same thing but at **k = 1** (double strength),
implemented by appending fake game terms `[1,0,1,1]` / `[0,1,1,1]` in
`playerday.py` when `is_first_day`.

Change:
- Remove the fake-term injection from `PlayerDay.won_game_terms()` /
  `lost_game_terms()`.
- Inject the anchor **directly** into the gradient/Hessian/log-likelihood on the
  first day, scaled by config `initial_prior_wins` (default **0.5**).
- Apply it at `days[0]` (the chronologically first day, since `self.days` is
  kept sorted) rather than relying on the `is_first_day` flag, which can be set
  on the wrong day when games are added out of chronological order.

Approach B (per-iteration re-centering / gauge fixing) is rejected: it is not in
Coulom's reference, and the scale-stability problem it targets is handled
separately by `RemoveDrift` (report point #2, phase 2).

### #4 — Damping: Coulom's Hessian damping, not a line search

Coulom uses no line search. Newton stability comes from a **Hessian damping**
term `HessianEpsilon = 1.0` subtracted from the diagonal (`CWHR.cpp:26,443-444`).
The Python `-0.001` is the same idea, ~1000× too weak.

Change:
- Replace the literal `- 0.001` in `Player.hessian()` with `- self.hessian_damping`,
  from config key `hessian_damping` (default **1.0**, matching Coulom; validated
  by tests). The damping does not bias the optimum: at convergence the gradient
  is 0, so the Newton step is 0 regardless of the damping — it only shrinks steps
  during iteration.
- Independently, **fix the buggy public `Player.log_likelihood()`** (it sums
  Gaussian *densities* over neighbours then takes `log`, and has `sys.maxsize`
  checks). This is a correctness bug in a public method, fixed regardless of the
  Newton mechanism. Corrected value = game log-likelihood + Wiener prior
  log-density + anchor log-prior (see below).
- Keep a **minimal safety guard**: raise `UnstableRatingException` only if a
  computed rating/gamma is non-finite (NaN/inf), never on an arbitrary elo
  threshold.

No backtracking line search (beyond WHR, unnecessary given the reference).

### #5 — Convergence on the gradient norm

`auto_iterate` currently stops on the delta of ratings between batches. Change it
to stop on **max‖gradient‖∞ < precision** (the principled stationarity
criterion), keeping the existing time/iteration guards. The `precision`
parameter now means the max-abs gradient tolerance in natural-rating units;
default tuned and locked by a test. This semantic change is noted in the
CHANGELOG.

### #6 — Guaranteed finite solution

Finiteness for undefeated/isolated players is provided by the `initial_prior_wins`
anchor (the prior pulls an unbounded rating back to a finite optimum); step
stability is provided by `hessian_damping`. With both in place:
- Remove `if candidate > 650: raise UnstableRatingException` in
  `update_by_ndim_newton`.
- Remove the `sys.maxsize` checks in `log_likelihood`.
- `UnstableRatingException` is retained only as a non-finite safety net and is
  documented as "should not occur in normal use".

## Detailed changes by file

### `whr/playerday.py`

- `won_game_terms()` / `lost_game_terms()`: drop the `is_first_day` virtual-term
  blocks. These become pure game-term lists.
- `log_likelihood()`, `log_likelihood_derivative()`,
  `log_likelihood_second_derivative()`: now reflect **games only** (anchor moved
  out).
- Add anchor helpers, non-zero only on the first day (strength read from
  `self.player.initial_prior_wins`):
  - `anchor_gradient() -> float`  = `k·(1 − 2γ/(1+γ))`
  - `anchor_hessian() -> float`   = `−2k·γ/(1+γ)²`
  - `anchor_log_likelihood() -> float` = `k·(log γ − 2·log(1+γ))`
- `update_by_1d_newtons_method()` (single-day players): include the anchor
  gradient/Hessian (a single-day player's only day is the first day).

### `whr/player.py`

- `__init__`: read `self.initial_prior_wins = config["initial_prior_wins"]` and
  `self.hessian_damping = config["hessian_damping"]`.
- `hessian()`: `diagonal[row] = day.log_likelihood_second_derivative() + prior
  - self.hessian_damping`; add `days[0].anchor_hessian()` on `row == 0`.
- `gradient()`: add `days[0].anchor_gradient()` on `idx == 0`. Drop the debug
  `print`.
- `update_by_ndim_newton()`: remove the `> 650` guard; add a non-finite check on
  the resulting ratings.
- `log_likelihood()`: rewrite correctly — per-day game LL + anchor LL on day 0 +
  Wiener prior log-density summed over consecutive day pairs
  `−(Δr)²/(2σ²) − ½·log(2π·σ²)`. Remove `sys.maxsize` checks.
- Add `gradient_infinity_norm() -> float` (max abs gradient component over this
  player's days) to support #5.

### `whr/whole_history_rating.py`

- `WHR.__init__`: `config.setdefault("initial_prior_wins", 0.5)` and
  `config.setdefault("hessian_damping", 1.0)`.
- `auto_iterate()`: after each batch compute the global max-abs gradient
  (max over players of `gradient_infinity_norm()`); converge when `< precision`.
  Keep `time_limit` and return signature `(iterations, converged)`.

## Config summary

| key | default | meaning |
|-----|---------|---------|
| `initial_prior_wins` | 0.5 | strength of the first-day BT anchor (Coulom) |
| `hessian_damping` | 1.0 | Newton Hessian damping (Coulom's HessianEpsilon) |

Existing keys (`w2`, `debug`, `uncased`) unchanged.

## Compatibility

- No public API removed or renamed; existing calls keep working.
- Ratings **values change** (less compression; different scale for
  weakly-connected players). `auto_iterate(precision=...)` changes meaning.
- Target release: **2.1.0** (minor — additive API, behavioral improvement),
  with a prominent CHANGELOG note. Open question for the user at release time:
  treat as minor (2.1.0) or major (3.0.0)?

## Testing plan (TDD, ~7 property tests)

Tests are written **first** and must fail against current code, pass after.

1. **Undefeated player is finite, no exception.** A player who wins every game
   converges to a finite elo; no `UnstableRatingException`. (Currently hits the
   `>650` raise / runs away.)
2. **Less compression.** With a spread-skill ladder (A≫B≫C≫D, consistent
   results), the top-vs-bottom elo spread is strictly larger at
   `initial_prior_wins=0.5` than at `1.0`. Asserts the anchor knob works and
   default halves compression.
3. **Monotone / stationary gradient.** After `auto_iterate`, the global
   max-abs gradient is `< precision` (the criterion actually holds).
4. **No overflow, damping respected.** A pathological dense first-day case that
   currently overflows converges without NaN/inf; a smaller `hessian_damping`
   still converges, a very large one converges more slowly (both finite).
5. **Config plumbing.** `initial_prior_wins` and `hessian_damping` are read from
   config, copied (not shared/mutated), and affect the result.
6. **Corrected log-likelihood.** `WHR.log_likelihood()` is finite, and increases
   (non-strictly) across iterations on a fixed dataset; a hand-computed
   two-day / one-anchor case matches the closed form.
7. **Faithfulness anchor value.** For a single isolated player with one win, the
   converged gamma matches the closed-form fixed point of
   `wins − γ·Σ + k·(1−2γ/(1+γ)) = 0` at `k=0.5`.

Coverage stays at the locked 95% floor.

## Out of scope (later phases)

- #2 `ComputeDrift`/`RemoveDrift` (anti-inflation) — the correct home for
  scale-stability.
- #3 handicap+komi as estimated BT parameters.
- #7 hyperparameter fitting (`w2`) by cross-validated log-loss (`Extract`).
- #8 draws; #9 joint-uncertainty API; #10 vectorization.
- #11 pluggable GP kernel / Kickscore — rejected (beyond WHR).
