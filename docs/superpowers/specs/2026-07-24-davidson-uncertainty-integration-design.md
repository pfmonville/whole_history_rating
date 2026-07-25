# WHR — uncertainty-integrated three-outcome prediction

Date: 2026-07-24
Status: design, approved

## Context

Phase 5 gave `probability_future_match` an opt-in `account_for_uncertainty`
flag implementing Coulom's `Predict` (`CWHR.cpp:609-691`): integrate the win
probability over the Gaussian implied by the two players' rating variances, so
predictions hedge when the ratings are poorly known.

`win_draw_loss_probabilities` — the Davidson three-outcome predictor added in
phase 6 — never got the equivalent. Its signature is `(name1, name2,
handicap=0, handicap_key=None, komi_key=None)`, so three-outcome predictions
are always point estimates.

This gap showed up while benchmarking against KickScore and TrueSkill Through
Time (`benchmarks/versus.py`). Both competitors fold their posterior variance
into every prediction. On the two-outcome sports WHR's point-estimate path was
measurably overconfident — equal or better accuracy but worse log-loss — and
that is fixable with `account_for_uncertainty=True`. The three-outcome path has
no such escape hatch.

## What ships

`account_for_uncertainty: bool = False` and `uncertainty_steps: int = 4` on
`win_draw_loss_probabilities`, matching `probability_future_match`'s spelling,
defaults, and error behaviour. Default `False` keeps existing behaviour
byte-identical.

## The integration variable

Divide the Davidson split through by `sqrt(s1*s2)`. Writing
`d = ln(s1) - ln(s2)` for the effective rating gap in natural units (both
players' advantages already folded into `s1`, `s2`):

```
s1 / sqrt(s1*s2) = e^(d/2)
s2 / sqrt(s1*s2) = e^(-d/2)
t  / sqrt(s1*s2) = nu
```

so with `Z(d) = e^(d/2) + e^(-d/2) + nu`:

```
P(win)  = e^(d/2)  / Z(d)
P(draw) =    nu    / Z(d)
P(loss) = e^(-d/2) / Z(d)
```

All three outcomes are a function of the **single scalar** `d`. That is what
makes this a clean extension rather than a new derivation: it is Coulom's
`Predict` with the same `sigma`, the same grid, and the same weights, applied
to a three-way split instead of a logistic.

Two consequences worth stating, because they are the reason to prefer this
formulation:

- **Normalisation is automatic, not enforced.** Each quadrature node
  contributes a triple summing to 1, so the weight-normalised average of those
  triples sums to 1 identically. There is no renormalisation step, and hence no
  place for one to introduce bias. Verified exactly (to 1e-12) across a
  3900-point `(d, nu, sigma)` grid.
- **At `nu == 0` it collapses onto the existing method.** `d` equals
  `probability_future_match`'s `delta_r`, so the integrated `(win, loss)` pair
  is *identical* to `probability_future_match(..., account_for_uncertainty=True)`.
  This is a free regression anchor tying the two paths together.

## Algorithm

Mirrors `probability_future_match` step for step:

1. Compute the point-estimate `s1`, `s2`, `t`, `z` exactly as today, and return
   `(s1/z, t/z, s2/z)` unchanged when `account_for_uncertainty` is `False`.
2. Raise `ValueError("uncertainty_steps must be >= 1")` when
   `uncertainty_steps < 1`.
3. `sigma = sqrt(var1 + var2)` from each player's last-day `uncertainty`,
   clamped at 0 (the field is `-1` before `iterate()` runs) and treating
   unknown/dayless players as variance 0.
4. Short-circuit to the point estimate when `sigma == 0`.
5. `d_hat = ln(s1) - ln(s2)`; accumulate `w_i = exp(-x_i^2/2)` weighted triples
   at `x_i = 0.5*i` for `i` in `-steps..steps`; divide each by `sum(w_i)`.

Shared with `probability_future_match` via two private helpers, so the two
methods cannot drift apart on the parts that must agree:

- `_prediction_sigma(player1, player2) -> float` — the clamped
  `sqrt(var1 + var2)`.
- `_gaussian_quadrature(uncertainty_steps) -> Iterator[tuple[float, float]]` —
  the `(x_i, w_i)` nodes.

`probability_future_match`'s arithmetic is unchanged by the extraction; its
results stay byte-identical.

### Overflow clamp (found during implementation)

Step 5 exponentiates the half-gap `h = d/2`. `math.exp` overflows just above
`709.78`, and `e^h + e^-h + nu` needs headroom on top, so a large enough gap
raised `OverflowError: math range error` on a matchup the *point* path returned
fine — the same failure class as the 3.0.1 advantage-step fix. Reproduced with
`s1 = exp(709)`, `s2 = 5e-324` (a ~1453-nat gap): point returned
`(1.0, 6.35e-316, 0.0)`, integrated raised.

An opt-in flag must not add a failure mode, so `h` is clamped to
`±_MAX_HALF_GAP = 700`. At that magnitude the split is already `(1, 0, 0)` to
full double precision (`h = 700` is ~486,000 elo), so the clamp changes no
prediction that carries information, and it leaves the arithmetic bit-identical
throughout the useful range — including the exact `nu == 0` agreement above.
A clamp rather than a log-sum-exp rewrite specifically to preserve that exact
agreement: the stable rewrite computes `e^d/(1+e^d)` instead of
`1/(1+e^-d)` for negative gaps, which is mathematically equal but not
bit-identical, and would have downgraded the cross-path test to approximate.

## Hedging direction — what is actually true

The intuition "uncertainty moves mass toward the neutral split" is *not*
correct here, and the tests must not assert it. Sweeping `d ∈ [0,4]`,
`nu ∈ [0,5]`, `sigma ∈ [0.05,3]`:

**Always true (zero violations):**

- `P(underdog wins)` never decreases.
- The win/loss odds ratio `P(win)/P(loss)` never moves away from 1 — it
  strictly compresses whenever `sigma > 0` and `d != 0`.

**Not monotone:**

- `P(draw)` *falls* for near-even matchups and *rises* for lopsided ones.
  Davidson's draw curve `nu/Z(d)` is concave near `d = 0` and convex in the
  tails, so spreading `d` pushes draw mass out near even and pulls it in when
  lopsided. At `d=0, nu=1, sigma=0.8`: `0.3333 -> 0.3193`. At
  `d=3, nu=1, sigma=0.8`: `0.1753 -> 0.1783`.
- `P(favourite wins)` *rises* slightly for a marginal favourite, because draw
  mass leaking outward splits to both sides. At `d=0.2, nu=1, sigma=0.8`:
  `0.3672 -> 0.3721`. It falls, as expected, once the favourite is clear
  (`d ≳ 0.5`).

So the mechanism that fixes log-loss is **odds compression**, not motion toward
`(1/3, 1/3, 1/3)`. Documented in the docstring in those terms.

## Tests

In `tests/test_draws.py`, alongside the existing `win_draw_loss` coverage.

Behaviour preservation:

- Default and explicit `account_for_uncertainty=False` return values identical
  to today's, on a fitted base with draws.
- `uncertainty_steps < 1` raises `ValueError` when the flag is on; ignored when off.
- `sigma == 0` (unknown players, never iterated) returns the point estimate.

Normalisation:

- The three probabilities sum to 1 and stay non-negative, with the flag on,
  across several `uncertainty_steps` values.

Hedging direction — the robust invariants:

- Clear favourite: `P(win)` falls, `P(draw) + P(loss)` rises.
- `P(underdog wins)` increases.
- The win/loss odds ratio compresses toward 1.

Hedging direction — the counterintuitive cases, pinned so a future change has
to notice it broke them:

- Near-even matchup: `P(draw)` *falls*.
- Marginal favourite: `P(win)` *rises*.

Cross-path equivalence:

- At `nu == 0`, the integrated win probability equals
  `probability_future_match(..., account_for_uncertainty=True)`'s bit-for-bit.
  The loss probability agrees only to float precision, because that method
  returns the forced complement `1.0 - p1` while this one integrates the loss
  independently — the asymmetry is the point, so the test asserts exactness on
  the win and `abs=1e-15` on the loss rather than hiding it.

Robustness (`tests/test_robustness.py`):

- An extreme rating gap, and a plausible gap with an implausible sigma, both
  return a finite normalised triple instead of raising `OverflowError`.

Coverage stays at or above the `--cov-fail-under=95` gate.

## Verification

Beyond the suite, backward compatibility was checked directly rather than
inferred: both predictors were dumped as exact float hex over a 4332-case grid
(3 seeds x draws/no-draws x keys/no-keys x 6 name pairs including unknown
players x 3 handicaps x 4 key combinations x flag on/off x 3 step counts) under
this branch and under `HEAD`, and the outputs are **bit-identical**. Every line
of the new code and helpers is covered.
