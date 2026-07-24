# WHR — Phase 5: uncertainty API (review point #9)

Date: 2026-07-24
Status: design, awaiting approval

## Context

Phase 5 of the WHR roadmap. Phases 1, cleanup, 2, 3, 4 (+ their cleanup) are
merged to local `master`. This phase adds review point #9: expose the model's
uncertainty in the forms that are actually meaningful, chosen with the user as
**A + B + C** (D — true cross-player covariance — is out of scope: it abandons
WHR's block-coordinate method).

Today only a per-day **marginal variance** is exposed (`uncertainty` in
`ratings_for_player`), in natural (`r`) units.

## What ships

- **A — rating-difference CI between two players.** The comparable quantity
  between players, with the independence approximation Coulom uses
  (`Var(r_A − r_B) ≈ Var(r_A) + Var(r_B)`, `CWHR.cpp:650`). Cross-player
  covariance is not computed by WHR, so this is an approximation — documented.
- **B — uncertainty-integrated win prediction.** Coulom's `Predict`
  (`CWHR.cpp:609-691`): integrate the win probability over the Gaussian of the
  rating difference, so predictions hedge toward 0.5 when the ratings are
  uncertain. Opt-in (default off), so existing prediction behaviour is unchanged.
- **C — within-player joint covariance.** Exact (already implicit in the
  tridiagonal Hessian): the full covariance among ONE player's days, enabling
  confidence bands on a trajectory and a correct "did this player change
  significantly?" test. This is the pure, exact centrepiece.

## Units

Internal ratings are natural `r = ln(gamma)`; the Hessian and its inverse are in
`r²`. Elo `= r · 400/ln(10)`, so a variance converts to elo² by
`(400/ln 10)²`. **All new public methods report ELO** (differences, standard
errors, covariance entries) — more intuitive than raw `r`. The existing
`uncertainty` field (raw `r`-variance) is left unchanged for back-compat.

Let `K = 400 / ln(10)` (the elo-per-natural-unit factor).

## A — `rating_difference`

```python
def rating_difference(
    self, name_a: str, name_b: str, day_a: int | None = None, day_b: int | None = None
) -> dict:
    """Elo difference (a − b) and its uncertainty, comparing two players.

    Uses each player's day rating (the given day, else their last day) and the
    INDEPENDENCE APPROXIMATION for the difference variance
    (Var(a−b) ≈ Var(a)+Var(b)); WHR does not compute cross-player covariance, so
    two players who played each other a lot have correlated errors this ignores.
    Returns {"difference": elo, "std_error": elo, "confidence_interval_95":
    (lo, hi)}. Raises ValueError for an unknown/unrated player.
    """
```

- `elo_diff = day_a.elo − day_b.elo`.
- `var_r = var_r_a + var_r_b` (each day's `uncertainty`, the `r`-variance);
  `se_elo = sqrt(var_r) · K`; `ci = (elo_diff − 1.96·se_elo, elo_diff + 1.96·se_elo)`.
- Requires uncertainties computed (they are, after `iterate`/`auto_iterate`).

## B — `probability_future_match(..., account_for_uncertainty=False)`

Add a parameter to the existing method (which already supports
`handicap`/`handicap_key`/`komi_key` from the cleanup):

```python
def probability_future_match(
    self, name1, name2, handicap=0, handicap_key=None, komi_key=None,
    account_for_uncertainty: bool = False, uncertainty_steps: int = 4,
) -> tuple[float, float]:
```

- Default `False` → current point-prediction behaviour, unchanged (non-breaking).
- `True` → compute the mean log-gamma difference `ΔR` (as today, incl.
  advantages) and `Σ = sqrt(var_r_1 + var_r_2)` (independence approx, natural
  units), then integrate the logistic over the Gaussian:
  `P = Σ_i w_i · σ(ΔR + Σ·x_i)` with symmetric quadrature nodes
  `x_i = step·i`, `i ∈ [−steps, steps]`, weights `∝ exp(−x_i²/2)` (Coulom's
  scheme, `CWHR.cpp:664-675`, `step = 0.5`). Returns the pair `(p1, p2)`
  summing to 1, as today. Unknown/unrated players contribute `Σ = 0` for their
  side (falls back to the point value).

## C — `rating_covariance` and `rating_change`

```python
def rating_covariance(self, name: str) -> tuple[list[int], "np.ndarray"]:
    """Full within-player covariance of a player's day ratings, in elo².

    Returns (days, matrix) where matrix[i][j] = Cov(elo_day_i, elo_day_j),
    the exact inverse of the player's negative tridiagonal Hessian scaled to
    elo². Diagonal matches the per-day marginal variance. Raises ValueError for
    an unknown/unrated player.
    """

def rating_change(self, name: str, day_from: int, day_to: int) -> dict:
    """Elo change of one player between two of their days, with uncertainty.

    Var(change) = C[to,to] + C[from,from] − 2·C[from,to] using the WITHIN-player
    covariance (exact — consecutive days are positively correlated via the
    Wiener prior, so a change is more certain than summing marginals). Returns
    {"change": elo, "std_error": elo, "confidence_interval_95": (lo,hi)}.
    Raises ValueError if the player or either day is unknown.
    """
```

Implementation of `rating_covariance`:
- Reuse `Player.hessian(days, sigma2, damping)` → `(diagonal, sub_diagonal)` of
  the Hessian `H` (negative definite). Build the dense symmetric tridiagonal
  `−H` (positive definite) and invert it (`numpy.linalg.inv`; per player `n` is
  small). Scale by `K²` to elo². Return `(days, matrix)`.
- **Consistency:** the diagonal (in `r²`, before the `K²` scale) must match each
  day's stored `uncertainty` within tolerance — a test asserts this (if it
  fails, the pre-existing specialised `covariance()` disagrees with the true
  inverse — flag, don't silently paper over).

`rating_change` uses `rating_covariance` and looks up the two days' indices.

## Compatibility

- Purely additive: three new methods + one new opt-in parameter (default
  preserves current behaviour). Non-breaking. `w2`/defaults unchanged.

## Testing plan (TDD, property-based)

1. **A independence formula.** For two players with known day variances,
   `rating_difference` returns `sqrt(var_a+var_b)·K` as `std_error` and a CI of
   `diff ± 1.96·se`; the difference equals `elo_a − elo_b`.
2. **A comparability.** A clearly-stronger player vs a clearly-weaker one over
   many games → difference CI excludes 0; two near-equal players with few games
   → CI includes 0.
3. **B hedges toward 0.5.** With high uncertainty, `account_for_uncertainty=True`
   gives a probability strictly closer to 0.5 than the point prediction; with
   ~0 uncertainty the two agree (within tolerance). `steps` larger → smooth,
   still summing to 1.
4. **B non-breaking.** Default (`False`) equals the pre-phase-5 output exactly.
5. **C consistency.** `rating_covariance` diagonal (converted back to `r²`)
   matches each day's `uncertainty` within tolerance; matrix is symmetric and
   positive semi-definite.
6. **C within-player change.** For a player whose rating provably moved across
   days, `rating_change(from,to)` change ≈ `elo_to − elo_from`, and its SE is
   SMALLER than the naive `sqrt(var_from+var_to)·K` (because
   `Cov(from,to) > 0`) — proving the joint covariance is used, not the marginals.
7. **Degenerate/errors.** Unknown player, unrated player, unknown day → clear
   `ValueError`.

Coverage stays at the locked 95% floor.

## Out of scope

- **D** — true cross-player covariance (full-Hessian inverse) — abandons WHR's
  method; rejected.
- Exposing uncertainty on the handicap/komi advantage estimates.
- Later roadmap points #8 (draws), #10 (vectorisation).

## Open review points

1. Public methods report **elo** (not natural `r`) — confirm.
2. Method names: `rating_difference`, `rating_change`, `rating_covariance`.
3. B as an opt-in parameter on `probability_future_match` (default off) vs a
   separate method; default `uncertainty_steps = 4` (Coulom's scheme).
4. `rating_covariance` returns the full dense matrix (fine — per-player `n` is
   small); acceptable vs only exposing adjacent-day covariances.
