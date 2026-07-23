# WHR — Phase 2: anti-drift (`RemoveDrift`, review point #2)

Date: 2026-07-23
Status: design, awaiting approval

## Context

Phase 2 of the WHR roadmap. Phase 1 (robustness/fidelity, points 1/4/5/6) is
merged to local `master`. This phase adds review point #2: a faithful port of
Rémi Coulom's `ComputeDrift`/`RemoveDrift` (`~/Documents/git/WHR`,
`src/CWHR.cpp:696-790`). It is the real anti-inflation mechanism behind the
stability of the goratings scale — far more than a display offset.

The library stays on the WHR model (point #11 is excluded).

## Problem

WHR ratings are only identified up to the per-first-day anchor. Over a long
history the *overall* scale can drift/inflate: the average player strength on
day N is not pinned to the average on day 1. RemoveDrift cancels this global
drift so ratings are comparable across eras.

## Design decision (settled with the user)

**Opt-in method**, not automatic. A new public method `WHR.remove_drift()` that
the user calls AFTER convergence. It does not change the output of
`iterate`/`auto_iterate`, so it is non-breaking. This mirrors Coulom, who runs
`RemoveDrift` as a separate post-convergence step in his pipeline.

## Algorithm (faithful port of Coulom)

Let the global day range be `MinDay = min(game.day)`, `MaxDay = max(game.day)`,
`N = MaxDay - MinDay + 1`. Let `R = drift_kernel_radius` (config, default 100).

1. **Accumulate per day** (`CWHR.cpp:717-725`). For every game on day `d`
   (index `i = d - MinDay`), using the current per-day elos of the two players
   (`game.bpd.elo`, `game.wpd.elo`):
   - `game_count[i] += 1`
   - `total_elo[i] += game.bpd.elo + game.wpd.elo`
   Arrays are padded by `R` on each side (zeros) so the convolution never reads
   out of range.

2. **Gaussian kernel** (`CWHR.cpp:730-745`), `sigma = R * 0.5` (the 0.5 factor
   is Coulom's; not exposed). Build `kernel[0..R-1]` with
   `kernel[k] = exp(-k² / (2·sigma²))`, normalised by
   `Total = 1 + 2·Σ_{k=1}^{R-1} exp(...)`, then `kernel[0] = (1/Total)·0.5` and
   `kernel[k>0] = (1/Total)·exp(...)`. (The centre is half-weighted because the
   convolution below sums the `k` and `-k` sides, double-counting `k = 0`.)

3. **Convolve** (`CWHR.cpp:750-761`). For each output day `i` in `0..N-1`
   (`j = i + R`):
   - `filtered_elo[i]   = Σ_{k=0}^{R-1} (total_elo[j+k]   + total_elo[j-k])   · kernel[k]`
   - `filtered_count[i] = Σ_{k=0}^{R-1} (game_count[j+k] + game_count[j-k]) · kernel[k]`

4. **Per-day correction** (`CWHR.cpp:775-783`). `drift_elo[i] = filtered_elo[i]
   / (2 · filtered_count[i])` (÷2 because each game contributes two players).
   The correction that cancels it, in elo, is `-drift_elo[i]`. If
   `filtered_count[i]` is 0 (a day with no smoothing support) or `drift_elo[i]`
   is non-finite, the correction is 0 elo (Coulom's nan/inf guard → gamma
   factor 1.0).

5. **Apply** (`CWHR.cpp:787-789`). For every `PlayerDay` on day `d`, shift its
   natural rating: `r += (-drift_elo[d-MinDay]) · ln(10) / 400`. Because the
   shift is identical for all players on the same day, within-day rating
   *differences* — and therefore same-day win probabilities — are unchanged;
   only cross-day comparisons move.

## API

```python
def remove_drift(self) -> dict[int, float]:
    """Cancel global rating drift over time (Coulom's RemoveDrift).

    Call after iterate()/auto_iterate(). Rescales each day's ratings so the
    smoothed average player strength per day is centred at 0 elo, making
    ratings comparable across eras. Mutates the stored ratings in place and
    returns the per-day elo correction applied ({day: correction_elo}).
    Within-day rating differences (hence same-day win probabilities) are
    unchanged. Uncertainties are not recomputed.
    """
```

Config key: `drift_kernel_radius` (int, default 100) — smoothing radius in days,
added via `WHR.__init__`'s `setdefault`.

## Behaviour & compatibility

- Non-breaking: default `iterate`/`auto_iterate` output unchanged; this is a
  separate, explicit call.
- Mutates `PlayerDay.r` (ratings). Does NOT recompute uncertainties — the
  curvature is essentially unchanged by a uniform per-day shift, and Coulom
  treats drift removal as a scale adjustment. Callers who want refreshed
  uncertainties can call `iterate(0)`-style update or `player.update_uncertainty()`.
- Idempotence: a second `remove_drift()` on already-de-drifted ratings applies
  near-zero corrections (the smoothed mean is already ~0).
- Empty base or a single distinct day: returns corrections that recentre that
  day (or an empty dict for no games); never raises.

## Testing plan (TDD, property-based)

1. **Removes injected drift.** Build a history where true skills are constant
   but results are generated so the fitted mean strength ramps upward over
   days (e.g. progressively stronger newcomers). After `iterate` +
   `remove_drift`, the games-weighted smoothed mean elo per day is ≈ 0 (within
   tolerance), whereas before it trended.
2. **Within-day invariance.** For two players who play on the same day, their
   win-probability (`probability_future_match` / `white_win_probability`) is
   unchanged by `remove_drift` (difference of their elos preserved).
3. **Cross-day shift applied.** A day whose smoothed mean is clearly non-zero
   has all its players shifted by the same reported correction.
4. **Config radius respected.** Different `drift_kernel_radius` values produce
   different (finite) corrections; the returned dict keys cover the day range.
5. **Degenerate inputs.** Empty base → `{}`, no raise. Single day → finite
   correction, no raise. A day with zero smoothing support → 0 correction (guard).
6. **Return contract.** `remove_drift()` returns a `dict[int, float]` of finite
   corrections keyed by day; ratings after the call reflect exactly those
   corrections.

Coverage stays at the locked 95% floor.

## Out of scope (this phase)

- Automatic application inside `auto_iterate` (rejected: opt-in chosen).
- Recomputing uncertainties after drift removal.
- Exposing the Gaussian `sigma` factor (kept at Coulom's `R·0.5`).
- Later roadmap points #3, #7, #8, #9, #10 (separate phases).

## Sequencing note

Implementation of this phase must start only AFTER the deferred-minors cleanup
task (running in a separate session; it edits `player.py`/`playerday.py` and
re-baselines tests) has landed on `master`, to avoid conflicting edits. The
plan is written against the post-cleanup `master`.
