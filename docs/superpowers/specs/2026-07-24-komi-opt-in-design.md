# WHR — komi opt-in (3.1.0)

Date: 2026-07-24
Status: design, awaiting approval

## Problem

`komi` is handled inconsistently and Go-centrically:

- `handicap` is a required, first-class `create_game` argument (explicit).
- `komi` is buried in `extras` with a **silent Go default of `6.5`** that is
  then **estimated** (`game.py:32` `self.extras = {"komi": 6.5}`).

In the model komi is a *categorical key*, not a points value, so `6.5` is just a
label every game silently shares. For any non-Go use this creates a spurious
global komi advantage that is estimated unasked; today the only way to switch it
off is the awkward `pinned_komi={6.5: 0.0}`. (It also caused the 3.0.1
`OverflowError`, now clamped, but the ergonomic wart remains.)

Decided with the maintainer: make komi **opt-in** with a neutral default, and
ship it as **3.1.0** (the maintainer's call, accepting the semver caveat below).

## Design

### Public API

```python
def create_game(self, black, white, winner, time_step,
                handicap, komi=None, extras=None) -> Game
```

`komi` is placed **before** `extras` (natural order — maintainer's choice). This
accepts a small positional break: an existing `create_game(b, w, winner, t,
handicap, extras)` call passing `extras` positionally would now bind that dict to
`komi`. `extras` was always documented as keyword, and `load_base` is updated to
pass `extras=` by keyword.

| `komi=` (or `extras["komi"]`) | Behaviour |
|---|---|
| absent / `None` (new default) | **No komi advantage.** No komi key is registered; the game's komi gamma is 1.0 (neutral); nothing is estimated. |
| a value (e.g. `6.5`) | That komi category is registered and **estimated** (or pinned via `pinned_komi`) — the current behaviour, now explicit. |

Resolution: `komi = komi if komi is not None else (extras or {}).get("komi")`.
`extras={"komi": X}` therefore keeps working (back-compat). No default `6.5` is
ever injected.

### Internal representation

- komi stays inside `extras` (present only when set) — minimal churn and
  save/load format is unchanged. A no-komi game simply has no `"komi"` entry.
- Every internal read changes from `extras["komi"]` (KeyError-prone) to
  `extras.get("komi")` → `None` means "no komi":
  - `Game.opponents_adjusted_gamma`, `Game.effective_gammas`, `Game.__str__`
    (game.py) — `komi_gamma.get(None, 1.0)` = 1.0, neutral.
  - `WHR._ensure_advantage_keys` — skip registering a `None` komi key.
  - `WHR._accumulate_handicap_komi` — a `None`-komi game still contributes to
    **handicap** estimation; its komi term is routed to a `None` bin that is
    dropped when scattering per-key results, so `None` never becomes an
    eligible/estimated key. komi-gamma lookups use `.get(k, 1.0)`.
  - Legacy `load_base` path — `extras.get("komi")`.

### save / load

Format unchanged: games serialize as `[black, white, winner, day, handicap,
extras]`; komi (when set) rides in `extras`. `load_base`'s reconstruction call
becomes `create_game(..., handicap, extras=extras)` (keyword `extras`, so the
new `komi` param is not accidentally bound positionally).

## Versioning

**3.1.0** (maintainer's decision). Honest caveat: this **changes ratings** for
anyone who relied on the implicit default komi being estimated (measurable even
on simple data), which strict semver would call major. It is shipped as a minor
by maintainer judgement (recent release cadence had several majors); the
breakage is minimised by honouring explicit `extras["komi"]`, and it is
documented as a **`Changed`** entry with a one-line migration ("pass `komi=6.5`
to restore the old default komi estimation"), not merely `Added`.

## Test / doc migration

- `tests/test_handicap_komi.py`: tests that relied on the implicit default komi
  (asserting `komi_gamma[6.5]` behaviour, "default komi game", the komi-recovery
  and auto-iterate-waits-for-komi tests) pass `komi=6.5` explicitly to keep
  exercising komi estimation. "handicap-0 default game has no adjustment" stays
  valid (now literally no komi).
- `tests/test_vectorize.py`: the frozen-constant equivalence scenario passes
  komi explicitly where it intends komi, so its frozen values stay valid
  (otherwise re-freeze).
- Any direct `Game(...)`/`extras["komi"]` reads in tests → `.get`.
- README: the komi example becomes `komi=7.5` (extras still noted as accepted);
  "Handicap and komi" gains a line that komi is opt-in as of 3.1.0.
- New tests: (a) no-komi game registers no komi key and is neutral; (b) a game
  with a handicap but no komi still estimates the handicap; (c) `komi=` and
  `extras={"komi":…}` are equivalent; (d) save/load round-trips a komi game and
  a no-komi game.

## TDD plan

Red first (new tests above + migrated tests), then implement game.py +
whole_history_rating.py, then re-green the suite, keep coverage ≥ 95%, ruff +
mypy clean.

## Open question for approval

**Signature ordering.** This spec puts `komi` **after** `extras`
(`…, handicap, extras=None, komi=None`) to guarantee positional back-compat. The
natural order would be `…, handicap, komi=None, extras=None`, but that would
rebind any existing positional `create_game(…, handicap, extras)` call's dict to
`komi`. Confirm the safe (after-`extras`) ordering, or accept the small
positional break for the nicer order.

## Out of scope

- Changing `handicap` (stays a required first-class arg).
- Full Davidson-aware komi estimation on drawn games (unchanged).
- Any deprecation-then-4.0.0 path (rejected in favour of a direct 3.1.0 change).
