# WHR — Phase 3: estimated handicap & komi (review point #3)

Date: 2026-07-23
Status: design, awaiting approval

## Context

Phase 3 of the WHR roadmap. Phases 1 (robustness), cleanup, and 2 (anti-drift)
are merged to local `master`. This phase adds review point #3: estimate the
handicap and komi advantages as Bradley-Terry parameters, co-estimated with the
players — a faithful port of Coulom's `NewtonKomiHandicap` and the
`vHandicapGamma` machinery (`~/Documents/git/WHR/src/CWHR.cpp:112-127,228-314`).
Per the user's decision, values are **estimated by default with the option to
pin known values**.

The library stays on the WHR model (#11 excluded).

## Current behaviour (what changes)

- `Game.handicap` is a **fixed elo constant** added to the opponent's elo in
  `opponents_adjusted_gamma` (`whr/game.py:53-55`).
- `komi` (from `extras["komi"]`, default 6.5) is stored but **ignored** in the
  likelihood.

## Model (faithful to Coulom)

Each game carries a handicap key `h` (`game.handicap`) and a komi key `k`
(`game.extras["komi"]`). Two advantage tables of gammas are maintained,
initialised to 1.0:

- `handicap_gamma[h]` multiplies **black**'s value,
- `komi_gamma[k]` multiplies **white**'s value.

Game probability: `P(black wins) = (γ_b·γ_h) / (γ_b·γ_h + γ_w·γ_k)`
(`CWHR.cpp:228-239`). This **generalises** the current model exactly: today's
fixed elo handicap equals `γ_h = 10^(handicap/400)`, `γ_k = 1`. So the existing
behaviour is reproduced by *pinning* (see below).

### Effect on `opponents_adjusted_gamma`

Rewritten to fold in the two tables (used by the per-day game terms, so the
player Newton step co-adapts to the current advantage gammas):

- opponent of **white** (i.e. black's adjusted gamma):
  `bpd.gamma() · γ_h / γ_k`
- opponent of **black** (i.e. white's adjusted gamma):
  `wpd.gamma() · γ_k / γ_h`

(Check: for `γ_k = 1`, `γ_h = 10^(handicap/400)`, this is
`bpd.gamma()·10^(handicap/400)` = today's `10^((bpd.elo+handicap)/400)`. ✓)

### Estimating the advantages — `_newton_handicap_komi()`

A new step run once per iteration (in `WHR._run_one_iteration`, after the player
updates), porting `NewtonKomiHandicap` (`CWHR.cpp:270-314`):

- For each game accumulate, per key:
  `CKomi = γ_w`, `DKomi = γ_b·γ_h`, `CHandicap = γ_b`, `DHandicap = γ_w·γ_k`,
  `Div = 1/(DKomi + DHandicap)`;
  `grad[h] += CHandicap·Div`, `hess[h] += CHandicap·DHandicap·Div²`;
  `grad[k] += CKomi·Div`, `hess[k] += CKomi·DKomi·Div²`.
- Track per key the number of games and wins (black-win credits the handicap
  key, white-win credits the komi key — `CWHR.cpp:118-127`).
- Update each key's gamma **only if** it has games, `0 < wins < games` (Coulom's
  degenerate guard — an all-win/all-loss category has no finite estimate), and
  it is **not pinned**:
  `G = wins − γ·grad`, `H = −γ·hess − hessian_damping`, `γ *= exp(−G/H)`.
  (Reuses the existing `hessian_damping` config as Coulom reuses `HessianEpsilon`.)

## Pinning known advantages

Config keys (default `{}`):

- `pinned_handicap: dict[key, float]` — elo advantage per handicap key,
- `pinned_komi: dict[key, float]` — elo advantage per komi key.

Pinned entries are converted to gamma (`γ = 10^(elo/400)`), used in the
likelihood, and **never re-estimated**. Reproducing the old fixed-elo handicap
is therefore `WHR(config={"pinned_handicap": {h: elo}})`.

### Baseline (identifiability)

`handicap` key `0` (no handicap) defaults to a **pinned γ = 1** (0 elo): "no
handicap ⇒ no advantage". This resolves the confound between the black baseline
(`handicap_gamma[0]`) and `komi_gamma` (only their ratio is identified per
game), letting `komi_gamma` absorb the systematic white/side advantage. A user
can override by putting `0` in `pinned_handicap` with a different value, or (if
they truly want it estimated) via a config escape hatch `estimate_handicap_zero:
bool = False`. **Reviewable:** this deviates slightly from Coulom (who estimates
key 0); it is the pragmatic choice for a general library.

## Generalisation beyond Go

With one komi category shared by all games (e.g. every game `komi = 6.5`),
`komi_gamma[6.5]` is exactly a single estimated **white/side advantage** — the
"colour advantage" in chess, or home advantage in sports (set all games to the
same komi key). No Go-specific assumption is baked in.

## Data flow / ownership

- `WHR` owns `self.handicap_gamma: dict` and `self.komi_gamma: dict`,
  initialised from the pinned config (plus the `0 → 1.0` baseline), and grown
  (new key → 1.0) as games are added in `create_game`/`load_games`.
- `Game` holds references to these two shared dicts (passed by the base at
  construction) so `opponents_adjusted_gamma` reads the live gammas each
  iteration (game-term caches are already cleared per iteration). The `Game`
  constructor gains two optional params `handicap_gamma`/`komi_gamma`
  (default `None` ⇒ treated as γ = 1, i.e. no advantage — for direct `Game`
  construction outside the base).
- A pinned-key set on the base marks which keys `_newton_handicap_komi` must
  skip.

## Compatibility

- **Breaking:** `handicap` changes meaning from an elo constant to a category
  key whose advantage is estimated. Old behaviour = pin it. Documented in the
  CHANGELOG with the migration one-liner.
- `komi` moves from an ignored extra to an estimated factor; games that never
  set komi share the single default key (6.5) whose gamma captures the white
  advantage.
- Ratings values change (handicap/komi now modelled). Same release as the other
  phases; version decided at release.

## Testing plan (TDD, property-based)

1. **Back-compat via pinning.** With `pinned_handicap={h: E}` and no komi
   variation, results match the pre-phase-3 fixed-elo handicap within tolerance
   (compare `white_win_probability` on a constructed game).
2. **Recovers a known handicap advantage.** Generate many games at handicap `h`
   between equal-strength players with a built-in black win-rate implying elo
   `E`; after `iterate`, `handicap_gamma[h]` ≈ `10^(E/400)` (within tolerance).
3. **Recovers a white/side (komi) advantage.** All games same komi key, a built
   in white edge; `komi_gamma[k]` recovers it.
4. **Pinned values are not moved.** A pinned handicap/komi key keeps its exact
   gamma after many iterations; an unpinned one moves.
5. **Degenerate guard.** A handicap key whose games are all black-wins (or all
   white-wins) is NOT updated (stays at its init/pinned value), no divide error.
6. **Baseline.** `handicap_gamma[0] == 1.0` by default and is never moved by
   estimation; `estimate_handicap_zero=True` lets it move.
7. **Config plumbing.** `pinned_handicap`/`pinned_komi` defaults `{}`, copied
   (not mutated/shared), converted to gamma correctly.
8. **Opt-nothing regression.** With no handicap (all key 0) and a single komi
   key, players-only ratings still converge sensibly (sanity vs phase-1
   behaviour), and same-day symmetric data stays symmetric.

Coverage stays at the locked 95% floor.

## Out of scope (this phase)

- Continuous/parametric handicap or komi models (Coulom's is categorical; we
  match it).
- Uncertainties on the handicap/komi estimates (Coulom leaves them; a later #9
  could expose them).
- Later roadmap points #7, #8, #9, #10.

## Open review points (please confirm)

1. The `handicap` semantic break (elo → category) with pinning as the migration
   path.
2. `handicap` key `0` pinned to γ = 1 by default (deviates from Coulom;
   escape hatch `estimate_handicap_zero`).
3. Pinning expressed in **elo** (not gamma) — friendlier, matches how the old
   `handicap` was expressed.
4. `Game` constructor gaining two optional dict-reference params.
