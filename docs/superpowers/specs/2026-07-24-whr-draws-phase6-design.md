# WHR — Phase 6: draws (Davidson model, review point #8)

Date: 2026-07-24
Status: design, awaiting approval

## Context

Phase 6 of the WHR roadmap. Phases 1, cleanup, 2, 3, 4, 5 are merged to local
`master`. This phase adds review point #8: model **draws** (ties). This is NOT
in Coulom's Go reference (Go has effectively no draws) — it is a genuine model
extension for a general-purpose library, using the **Davidson** tie model
(chosen with the user). #11 remains out of scope.

This is the most invasive phase: it changes the core game likelihood from
win/loss to win/draw/loss, which feeds the per-player Newton updates and the
handicap/komi estimation.

## Model (Davidson)

For a game, let `S` = the player-of-interest's effective gamma (their `gamma`
times their side advantage: `handicap_gamma` for black, `komi_gamma` for white),
`O` = the opponent's effective gamma, and `nu ≥ 0` a global **draw tendency**.
Let `T = nu·sqrt(S·O)` (the "draw mass") and `Z = S + O + T`. Then:

- `P(player wins) = S / Z`
- `P(player loses) = O / Z`
- `P(draw) = T / Z`

`nu = 0` recovers exactly the current Bradley-Terry win/loss model
(`T = 0`, `P(win) = S/(S+O)`).

## Per-player gradient & Hessian (derived)

Updating a player (their `r = log gamma`, so `S = adv·e^r`, `dS/dr = S`;
`d sqrt(SO)/dr = sqrt(SO)/2`, so `dT/dr = T/2`, `dZ/dr = S + T/2`). For one game,
with outcome weight `w = 1` (player won), `0.5` (draw), `0` (player lost):

- **gradient** `= w − (S + T/2)/Z`
- **Hessian** `= (N/Z)² − N'/Z`, where `N = S + T/2`, `N' = S + T/4`.

Summed over the player's games. Sanity check at `nu = 0` (`T = 0`): gradient
`= w − S/(S+O)`, Hessian `= (S/(S+O))² − S/(S+O) = −S·O/(S+O)²` — exactly the
current BT win/loss derivatives (win `w=1`, loss `w=0`). Verified.

## Estimating `nu` (Newton step, in `log nu` space)

`nu` is a single global parameter, estimated each iteration like the
handicap/komi advantages. With `v = log nu` (`dT/dv = T`, `dZ/dv = T`), per game:

- **gradient** `= [1 if draw else 0] − T/Z`
- **Hessian** `= −(T/Z)·(1 − T/Z)`

Summed over ALL games. Newton: `v -= gradient / (Hessian − hessian_damping)`,
then `nu = exp(v)`. Skipped entirely when there are no draws (see below).

## Compatibility strategy (avoids re-baselining)

Because `nu = 0` reduces Davidson EXACTLY to the current model, and every
existing test uses draw-free data, the draw machinery is **activated only when
the base contains at least one draw** (a `"D"` game):

- `WHR` tracks `_has_draws` (set when a `"D"` game is added).
- If `not _has_draws`: `nu` stays `0`, the `nu` Newton step is skipped, and the
  per-player derivatives use the EXISTING win/loss code path **unchanged** —
  so draw-free datasets (all current tests) are bit-for-bit identical and need
  NO re-baselining.
- If `_has_draws`: `nu` is initialised to a positive seed (`1.0`) and estimated;
  the per-player derivatives use the Davidson path (which at the fitted `nu`
  models the draws). Ratings for draw-containing data change (correct — draws
  are information).

`nu` is also pinnable via config `pinned_draw` (elo-free scalar; if set, `nu` is
fixed and not estimated) — consistent with the handicap/komi pinning; default
unset.

## Data & API changes

- `create_game`/`load_games`: accept `winner == "D"` (draw). `Game.winner`
  stores `"D"`. `PlayerDay` gains `drawn_games`; `add_game` routes a draw there.
- `Game`: add `effective_gammas(player) -> (S, O)` (the player's and opponent's
  effective gammas, folding in handicap/komi) for the Davidson computation.
- New prediction method `win_draw_loss_probabilities(name1, name2, handicap=0,
  handicap_key=None, komi_key=None) -> tuple[float, float, float]` returning
  `(P(name1 wins), P(draw), P(name2 wins))` summing to 1, using the fitted `nu`.
  When `nu = 0` the draw probability is 0 and the win pair matches
  `probability_future_match`. `probability_future_match` is unchanged (its
  `(p1, p2)` remain the win/loss split ignoring draw mass; documented).
- Expose the fitted draw tendency read-only as `WHR.draw_tendency` (`= nu`).

## Testing plan (TDD, property-based)

1. **No-draw data unchanged.** A draw-free base produces bit-identical ratings,
   `log_likelihood`, and predictions to before this phase (assert against a
   couple of the existing golden scenarios); `_has_draws is False`, `nu == 0`.
2. **`"D"` accepted.** `create_game(..., "D", ...)` and a `"D"` line in
   `load_games` are parsed; the game lands in the players' `drawn_games`.
3. **Recovers a known draw tendency.** Generate a balanced round-robin (equal
   players, colour-swapped, single day) where outcomes are sampled from Davidson
   with a known `nu*` (so a known draw fraction); after `iterate`, the fitted
   `WHR.draw_tendency` recovers `nu*` within tolerance. (Balanced so player
   strength doesn't confound `nu`, mirroring the handicap recovery test.)
4. **Davidson reduces to BT at nu=0.** With `pinned_draw` forcing `nu = 0` on
   draw-containing data, the win/loss conditional probabilities equal the plain
   BT values.
5. **Prediction sums to 1.** `win_draw_loss_probabilities` returns three
   non-negative values summing to 1; higher fitted `nu` → larger `P(draw)`.
6. **Pinned nu not moved.** `pinned_draw` keeps `nu` fixed across iterations.
7. **Gradient/Hessian sanity.** For a hand-set S/O/nu, the per-game Davidson
   gradient and Hessian match the closed forms above (unit test of the term
   computation).
8. **Degenerate.** All-draw or single-game draw data converges finitely, no
   exception.

Coverage stays at the locked 95% floor.

## Out of scope

- Rao-Kupper (Davidson chosen).
- Draw-specific handicap/komi interactions beyond the effective-gamma folding.
- #10 vectorization (next).

## Open review points

1. Davidson (settled). `nu` estimated in `log nu` space, reused `hessian_damping`.
2. Draw machinery activated only when draws present (keeps no-draw behaviour
   bit-identical — no re-baselining). Confirm this compatibility strategy.
3. New method `win_draw_loss_probabilities` for the 3-way prediction;
   `probability_future_match` left as the win/loss split (documented).
4. `pinned_draw` config to fix `nu`; `WHR.draw_tendency` to read it.
5. `"D"` as the draw marker in the game format.
