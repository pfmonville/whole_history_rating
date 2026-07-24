# WHR Vectorization (Phase 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** numpy-vectorize the per-game hot loops (review point #10) — algorithm-preserving, results identical up to float reordering, no new dependency.

**Architecture:** Keep the per-player Gauss-Seidel loop and the tridiagonal solve. Replace the Python per-game loops with numpy array arithmetic in (1) the global all-games accumulators (`_accumulate_handicap_komi`, `_nu_gradient_hessian`) and (2) the per-player per-day term computations (BT + Davidson). A prepared safety net (loosened exact assertions + an equivalence regression) guards behaviour.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest+pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python ≥3.11, numpy ≥2.0; ruff/ruff-format/mypy clean; `uv run pytest` passes at ≥95% coverage.
- **Algorithm-preserving:** do NOT change the Gauss-Seidel per-player loop or the tridiagonal solve; do NOT batch across players. Results must match the pre-vectorization values within `rel=1e-9`/`abs=1e-9`.
- numpy only (NO numba). No public API/behaviour change beyond float-level reordering.
- Draws still skipped in `_accumulate_handicap_komi`.
- Branch `feat/vectorize-phase7`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Safety net — loosen exact assertions + equivalence regression

Prepare the tests to tolerate float reordering BEFORE any code change (this task changes NO production code; the suite stays green because the values are unchanged and `approx` trivially holds).

**Files:**
- Modify: `tests/whr_test.py` (loosen exact-equality golden assertions to `pytest.approx`; convert print-string assertions to parse-and-approx)
- Test: `tests/test_vectorize.py` (create — equivalence regression)

- [ ] **Step 1: Loosen exact golden assertions.** In `tests/whr_test.py`, find the tests asserting exact computed floats — `test_output`, `test_output2`, `test_loading_several_games_at_once` (ratings lists, `ratings_for_player`, `probability_future_match`, `log_likelihood`, `get_ordered_ratings`) — and change exact `==` comparisons to `pytest.approx(<same value>, rel=1e-9, abs=1e-9)`. For the `print_ordered_ratings` display-string assertions in `test_loading_several_games_at_once`, replace the exact string compare with: capture the output, parse the per-name float(s) out of each line, and assert them `== pytest.approx(<same values>, rel=1e-9)`. Do NOT change the expected numbers — only add tolerance / parse.

- [ ] **Step 2: Equivalence regression test.** Create `tests/test_vectorize.py` capturing, as hard-coded constants gathered from the CURRENT code, the key outputs for representative scenarios, and asserting the live code matches within tolerance:

```python
import math
import pytest
from whr.whole_history_rating import WHR

def _scenario_draw_free():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "c b B 2", "a c B 3", "a b B 4"])
    w.iterate(50)
    return w

def _scenario_handicap_komi():
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    for d in range(1, 8):
        w.create_game("x", "y", "B", d, 2)
        w.create_game("y", "x", "W", d, 2)
    w.iterate(50)
    return w

def _scenario_draws():
    w = WHR()
    for d in range(1, 8):
        w.create_game("p", "q", "D", d, 0)
        w.create_game("p", "q", "B", d, 0)
    w.iterate(50)
    return w

def test_equivalence_draw_free_ratings():
    w = _scenario_draw_free()
    # Capture the current values ONCE (run this test on the pre-vectorization
    # code, paste the printed numbers here), then assert they survive vectorization.
    got = dict(w.get_ordered_ratings(current=True))
    # EXPECTED filled from the pre-vectorization run:
    expected = {name: elo for name, elo in got.items()}  # PLACEHOLDER — see Step 3
    for name, elo in expected.items():
        assert got[name] == pytest.approx(elo, rel=1e-9, abs=1e-9)

# ... similar equivalence tests for handicap/komi (handicap_gamma[2], ratings)
# and draws (draw_tendency, win_draw_loss_probabilities, log_likelihood).
```

- [ ] **Step 3: Freeze the expected values.** Run each scenario on the CURRENT code, read the actual outputs (`get_ordered_ratings`, `handicap_gamma[2]`, `draw_tendency`, `win_draw_loss_probabilities("p","q")`, `log_likelihood()`), and hard-code them as the `expected` constants in `tests/test_vectorize.py` (replace the placeholder). These constants are the pre-vectorization ground truth that Tasks 2–3 must reproduce within `rel=1e-9`.

- [ ] **Step 4: Verify green.** `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`. All pass (no code changed; loosened tests still hold exactly, equivalence constants match the current code).

- [ ] **Step 5: Commit.**
```bash
git add tests/whr_test.py tests/test_vectorize.py
git commit -m "Prepare vectorization safety net: approx tolerances + equivalence regression

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Vectorize the global all-games accumulators

**Files:**
- Modify: `whr/whole_history_rating.py` (`_accumulate_handicap_komi`, `_nu_gradient_hessian`; add `import numpy as np` if absent)
- Test: `tests/test_vectorize.py`

- [ ] **Step 1: Unit-equivalence test.** Append a test that builds a handicap/komi/draw base, computes the per-key gradient/Hessian/counts BOTH via the vectorized method and via a small reference Python loop kept inside the test, and asserts they match within `rel=1e-9`. (Access `_accumulate_handicap_komi`'s outputs and `_nu_gradient_hessian()` directly.)

- [ ] **Step 2: Run to see it fail** only if the reference and implementation would diverge — initially they match (no change yet), so this test passes; it becomes the guard. (If the method's outputs aren't directly inspectable, refactor minimally so the test can call it. Keep behaviour identical.)

- [ ] **Step 3: Vectorize `_accumulate_handicap_komi`.** Replace the `for g in self.games` loop with numpy batching over DECISIVE games (skip `winner == "D"`):
  - Build arrays: `gb`, `gw` (black/white `bpd.gamma()`/`wpd.gamma()`), `gh = handicap_gamma[key]`, `gk = komi_gamma[key]`, integer index arrays `hi`, `ki` mapping each game's handicap/komi key to a contiguous key index, and `black_win` (bool).
  - Per-game (vectorized): `c_komi=gw`, `d_komi=gb*gh`, `c_handicap=gb`, `d_handicap=gw*gk`, `div=1.0/(d_komi+d_handicap)`; `h_grad_g=c_handicap*div`, `h_hess_g=c_handicap*d_handicap*div**2`, `k_grad_g=c_komi*div`, `k_hess_g=c_komi*d_komi*div**2`.
  - Accumulate per key with `np.bincount(hi, weights=h_grad_g, minlength=nh)` etc. (and `np.bincount(hi, weights=black_win)` for wins, `np.bincount(hi)` for games; likewise komi with `~black_win`). Convert back to the same dict/return shape the callers expect.
  - The update loop (applying the Newton step per non-pinned key with the `0<wins<games` guard) stays as-is — only the accumulation is vectorized.

- [ ] **Step 4: Vectorize `_nu_gradient_hessian`.** Build arrays `s`, `o` (via `effective_gammas(black_player)` — or directly `gb*gh`, `gw*gk`), `t = nu*sqrt(s*o)`, `z = s+o+t`, `ratio = t/z`, `is_draw`; `gradient = np.sum(is_draw) - np.sum(ratio)`; `hessian = -np.sum(ratio*(1-ratio))`. Return `(gradient, hessian)` as before (the caller subtracts damping / does the Newton step).

- [ ] **Step 5: Verify.** `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`. All pass — the equivalence regression (Task 1) and the unit-equivalence test must hold within tolerance; the loosened golden tests pass.

- [ ] **Step 6: Commit.**
```bash
git add whr/whole_history_rating.py tests/test_vectorize.py
git commit -m "Vectorize the global handicap/komi/nu accumulation with numpy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Vectorize the per-player per-day terms

**Files:**
- Modify: `whr/playerday.py` (`log_likelihood_derivative`, `log_likelihood_second_derivative`, `log_likelihood`, `davidson_derivatives`, `davidson_log_likelihood`)
- Test: `tests/test_vectorize.py`

- [ ] **Step 1: Unit-equivalence test.** Append a test that, for a `PlayerDay` with several games (some won/lost, and a Davidson case with draws), asserts the vectorized `log_likelihood_derivative`/`_second_derivative`/`log_likelihood` and `davidson_derivatives`/`davidson_log_likelihood` equal a reference scalar-loop computation (kept in the test) within `rel=1e-9`.

- [ ] **Step 2: Vectorize the BT terms.** In `PlayerDay`, replace the scalar loops. For the day's decisive games, build a numpy array `d` of opponents' adjusted gammas (`opponents_adjusted_gamma(self.player)` for won ∪ lost) and `n_wins = len(self.won_games)`. Then:
  - `log_likelihood_derivative = n_wins - gamma*np.sum(1.0/(gamma+d))`
  - `log_likelihood_second_derivative = -gamma*np.sum(d/(gamma+d)**2)`
  - `log_likelihood = np.sum(np.log(gamma) - np.log(gamma+d_won)) + np.sum(np.log(d_lost) - np.log(gamma+d_lost))` where `d_won`/`d_lost` are the won/lost opponents' adjusted gammas (won term `[1,0,1,d]`→`log(gamma)-log(gamma+d)`, lost term `[0,d,1,d]`→`log(d)-log(gamma+d)`).
  Handle the empty-day case (no games → 0.0) so numpy over empty arrays is fine (`np.sum([])==0`). Keep the same public return values.

- [ ] **Step 3: Vectorize the Davidson terms.** In `davidson_derivatives(nu)` and `davidson_log_likelihood(nu)`, build arrays over `_weighted_games()`: `s`, `o` (from `effective_gammas`), `w` (weights 1/0/0.5). Then `t=nu*np.sqrt(s*o)`, `z=s+o+t`, `n=s+t/2`, `n2=s+t/4`, `ratio=n/z`:
  - `davidson_derivatives`: `gradient=np.sum(w-ratio)`, `hessian=np.sum(ratio**2 - n2/z)`.
  - `davidson_log_likelihood`: `num = where(w==1, s, where(w==0, o, t))`; `= np.sum(np.log(num) - np.log(z))`.
  Empty-day → `(0.0, 0.0)` / `0.0`.

- [ ] **Step 4: Verify.** `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`. All pass within tolerance; the equivalence regression + unit tests hold; `fit_w2`'s recovery test still passes (margins >> float noise).

- [ ] **Step 5: Commit.**
```bash
git add whr/playerday.py tests/test_vectorize.py
git commit -m "Vectorize per-player per-day BT and Davidson terms with numpy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Benchmark smoke test + docs

**Files:**
- Test: `tests/test_vectorize.py`
- Modify: `CHANGELOG.md`, `README.md`

- [ ] **Step 1: Benchmark smoke test.** Append a test building a moderately large history (e.g. ~3000 games across many players/days) and running `whr.auto_iterate(time_limit=30)` (or `iterate(20)`), asserting it completes and returns finite ratings. NO hard wall-clock assertion (machine-dependent) — this only guards that the vectorized path runs at scale without error.

- [ ] **Step 2: CHANGELOG.** Under `## [2.1.0] - unreleased` `### Changed`: the per-game hot loops (handicap/komi/nu accumulation and per-player term computation) are now numpy-vectorized for large histories; results are unchanged up to floating-point reordering; no new dependency.

- [ ] **Step 3: README.** Add a brief performance note (e.g. in a "Performance"/notes section): ratings computation is numpy-vectorized and scales to large histories; the algorithm and results are unchanged. Keep it short.

- [ ] **Step 4: Verify.** `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy` (all clean, ≥95%).

- [ ] **Step 5: Commit.**
```bash
git add tests/test_vectorize.py CHANGELOG.md README.md
git commit -m "Add vectorization benchmark smoke test and docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** safety net + equivalence regression (T1); global accumulators vectorized (T2); per-player BT+Davidson terms vectorized (T3); benchmark + docs (T4). numba and cross-player batching excluded.
- **Algorithm-preserving:** only per-game arithmetic is batched; the Gauss-Seidel player loop, tridiagonal solve, Newton-step application, and `0<wins<games` guards are untouched. Draws still skipped in handicap/komi.
- **Guard:** the equivalence regression (frozen pre-vectorization constants, `rel=1e-9`) plus per-method unit-equivalence tests are the proof that behaviour is preserved; the loosened golden tests + `fit_w2` recovery confirm nothing drifts beyond float noise.
- **Empty arrays:** each vectorized path handles a day/base with zero relevant games (`np.sum` of empty → 0.0).
