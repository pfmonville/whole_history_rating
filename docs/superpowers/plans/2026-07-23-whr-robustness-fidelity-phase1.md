# WHR Robustness & Fidelity (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make WHR's Newton optimisation numerically robust and faithful to Coulom's C++ reference (review points 1, 4, 5, 6), removing three ad-hoc hacks and the excessive compression of weakly-connected players.

**Architecture:** Move the first-day Bradley-Terry anchor out of the fake game-term lists into a direct gradient/Hessian/log-likelihood contribution scaled by `initial_prior_wins` (default 0.5, Coulom). Replace the `-0.001` Hessian nudge with a configurable damping `hessian_damping` (default 1.0, Coulom's `HessianEpsilon`). Converge `auto_iterate` on the gradient infinity-norm. Delete the `>650` and `sys.maxsize` guards, keeping only a non-finite safety net. Fix the buggy public `log_likelihood`.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest + pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python floor: `>=3.11`; numpy `>=2.0`. Copied verbatim from spec.
- `ruff check whr tests` and `mypy` must stay clean; `uv run pytest` must pass with coverage `--cov-fail-under=95`.
- Keep the exact Gaussian Wiener temporal prior `sigma2 = |Δdays| · w2` — do NOT replace it with Coulom's virtual-wins approximation.
- New behaviour is the default; new config keys have defaults `initial_prior_wins=0.5`, `hessian_damping=1.0`.
- Faithful to Coulom's `CWHR.cpp`: anchor on the first player-day only; damping subtracted from the Hessian diagonal; no line search; no arbitrary elo cap.
- Run individual tests with `--no-cov` (the global `--cov-fail-under=95` in `addopts` fails single-file runs); run the full `uv run pytest` only where the plan says to.
- All commits on branch `feat/robustness-fidelity-phase1`. Use `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` in commit messages.

---

### Task 1: Config plumbing + quarantine value-sensitive tests

Adds the two config keys and Player attributes (no behavioural change yet), and skips the existing golden/semantic tests that Tasks 2–6 will invalidate, so the suite stays green during the transition. They are re-baselined in Task 7.

**Files:**
- Modify: `whr/whole_history_rating.py` (`WHR.__init__`, ~lines 18-21)
- Modify: `whr/player.py` (`Player.__init__`, ~lines 17-21)
- Modify: `tests/whr_test.py` (add skip markers)
- Test: `tests/test_robustness.py` (create)

**Interfaces:**
- Produces: `Player.initial_prior_wins: float`, `Player.hessian_damping: float`; config keys `initial_prior_wins` (default 0.5), `hessian_damping` (default 1.0).

- [ ] **Step 1: Write the failing test**

Create `tests/test_robustness.py`:

```python
import math

import pytest

from whr import utils, whole_history_rating


def test_new_config_keys_defaults_and_copy():
    src = {"w2": 300}
    w = whole_history_rating.WHR(config=src)
    assert w.config["initial_prior_wins"] == 0.5
    assert w.config["hessian_damping"] == 1.0
    assert "initial_prior_wins" not in src  # caller dict not mutated
    w.create_game("a", "b", "B", 1, 0)
    player = w.player_by_name("a")
    assert player.initial_prior_wins == 0.5
    assert player.hessian_damping == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robustness.py::test_new_config_keys_defaults_and_copy --no-cov -v`
Expected: FAIL with `KeyError: 'initial_prior_wins'` (or AttributeError).

- [ ] **Step 3: Add config defaults**

In `whr/whole_history_rating.py`, `WHR.__init__`, after the existing `setdefault` lines:

```python
        self.config.setdefault("debug", False)
        self.config.setdefault("w2", 300.0)
        self.config.setdefault("uncased", False)
        self.config.setdefault("initial_prior_wins", 0.5)
        self.config.setdefault("hessian_damping", 1.0)
```

In `whr/player.py`, `Player.__init__`:

```python
    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.debug = config["debug"]
        self.w2 = (math.sqrt(config["w2"]) * math.log(10) / 400) ** 2
        self.initial_prior_wins = config["initial_prior_wins"]
        self.hessian_damping = config["hessian_damping"]
        self.days: list[PD.PlayerDay] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_robustness.py --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Quarantine value-sensitive existing tests**

The following tests in `tests/whr_test.py` assert exact ratings / uncertainties / log-likelihoods, or old-hack semantics, that change in Tasks 2–6. Add a skip marker directly above each, with this exact reason, so the suite stays green until Task 7 re-baselines them:

```python
@pytest.mark.skip(reason="re-baselined in phase-1 robustness plan, Task 7")
```

Apply to every test function that contains any of these (identify by reading the file): asserts a specific value from `ratings_for_player(...)`, asserts `player.log_likelihood()` / `day.log_likelihood()` magic numbers (`test_log_likelihood`), asserts `whr.log_likelihood()` magic numbers, the handicap-600 test that does `with pytest.raises(utils.UnstableRatingException): whr.iterate(...)`, `test_auto_iterate`, `test_auto_iterate_returns_not_stable_on_timeout`, `test_log_likelihood_raises_instead_of_exiting_on_overflow`, and any uncertainty-value assertion. Do NOT skip: save/load equality, unknown-player `ValueError`, invalid-format errors, game-type checks, deprecation-warning test.

- [ ] **Step 6: Verify full suite is green (with skips)**

Run: `uv run pytest --no-cov`
Expected: PASS, several tests reported as skipped. Coverage is intentionally
NOT enforced during quarantine (Tasks 1–6), because skipped tests can transiently
drop coverage below the 95% floor; the floor is re-enforced with a full
`uv run pytest` in Task 7 once the quarantined tests are re-baselined.

- [ ] **Step 7: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add whr/whole_history_rating.py whr/player.py tests/whr_test.py tests/test_robustness.py
git commit -m "Add phase-1 config keys and quarantine value tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Faithful first-day anchor (#1)

Removes the fake first-day game terms and injects Coulom's anchor directly, scaled by `initial_prior_wins`.

**Files:**
- Modify: `whr/playerday.py` (`won_game_terms`, `lost_game_terms`, `update_by_1d_newtons_method`; add anchor methods)
- Modify: `whr/player.py` (`gradient`, `hessian` — add anchor at index 0)
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `Player.initial_prior_wins` (Task 1).
- Produces: `PlayerDay.anchor_gradient() -> float`, `PlayerDay.anchor_hessian() -> float`, `PlayerDay.anchor_log_likelihood() -> float`. `Player.hessian` gains a `damping` parameter in Task 3 — in this task it keeps its current signature and the anchor is added at `diagonal[0]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_robustness.py`:

```python
def test_anchor_terms_match_coulom_formula():
    w = whole_history_rating.WHR()  # initial_prior_wins == 0.5
    w.create_game("a", "b", "B", 1, 0)
    day = w.player_by_name("a").days[0]
    day.set_gamma(2.0)
    k, g = 0.5, 2.0
    assert day.anchor_gradient() == pytest.approx(k * (1 - 2 * g / (1 + g)))
    assert day.anchor_hessian() == pytest.approx(-2 * k * g / ((1 + g) ** 2))
    assert day.anchor_log_likelihood() == pytest.approx(
        k * (math.log(g) - 2 * math.log(1 + g))
    )


def test_anchor_strength_scales_with_config():
    w = whole_history_rating.WHR(config={"initial_prior_wins": 1.0})
    w.create_game("a", "b", "B", 1, 0)
    day = w.player_by_name("a").days[0]
    day.set_gamma(2.0)
    assert day.anchor_gradient() == pytest.approx(1.0 * (1 - 2 * 2.0 / 3.0))


def test_lower_prior_reduces_compression():
    def spread(k):
        w = whole_history_rating.WHR(config={"initial_prior_wins": k})
        for d in range(1, 21):
            w.create_game("strong", "weak", "B", d, 0)  # strong (black) always wins
        w.iterate(300)
        elos = dict(w.get_ordered_ratings(current=True))
        return abs(elos["strong"] - elos["weak"])

    assert spread(0.5) > spread(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_robustness.py -k "anchor or compression" --no-cov -v`
Expected: FAIL with `AttributeError: 'PlayerDay' object has no attribute 'anchor_gradient'`.

- [ ] **Step 3: Remove fake terms and add anchor methods in `whr/playerday.py`**

Replace `won_game_terms` and `lost_game_terms` bodies to drop the `is_first_day` blocks:

```python
    def won_game_terms(self) -> list[list[float]]:
        if self._won_game_terms is None:
            self._won_game_terms = []
            for g in self.won_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                self._won_game_terms.append([1.0, 0.0, 1.0, other_gamma])
        return self._won_game_terms

    def lost_game_terms(self) -> list[list[float]]:
        if self._lost_game_terms is None:
            self._lost_game_terms = []
            for g in self.lost_games:
                other_gamma = g.opponents_adjusted_gamma(self.player)
                self._lost_game_terms.append([0.0, other_gamma, 1.0, other_gamma])
        return self._lost_game_terms
```

Add anchor methods (after `log_likelihood`, before `add_game`). Coulom's first-day Bradley-Terry prior, strength `k = initial_prior_wins`:

```python
    def anchor_gradient(self) -> float:
        """First-day Bradley-Terry prior gradient (Coulom's InitialPriorWins)."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return k * (1.0 - 2.0 * gamma / (1.0 + gamma))

    def anchor_hessian(self) -> float:
        """Second derivative of the first-day prior."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return -2.0 * k * gamma / ((1.0 + gamma) ** 2)

    def anchor_log_likelihood(self) -> float:
        """Log-likelihood contribution of the first-day prior."""
        k = self.player.initial_prior_wins
        gamma = self.gamma()
        return k * (math.log(gamma) - 2.0 * math.log(1.0 + gamma))
```

Update `update_by_1d_newtons_method` (single-day players — their only day is the first day, so the anchor always applies):

```python
    def update_by_1d_newtons_method(self) -> None:
        """Updates the player's rating using one-dimensional Newton's method."""
        dlogp = self.log_likelihood_derivative() + self.anchor_gradient()
        d2logp = self.log_likelihood_second_derivative() + self.anchor_hessian()
        dr = dlogp / d2logp
        self.r = self.r - dr
```

- [ ] **Step 4: Wire the anchor into `whr/player.py` gradient and hessian at index 0**

In `gradient`, add the anchor on the first day and drop the debug print:

```python
    def gradient(
        self, r: list[float], days: list[PD.PlayerDay], sigma2: list[float]
    ) -> list[float]:
        g = []
        n = len(days)
        for idx, day in enumerate(days):
            prior = 0.0
            if idx < (n - 1):
                prior += -(r[idx] - r[idx + 1]) / sigma2[idx]
            if idx > 0:
                prior += -(r[idx] - r[idx - 1]) / sigma2[idx - 1]
            term = day.log_likelihood_derivative() + prior
            if idx == 0:
                term += day.anchor_gradient()
            g.append(term)
        return g
```

In `hessian` (still `-0.001` here — replaced in Task 3), add the anchor at `diagonal[0]`:

```python
        for row in range(n):
            prior = 0.0
            if row < (n - 1):
                prior += -1 / sigma2[row]
            if row > 0:
                prior += -1 / sigma2[row - 1]
            diagonal[row] = days[row].log_likelihood_second_derivative() + prior - 0.001
        diagonal[0] += days[0].anchor_hessian()
        for i in range(n - 1):
            sub_diagonal[i] = 1 / sigma2[i]
        return (diagonal, sub_diagonal)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_robustness.py -k "anchor or compression" --no-cov -v`
Expected: PASS.

- [ ] **Step 6: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add whr/playerday.py whr/player.py tests/test_robustness.py
git commit -m "Inject first-day anchor directly (initial_prior_wins, default 0.5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Configurable Hessian damping (#4, part 1)

Replaces the `-0.001` nudge with `hessian_damping` (default 1.0, Coulom's `HessianEpsilon`), threaded through `hessian` and both call sites.

**Files:**
- Modify: `whr/player.py` (`hessian` signature + body; callers `update_by_ndim_newton`, `covariance`)
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `Player.hessian_damping` (Task 1); `PlayerDay.anchor_hessian` (Task 2).
- Produces: `Player.hessian(days, sigma2, damping)` — static, now takes a third positional `damping: float`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_robustness.py`:

```python
def test_hessian_damping_configurable_and_stable():
    for damping in (0.1, 1.0, 10.0):
        w = whole_history_rating.WHR(config={"hessian_damping": damping})
        for d in range(1, 6):
            w.create_game("a", "b", "B", d, 0)
            w.create_game("a", "b", "W", d, 0)
        w.iterate(50)
        elo, _ = w.ratings_for_player("a", current=True)
        assert math.isfinite(elo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robustness.py::test_hessian_damping_configurable_and_stable --no-cov -v`
Expected: FAIL — the config value is ignored while `hessian` hardcodes `-0.001` (the assertion may still pass; if so, this test is confirmed meaningful only after Step 3, so treat a pre-change PASS as acceptable and proceed — the behavioural link is what Step 3 establishes). To make the failure explicit, temporarily assert the wiring instead:

```python
def test_hessian_uses_damping_param():
    w = whole_history_rating.WHR(config={"hessian_damping": 5.0})
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    p = w.player_by_name("a")
    sigma2 = p.compute_sigma2()
    diag_small, _ = whole_history_rating.Player.hessian(p.days, sigma2, 0.0)
    diag_big, _ = whole_history_rating.Player.hessian(p.days, sigma2, 5.0)
    assert diag_big[1] == pytest.approx(diag_small[1] - 5.0)
```

Run: `uv run pytest tests/test_robustness.py::test_hessian_uses_damping_param --no-cov -v`
Expected: FAIL with `TypeError: hessian() takes 2 positional arguments but 3 were given`.

Note: `whole_history_rating.Player` is re-exported? If not, import `from whr.player import Player` in the test.

- [ ] **Step 3: Change `hessian` to take `damping` and update call sites**

Signature and body in `whr/player.py`:

```python
    @staticmethod
    def hessian(
        days: list[PD.PlayerDay], sigma2: list[float], damping: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n = len(days)
        diagonal = np.zeros((n,))
        sub_diagonal = np.zeros((n - 1,))
        for row in range(n):
            prior = 0.0
            if row < (n - 1):
                prior += -1 / sigma2[row]
            if row > 0:
                prior += -1 / sigma2[row - 1]
            diagonal[row] = (
                days[row].log_likelihood_second_derivative() + prior - damping
            )
        diagonal[0] += days[0].anchor_hessian()
        for i in range(n - 1):
            sub_diagonal[i] = 1 / sigma2[i]
        return (diagonal, sub_diagonal)
```

In `update_by_ndim_newton`: `diag, sub_diag = Player.hessian(self.days, sigma2, self.hessian_damping)`.
In `covariance`: `diag, sub_diag = Player.hessian(self.days, sigma2, self.hessian_damping)`.

Update the failing-test import if needed: add `from whr.player import Player` at the top of `tests/test_robustness.py` and use `Player.hessian(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_robustness.py -k "damping" --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/player.py tests/test_robustness.py
git commit -m "Replace -0.001 nudge with configurable hessian_damping (default 1.0)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Correct the public log-likelihood (#4, part 2)

Rewrites `Player.log_likelihood` to a correct log-posterior (game LL + anchor LL + Gaussian Wiener prior log-density), removes the `sys.maxsize` checks and the now-unused `import sys`, and deletes the obsolete overflow test.

**Files:**
- Modify: `whr/player.py` (`log_likelihood`, remove `import sys`)
- Modify: `tests/whr_test.py` (delete `test_log_likelihood_raises_instead_of_exiting_on_overflow`; remove `import sys` if unused)
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `PlayerDay.log_likelihood` (games only, Task 2), `PlayerDay.anchor_log_likelihood` (Task 2), `Player.compute_sigma2`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_robustness.py`:

```python
def test_player_log_likelihood_closed_form():
    w = whole_history_rating.WHR()  # initial_prior_wins == 0.5
    w.create_game("a", "b", "B", 1, 0)  # a (black) beats b on day 1
    a = w.player_by_name("a")
    b = w.player_by_name("b")
    a.days[0].set_gamma(3.0)
    b.days[0].set_gamma(1.0)
    ga, k = 3.0, 0.5
    expected_game = math.log(ga / (ga + 1.0))  # one day, opponent gamma == 1
    expected_anchor = k * (math.log(ga) - 2 * math.log(1 + ga))
    assert a.log_likelihood() == pytest.approx(expected_game + expected_anchor)


def test_total_log_likelihood_finite_and_improves():
    w = whole_history_rating.WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    w.create_game("a", "b", "B", 3, 0)
    w.iterate(1)
    start = w.log_likelihood()
    w.iterate(30)
    end = w.log_likelihood()
    assert math.isfinite(start) and math.isfinite(end)
    assert end >= start - 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_robustness.py -k "log_likelihood" --no-cov -v`
Expected: FAIL (closed-form mismatch — current formula sums densities then logs).

- [ ] **Step 3: Rewrite `Player.log_likelihood` and remove `import sys`**

Delete `import sys` (line 5). Replace the whole `log_likelihood` method:

```python
    def log_likelihood(self) -> float:
        """Log-posterior contribution of this player.

        Sum of the per-day game log-likelihoods, the first-day anchor prior,
        and the Gaussian Wiener prior log-density over consecutive days.
        """
        result = 0.0
        for day in self.days:
            result += day.log_likelihood()
        if self.days:
            result += self.days[0].anchor_log_likelihood()
        sigma2 = self.compute_sigma2()
        for i, s2 in enumerate(sigma2):
            rd = self.days[i + 1].r - self.days[i].r
            result += -(rd**2) / (2 * s2) - 0.5 * math.log(2 * math.pi * s2)
        return result
```

- [ ] **Step 4: Delete the obsolete overflow test**

In `tests/whr_test.py`, delete `test_log_likelihood_raises_instead_of_exiting_on_overflow` entirely (its premise — a `sys.maxsize` guard in `log_likelihood` — is gone; the non-finite safety net moves to `update_by_ndim_newton` in Task 5). If `import sys` in `tests/whr_test.py` is now unused, remove it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_robustness.py -k "log_likelihood" --no-cov -v`
Expected: PASS.

- [ ] **Step 6: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean (no unused `sys` import).

- [ ] **Step 7: Commit**

```bash
git add whr/player.py tests/whr_test.py tests/test_robustness.py
git commit -m "Fix public log_likelihood; drop sys.maxsize guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Remove the elo cap, add a non-finite safety net (#6)

Deletes the `if candidate > 650: raise` guard; `UnstableRatingException` now fires only on a non-finite result.

**Files:**
- Modify: `whr/player.py` (`update_by_ndim_newton`)
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `Player.hessian_damping` (Task 3), anchor (Task 2). `UnstableRatingException` from `whr.utils`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_robustness.py`:

```python
def test_undefeated_player_is_finite_no_exception():
    w = whole_history_rating.WHR()
    for d in range(1, 11):
        w.create_game("winner", "loser", "B", d, 0)  # winner (black) always wins
    w.iterate(100)  # must NOT raise
    elo, unc = w.ratings_for_player("winner", current=True)
    assert math.isfinite(elo) and math.isfinite(unc)


def test_non_finite_step_raises(monkeypatch):
    from whr import playerday

    w = whole_history_rating.WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 2, 0)
    monkeypatch.setattr(
        playerday.PlayerDay, "log_likelihood_derivative", lambda self: float("nan")
    )
    with pytest.raises(utils.UnstableRatingException):
        w.iterate(1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_robustness.py -k "undefeated or non_finite" --no-cov -v`
Expected: `test_undefeated...` FAILs (raises `UnstableRatingException` via the `>650` guard); `test_non_finite_step_raises` may already pass or error differently.

- [ ] **Step 3: Replace the guard with a non-finite check**

In `update_by_ndim_newton`, replace this block:

```python
        new_r = [ri - xi for ri, xi in zip(r, x, strict=True)]

        for candidate in new_r:
            if candidate > 650:
                raise UnstableRatingException("unstable r on player")

        for idx, day in enumerate(self.days):
            day.r = float(day.r - x[idx])
```

with:

```python
        for idx, day in enumerate(self.days):
            new_r = float(day.r - x[idx])
            if not math.isfinite(new_r):
                raise UnstableRatingException(
                    f"Non-finite rating for {self.name} on day {day.day}"
                )
            day.r = new_r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_robustness.py -k "undefeated or non_finite" --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/player.py tests/test_robustness.py
git commit -m "Drop the >650 elo cap; keep only a non-finite safety net

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Gradient-norm convergence (#5)

`auto_iterate` stops on the global gradient infinity-norm.

**Files:**
- Modify: `whr/player.py` (add `gradient_infinity_norm`)
- Modify: `whr/whole_history_rating.py` (`auto_iterate`, add `max_gradient_norm`)
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `Player.gradient`, `Player.compute_sigma2`.
- Produces: `Player.gradient_infinity_norm() -> float`; `WHR.max_gradient_norm() -> float`; `WHR.auto_iterate(time_limit=None, precision=1e-3, batch_size=10) -> tuple[int, bool]` (unchanged signature; `precision` now means the max-abs gradient tolerance in natural-rating units).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_robustness.py`:

```python
def test_auto_iterate_converges_on_gradient_norm():
    w = whole_history_rating.WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
        w.create_game("a", "b", "W", d, 0)
    iterations, converged = w.auto_iterate(precision=1e-2, time_limit=10)
    assert converged is True
    assert iterations > 0
    assert w.max_gradient_norm() < 1e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robustness.py::test_auto_iterate_converges_on_gradient_norm --no-cov -v`
Expected: FAIL with `AttributeError: 'WHR' object has no attribute 'max_gradient_norm'`.

- [ ] **Step 3: Add `gradient_infinity_norm` to `whr/player.py`**

```python
    def gradient_infinity_norm(self) -> float:
        """Max absolute gradient component over this player's days."""
        if not self.days:
            return 0.0
        r = [d.r for d in self.days]
        sigma2 = self.compute_sigma2()
        return max(abs(gi) for gi in self.gradient(r, self.days, sigma2))
```

- [ ] **Step 4: Rewrite `auto_iterate` and add `max_gradient_norm` in `whr/whole_history_rating.py`**

```python
    def max_gradient_norm(self) -> float:
        """Largest gradient infinity-norm across all players (stationarity gauge)."""
        norm = 0.0
        for p in self.players.values():
            if len(p.days) > 0:
                norm = max(norm, p.gradient_infinity_norm())
        return norm

    def auto_iterate(
        self,
        time_limit: int | None = None,
        precision: float = 1e-3,
        batch_size: int = 10,
    ) -> tuple[int, bool]:
        """Iterate until the gradient infinity-norm drops below ``precision``.

        Args:
            time_limit: max seconds before giving up. None means no timeout.
            precision: convergence tolerance on the max absolute gradient
                component (natural-rating units).
            batch_size: iterations per convergence/timeout check.

        Returns:
            (iterations performed, whether convergence was reached).
        """
        start = time.time()
        i = 0
        while True:
            self.iterate(batch_size)
            i += batch_size
            if self.max_gradient_norm() < precision:
                return i, True
            if time_limit is not None and time.time() - start > time_limit:
                return i, False
```

Remove the now-unused `cast` import and the `test_stability` import from `whr.utils` in `whole_history_rating.py` **only if** they are no longer used anywhere in the file (check first; `cast` was used solely for the old delta-based `auto_iterate`).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_robustness.py::test_auto_iterate_converges_on_gradient_norm --no-cov -v`
Expected: PASS.

- [ ] **Step 6: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean (no unused imports).

- [ ] **Step 7: Commit**

```bash
git add whr/player.py whr/whole_history_rating.py tests/test_robustness.py
git commit -m "Converge auto_iterate on the gradient infinity-norm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Re-baseline and update quarantined tests

Un-skips the Task-1 quarantined tests, recomputes changed golden values, and rewrites the two semantic tests for the new behaviour. This is the task that returns the FULL suite to green with coverage.

**Files:**
- Modify: `tests/whr_test.py`

**Interfaces:**
- Consumes: all prior tasks (final behaviour).

- [ ] **Step 1: Remove all `@pytest.mark.skip(...)` markers added in Task 1**

Delete every `reason="re-baselined in phase-1 robustness plan, Task 7"` skip line.

- [ ] **Step 2: Rewrite the two semantic tests**

Handicap-600 test (was `pytest.raises(UnstableRatingException)`): the huge handicap no longer explodes. Replace the body's assertion so it iterates without raising and checks finiteness:

```python
    whr.iterate(10)  # no longer raises
    for _, elo, unc in whr.ratings_for_player("player"):
        assert math.isfinite(elo) and math.isfinite(unc)
```

Add `import math` to `tests/whr_test.py` if not present.

`test_auto_iterate`: `precision` is now a gradient-norm tolerance, so the old iteration-count/stability magic numbers are invalid. Replace with property assertions:

```python
def test_auto_iterate():
    def run(precision):
        w = whole_history_rating.WHR()
        for d in range(1, 6):
            w.create_game("a", "b", "B", d, 0)
            w.create_game("a", "b", "W", d, 0)
        return w.auto_iterate(precision=precision, batch_size=1, time_limit=10)

    it_loose, stable_loose = run(1e-1)
    it_tight, stable_tight = run(1e-3)
    assert stable_loose is True and stable_tight is True
    assert it_loose <= it_tight  # looser tolerance converges no later
```

`test_auto_iterate_returns_not_stable_on_timeout`: keep it; `time_limit=0` still returns `(batch_size, False)` because one batch cannot reach the tolerance instantly. Verify it passes; if the single batch already converges on the tiny dataset, tighten `precision` to `1e-9` in that test so convergence cannot happen in one batch.

- [ ] **Step 3: Recompute golden values for the remaining quarantined tests**

For each still-failing value assertion (ratings lists, `current=True` tuples, `player.log_likelihood()`, `day.log_likelihood()`, `whr.log_likelihood()`, uncertainty values), run the specific test to see the actual vs expected, then update the expected literal to the new computed value. Command per test:

```bash
uv run pytest "tests/whr_test.py::<test_name>" --no-cov -v
```

For each, before pasting the new number, sanity-check it: ratings remain correctly ordered (stronger player higher), all values finite, uncertainties positive. Note in a one-line comment above any re-baselined block: `# re-baselined for phase-1 (anchor 0.5, damping 1.0)`.

- [ ] **Step 4: Run the full suite with coverage**

Run: `uv run pytest`
Expected: PASS, 0 skipped, coverage ≥95%.

If coverage <95%, add targeted tests in `tests/test_robustness.py` for any uncovered new lines (e.g., `max_gradient_norm` with a player that has zero days, `gradient_infinity_norm` empty case).

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/whr_test.py tests/test_robustness.py
git commit -m "Re-baseline tests for phase-1 robustness behaviour

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Documentation (CHANGELOG + README)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md` (or `README.rst` — match the repo's existing README format)

- [ ] **Step 1: Add a CHANGELOG entry**

Under a new top section (target 2.1.0; final version number confirmed with the maintainer at release):

```markdown
## [2.1.0] - unreleased

### Changed
- **Ratings values change.** The first-day anchor now uses Coulom's
  `initial_prior_wins` strength (default 0.5 instead of an implicit 1.0),
  reducing the compression of weakly-connected players toward 0 elo.
- Newton stability now comes from a configurable `hessian_damping`
  (default 1.0, Coulom's `HessianEpsilon`) instead of a fixed `-0.001` nudge.
- `auto_iterate(precision=...)` now converges on the gradient infinity-norm
  (natural-rating units) rather than the change in ratings between batches.
- `WHR.log_likelihood()` is now a correct log-posterior (game likelihood +
  first-day prior + Gaussian Wiener prior).

### Added
- Config keys `initial_prior_wins` (default 0.5) and `hessian_damping`
  (default 1.0).

### Removed
- The `> 650` elo guard and the `sys.maxsize` log-likelihood guard.
  `UnstableRatingException` now fires only on a genuinely non-finite result;
  undefeated/isolated players converge to a finite rating via the prior.
```

- [ ] **Step 2: Document the config keys in the README**

Add the two keys to the config documentation, and note the `auto_iterate` `precision` semantic change (gradient-norm tolerance). Match existing README style.

- [ ] **Step 3: Full verification**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document phase-1 robustness changes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** #1 → Tasks 2; #4 → Tasks 3 (damping) + 4 (log-likelihood fix); #5 → Task 6; #6 → Task 5. Config exposure → Task 1. Behaviour-change docs/compat → Task 8. Test plan's 7 properties map to: undefeated-finite (T5), less-compression (T2), gradient-stationarity (T6), damping/no-overflow (T3/T5), config plumbing (T1), corrected log-likelihood (T4), anchor closed-form (T2).
- **Gaussian Wiener prior** preserved (only the anchor and damping change; `compute_sigma2` and the prior terms in `gradient`/`hessian`/`log_likelihood` keep the `sigma2 = Δdays·w2` form).
- **Type consistency:** `hessian(days, sigma2, damping)` used identically in `update_by_ndim_newton` and `covariance`; `anchor_gradient/hessian/log_likelihood`, `gradient_infinity_norm`, `max_gradient_norm` names used consistently across tasks.
- **Known unknowns handled procedurally:** exact re-baselined golden numbers are recomputed in Task 7 (they depend on the final implementation) with sanity checks, not guessed here.
