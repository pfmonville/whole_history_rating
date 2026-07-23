# WHR Anti-Drift (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `WHR.remove_drift()` that cancels global rating drift over time (review point #2), a faithful port of Coulom's `ComputeDrift`/`RemoveDrift`.

**Architecture:** A private `_compute_drift()` builds a padded per-day mean-strength field from the games, smooths it with a Gaussian kernel (radius `drift_kernel_radius`, default 100; sigma = radius·0.5), and returns a per-day drift in elo. `remove_drift()` shifts every `PlayerDay`'s natural rating by the negated per-day drift and returns the applied corrections. The shift is uniform per day, so within-day rating differences (and same-day win probabilities) are preserved.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest + pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python floor `>=3.11`; numpy `>=2.0`.
- `ruff check whr tests`, `ruff format --check whr tests`, and `mypy` must be clean; `uv run pytest` must pass with coverage `--cov-fail-under=95`.
- Faithful to Coulom's `~/Documents/git/WHR/src/CWHR.cpp:696-790`: per-game accumulation of `elo_black+elo_white` per day; Gaussian kernel radius = config, sigma = radius·0.5, centre half-weighted; `drift = filtered_elo/(2·filtered_count)`; nan/inf or zero-support guard → 0 correction.
- Opt-in only: `iterate`/`auto_iterate` output MUST be unchanged. `remove_drift` mutates `PlayerDay.r` in place, does NOT recompute uncertainties.
- New config key `drift_kernel_radius` default 100.
- All commits on branch `feat/antidrift-phase2`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `remove_drift()` and the drift computation

**Files:**
- Modify: `whr/whole_history_rating.py` (add `import math`; `drift_kernel_radius` default in `__init__`; add `_compute_drift` and `remove_drift`)
- Test: `tests/test_antidrift.py` (create)

**Interfaces:**
- Consumes: `self.games` (each `Game` has `bpd`/`wpd` set by `create_game`, with `.elo`), `self.players` (each `Player.days` of `PlayerDay` with `.day`, `.elo`, `.r`), `self.config["drift_kernel_radius"]`.
- Produces: `WHR._compute_drift() -> dict[int, float]` (day → drift elo over the full `[min_day, max_day]` range); `WHR.remove_drift() -> dict[int, float]` (day → applied correction elo, keyed by days that have player-days).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_antidrift.py`:

```python
import math

import pytest

from whr.whole_history_rating import WHR


def test_drift_kernel_radius_default_and_configurable():
    assert WHR().config["drift_kernel_radius"] == 100
    assert WHR(config={"drift_kernel_radius": 30}).config["drift_kernel_radius"] == 30


def test_remove_drift_empty_base_returns_empty():
    assert WHR().remove_drift() == {}


def test_remove_drift_single_day_is_finite_and_no_raise():
    w = WHR()
    w.load_games(["a b B 5"])
    w.iterate(10)
    corrections = w.remove_drift()
    assert all(math.isfinite(c) for c in corrections.values())


def test_remove_drift_return_contract():
    w = WHR()
    for d in range(1, 11):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(20)
    corrections = w.remove_drift()
    day_set = {pd.day for p in w.players.values() for pd in p.days}
    assert set(corrections) == day_set
    assert all(
        isinstance(k, int) and isinstance(v, float) and math.isfinite(v)
        for k, v in corrections.items()
    )


def test_remove_drift_preserves_same_day_win_probability():
    w = WHR()
    w.load_games(["a b B 1", "a b W 2", "c b B 2", "a b B 3"])
    w.iterate(50)
    game = w.games[2]  # c vs b on day 2 — both players share that day
    before = game.white_win_probability()
    w.remove_drift()
    after = game.white_win_probability()
    assert after == pytest.approx(before, abs=1e-9)


def test_remove_drift_cancels_linear_drift():
    # One fresh, independent matchup per day for 300 days; inject a linear
    # drift by setting every player-day's elo equal to its day number, so the
    # mean strength on day d is exactly d. Symmetric Gaussian smoothing of a
    # linear field returns the centre value, so fully-interior days must be
    # recentred to ~0 after remove_drift.
    w = WHR()
    for d in range(1, 301):
        w.create_game(f"b{d}", f"w{d}", "B", d, 0)
    for game in w.games:
        assert game.bpd is not None and game.wpd is not None
        game.bpd.elo = float(game.day)
        game.wpd.elo = float(game.day)
    corrections = w.remove_drift()
    assert all(math.isfinite(c) for c in corrections.values())
    for d in (120, 150, 180):  # fully interior (>100 from both ends)
        elos = [pd.elo for p in w.players.values() for pd in p.days if pd.day == d]
        assert elos
        for elo in elos:
            assert abs(elo) < 1.0  # was ~d, now recentred to ~0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_antidrift.py --no-cov -v`
Expected: FAIL — `test_drift_kernel_radius_default_and_configurable` with `KeyError: 'drift_kernel_radius'`, the rest with `AttributeError: 'WHR' object has no attribute 'remove_drift'`.

- [ ] **Step 3: Add the config default**

In `whr/whole_history_rating.py`, add `import math` to the imports, and in `WHR.__init__` after the existing `setdefault` calls:

```python
        self.config.setdefault("drift_kernel_radius", 100)
```

- [ ] **Step 4: Implement `_compute_drift` and `remove_drift`**

Add these two methods to the `WHR` class (e.g. after `auto_iterate`):

```python
    def _compute_drift(self) -> dict[int, float]:
        """Smoothed per-day drift in elo (Coulom's ComputeDrift).

        For every game, accumulate ``elo_black + elo_white`` and a game count
        on its day; convolve both with a Gaussian kernel (radius
        ``drift_kernel_radius``, sigma = radius * 0.5, centre half-weighted);
        the per-day drift is ``filtered_elo / (2 * filtered_count)``. Days with
        no smoothing support (or a non-finite result) get a 0 drift.
        """
        if not self.games:
            return {}
        days = [g.day for g in self.games]
        min_day, max_day = min(days), max(days)
        n = max_day - min_day + 1
        radius = self.config["drift_kernel_radius"]

        total_elo = [0.0] * (n + 2 * radius)
        game_count = [0.0] * (n + 2 * radius)
        for g in self.games:
            if g.bpd is None or g.wpd is None:
                continue
            j = g.day - min_day + radius
            total_elo[j] += g.bpd.elo + g.wpd.elo
            game_count[j] += 1.0

        sigma = radius * 0.5
        kernel = [0.0] * radius
        total = 1.0
        for k in range(1, radius):
            x = math.exp(-(k * k) / (2.0 * sigma * sigma))
            kernel[k] = x
            total += 2.0 * x
        norm = 1.0 / total
        kernel[0] = norm * 0.5
        for k in range(1, radius):
            kernel[k] *= norm

        drift: dict[int, float] = {}
        for i in range(n):
            j = i + radius
            filtered_elo = 0.0
            filtered_count = 0.0
            for k in range(radius):
                filtered_elo += (total_elo[j + k] + total_elo[j - k]) * kernel[k]
                filtered_count += (game_count[j + k] + game_count[j - k]) * kernel[k]
            if filtered_count > 0:
                d = filtered_elo / (2.0 * filtered_count)
                drift[min_day + i] = d if math.isfinite(d) else 0.0
            else:
                drift[min_day + i] = 0.0
        return drift

    def remove_drift(self) -> dict[int, float]:
        """Cancel global rating drift over time (Coulom's RemoveDrift).

        Call after iterate()/auto_iterate(). Shifts every player-day's rating by
        the negated smoothed per-day drift so the average player strength per day
        is recentred near 0 elo, making ratings comparable across eras. Mutates
        the stored ratings in place and returns the applied per-day corrections
        ({day: correction_elo}). Because the shift is uniform per day, within-day
        rating differences (hence same-day win probabilities) are unchanged.
        Uncertainties are not recomputed.
        """
        drift = self._compute_drift()
        factor = math.log(10) / 400.0
        applied: dict[int, float] = {}
        for player in self.players.values():
            for pd in player.days:
                correction_elo = -drift.get(pd.day, 0.0)
                pd.r += correction_elo * factor
                applied[pd.day] = correction_elo
        return applied
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_antidrift.py --no-cov -v`
Expected: PASS (all 6).

- [ ] **Step 6: Full suite + lint + types**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: all pass, coverage ≥95%. (The default `iterate`/`auto_iterate` tests are untouched, confirming opt-in / non-breaking.) If coverage dipped, add a targeted test for any uncovered branch (e.g. the `filtered_count == 0` guard via a `drift_kernel_radius` small enough to leave an unsupported day, or the non-finite guard).

- [ ] **Step 7: Commit**

```bash
git add whr/whole_history_rating.py tests/test_antidrift.py
git commit -m "Add opt-in remove_drift() (Coulom anti-drift)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Documentation

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`

- [ ] **Step 1: CHANGELOG entry**

Under the existing `## [2.1.0] - unreleased` section's `### Added` list (create the list if absent), add:

```markdown
- `WHR.remove_drift()`: opt-in anti-inflation step (faithful port of Coulom's
  `RemoveDrift`). Call it after convergence to cancel global rating drift over
  time so ratings are comparable across eras; it mutates ratings in place,
  returns the per-day elo corrections, and preserves same-day win
  probabilities. New config key `drift_kernel_radius` (default 100).
```

- [ ] **Step 2: README section**

Add a short subsection (match the existing README.md style, e.g. near the "Optional Configuration" / iteration sections) documenting `remove_drift()`:
- What it does (recentre per-day mean strength to cancel drift/inflation across eras).
- When to call it (after `iterate`/`auto_iterate`).
- That it is opt-in and does not change `iterate` output; it mutates ratings and returns `{day: correction_elo}`; within-day win probabilities are unchanged; uncertainties are not recomputed.
- The `drift_kernel_radius` config key (default 100), shown with the same `WHR({'drift_kernel_radius': ...})` example format as the other config keys.

Include a minimal usage example:

```python
whr = WHR()
whr.load_games([...])
whr.auto_iterate()
corrections = whr.remove_drift()  # optional, after convergence
```

- [ ] **Step 3: Verify**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document remove_drift() and drift_kernel_radius

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** opt-in `remove_drift()` (Task 1); faithful ComputeDrift/RemoveDrift math incl. kernel, convolution, ÷2, guards (Task 1 Step 4); `drift_kernel_radius` config (Task 1 Step 3); non-breaking / within-day invariance / degenerate cases / return contract (Task 1 tests); docs (Task 2). Uncertainties-not-recomputed and auto-application are explicitly out of scope per spec.
- **Type safety:** `_compute_drift` guards `g.bpd`/`g.wpd` for None (mypy-clean) and defensively skips unattached games.
- **No placeholder:** full implementation and full test code included.
