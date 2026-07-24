# WHR Uncertainty API (Phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Expose model uncertainty as review point #9: `rating_difference` (A), an uncertainty-integrated prediction option (B), and `rating_covariance`/`rating_change` (C). Additive/non-breaking.

**Architecture:** A and B use the already-stored per-day marginal variance (`PlayerDay.uncertainty`, an `r`-space variance) with the independence approximation across players. C computes the exact within-player joint covariance by inverting the player's negative tridiagonal Hessian. All new public methods report ELO.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest+pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python ≥3.11, numpy ≥2.0; `ruff check whr tests`, `ruff format --check`, `mypy` clean; `uv run pytest` passes at coverage `--cov-fail-under=95`.
- Elo conversion: `elo = r · 400/ln(10)`; variance to elo² multiplies by `K² = (400/ln 10)²`. New public methods report elo; the existing `uncertainty` field is unchanged.
- A and B use the independence approximation `Var(r_A−r_B) ≈ Var(r_A)+Var(r_B)` (WHR computes no cross-player covariance) — documented.
- C is exact within-player (inverse of the negative tridiagonal Hessian); its diagonal must match each day's stored `uncertainty`.
- Additive/non-breaking: three new methods + one opt-in parameter defaulting to current behaviour.
- Branch `feat/uncertainty-api-phase5`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: A — `rating_difference`

**Files:**
- Modify: `whr/whole_history_rating.py` (add an `_ELO_PER_NAT` module constant, a `_player_day` helper, and `rating_difference`)
- Test: `tests/test_uncertainty_api.py` (create)

**Interfaces:**
- Consumes: `_existing_player`, `PlayerDay.elo`, `PlayerDay.uncertainty` (r-variance).
- Produces: `WHR.rating_difference(name_a, name_b, day_a=None, day_b=None) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uncertainty_api.py`:

```python
import math

import pytest

from whr.whole_history_rating import WHR

_K = 400.0 / math.log(10)


def _rated(games, iters=50):
    w = WHR()
    w.load_games(games)
    w.iterate(iters)
    return w


def test_rating_difference_formula_and_ci():
    w = _rated(["a b B 1", "a b W 2", "a b B 3", "c b B 1", "c b W 2"])
    a = w.player_by_name("a").days[-1]
    b = w.player_by_name("b").days[-1]
    res = w.rating_difference("a", "b")
    assert res["difference"] == pytest.approx(a.elo - b.elo)
    expected_se = math.sqrt(a.uncertainty + b.uncertainty) * _K
    assert res["std_error"] == pytest.approx(expected_se)
    lo, hi = res["confidence_interval_95"]
    assert lo == pytest.approx(res["difference"] - 1.96 * expected_se)
    assert hi == pytest.approx(res["difference"] + 1.96 * expected_se)


def test_rating_difference_specific_days():
    w = _rated(["a b B 1", "a b W 5", "a b B 9"])
    res = w.rating_difference("a", "b", day_a=1, day_b=5)
    a1 = next(d for d in w.player_by_name("a").days if d.day == 1)
    b5 = next(d for d in w.player_by_name("b").days if d.day == 5)
    assert res["difference"] == pytest.approx(a1.elo - b5.elo)


def test_rating_difference_unknown_player_raises():
    w = _rated(["a b B 1", "a b W 2"])
    with pytest.raises(ValueError):
        w.rating_difference("a", "ghost")
    with pytest.raises(ValueError):
        w.rating_difference("a", "b", day_a=999)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_uncertainty_api.py --no-cov -v`
Expected: FAIL (`AttributeError: ... 'rating_difference'`).

- [ ] **Step 3: Implement**

In `whr/whole_history_rating.py`, add a module-level constant near the top (after imports):

```python
_ELO_PER_NAT = 400.0 / math.log(10)
```

Add to the `WHR` class:

```python
    @staticmethod
    def _player_day(player: Player, day: int | None) -> "PD.PlayerDay":
        if day is None:
            return player.days[-1]
        for d in player.days:
            if d.day == day:
                return d
        raise ValueError(f"player {player.name!r} has no rated day {day}")

    def rating_difference(
        self,
        name_a: str,
        name_b: str,
        day_a: int | None = None,
        day_b: int | None = None,
    ) -> dict[str, Any]:
        """Elo difference (a - b) between two players and its uncertainty.

        Uses each player's given day (else their last day). The difference
        variance uses the INDEPENDENCE APPROXIMATION Var(a-b) ~= Var(a)+Var(b);
        WHR computes no cross-player covariance, so this ignores the correlated
        errors of players who have faced each other. Returns
        {"difference", "std_error", "confidence_interval_95"} in elo. Raises
        ValueError for an unknown/unrated player or day.
        """
        pa = self._existing_player(name_a)
        pb = self._existing_player(name_b)
        if pa is None or not pa.days:
            raise ValueError(f"No ratings available for player {name_a!r}")
        if pb is None or not pb.days:
            raise ValueError(f"No ratings available for player {name_b!r}")
        da = self._player_day(pa, day_a)
        db = self._player_day(pb, day_b)
        if da.uncertainty < 0 or db.uncertainty < 0:
            raise ValueError("uncertainties not computed; call iterate() first")
        difference = da.elo - db.elo
        se = math.sqrt(da.uncertainty + db.uncertainty) * _ELO_PER_NAT
        return {
            "difference": difference,
            "std_error": se,
            "confidence_interval_95": (difference - 1.96 * se, difference + 1.96 * se),
        }
```

(`PD` is already imported as the playerday module alias — verify the correct alias used in the file, e.g. `from whr import playerday as PD` or via `Player`; use whatever the file already uses for the type reference, or just annotate the helper return as `Any` if no alias is imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_uncertainty_api.py --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_uncertainty_api.py
git commit -m "Add rating_difference: elo gap between players with CI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: C — `rating_covariance` and `rating_change`

**Files:**
- Modify: `whr/whole_history_rating.py` (add `import numpy as np` if not present; add both methods)
- Test: `tests/test_uncertainty_api.py`

**Interfaces:**
- Consumes: `_existing_player`, `Player.compute_sigma2`, `Player.hessian`, `Player.hessian_damping`, `PlayerDay.elo`/`.uncertainty`/`.day`.
- Produces: `WHR.rating_covariance(name) -> tuple[list[int], np.ndarray]` (elo² within-player covariance); `WHR.rating_change(name, day_from, day_to) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uncertainty_api.py`:

```python
import numpy as np


def test_rating_covariance_diagonal_matches_uncertainty():
    w = _rated(["a b B 1", "a b W 5", "a b B 9", "a b W 13"])
    days, cov = w.rating_covariance("a")
    assert days == [1, 5, 9, 13]
    p = w.player_by_name("a")
    # diagonal, converted back to r-space, matches stored per-day uncertainty
    for i, d in enumerate(p.days):
        assert cov[i, i] / (_K**2) == pytest.approx(d.uncertainty, rel=1e-6, abs=1e-9)


def test_rating_covariance_symmetric_and_psd():
    w = _rated(["a b B 1", "a b W 5", "a b B 9"])
    _, cov = w.rating_covariance("a")
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > -1e-9).all()  # positive semi-definite


def test_rating_change_uses_joint_covariance_not_marginals():
    w = _rated(["a b B 1", "a b W 5", "a b B 9", "a b W 13"], iters=60)
    p = w.player_by_name("a")
    res = w.rating_change("a", 1, 13)
    d_from = next(d for d in p.days if d.day == 1)
    d_to = next(d for d in p.days if d.day == 13)
    assert res["change"] == pytest.approx(d_to.elo - d_from.elo)
    naive_se = math.sqrt(d_from.uncertainty + d_to.uncertainty) * _K
    # consecutive days positively correlated -> joint SE strictly smaller than naive
    assert res["std_error"] < naive_se


def test_rating_covariance_and_change_errors():
    w = _rated(["a b B 1", "a b W 2"])
    with pytest.raises(ValueError):
        w.rating_covariance("ghost")
    with pytest.raises(ValueError):
        w.rating_change("a", 1, 999)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_uncertainty_api.py -k "covariance or rating_change" --no-cov -v`
Expected: FAIL (`AttributeError: ... 'rating_covariance'`).

- [ ] **Step 3: Implement**

Add `import numpy as np` to `whr/whole_history_rating.py` if not already imported. Add to the `WHR` class:

```python
    def rating_covariance(self, name: str) -> tuple[list[int], "np.ndarray"]:
        """Full within-player covariance of a player's day ratings, in elo^2.

        Returns (days, matrix) where matrix[i][j] = Cov(elo on days[i], elo on
        days[j]) — the exact inverse of the player's negative tridiagonal
        Hessian scaled to elo^2. The diagonal equals the per-day marginal
        variance. Raises ValueError for an unknown/unrated player.
        """
        player = self._existing_player(name)
        if player is None or not player.days:
            raise ValueError(f"No ratings available for player {name!r}")
        n = len(player.days)
        sigma2 = player.compute_sigma2()
        diagonal, sub_diagonal = Player.hessian(
            player.days, sigma2, player.hessian_damping
        )
        neg_h = np.zeros((n, n))
        for i in range(n):
            neg_h[i, i] = -diagonal[i]
        for i in range(n - 1):
            neg_h[i, i + 1] = -sub_diagonal[i]
            neg_h[i + 1, i] = -sub_diagonal[i]
        cov = np.linalg.inv(neg_h) * (_ELO_PER_NAT**2)
        days = [d.day for d in player.days]
        return days, cov

    def rating_change(self, name: str, day_from: int, day_to: int) -> dict[str, Any]:
        """Elo change of one player between two of their days, with uncertainty.

        Var(change) = C[to,to] + C[from,from] - 2*C[from,to] from the WITHIN-
        player covariance (exact; consecutive days are positively correlated via
        the Wiener prior, so a change is more certain than summing marginals).
        Returns {"change", "std_error", "confidence_interval_95"} in elo. Raises
        ValueError for an unknown player or an unknown day.
        """
        player = self._existing_player(name)
        if player is None or not player.days:
            raise ValueError(f"No ratings available for player {name!r}")
        days, cov = self.rating_covariance(name)
        index = {d: i for i, d in enumerate(days)}
        if day_from not in index or day_to not in index:
            raise ValueError(f"player {name!r} has no rated day {day_from} / {day_to}")
        i, j = index[day_from], index[day_to]
        change = player.days[j].elo - player.days[i].elo
        var = cov[j, j] + cov[i, i] - 2 * cov[i, j]
        se = math.sqrt(max(var, 0.0))
        return {
            "change": change,
            "std_error": se,
            "confidence_interval_95": (change - 1.96 * se, change + 1.96 * se),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_uncertainty_api.py -k "covariance or rating_change" --no-cov -v`
Expected: PASS. If `test_rating_covariance_diagonal_matches_uncertainty` fails, the pre-existing `Player.covariance()` (which feeds `uncertainty`) disagrees with the true inverse — STOP and report (do not adjust the tolerance); it means one of the two is wrong and must be reconciled.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_uncertainty_api.py
git commit -m "Add rating_covariance/rating_change: exact within-player joint covariance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: B — uncertainty-integrated prediction

**Files:**
- Modify: `whr/whole_history_rating.py` (`probability_future_match`)
- Test: `tests/test_uncertainty_api.py`

**Interfaces:**
- Modifies: `WHR.probability_future_match(name1, name2, handicap=0, handicap_key=None, komi_key=None, account_for_uncertainty=False, uncertainty_steps=4) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uncertainty_api.py`:

```python
def test_prediction_uncertainty_default_unchanged():
    w = _rated(["a b B 1", "a b B 2", "a b B 3", "a b B 4"])
    point = w.probability_future_match("a", "b")
    also = w.probability_future_match("a", "b", account_for_uncertainty=False)
    assert also == point


def test_prediction_uncertainty_hedges_toward_half():
    # few games -> high uncertainty; integrating should pull the favourite's
    # probability toward 0.5.
    w = _rated(["a b B 1", "a b B 2"], iters=50)
    p_point, _ = w.probability_future_match("a", "b")
    p_unc, _ = w.probability_future_match("a", "b", account_for_uncertainty=True)
    assert p_point > 0.5
    assert 0.5 < p_unc < p_point  # hedged toward 0.5 but same side
    # both pairs still sum to 1
    p1, p2 = w.probability_future_match("a", "b", account_for_uncertainty=True)
    assert p1 + p2 == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_uncertainty_api.py -k "prediction_uncertainty" --no-cov -v`
Expected: FAIL — `probability_future_match` has no `account_for_uncertainty` kwarg (`TypeError`).

- [ ] **Step 3: Implement**

Read the current `probability_future_match` body. Keep the existing point computation that yields `(player1_proba, player2_proba)`. Add the two parameters to the signature and, at the end (before returning), insert the integration:

```python
        # ... existing point computation producing player1_proba, player2_proba ...
        if not account_for_uncertainty:
            return player1_proba, player2_proba

        var1 = player1.days[-1].uncertainty if (player1 and player1.days) else 0.0
        var2 = player2.days[-1].uncertainty if (player2 and player2.days) else 0.0
        var1 = max(var1, 0.0)  # uncertainty is -1 before iterate()
        var2 = max(var2, 0.0)
        sigma = math.sqrt(var1 + var2)
        if sigma == 0.0:
            return player1_proba, player2_proba

        eps = 1e-12
        p1c = min(max(player1_proba, eps), 1.0 - eps)
        delta_r = math.log(p1c / (1.0 - p1c))  # logit of the point probability
        total_weight = 0.0
        integral = 0.0
        for i in range(-uncertainty_steps, uncertainty_steps + 1):
            x = 0.5 * i
            weight = math.exp(-x * x / 2.0)
            integral += weight / (1.0 + math.exp(-(delta_r + sigma * x)))
            total_weight += weight
        p1 = integral / total_weight
        return p1, 1.0 - p1
```

(Ensure `player1`/`player2` are the `_existing_player` lookups already done in the method; if the method uses different local names, adapt. The point path and unknown-player handling stay exactly as they are today.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_uncertainty_api.py -k "prediction_uncertainty" --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_uncertainty_api.py
git commit -m "Add opt-in uncertainty integration to probability_future_match

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CHANGELOG.md`, `README.md`

- [ ] **Step 1: CHANGELOG** — under `## [2.1.0] - unreleased` `### Added`: `rating_difference` (elo gap + CI between two players, independence approximation), `rating_covariance`/`rating_change` (exact within-player joint covariance — trajectory bands and significant-change tests), and an opt-in `account_for_uncertainty` on `probability_future_match` (Coulom-style integration over rating uncertainty, hedging toward 0.5 when unsure).

- [ ] **Step 2: README** — add an "Uncertainty" subsection: the difference between comparing two players (`rating_difference`, approximate — independence) and a single player's trajectory/change (`rating_covariance`/`rating_change`, exact), why differences (not absolute ratings) are the comparable quantity, and the opt-in uncertainty-aware prediction. Note all report elo, and that uncertainties require `iterate()`/`auto_iterate()` first. Short examples. Match existing README style.

- [ ] **Step 3: Verify** — `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy` (all clean; coverage ≥95%).

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document the uncertainty API (rating_difference/covariance/change)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** A `rating_difference` (Task 1); C `rating_covariance`/`rating_change` (Task 2, with the diagonal-vs-`uncertainty` consistency check that STOPS on mismatch); B opt-in integration (Task 3, default preserves behaviour); docs (Task 4). D rejected (not implemented).
- **Units:** all new methods report elo (`_ELO_PER_NAT`, `K²`); existing `uncertainty` untouched.
- **Additive/non-breaking:** verified by `test_prediction_uncertainty_default_unchanged`.
- **Independence approximation** (A, B) documented in docstrings + README; C is exact within-player.
- **Consistency guard:** Task 2 asserts C's diagonal matches the stored `uncertainty`; a mismatch is escalated, not silently tolerated.
