# WHR Fit `w2` by Temporal CV (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `WHR.fit_w2()` that picks the `w2` hyperparameter by temporal expanding-window cross-validated predictive log-loss (review point #7).

**Architecture:** `_temporal_folds(n_splits)` splits the games (by distinct day) into `n_splits` expanding-window (train, test) folds. `fit_w2` trains a fresh `WHR` (this instance's config, each candidate `w2`) on each fold's train games, scores pooled predictive log-loss on the test games via `_predict_black_win_probability`, and returns the best candidate. Pure query — the instance is not mutated. Additive and non-breaking.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest+pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python floor `>=3.11`; numpy `>=2.0`.
- `ruff check whr tests`, `ruff format --check whr tests`, `mypy` clean; `uv run pytest` passes at coverage `--cov-fail-under=95`.
- Temporal only (no random k-fold — it leaks future info). Training days strictly precede test days in every fold.
- Prediction reuses the Bradley-Terry model with the trained `handicap_gamma`/`komi_gamma`; cold-start test games (a player absent from train) are skipped and counted.
- `fit_w2` MUST NOT mutate the instance (config, players, ratings) — pure query.
- Additive: `w2` default stays 300; nothing changes unless the user acts on the result.
- Branch `feat/fit-w2-phase4`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `_temporal_folds` (and `_game_descriptions`)

**Files:**
- Modify: `whr/whole_history_rating.py` (add `import math` is already present; add the two helpers)
- Test: `tests/test_fit_w2.py` (create)

**Interfaces:**
- Produces: `WHR._game_descriptions() -> list[tuple]` (per game: `(black_name, white_name, winner, day, handicap, extras_copy)`); `WHR._temporal_folds(n_splits: int) -> list[tuple[list, list]]` (expanding-window (train, test) description lists, split on distinct days).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fit_w2.py`:

```python
import pytest

from whr.whole_history_rating import WHR


def _linear_history(n_days):
    w = WHR()
    for d in range(1, n_days + 1):
        w.create_game("a", "b", "B", d, 0)
    return w


def test_temporal_folds_are_expanding_and_leak_free():
    w = _linear_history(12)
    folds = w._temporal_folds(3)
    assert len(folds) == 3
    prev_train = -1
    for train, test in folds:
        assert train and test
        max_train_day = max(d[3] for d in train)
        min_test_day = min(d[3] for d in test)
        assert max_train_day < min_test_day  # no future leakage, no same-day split
        assert len(train) > prev_train  # expanding window
        prev_train = len(train)


def test_temporal_folds_cover_later_games_and_copy_extras():
    w = WHR()
    for d in range(1, 7):
        w.create_game("a", "b", "B", d, 0, {"komi": 6.5})
    folds = w._temporal_folds(2)
    # extras are copies, not the live Game dict
    train0 = folds[0][0]
    assert train0[0][5] == {"komi": 6.5}
    train0[0][5]["komi"] = 999
    assert w.games[0].extras["komi"] == 6.5  # original untouched


def test_temporal_folds_raise_when_too_few_distinct_days():
    w = _linear_history(2)
    with pytest.raises(ValueError):
        w._temporal_folds(5)  # needs >= 6 distinct days


def test_temporal_folds_rejects_bad_n_splits():
    w = _linear_history(10)
    with pytest.raises(ValueError):
        w._temporal_folds(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fit_w2.py --no-cov -v`
Expected: FAIL (`AttributeError: 'WHR' object has no attribute '_temporal_folds'`).

- [ ] **Step 3: Implement the helpers**

Add to the `WHR` class in `whr/whole_history_rating.py`:

```python
    def _game_descriptions(self) -> list[tuple[str, str, str, int, float, dict[str, Any]]]:
        """Replayable (black, white, winner, day, handicap, extras) tuples.

        ``extras`` is copied so replaying into a fresh WHR cannot mutate this
        instance's game state.
        """
        return [
            (
                g.black_player.name,
                g.white_player.name,
                g.winner,
                g.day,
                g.handicap,
                dict(g.extras),
            )
            for g in self.games
        ]

    def _temporal_folds(
        self, n_splits: int
    ) -> list[tuple[list[tuple], list[tuple]]]:
        """Expanding-window temporal (train, test) folds, split on distinct days.

        Distinct days are cut into ``n_splits + 1`` contiguous groups; fold ``i``
        trains on the first ``i`` groups and tests on group ``i`` (1-indexed), so
        every train day is strictly earlier than every test day.
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        descs = self._game_descriptions()
        days = sorted({d[3] for d in descs})
        if len(days) < n_splits + 1:
            raise ValueError(
                f"need at least {n_splits + 1} distinct days for "
                f"n_splits={n_splits}, got {len(days)}"
            )
        n_days = len(days)
        cuts = [round(n_days * i / (n_splits + 1)) for i in range(n_splits + 2)]
        day_index = {day: idx for idx, day in enumerate(days)}
        folds = []
        for i in range(1, n_splits + 1):
            train_end = cuts[i]
            test_end = cuts[i + 1]
            train = [d for d in descs if day_index[d[3]] < train_end]
            test = [d for d in descs if train_end <= day_index[d[3]] < test_end]
            folds.append((train, test))
        return folds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fit_w2.py --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_fit_w2.py
git commit -m "Add temporal expanding-window fold splitting for w2 CV

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_predict_black_win_probability` and `fit_w2`

**Files:**
- Modify: `whr/whole_history_rating.py`
- Test: `tests/test_fit_w2.py`

**Interfaces:**
- Consumes: `_temporal_folds` (Task 1), `_existing_player`, `create_game`, `iterate`, `handicap_gamma`/`komi_gamma`.
- Produces: `WHR._predict_black_win_probability(black_name, white_name, handicap, komi) -> float | None` (None on cold-start); `WHR.fit_w2(candidates=None, n_splits=5, iterations=50) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_w2.py`:

```python
import math
import random


def _round_robin_drifting_history(rng, n_players=8, n_days=12, drift=40.0):
    """Players whose true elo ramps over time; outcomes sampled from the true
    Bradley-Terry probability. Moderate drift => a middle w2 should predict
    best (tiny w2 can't track the drift, huge w2 overfits noise)."""
    w = WHR()
    names = [f"p{i}" for i in range(n_players)]
    base = {n: (i - n_players / 2) * 60.0 for i, n in enumerate(names)}
    for day in range(1, n_days + 1):
        true_elo = {n: base[n] + drift * day * (1 if i % 2 else -1)
                    for i, n in enumerate(names)}
        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    continue
                black, white = names[i], names[j]
                pb = 1.0 / (1.0 + 10 ** ((true_elo[white] - true_elo[black]) / 400.0))
                winner = "B" if rng.random() < pb else "W"
                w.create_game(black, white, winner, day, 0)
    return w


def test_fit_w2_prefers_middle_over_extremes_on_drifting_data():
    rng = random.Random(1234)
    w = _round_robin_drifting_history(rng)
    result = w.fit_w2(candidates=[1.0, 300.0, 100000.0], n_splits=2, iterations=25)
    ll = result["log_loss"]
    assert ll[300.0] < ll[1.0]
    assert ll[300.0] < ll[100000.0]
    assert result["best_w2"] == 300.0


def test_fit_w2_is_a_pure_query():
    w = _linear_history(8)
    w.iterate(5)
    before_w2 = w.config["w2"]
    before = w.ratings_for_player("a")
    w.fit_w2(candidates=[100.0, 300.0], n_splits=2, iterations=10)
    assert w.config["w2"] == before_w2
    assert w.ratings_for_player("a") == before


def test_fit_w2_skips_cold_start_test_games():
    # "newbie" appears only in the final period (test block) -> its games are skipped.
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
    w.create_game("newbie", "a", "B", 6, 0)
    result = w.fit_w2(candidates=[300.0], n_splits=1, iterations=10)
    assert result["n_test_skipped"] >= 1


def test_fit_w2_return_contract():
    w = _linear_history(10)
    result = w.fit_w2(candidates=[100.0, 300.0], n_splits=2, iterations=10)
    assert set(result) == {"best_w2", "log_loss", "n_splits", "n_test_scored", "n_test_skipped"}
    assert result["best_w2"] in (100.0, 300.0)
    assert all(math.isfinite(v) for v in result["log_loss"].values())
    assert result["n_splits"] == 2


def test_fit_w2_raises_on_single_day():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    w.create_game("a", "b", "W", 1, 0)
    with pytest.raises(ValueError):
        w.fit_w2(n_splits=5)


def test_predict_black_win_probability_cold_start_is_none():
    w = _linear_history(4)
    w.iterate(5)
    assert w._predict_black_win_probability("a", "ghost", 0, 6.5) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fit_w2.py -k "fit_w2 or predict_black" --no-cov -v`
Expected: FAIL (`AttributeError: ... 'fit_w2'`).

- [ ] **Step 3: Implement the prediction helper and `fit_w2`**

Add to the `WHR` class:

```python
    def _predict_black_win_probability(
        self, black_name: str, white_name: str, handicap: float, komi: Any
    ) -> float | None:
        """P(black wins) from the current ratings, or None if either player is
        unknown / unrated (cold start)."""
        black = self._existing_player(black_name)
        white = self._existing_player(white_name)
        if black is None or white is None or not black.days or not white.days:
            return None
        gb = black.days[-1].gamma() * self.handicap_gamma.get(handicap, 1.0)
        gw = white.days[-1].gamma() * self.komi_gamma.get(komi, 1.0)
        return gb / (gb + gw)

    def fit_w2(
        self,
        candidates: list[float] | None = None,
        n_splits: int = 5,
        iterations: int = 50,
    ) -> dict[str, Any]:
        """Pick w2 by temporal expanding-window CV predictive log-loss.

        Trains a fresh model (this instance's config, each candidate w2) on each
        fold's earlier games and scores pooled predictive log-loss on the fold's
        held-out later games. Does NOT mutate this instance. See the design spec
        for details. Raises ValueError if a temporal split is impossible.
        """
        if candidates is None:
            candidates = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
        folds = self._temporal_folds(n_splits)
        eps = 1e-15
        log_loss: dict[float, float] = {}
        n_scored = 0
        n_skipped = 0
        for w2 in candidates:
            sub_config = {**self.config, "w2": w2}
            total = 0.0
            scored = 0
            skipped = 0
            for train, test in folds:
                model = WHR(sub_config)
                for black, white, winner, day, handicap, extras in train:
                    model.create_game(black, white, winner, day, handicap, extras)
                model.iterate(iterations)
                for black, white, winner, day, handicap, extras in test:
                    komi = extras.get("komi", 6.5)
                    p_black = model._predict_black_win_probability(
                        black, white, handicap, komi
                    )
                    if p_black is None:
                        skipped += 1
                        continue
                    p_actual = p_black if winner == "B" else 1.0 - p_black
                    p_actual = min(max(p_actual, eps), 1.0 - eps)
                    total += -math.log(p_actual)
                    scored += 1
            log_loss[w2] = total / scored if scored else float("inf")
            n_scored, n_skipped = scored, skipped
        best_w2 = min(candidates, key=lambda w: log_loss[w])
        return {
            "best_w2": best_w2,
            "log_loss": log_loss,
            "n_splits": n_splits,
            "n_test_scored": n_scored,
            "n_test_skipped": n_skipped,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fit_w2.py --no-cov -v`
Expected: PASS. If `test_fit_w2_prefers_middle_over_extremes_on_drifting_data` is flaky or fails, the signal is too weak — increase the dataset (more players/days) or the drift so a middle w2 clearly wins, but do NOT weaken the assertion into a tautology; if it cannot be made to hold, STOP and report (it would mean the CV isn't discriminating w2, which is the whole point).

- [ ] **Step 5: Lint & types**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_fit_w2.py
git commit -m "Add WHR.fit_w2: temporal cross-validated w2 selection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Documentation

**Files:**
- Modify: `CHANGELOG.md`, `README.md`

- [ ] **Step 1: CHANGELOG** — under `## [2.1.0] - unreleased` `### Added`: `WHR.fit_w2()` — selects `w2` by temporal expanding-window cross-validated predictive log-loss (train on past games, predict future); a pure query returning `{best_w2, log_loss, ...}` without mutating the instance; the user applies the chosen `w2` by constructing/iterating with it.

- [ ] **Step 2: README** — add a subsection "Choosing `w2` from data": what `w2` controls (rating volatility over time), that `fit_w2()` scores candidates by out-of-sample predictive log-loss on a temporal split (no future leakage), the `candidates`/`n_splits`/`iterations` params, that it is a pure query (apply the result yourself, e.g. `WHR({'w2': result['best_w2']})`), and a cost caveat (`candidates × n_splits × iterations` model fits — expensive on large histories until vectorisation). Short example. Match existing README style.

- [ ] **Step 3: Verify** — `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy` (all clean; coverage ≥95%).

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document WHR.fit_w2 (temporal cross-validated w2 selection)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** temporal expanding-window folds (Task 1); pooled predictive log-loss + grid + fresh-model training + cold-start skip + pure query + degenerate raises + return contract (Task 2); docs incl. cost caveat (Task 3). Random k-fold intentionally absent (leakage).
- **Purity:** `fit_w2` builds fresh `WHR` sub-models and never touches `self` — asserted by `test_fit_w2_is_a_pure_query`.
- **No golden-test churn:** additive; existing tests untouched, so no quarantine/re-baseline needed this phase.
- **Type safety:** `_predict_black_win_probability` returns `float | None`; extras copied in `_game_descriptions`.
- **Known cost:** documented; the flaky-risk recovery test is flagged with explicit "strengthen the signal, don't weaken the assertion" guidance.
