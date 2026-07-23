# WHR Estimated Handicap & Komi (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Estimate handicap and komi advantages as Bradley-Terry parameters co-estimated with players (review point #3), faithful to Coulom's `NewtonKomiHandicap`, with the option to pin known values.

**Architecture:** `WHR` owns two gamma tables `handicap_gamma` (boosts black) and `komi_gamma` (boosts white), initialised from pinned config (+ a pinned `handicap[0]=1` baseline) and grown to 1.0 for new keys as games arrive. `Game` holds references to these shared tables; `opponents_adjusted_gamma` folds them in. A new `_newton_handicap_komi()` step runs each iteration to Newton-update the non-pinned keys.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest+pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python floor `>=3.11`; numpy `>=2.0`.
- `ruff check whr tests`, `ruff format --check whr tests`, `mypy` clean; `uv run pytest` passes with coverage `--cov-fail-under=95`.
- Faithful to Coulom (`~/Documents/git/WHR/src/CWHR.cpp:112-127,228-314`): `P(black)=γ_b·γ_h/(γ_b·γ_h+γ_w·γ_k)`; per-key Newton update `G=wins−γ·grad`, `H=−γ·hess−hessian_damping`, `γ*=exp(−G/H)`, applied only when the key has games and `0<wins<games` and is not pinned.
- The Gaussian Wiener prior and the phase-1/2 mechanics stay intact.
- New config: `pinned_handicap: dict={}`, `pinned_komi: dict={}`, `estimate_handicap_zero: bool=False`. Pins are elo, converted `γ=10^(elo/400)`. `handicap[0]` pinned to `γ=1` unless `estimate_handicap_zero` or overridden in `pinned_handicap`.
- Breaking: `handicap` becomes a category key (was elo constant); old behaviour = pin it.
- During quarantine (Tasks 1–4) run the full suite with `uv run pytest --no-cov`; the 95% floor is re-enforced in Task 5. Targeted runs use `--no-cov`.
- Branch `feat/handicap-komi-phase3`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Config, gamma tables, and quarantine

**Files:**
- Modify: `whr/whole_history_rating.py` (`WHR.__init__`: config + tables; add `_ensure_advantage_keys`)
- Modify: `tests/whr_test.py` (quarantine value-assertion golden tests)
- Test: `tests/test_handicap_komi.py` (create)

**Interfaces:**
- Produces: `WHR.handicap_gamma: dict`, `WHR.komi_gamma: dict`, `WHR._pinned_handicap_keys: set`, `WHR._pinned_komi_keys: set`, `WHR._ensure_advantage_keys(handicap, komi) -> None`; config keys `pinned_handicap`/`pinned_komi`/`estimate_handicap_zero`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handicap_komi.py`:

```python
import math

import pytest

from whr.whole_history_rating import WHR


def test_advantage_config_defaults():
    w = WHR()
    assert w.config["pinned_handicap"] == {}
    assert w.config["pinned_komi"] == {}
    assert w.config["estimate_handicap_zero"] is False


def test_handicap_zero_baseline_pinned_by_default():
    w = WHR()
    assert w.handicap_gamma[0] == 1.0
    assert 0 in w._pinned_handicap_keys


def test_estimate_handicap_zero_unpins_baseline():
    w = WHR(config={"estimate_handicap_zero": True})
    assert 0 not in w._pinned_handicap_keys


def test_pinned_values_converted_to_gamma_and_marked():
    w = WHR(config={"pinned_handicap": {2: 200.0}, "pinned_komi": {6.5: -30.0}})
    assert w.handicap_gamma[2] == pytest.approx(10 ** (200.0 / 400.0))
    assert w.komi_gamma[6.5] == pytest.approx(10 ** (-30.0 / 400.0))
    assert 2 in w._pinned_handicap_keys and 6.5 in w._pinned_komi_keys


def test_config_not_mutated():
    src = {"pinned_handicap": {2: 200.0}}
    WHR(config=src)
    assert src == {"pinned_handicap": {2: 200.0}}


def test_ensure_advantage_keys_grows_tables_to_one():
    w = WHR()
    w._ensure_advantage_keys(3, 7.5)
    assert w.handicap_gamma[3] == 1.0
    assert w.komi_gamma[7.5] == 1.0
    # does not overwrite an existing (pinned) key
    w2 = WHR(config={"pinned_handicap": {3: 100.0}})
    before = w2.handicap_gamma[3]
    w2._ensure_advantage_keys(3, 6.5)
    assert w2.handicap_gamma[3] == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_handicap_komi.py --no-cov -v`
Expected: FAIL (`KeyError`/`AttributeError` — config keys and tables don't exist yet).

- [ ] **Step 3: Add config and tables in `WHR.__init__`**

In `whr/whole_history_rating.py`, `WHR.__init__`, after the existing `setdefault`s (and with `import math` already present from phase 2):

```python
        self.config.setdefault("pinned_handicap", {})
        self.config.setdefault("pinned_komi", {})
        self.config.setdefault("estimate_handicap_zero", False)
        self.games: list[Game] = []
        self.players: dict[str, Player] = {}
        self.handicap_gamma: dict[Any, float] = {}
        self.komi_gamma: dict[Any, float] = {}
        self._pinned_handicap_keys: set[Any] = set()
        self._pinned_komi_keys: set[Any] = set()
        for key, elo in self.config["pinned_handicap"].items():
            self.handicap_gamma[key] = 10 ** (elo / 400.0)
            self._pinned_handicap_keys.add(key)
        for key, elo in self.config["pinned_komi"].items():
            self.komi_gamma[key] = 10 ** (elo / 400.0)
            self._pinned_komi_keys.add(key)
        if not self.config["estimate_handicap_zero"] and 0 not in self.handicap_gamma:
            self.handicap_gamma[0] = 1.0
            self._pinned_handicap_keys.add(0)
```

(Keep the existing `self.games`/`self.players` initialisation — shown here for placement; do not duplicate.) Add the method:

```python
    def _ensure_advantage_keys(self, handicap: Any, komi: Any) -> None:
        """Ensure the advantage tables have an entry (default gamma 1.0) for a
        game's handicap and komi keys, without overwriting existing/pinned ones."""
        if handicap not in self.handicap_gamma:
            self.handicap_gamma[handicap] = 1.0
        if komi not in self.komi_gamma:
            self.komi_gamma[komi] = 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_handicap_komi.py --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Quarantine value-assertion golden tests**

In `tests/whr_test.py`, add `@pytest.mark.skip(reason="re-baselined in phase-3 handicap/komi plan, Task 5")` above each of these (they assert hardcoded ratings/log-likelihood/print output that changes once komi is estimated): `test_output`, `test_output2`, `test_log_likelihood`, `test_loading_several_games_at_once`. Also quarantine `test_large_handicap_converges_to_finite_ratings` (handicap 600 changes from a +600-elo constant to an unlearned category) and `test_auto_iterate` (convergence dynamics shift). Do NOT quarantine save/load, config, error, deprecation, or the `test_antidrift.py` tests. `pytest` is already imported.

- [ ] **Step 6: Full suite (quarantined) + lint + types**

Run: `uv run pytest --no-cov && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS with skips; lint/types clean.

- [ ] **Step 7: Commit**

```bash
git add whr/whole_history_rating.py tests/whr_test.py tests/test_handicap_komi.py
git commit -m "Add handicap/komi advantage tables, config, and quarantine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Game wiring and `opponents_adjusted_gamma` rework

**Files:**
- Modify: `whr/game.py` (`Game.__init__` gains table refs; rewrite `opponents_adjusted_gamma`; drop `import sys`, add `import math`)
- Modify: `whr/whole_history_rating.py` (`_setup_game`/`create_game` pass tables and ensure keys)
- Test: `tests/test_handicap_komi.py`

**Interfaces:**
- Consumes: `WHR.handicap_gamma`/`komi_gamma`, `WHR._ensure_advantage_keys` (Task 1).
- Produces: `Game(black, white, winner, time_step, handicap=0, extras=None, handicap_gamma=None, komi_gamma=None)`; reworked `Game.opponents_adjusted_gamma`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_handicap_komi.py`:

```python
def test_pinned_handicap_reproduces_legacy_elo_behaviour():
    # A handicap pinned to E elo must behave exactly like the old fixed-elo
    # handicap: black's win prob equals the equal-komi Bradley-Terry value with
    # black boosted by E elo.
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    w.create_game("black", "white", "B", 1, 2)  # handicap key 2, pinned to +200
    w.player_by_name("black").days[0].elo = 0.0
    w.player_by_name("white").days[0].elo = 0.0
    game = w.games[0]
    # black gamma boosted by 200 elo vs white (komi default gamma == 1)
    gb = 10 ** (200.0 / 400.0)
    assert game.white_win_probability() == pytest.approx(1.0 / (1.0 + gb))


def test_handicap_zero_default_komi_game_has_no_adjustment():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    game = w.games[0]
    w.player_by_name("a").days[0].elo = 0.0
    w.player_by_name("b").days[0].elo = 0.0
    # γ_h[0]=1 (baseline), γ_k[6.5]=1 (init) -> even game
    assert game.white_win_probability() == pytest.approx(0.5)


def test_direct_game_without_tables_treats_advantages_as_one(monkeypatch):
    from whr.player import Player
    from whr import playerday

    cfg = {
        "debug": False, "w2": 300.0, "uncased": False,
        "initial_prior_wins": 0.5, "hessian_damping": 1.0,
    }
    b = Player("b", {**cfg}); wp = Player("w", {**cfg})
    game = __import__("whr.game", fromlist=["Game"]).Game(b, wp, "B", 1, 5)
    game.bpd = playerday.PlayerDay(b, 1); game.wpd = playerday.PlayerDay(wp, 1)
    game.bpd.elo = 0.0; game.wpd.elo = 0.0
    # No tables -> handicap/komi treated as gamma 1 -> even
    assert game.white_win_probability() == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_handicap_komi.py -k "legacy or no_adjustment or without_tables" --no-cov -v`
Expected: FAIL — the current `opponents_adjusted_gamma` still uses the elo-constant handicap and ignores komi, so `test_handicap_zero_default_komi_game_has_no_adjustment` may pass but the legacy-pinning and no-tables tests fail (Game has no `handicap_gamma`/`komi_gamma`).

- [ ] **Step 3: Rework `whr/game.py`**

Replace `import sys` with `import math` at the top. Change the constructor signature and store the tables:

```python
    def __init__(
        self,
        black: P.Player,
        white: P.Player,
        winner: str,
        time_step: int,
        handicap: float = 0,
        extras: dict[str, Any] | None = None,
        handicap_gamma: dict[Any, float] | None = None,
        komi_gamma: dict[Any, float] | None = None,
    ):
        self.day = time_step
        self.white_player = white
        self.black_player = black
        self.winner = winner.upper()
        self.handicap = handicap
        self.handicap_gamma = handicap_gamma
        self.komi_gamma = komi_gamma
        self.bpd: PD.PlayerDay | None = None
        self.wpd: PD.PlayerDay | None = None
        if extras is None:
            self.extras = {"komi": 6.5}
        else:
            self.extras = extras
            self.extras.setdefault("komi", 6.5)
```

Rewrite `opponents_adjusted_gamma`:

```python
    def opponents_adjusted_gamma(self, player: P.Player) -> float:
        """Opponent's gamma folding in the handicap/komi advantages.

        With handicap boosting black (γ_h) and komi boosting white (γ_k):
        the opponent of white is black with effective gamma γ_b·γ_h/γ_k, and
        the opponent of black is white with effective gamma γ_w·γ_k/γ_h. When
        the tables are absent (direct construction) advantages are 1.
        """
        if self.bpd is None or self.wpd is None:
            raise AttributeError("black player day and white player day must be set")
        gh = 1.0 if self.handicap_gamma is None else self.handicap_gamma.get(self.handicap, 1.0)
        gk = 1.0 if self.komi_gamma is None else self.komi_gamma.get(self.extras["komi"], 1.0)
        if player == self.white_player:
            rval = self.bpd.gamma() * gh / gk
        elif player == self.black_player:
            rval = self.wpd.gamma() * gk / gh
        else:
            raise AttributeError(
                f"No opponent for {player.__str__()}, since they're not in this game: {self.__str__()}."
            )
        if not math.isfinite(rval) or rval <= 0:
            raise AttributeError("bad adjusted gamma")
        return rval
```

- [ ] **Step 4: Pass tables and ensure keys in `whr/whole_history_rating.py`**

In `_setup_game`, pass the base's tables into `Game`:

```python
        game = Game(
            black_player, white_player, winner, time_step, handicap, extras,
            handicap_gamma=self.handicap_gamma, komi_gamma=self.komi_gamma,
        )
```

In `create_game`, after building the game and before `_add_game`, ensure the keys exist:

```python
        game = self._setup_game(black, white, winner, time_step, handicap, extras)
        self._ensure_advantage_keys(game.handicap, game.extras["komi"])
        return self._add_game(game)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_handicap_komi.py --no-cov -v`
Expected: PASS (all).

- [ ] **Step 6: Full suite (quarantined) + lint + types**

Run: `uv run pytest --no-cov && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS with skips; clean (no unused `sys` import in game.py).

- [ ] **Step 7: Commit**

```bash
git add whr/game.py whr/whole_history_rating.py tests/test_handicap_komi.py
git commit -m "Fold estimated handicap/komi advantages into game probabilities

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `_newton_handicap_komi` estimation step

**Files:**
- Modify: `whr/whole_history_rating.py` (add `_newton_handicap_komi`; call it in `_run_one_iteration`)
- Test: `tests/test_handicap_komi.py`

**Interfaces:**
- Consumes: `WHR.handicap_gamma`/`komi_gamma`/`_pinned_*_keys`, `self.config["hessian_damping"]`, `self.games` (each with `bpd`/`wpd`, `handicap`, `extras["komi"]`, `winner`).
- Produces: `WHR._newton_handicap_komi() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_handicap_komi.py`:

```python
def _elo(gamma):
    return math.log10(gamma) * 400.0


def test_recovers_known_handicap_advantage():
    # Two equal-strength players; at handicap 2, black wins ~76% of games,
    # which corresponds to ~ +200 elo. handicap_gamma[2] should recover it.
    w = WHR()
    day = 1
    # 76 black wins, 24 white wins over distinct days (handicap 2 each)
    for _ in range(76):
        w.create_game("a", "b", "B", day, 2); day += 1
    for _ in range(24):
        w.create_game("a", "b", "W", day, 2); day += 1
    w.iterate(200)
    assert _elo(w.handicap_gamma[2]) == pytest.approx(200.0, abs=40.0)


def test_recovers_white_side_advantage_via_komi():
    # All games share komi 6.5, handicap 0; white wins ~64% (~ +100 elo for white).
    w = WHR()
    day = 1
    for _ in range(64):
        w.create_game("a", "b", "W", day, 0); day += 1
    for _ in range(36):
        w.create_game("a", "b", "B", day, 0); day += 1
    w.iterate(200)
    assert _elo(w.komi_gamma[6.5]) == pytest.approx(100.0, abs=40.0)


def test_pinned_key_is_not_moved_by_estimation():
    w = WHR(config={"pinned_handicap": {2: 300.0}})
    for d in range(1, 21):
        w.create_game("a", "b", "B", d, 2)
    w.iterate(50)
    assert w.handicap_gamma[2] == pytest.approx(10 ** (300.0 / 400.0))


def test_baseline_handicap_zero_not_moved():
    w = WHR()
    for d in range(1, 21):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(50)
    assert w.handicap_gamma[0] == 1.0


def test_estimate_handicap_zero_lets_it_move():
    w = WHR(config={"estimate_handicap_zero": True})
    for d in range(1, 41):
        w.create_game("a", "b", "B" if d % 3 else "W", d, 0)
    w.iterate(50)
    # unpinned and mixed results -> generally moves off 1.0
    assert w.handicap_gamma[0] != 1.0


def test_all_win_category_guard_leaves_gamma_untouched():
    # A handicap key with only black wins has no finite estimate -> not updated.
    w = WHR()
    for d in range(1, 11):
        w.create_game("a", "b", "B", d, 3)  # handicap 3, always black win
    w.iterate(50)
    assert w.handicap_gamma[3] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_handicap_komi.py -k "recovers or pinned_key or baseline or estimate_handicap_zero or all_win" --no-cov -v`
Expected: FAIL — `_newton_handicap_komi` does not exist, so advantages never move (recovers/estimate tests fail; guard tests may pass trivially).

- [ ] **Step 3: Implement the estimation step**

Add to `WHR` in `whr/whole_history_rating.py`:

```python
    def _newton_handicap_komi(self) -> None:
        """One Newton step on each non-pinned handicap/komi advantage gamma
        (Coulom's NewtonKomiHandicap)."""
        h_grad: dict[Any, float] = {}
        h_hess: dict[Any, float] = {}
        k_grad: dict[Any, float] = {}
        k_hess: dict[Any, float] = {}
        h_games: dict[Any, int] = {}
        h_wins: dict[Any, int] = {}
        k_games: dict[Any, int] = {}
        k_wins: dict[Any, int] = {}
        for g in self.games:
            if g.bpd is None or g.wpd is None:
                continue
            h = g.handicap
            k = g.extras["komi"]
            gh = self.handicap_gamma[h]
            gk = self.komi_gamma[k]
            gb = g.bpd.gamma()
            gw = g.wpd.gamma()
            c_komi = gw
            d_komi = gb * gh
            c_handicap = gb
            d_handicap = gw * gk
            div = 1.0 / (d_komi + d_handicap)
            h_grad[h] = h_grad.get(h, 0.0) + c_handicap * div
            h_hess[h] = h_hess.get(h, 0.0) + c_handicap * d_handicap * div * div
            k_grad[k] = k_grad.get(k, 0.0) + c_komi * div
            k_hess[k] = k_hess.get(k, 0.0) + c_komi * d_komi * div * div
            h_games[h] = h_games.get(h, 0) + 1
            k_games[k] = k_games.get(k, 0) + 1
            if g.winner == "B":
                h_wins[h] = h_wins.get(h, 0) + 1
            else:
                k_wins[k] = k_wins.get(k, 0) + 1
        damping = self.config["hessian_damping"]
        for h in list(self.handicap_gamma):
            if h in self._pinned_handicap_keys:
                continue
            games = h_games.get(h, 0)
            wins = h_wins.get(h, 0)
            if games > 0 and 0 < wins < games:
                gamma = self.handicap_gamma[h]
                grad = wins - gamma * h_grad.get(h, 0.0)
                hess = -gamma * h_hess.get(h, 0.0) - damping
                self.handicap_gamma[h] = gamma * math.exp(-grad / hess)
        for k in list(self.komi_gamma):
            if k in self._pinned_komi_keys:
                continue
            games = k_games.get(k, 0)
            wins = k_wins.get(k, 0)
            if games > 0 and 0 < wins < games:
                gamma = self.komi_gamma[k]
                grad = wins - gamma * k_grad.get(k, 0.0)
                hess = -gamma * k_hess.get(k, 0.0) - damping
                self.komi_gamma[k] = gamma * math.exp(-grad / hess)
```

Call it at the end of `_run_one_iteration`:

```python
    def _run_one_iteration(self) -> None:
        """Runs one iteration of the WHR algorithm."""
        for player in self.players.values():
            player.run_one_newton_iteration()
        self._newton_handicap_komi()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_handicap_komi.py --no-cov -v`
Expected: PASS. If a `recovers_*` test misses its tolerance, the estimation math is wrong — debug it; do NOT widen the tolerance blindly.

- [ ] **Step 5: Full suite (quarantined) + lint + types**

Run: `uv run pytest --no-cov && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS with skips; clean.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_handicap_komi.py
git commit -m "Estimate handicap/komi advantages each iteration (NewtonKomiHandicap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Re-baseline the quarantined tests

**Files:**
- Modify: `tests/whr_test.py`

- [ ] **Step 1: Remove the six `@pytest.mark.skip(...)` markers added in Task 1.**

- [ ] **Step 2: Re-verify the two semantic/property tests.**

`test_large_handicap_converges_to_finite_ratings`: handicap 600 is now an unlearned category (guard-skipped: too few games / all one-sided), so it no longer confers +600 elo. The test only asserts finiteness — confirm it still passes; if the handicap-600 framing now reads as misleading, update the comment (do not re-add an elo expectation). `test_auto_iterate`: property-based (looser ≤ tighter, both stable) — confirm it still holds; if iteration counts shifted but the property holds, no change needed.

- [ ] **Step 3: Re-baseline the four value-assertion tests** (`test_output`, `test_output2`, `test_log_likelihood`, `test_loading_several_games_at_once`): run each with `uv run pytest "tests/whr_test.py::<name>" --no-cov -v`, read the new actual values, update the expected literals (INCLUDING `print_ordered_ratings` display strings and `get_ordered_ratings` lists). Before pasting each value, sanity-check: ratings still correctly ordered (stronger player higher), all finite, uncertainties positive; the komi-6.5 white advantage now being modelled should shift values modestly, not wildly. If anything looks like a regression (ordering flip, non-finite, absurd jump), STOP and report. Annotate re-baselined blocks with `# re-baselined for phase-3 (estimated handicap/komi)`.

- [ ] **Step 4: Full suite WITH coverage.**

Run: `uv run pytest`
Expected: PASS, 0 skipped, coverage ≥95%. If <95%, add targeted tests in `tests/test_handicap_komi.py` for any uncovered new branch.

- [ ] **Step 5: Lint + types.**

Run: `uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/whr_test.py tests/test_handicap_komi.py
git commit -m "Re-baseline tests for estimated handicap/komi

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CHANGELOG.md`, `README.md`

- [ ] **Step 1: CHANGELOG** — under `## [2.1.0] - unreleased`:
  - `### Changed`: a bullet that `handicap` is now an estimated Bradley-Terry category (was a fixed elo constant); komi is now modelled (was ignored); ratings change; migration = pin known values via `pinned_handicap`.
  - `### Added`: config keys `pinned_handicap` / `pinned_komi` (elo advantages per key), `estimate_handicap_zero`; the handicap/komi advantages are co-estimated with players and readable via `WHR.handicap_gamma` / `WHR.komi_gamma`.

- [ ] **Step 2: README** — add a subsection "Handicap and komi" explaining: handicap boosts black, komi boosts white; both are categories whose advantage (in elo) is learned from the data and exposed via `handicap_gamma`/`komi_gamma`; `handicap` key `0` is a pinned no-advantage baseline by default; how to pin known values (`WHR({'pinned_handicap': {2: 200}})`, elo); the generalisation to a single-category white/side advantage via komi; and that this changes the meaning of `handicap` vs earlier versions. Match the existing README config style; include a short example.

- [ ] **Step 3: Verify** — `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy` (all clean).

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document estimated handicap/komi and pinning

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** model & `opponents_adjusted_gamma` rework (Task 2, faithful `P=γ_b·γ_h/(γ_b·γ_h+γ_w·γ_k)`); `_newton_handicap_komi` with the `0<wins<games` guard and `hessian_damping` (Task 3); pinning + `handicap[0]` baseline + `estimate_handicap_zero` (Task 1); back-compat via pinning (Task 2 test); generalisation to a white/side advantage (Task 3 test); breaking-change docs (Task 5). Uncertainties on the advantages are out of scope per spec.
- **Faithfulness check:** legacy fixed-elo handicap ≡ `pinned_handicap={h: elo}` with `γ_k=1` — asserted in `test_pinned_handicap_reproduces_legacy_elo_behaviour`.
- **Golden-test churn** handled via quarantine (Task 1) → re-baseline (Task 4), mirroring phase 1; `--no-cov` during quarantine, 95% floor re-enforced in Task 4.
- **Type safety:** `_newton_handicap_komi` and `opponents_adjusted_gamma` guard `bpd`/`wpd` None; tables typed `dict[Any, float]`.
