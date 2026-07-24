# WHR Draws — Davidson (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Model draws with the Davidson tie model (review point #8): a global draw tendency `nu`, estimated like the handicap/komi advantages, activated ONLY when the base contains draws so draw-free data stays bit-identical.

**Architecture:** A `"D"` outcome is recorded per game. When any draw exists, the per-player Newton update and a new `nu` Newton step use the Davidson likelihood `P(win)=S/Z, P(loss)=O/Z, P(draw)=T/Z` with `T=nu·sqrt(S·O)`, `Z=S+O+T`. When no draw exists, `nu=0` and the existing win/loss code path runs unchanged.

**Tech Stack:** Python ≥3.11, numpy ≥2.0, pytest+pytest-cov, ruff, mypy, uv.

## Global Constraints

- Python ≥3.11, numpy ≥2.0; ruff/ruff-format/mypy clean; `uv run pytest` passes at coverage ≥95%.
- **Compatibility invariant:** draw-free data (all current tests) must be BIT-IDENTICAL — the Davidson path runs only when `_has_draws`. No re-baselining of existing golden tests.
- Davidson derivatives (per game, player-side; own effective gamma `S`, opponent `O`, `T=nu·sqrt(S·O)`, `Z=S+O+T`, weight `w`=1 win / 0.5 draw / 0 loss):
  - gradient `= w − (S+T/2)/Z`; Hessian `= (N/Z)² − N'/Z` with `N=S+T/2`, `N'=S+T/4`.
- `nu` Newton (log-`nu` space, per game, `ratio=T/Z`): gradient `= #draws − Σ ratio`; Hessian `= −Σ ratio·(1−ratio)`; damped by `hessian_damping`; skipped if no draws or `nu` pinned.
- Branch `feat/draws-phase6`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Data layer — `"D"` outcome, `nu` plumbing, `effective_gammas`

**Files:**
- Modify: `whr/game.py` (`effective_gammas`), `whr/playerday.py` (`drawn_games`, `add_game` routing), `whr/whole_history_rating.py` (`"D"` validation, `_has_draws`, `nu`, `pinned_draw` config, `draw_tendency` property), `whr/player.py` (`draw_tendency` attribute)
- Test: `tests/test_draws.py` (create)

**Interfaces produced:**
- `Game.effective_gammas(player) -> tuple[float, float]` → `(S_player, O_opponent)` effective gammas (folding handicap/komi; `None` tables → factor 1).
- `PlayerDay.drawn_games: list[Game]`; `add_game` routes `winner=="D"` there.
- `WHR._has_draws: bool`; `WHR.nu: float` (0.0 when no draws, else a positive seed); `WHR.draw_tendency` property (returns `nu`); config `pinned_draw` (float | None, default None); `Player.draw_tendency: float` (refreshed each iteration).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_draws.py`:

```python
import math

import pytest

from whr.whole_history_rating import WHR


def test_draw_result_accepted_and_routed():
    w = WHR()
    w.create_game("a", "b", "D", 1, 0)
    w.load_games(["a b D 2"])
    assert w._has_draws is True
    a = w.player_by_name("a")
    drawn_days = [d for d in a.days if d.drawn_games]
    assert len(drawn_days) == 2  # one draw on each of days 1 and 2


def test_no_draws_leaves_nu_zero():
    w = WHR()
    w.create_game("a", "b", "B", 1, 0)
    assert w._has_draws is False
    assert w.nu == 0.0
    assert w.draw_tendency == 0.0


def test_effective_gammas_fold_advantages():
    w = WHR(config={"pinned_handicap": {2: 200.0}})
    g = w.create_game("black", "white", "B", 1, 2)
    w.player_by_name("black").days[0].set_gamma(3.0)
    w.player_by_name("white").days[0].set_gamma(2.0)
    s_black, o_white = g.effective_gammas(g.black_player)
    assert s_black == pytest.approx(3.0 * 10 ** (200.0 / 400.0))  # handicap boosts black
    assert o_white == pytest.approx(2.0)  # komi default gamma 1
    # symmetry: querying as white swaps S/O
    s_white, o_black = g.effective_gammas(g.white_player)
    assert (s_white, o_black) == pytest.approx((o_white, s_black))


def test_pinned_draw_config_default_none():
    assert WHR().config["pinned_draw"] is None
    assert WHR(config={"pinned_draw": 1.5}).config["pinned_draw"] == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_draws.py --no-cov -v`
Expected: FAIL (no `_has_draws`, `effective_gammas`, etc.).

- [ ] **Step 3: Implement the data layer**

`whr/playerday.py` — add `self.drawn_games: list[G.Game] = []` in `__init__`, and route draws in `add_game`:

```python
    def add_game(self, game: G.Game) -> None:
        if game.winner == "D":
            self.drawn_games.append(game)
        elif (game.winner == "W" and game.white_player == self.player) or (
            game.winner == "B" and game.black_player == self.player
        ):
            self.won_games.append(game)
        else:
            self.lost_games.append(game)
```

`whr/game.py` — add:

```python
    def effective_gammas(self, player: P.Player) -> tuple[float, float]:
        """(player's, opponent's) effective gammas, folding in handicap/komi.

        Black's effective gamma is gamma*handicap_gamma; white's is
        gamma*komi_gamma. Returns (S, O) from ``player``'s perspective.
        """
        if self.bpd is None or self.wpd is None:
            raise AttributeError("black player day and white player day must be set")
        gh = 1.0 if self.handicap_gamma is None else self.handicap_gamma.get(self.handicap, 1.0)
        gk = 1.0 if self.komi_gamma is None else self.komi_gamma.get(self.extras["komi"], 1.0)
        black_eff = self.bpd.gamma() * gh
        white_eff = self.wpd.gamma() * gk
        if player == self.black_player:
            return black_eff, white_eff
        if player == self.white_player:
            return white_eff, black_eff
        raise AttributeError(f"{player.name!r} is not in this game")
```

`whr/player.py` — in `Player.__init__` add `self.draw_tendency: float = 0.0`.

`whr/whole_history_rating.py`:
- In `WHR.__init__`: `self.config.setdefault("pinned_draw", None)`, `self._has_draws = False`, `self.nu = 0.0`.
- Validate `"D"` as a legal winner where winners are checked (`_setup_game`/`create_game`/`load_games` already accept the raw string; `Game.winner` upstreams `.upper()`, so `"D"`/`"d"` work — just make sure no code rejects a non-B/W winner). In `_add_game` (or `create_game`), when `game.winner == "D"`, set `self._has_draws = True` and, if `self.nu == 0.0` and `pinned_draw is None`, seed `self.nu = 1.0`; if `pinned_draw is not None`, set `self.nu = self.config["pinned_draw"]`.
- Add a read-only property:
  ```python
  @property
  def draw_tendency(self) -> float:
      return self.nu
  ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_draws.py --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Full suite (must be unchanged for no-draw data) + lint + types**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS, coverage ≥95%. All existing tests still pass unchanged (draws are only recorded, not yet in the likelihood — no existing test uses `"D"`).

- [ ] **Step 6: Commit**

```bash
git add whr/ tests/test_draws.py
git commit -m "Record draws: 'D' outcome, effective_gammas, nu plumbing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Davidson per-player derivatives

**Files:**
- Modify: `whr/playerday.py` (add `davidson_derivatives`), `whr/player.py` (`gradient`/`hessian`/`update_by_1d_newtons_method` use the Davidson path when draws are active)
- Test: `tests/test_draws.py`

**Interfaces:**
- Consumes: `Game.effective_gammas` (T1), `Player.draw_tendency` (T1).
- Produces: `PlayerDay.davidson_derivatives(nu) -> tuple[float, float]` (game-part gradient, Hessian summed over the day's games under Davidson). `Player.gradient`/`hessian` select the Davidson game-part when `self.draw_tendency > 0`, else the existing win/loss path (unchanged).

- [ ] **Step 1: Write the failing test (closed-form sanity)**

Append to `tests/test_draws.py`:

```python
def test_davidson_derivatives_match_closed_form():
    w = WHR()
    g = w.create_game("a", "b", "D", 1, 0)  # a draw on day 1
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    nu = 1.5
    s, o = g.effective_gammas(a_day.player)  # S=2, O=1 (no advantages)
    t = nu * math.sqrt(s * o)
    z = s + o + t
    n = s + t / 2.0
    n_prime = s + t / 4.0
    w_weight = 0.5  # draw
    exp_grad = w_weight - n / z
    exp_hess = (n / z) ** 2 - n_prime / z
    grad, hess = a_day.davidson_derivatives(nu)
    assert grad == pytest.approx(exp_grad)
    assert hess == pytest.approx(exp_hess)


def test_davidson_reduces_to_bt_at_nu_zero():
    # Same win/loss data, compute the day's game-part gradient both ways at nu=0.
    w = WHR()
    g = w.create_game("a", "b", "B", 1, 0)  # a (black) won
    a_day = w.player_by_name("a").days[0]
    a_day.set_gamma(2.0)
    w.player_by_name("b").days[0].set_gamma(1.0)
    davidson_grad, davidson_hess = a_day.davidson_derivatives(0.0)
    assert davidson_grad == pytest.approx(a_day.log_likelihood_derivative())
    assert davidson_hess == pytest.approx(a_day.log_likelihood_second_derivative())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_draws.py -k davidson --no-cov -v`
Expected: FAIL (`AttributeError: ... 'davidson_derivatives'`).

- [ ] **Step 3: Implement**

`whr/playerday.py` — add:

```python
    def _weighted_games(self):
        """Yield (game, outcome_weight): 1.0 won, 0.0 lost, 0.5 drawn."""
        for g in self.won_games:
            yield g, 1.0
        for g in self.lost_games:
            yield g, 0.0
        for g in self.drawn_games:
            yield g, 0.5

    def davidson_derivatives(self, nu: float) -> tuple[float, float]:
        """(gradient, Hessian) of this day's game log-likelihood under Davidson.

        Reduces to the plain Bradley-Terry win/loss derivatives at nu=0.
        """
        gradient = 0.0
        hessian = 0.0
        for game, weight in self._weighted_games():
            s, o = game.effective_gammas(self.player)
            t = nu * math.sqrt(s * o)
            z = s + o + t
            n = s + t / 2.0
            n_prime = s + t / 4.0
            ratio = n / z
            gradient += weight - ratio
            hessian += ratio * ratio - n_prime / z
        return gradient, hessian
```

`whr/player.py` — in `gradient` and `hessian`, choose the game-part per day: if `self.draw_tendency > 0.0`, use `day.davidson_derivatives(self.draw_tendency)` for the game gradient/Hessian; else keep `day.log_likelihood_derivative()`/`day.log_likelihood_second_derivative()` (existing path, unchanged). Keep the temporal prior and anchor terms exactly as they are. Do the same in `PlayerDay.update_by_1d_newtons_method` (single-day player) — but that method has no access to `self.player.draw_tendency`... it does via `self.player`. Adapt: in `update_by_1d_newtons_method`, if `self.player.draw_tendency > 0.0`, use `self.davidson_derivatives(self.player.draw_tendency)` for `(dlogp, d2logp)` (plus the anchor), else the current `log_likelihood_derivative()/second_derivative()`.

Example for `Player.gradient`:

```python
            if self.draw_tendency > 0.0:
                game_grad, _ = day.davidson_derivatives(self.draw_tendency)
            else:
                game_grad = day.log_likelihood_derivative()
            term = game_grad + prior
            if idx == 0:
                term += day.anchor_gradient()
            g.append(term)
```

and symmetrically in `hessian` (`day.davidson_derivatives(...)[1]` vs `day.log_likelihood_second_derivative()`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_draws.py -k davidson --no-cov -v`
Expected: PASS (including the nu=0 equivalence test — if that fails, the Davidson math does not reduce to BT and must be fixed, NOT the test).

- [ ] **Step 5: Full suite + lint + types**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS, ≥95%. Existing no-draw tests still bit-identical (their players have `draw_tendency == 0.0`, so the else-branch runs).

- [ ] **Step 6: Commit**

```bash
git add whr/playerday.py whr/player.py tests/test_draws.py
git commit -m "Davidson per-player derivatives (draw-aware), BT at nu=0

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Estimate `nu` and wire activation

**Files:**
- Modify: `whr/whole_history_rating.py` (`_newton_draw`; refresh `Player.draw_tendency` and call `_newton_draw` in `_run_one_iteration`)
- Test: `tests/test_draws.py`

**Interfaces:**
- Produces: `WHR._newton_draw() -> None`. `_run_one_iteration` sets each player's `draw_tendency = self.nu` before its Newton update, then (after the player loop) calls `_newton_draw()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draws.py`:

```python
import random


def _davidson_balanced_history(rng, nu_true, n_pairs=40, n_games=60):
    """Equal players, colour-swapped, single day; outcomes sampled from Davidson
    with a known nu_true (equal gammas => S=O=1 => P(draw)=nu/(2+nu))."""
    w = WHR()
    p_draw = nu_true / (2.0 + nu_true)
    p_win = 1.0 / (2.0 + nu_true)
    for k in range(n_pairs):
        a, b = f"a{k}", f"b{k}"
        for _ in range(n_games):
            r = rng.random()
            outcome = "D" if r < p_draw else ("B" if r < p_draw + p_win else "W")
            w.create_game(a, b, outcome, 1, 0)
            w.create_game(b, a, outcome if outcome == "D" else ("W" if outcome == "B" else "B"), 1, 0)
    return w


def test_recovers_known_draw_tendency():
    rng = random.Random(7)
    nu_true = 1.5
    w = _davidson_balanced_history(rng, nu_true)
    w.iterate(80)
    assert w.draw_tendency == pytest.approx(nu_true, abs=0.3)


def test_pinned_draw_is_not_moved():
    w = WHR(config={"pinned_draw": 0.8})
    for d in range(1, 11):
        w.create_game("a", "b", "D", d, 0)
        w.create_game("a", "b", "B", d, 0)
    w.iterate(30)
    assert w.draw_tendency == pytest.approx(0.8)


def test_no_draw_iteration_still_matches_baseline():
    # A draw-free scenario iterates with the win/loss path untouched.
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
        w.create_game("a", "b", "W", d, 0)
    w.iterate(30)
    assert w.nu == 0.0
    elo, _ = w.ratings_for_player("a", current=True)
    assert math.isfinite(elo)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_draws.py -k "recovers_known_draw or pinned_draw or no_draw_iteration" --no-cov -v`
Expected: FAIL — `nu` never moves (`_newton_draw` missing / not wired), so `test_recovers_known_draw_tendency` fails.

- [ ] **Step 3: Implement**

`whr/whole_history_rating.py`:

```python
    def _newton_draw(self) -> None:
        """One Newton step on the global draw tendency nu (Davidson), in log-nu
        space. Skipped when there are no draws or nu is pinned."""
        if not self._has_draws or self.config["pinned_draw"] is not None:
            return
        gradient = 0.0
        hessian = 0.0
        for game in self.games:
            if game.bpd is None or game.wpd is None:
                continue
            s, o = game.effective_gammas(game.black_player)
            t = self.nu * math.sqrt(s * o)
            z = s + o + t
            ratio = t / z
            gradient += (1.0 if game.winner == "D" else 0.0) - ratio
            hessian += -ratio * (1.0 - ratio)
        hessian -= self.config["hessian_damping"]
        v = math.log(self.nu) - gradient / hessian
        self.nu = math.exp(v)

    def _run_one_iteration(self) -> None:
        for player in self.players.values():
            player.draw_tendency = self.nu
            player.run_one_newton_iteration()
        self._newton_handicap_komi()
        self._newton_draw()
```

(Adapt `_run_one_iteration` to whatever it currently contains — it already calls `_newton_handicap_komi`; add the `draw_tendency` refresh in the player loop and the `_newton_draw()` call.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_draws.py -k "recovers_known_draw or pinned_draw or no_draw_iteration" --no-cov -v`
Expected: PASS. If `test_recovers_known_draw_tendency` misses tolerance, the `nu` Newton math is wrong — DEBUG it (or strengthen the data with more games), do NOT loosen the assertion into meaninglessness; if it cannot recover, STOP and report.

- [ ] **Step 5: Full suite + lint + types**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS, ≥95%. Existing no-draw tests unchanged (`_has_draws` False → `_newton_draw` returns immediately, `draw_tendency` stays 0).

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_draws.py
git commit -m "Estimate the Davidson draw tendency nu each iteration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 3-way prediction `win_draw_loss_probabilities`

**Files:**
- Modify: `whr/whole_history_rating.py`
- Test: `tests/test_draws.py`

**Interfaces:**
- Produces: `WHR.win_draw_loss_probabilities(name1, name2, handicap=0, handicap_key=None, komi_key=None) -> tuple[float, float, float]` → `(P(name1 wins), P(draw), P(name2 wins))`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draws.py`:

```python
def test_win_draw_loss_sums_to_one_and_reflects_nu():
    rng = random.Random(3)
    w = _davidson_balanced_history(rng, 1.5, n_pairs=20, n_games=40)
    w.iterate(60)
    p1, pd, p2 = w.win_draw_loss_probabilities("a0", "b0")
    assert p1 + pd + p2 == pytest.approx(1.0)
    assert all(p >= 0.0 for p in (p1, pd, p2))
    assert pd > 0.05  # meaningful draw mass with nu ~ 1.5


def test_win_draw_loss_no_draws_gives_zero_draw():
    w = WHR()
    for d in range(1, 6):
        w.create_game("a", "b", "B", d, 0)
    w.iterate(20)
    p1, pd, p2 = w.win_draw_loss_probabilities("a", "b")
    assert pd == pytest.approx(0.0)
    assert p1 + p2 == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_draws.py -k win_draw_loss --no-cov -v`
Expected: FAIL (method missing).

- [ ] **Step 3: Implement**

Mirror `probability_future_match`'s player lookup + advantage handling to obtain the two effective gammas `s1` (name1) and `s2` (name2) at their last days (unknown/unrated → gamma 1, elo 0; reuse the existing helper logic), then apply Davidson with `self.nu`:

```python
    def win_draw_loss_probabilities(
        self, name1, name2, handicap=0, handicap_key=None, komi_key=None
    ) -> tuple[float, float, float]:
        """(P(name1 wins), P(draw), P(name2 wins)) under the fitted Davidson
        model. Draw probability is 0 when nu == 0."""
        # ... reuse probability_future_match's setup to get s1, s2 (effective
        # gammas incl. handicap/komi advantages) ...
        t = self.nu * math.sqrt(s1 * s2)
        z = s1 + s2 + t
        return s1 / z, t / z, s2 / z
```

Factor the shared "compute s1, s2" logic out of `probability_future_match` into a small helper if that keeps both methods DRY; otherwise duplicate minimally. Keep `probability_future_match`'s public behaviour unchanged.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_draws.py -k win_draw_loss --no-cov -v`
Expected: PASS.

- [ ] **Step 5: Full suite + lint + types**

Run: `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy`
Expected: PASS, ≥95%.

- [ ] **Step 6: Commit**

```bash
git add whr/whole_history_rating.py tests/test_draws.py
git commit -m "Add win_draw_loss_probabilities (Davidson 3-way prediction)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Documentation + no-draw regression

**Files:**
- Modify: `CHANGELOG.md`, `README.md`
- Test: `tests/test_draws.py` (one explicit no-draw-unchanged regression)

- [ ] **Step 1: Regression test** — append a test asserting that a specific draw-free scenario's ratings equal a hard-coded pre-phase-6 value (compute it once on the current code, before this phase's behaviour could differ — or simply assert equality to the same scenario built and iterated identically, confirming `_has_draws is False` and `nu == 0.0`). The point: prove no-draw behaviour is untouched.

- [ ] **Step 2: CHANGELOG** — under `## [2.1.0] - unreleased` `### Added`: draws via the Davidson model — pass `"D"` as the winner; a global draw tendency `WHR.draw_tendency` is estimated (pinnable via `pinned_draw`); `win_draw_loss_probabilities` gives the 3-way prediction; draw-free data is unaffected (`nu=0` reduces to Bradley-Terry).

- [ ] **Step 3: README** — add a "Draws" subsection: pass `"D"`, the Davidson model + `nu` (estimated, pinnable), the new 3-way prediction, and that draw-free bases behave exactly as before. Short example. Match existing style.

- [ ] **Step 4: Verify** — `uv run pytest && uv run ruff check whr tests && uv run ruff format --check whr tests && uv run mypy` (all clean; ≥95%).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md README.md tests/test_draws.py
git commit -m "Document draws (Davidson) and lock no-draw invariance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** data layer + `nu` plumbing (T1); Davidson per-player derivatives with BT-at-nu=0 equivalence (T2); `nu` estimation + activation + pinning (T3); 3-way prediction (T4); docs + no-draw regression (T5). Rao-Kupper excluded.
- **Compatibility invariant** enforced by `draw_tendency > 0` / `_has_draws` gating everywhere the Davidson path could run; T2/T3/T5 assert no-draw data is unchanged (else branch = existing code).
- **Math** is the closed form derived and sanity-checked in the spec; T2 unit-tests the gradient/Hessian against the closed form AND the nu=0→BT reduction; T3 recovers a planted `nu`.
- **Numerical care:** `nu` estimated in log-space (seeded to 1.0 when draws appear); `sqrt(S·O)` well-defined (gammas > 0); `hessian_damping` reused.
