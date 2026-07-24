
# Whole History Rating (WHR) Python Implementation

This Python library is a conversion from the original Ruby implementation of Rémi Coulom's Whole-History Rating (WHR) algorithm, designed to provide a dynamic rating system for games or matches where players' skills are continuously estimated over time.

The original Ruby code is available here at [goshrine](https://github.com/goshrine/whole_history_rating).

## Installation

To install the library, use the following command:

```shell
pip install whole-history-rating
```

## Usage

### Basic Setup

Start by importing the library and initializing the WHR object:

```python
from whr import WHR

whr = WHR()
```

> The class used to be called `Base`. That name still works as a deprecated
> alias (it emits a `DeprecationWarning`); prefer `WHR` in new code.

### Creating Games

Add games to the system using `create_game()` method. It takes the names of the black and white players, the winner ('B' for black, 'W' for white), the day number, and an optional handicap (generally less than 500 elo).

```python
whr.create_game("shusaku", "shusai", "B", 1, 0)
whr.create_game("shusaku", "shusai", "W", 2, 0)
whr.create_game("shusaku", "shusai", "W", 3, 0)
```


### Refining Ratings Towards Stability

To achieve accurate and stable ratings, the WHR algorithm allows for iterative refinement. This process can be controlled manually or handled automatically to adjust player ratings until they reach a stable state.

#### Manual Iteration

For manual control over the iteration process, specify the number of iterations you wish to perform. This approach gives you direct oversight over the refinement steps.

```python
whr.iterate(50)
```

This command will perform 50 iterations, incrementally adjusting player ratings towards stability with each step.

#### Automatic Iteration

For a more hands-off approach, the algorithm can automatically iterate until the Elo ratings stabilize within a specified precision. Automatic iteration is particularly useful when dealing with large datasets or when seeking to automate the rating process.

```python
whr.auto_iterate(time_limit=10, precision=1e-3, batch_size=10)
```

- `time_limit` (optional): Sets a maximum duration (in seconds) for the iteration process. If `None` (the default), the algorithm will run indefinitely until the specified precision is achieved.
- `precision` (optional): Defines the desired level of accuracy for the ratings' stability. The default value is `0.001`. Convergence is measured on the gradient infinity-norm (the largest absolute gradient component across all player-days, in natural-rating units); iteration stops once that value drops below this threshold.
- `batch_size` (optional): Determines the number of iterations to perform before checking for convergence and, if a `time_limit` is set, before evaluating whether the time limit has been reached. The default value is `10`, balancing between frequent convergence checks and computational efficiency.

This automated process allows the algorithm to efficiently converge to stable ratings, adjusting the number of iterations dynamically based on the complexity of the data and the specified precision and time constraints.


### Viewing Ratings

Retrieve and view player ratings, which include the day number, elo rating, and uncertainty:

```python
# Example output for whr.ratings_for_player("shusaku")
print(whr.ratings_for_player("shusaku"))
# Output (one (day, elo, uncertainty) tuple per playing day):
#   [(1, -43, 0.84),
#    (2, -45, 0.84),
#    (3, -45, 0.84)]

# Example output for whr.ratings_for_player("shusai")
print(whr.ratings_for_player("shusai"))
# Output:
#   [(1, 43, 0.84),
#    (2, 45, 0.84),
#    (3, 45, 0.84)]
```

Querying an unknown player raises a `ValueError`.

You can also view or retrieve all ratings in order:

```python
whr.print_ordered_ratings(current=False)  # Set `current=True` for the latest rankings only.
ratings = whr.get_ordered_ratings(current=False, compact=False)  # Set `compact=True` for a condensed list.
```

### Predicting Match Outcomes

Predict the outcome of future matches, including between non-existent players:

```python
# Example of predicting a future match outcome. It is a pure query: it does
# not print anything, and unknown players are treated as an even (gamma = 1)
# reference without being added to the base.
probability = whr.probability_future_match("shusaku", "shusai", 0)
print(f"Win probability: shusaku: {probability[0]*100}%; shusai: {probability[1]*100}%")
# Output:
#   Win probability: shusaku: 37.24%; shusai: 62.76%
```


### Uncertainty

Beyond the per-day `uncertainty` from `ratings_for_player`, three methods turn
that raw variance into comparisons and predictions. All of them report **elo**
and require `iterate()`/`auto_iterate()` to have run first (an unrated player
raises `ValueError`).

**Comparing two players.** A player's own elo doesn't by itself say how
confidently they're ahead of a rival — the *difference* between the two is
the comparable quantity, given by `rating_difference()`:

```python
whr = WHR()
for day in range(1, 11):
    whr.create_game("north", "referee", "B", day, 0)  # north usually wins
for day in range(1, 11):
    whr.create_game("south", "referee", "W", day, 0)  # south usually loses
whr.auto_iterate()

whr.rating_difference("north", "south")
# {'difference': 1054.66, 'std_error': 85.73,
#  'confidence_interval_95': (886.63, 1222.69)}
```

This is an **approximation**: WHR never computes cross-player covariance, so
the difference's variance is `Var(a) + Var(b)` (an independence assumption).
Two players who have played each other a lot have correlated errors this
ignores, so treat the CI as indicative rather than exact. Pass `day_a=`/
`day_b=` to compare specific days instead of each player's latest.

**One player's trajectory over time.** `rating_covariance()` /
`rating_change()` instead use the *exact* joint covariance among a single
player's own day ratings (already implicit in WHR's per-player Hessian —
no approximation involved):

```python
whr = WHR()
whr.load_games(["casey dana B 1", "casey dana W 5", "casey dana B 9", "casey dana W 13"])
whr.iterate(60)

days, cov = whr.rating_covariance("casey")
# days == [1, 5, 9, 13]; cov is a 4x4 elo^2 matrix, cov[i][j] = Cov(elo on days[i], days[j])

whr.rating_change("casey", day_from=1, day_to=13)
# {'change': -6.70, 'std_error': 57.48,
#  'confidence_interval_95': (-119.35, 105.95)}
```

`rating_change`'s standard error comes from `Var(to) + Var(from) -
2*Cov(from, to)`, not from naively summing the two days' marginal variances
(here, `57.48` vs a naive `115.83`) — consecutive days are positively
correlated through the Wiener prior, so a real change is more certain than
that naive sum suggests. Use this — not `rating_difference` — to ask "did
this player change significantly between day X and day Y?"

**Uncertainty-aware predictions.** `probability_future_match()` takes an
opt-in `account_for_uncertainty` flag:

```python
whr = WHR()
whr.load_games(["rookie champ B 1", "rookie champ B 2"])
whr.iterate(50)

whr.probability_future_match("rookie", "champ")
# (0.883, 0.117) -- point prediction (default, unchanged from before)
whr.probability_future_match("rookie", "champ", account_for_uncertainty=True)
# (0.856, 0.144) -- hedged toward 0.5: still favours rookie, less confidently
```

The default (`False`) is exactly the pre-existing point prediction
(non-breaking). `True` integrates the win probability over the Gaussian
implied by both players' rating uncertainty (Coulom's `Predict`), pulling the
result toward 0.5 when the ratings involved are uncertain — as above, where
only two games have been played. `uncertainty_steps` (default `4`) sets the
number of Gaussian-quadrature steps used on each side of the integration grid,
which spans `±0.5 * uncertainty_steps` standard deviations; raise it for a
finer integral at some extra compute cost.

### Enhanced Batch Loading of Games

This feature facilitates the batch loading of multiple games simultaneously by accepting a list of strings, where each string encapsulates the details of a single game. To accommodate names with both first and last names and ensure flexibility in data formatting, you can specify a custom separator (e.g., a comma) to delineate the game attributes.

#### Standard Loading

Without specifying a separator, the default space (' ') is used to split the game details:

```python
whr.load_games([
    "shusaku shusai B 1 0",  # Game 1: Shusaku vs. Shusai, Black wins, Day 1, no handicap.
    "shusaku shusai W 2 0",  # Game 2: Shusaku vs. Shusai, White wins, Day 2, no handicap.
    "shusaku shusai W 3 0"   # Game 3: Shusaku vs. Shusai, White wins, Day 3, no handicap.
])
```

#### Custom Separator for Complex Names

When game details include names with spaces, such as first and last names, utilize the `separator` parameter to define an alternative delimiter, ensuring the integrity of each data point:

```python
whr.load_games([
    "John Doe, Jane Smith, W, 1, 0",  # Game 1: John Doe vs. Jane Smith, White wins, Day 1, no handicap.
    "Emily Chen, Liam Brown, B, 2, 0"  # Game 2: Emily Chen vs. Liam Brown, Black wins, Day 2, no handicap.
], separator=",")
```

This method allows for a clear and error-free way to load game data, especially when player names or game details include spaces, providing a robust solution for managing diverse datasets.


### Saving and Loading States

Save the current state to a file and reload it later to avoid recalculating:

```python
whr.save_base('path_to_save.whr')
whr2 = WHR.load_base('path_to_save.whr')
```

The state is serialized as a flat description (config, games and computed
ratings) rather than the raw object graph, so saving and loading works for a
history of any size and the computed ratings are preserved on reload. Files
written by older versions are still readable.

## Optional Configuration

Adjust the `w2` parameter, which influences the variance of rating change over time, allowing for faster or slower progression. The default is set to 300, but Rémi Coulom used a value of 14 in his paper to achieve his results.

```python
whr = WHR({'w2': 14})
```

Enable case-insensitive player names to treat "shusaku" and "ShUsAkU" as the same entity:

```python
whr = WHR({'uncased': True})
```

Adjust `initial_prior_wins`, the strength of the first-day Bradley-Terry anchor (Coulom's `InitialPriorWins`). The default is `0.5`; lower values reduce the compression of weakly-connected players toward 0 elo.

```python
whr = WHR({'initial_prior_wins': 0.5})
```

Adjust `hessian_damping`, the damping subtracted from the Newton Hessian diagonal (Coulom's `HessianEpsilon`) for numerical stability. The default is `1.0`; it does not bias the converged ratings, but it does change the reported uncertainties, since it flows through the covariance computation.

```python
whr = WHR({'hessian_damping': 1.0})
```

Adjust `drift_kernel_radius`, the half-width (in days) of the Gaussian kernel used by `remove_drift()` to smooth per-day drift (Coulom's `RemoveDrift`). The default is `100`.

```python
whr = WHR({'drift_kernel_radius': 100})
```

Pin a known handicap or komi advantage (in elo) instead of letting it be estimated from the data, via `pinned_handicap` / `pinned_komi` — each a `{key: elo}` dict. Both default to `{}` (nothing pinned, other than the `handicap` key `0` baseline described below).

```python
whr = WHR({'pinned_handicap': {2: 200}})
```

Set `estimate_handicap_zero` to let the `handicap` key `0` (no handicap) be estimated instead of pinned to a 0-elo baseline. The default is `False`. See "Handicap and komi" below for why this baseline exists.

```python
whr = WHR({'estimate_handicap_zero': True})
```

### Removing Rating Drift

Over long histories, the whole population's average strength can drift or inflate over time even though individual ratings are locally accurate, making players from different eras hard to compare. `remove_drift()` (a faithful port of Coulom's `RemoveDrift`) corrects for this by recentring the per-day mean player strength near 0 elo, using a Gaussian-smoothed estimate of the drift at each day (controlled by `drift_kernel_radius`).

Call it once after ratings have converged, i.e. after `iterate()` or `auto_iterate()` — and call it last, since a subsequent `iterate()`/`auto_iterate()` call would revert the correction:

```python
whr = WHR()
whr.load_games([...])
whr.auto_iterate()
corrections = whr.remove_drift()  # optional, after convergence; call last
```

This step is opt-in: it does not run automatically and does not change what `iterate()`/`auto_iterate()` compute. It mutates the stored ratings in place, shifting every player-day's rating by that day's negated drift, and returns the applied corrections as `{day: correction_elo}`. Because the shift is uniform within a day, the relative rating (and thus win probability, e.g. `Game.white_win_probability()`) of two players active on the *same* day is unchanged; `probability_future_match` is only invariant when the two players' last recorded days happen to coincide, and generally is not, since it compares each player's own last day, which typically receive different corrections. Uncertainties (from `ratings_for_player`) are not recomputed by this step; this is only approximate, since the first-day anchor curvature is not exactly invariant under the shift, but the effect is output-only and has no downstream effect on iteration.

`time_step` must be a compact day index counted from some origin (e.g. a day number), not an epoch timestamp: `remove_drift()`'s cost scales with the CALENDAR SPAN of day values (`max_day - min_day`), not with the number of games, so an epoch timestamp will silently hang or exhaust memory.

### Handicap and komi

Every game carries a `handicap` key (the `handicap` argument to `create_game`/`load_games`) and a `komi` key (`extras['komi']`, default `6.5`). Handicap boosts **black**; komi boosts **white**. Rather than a fixed elo constant, each distinct key is a Bradley-Terry *category*: its advantage, in elo, is a parameter co-estimated alongside the player ratings on every iteration (a faithful port of Coulom's `NewtonKomiHandicap`), and is readable at any time from `whr.handicap_gamma` / `whr.komi_gamma` — dicts mapping each key to its estimated gamma (convert to elo with `400 * log10(gamma)`).

The `handicap` key `0` (no handicap) is a pinned no-advantage baseline (gamma `1.0`, i.e. `0` elo) by default and is never moved by estimation — this resolves an identifiability confound between the black/white baseline and the komi advantage. Set `estimate_handicap_zero=True` if you want it estimated instead.

To pin a handicap or komi value you already know (rather than estimating it), use `pinned_handicap` / `pinned_komi`:

```python
whr = WHR({'pinned_handicap': {2: 200}})  # a 2-stone handicap is worth +200 elo to black
whr.create_game("weaker", "stronger", "B", 1, 2)
```

Pinning a handicap key to its known elo value reproduces the fixed-elo handicap behaviour of earlier versions of this library (see below).

The same mechanism generalises beyond Go: if every game shares a single komi key (e.g. all games use the default `6.5`, or you set a constant key of your own choosing), `komi_gamma` for that key becomes a single learned **white/side advantage** — the colour advantage in chess, or a home advantage in other sports. No Go-specific assumption is involved.

**This changes the meaning of `handicap` versus earlier versions**, where it was a fixed elo bonus added directly to black's elo. It is now a category label whose advantage is learned (or pinned). If you relied on the old fixed-elo behaviour, pin every handicap value you use, e.g. `WHR({'pinned_handicap': {h: elo_value, ...}})` for each handicap `h` your data contains.

**Caveat — don't estimate what your data can't support.** Estimating a handicap/komi advantage well requires enough games where player strengths and colours are reasonably balanced (as in the recovery tests: many games, colours swapped, roughly even overall). With a small or skewed sample, the estimated advantage — especially komi, since most games share the same default komi key — can silently absorb what is really just a player-strength difference (e.g. if white happens to be the stronger player more often in your data, `komi_gamma` will drift up to explain it instead of the player ratings doing so). If you don't have enough data to support estimating it, pin the value to disable estimation, e.g. `WHR({'pinned_komi': {6.5: 0}})`.

**Note — `probability_future_match`'s `handicap` is not this mechanism (but its `handicap_key`/`komi_key` are).** The positional `handicap` argument is a raw elo adjustment that shifts the effective elo gap in name1's favour for a what-if query; it is *not* a category key and applies no learned advantage. To have a prediction reflect the estimated advantages, pass the category keys explicitly via `handicap_key` (favouring name1, the black role) and/or `komi_key` (favouring name2, the white role); their learned/pinned gammas are then folded in exactly as in a real game, and any raw `handicap` elo stacks on top. Unseen keys default to no advantage.

```python
# Fold the learned 2-stone handicap advantage into the prediction:
whr.probability_future_match("weaker", "stronger", 0, handicap_key=2)
```

### Choosing `w2` from data

`w2` controls how much a player's rating is allowed to drift from one playing day to the next (the variance of Coulom's Wiener prior over time) — a larger `w2` lets ratings move faster in response to recent results, a smaller `w2` keeps them stable and slow-moving. Picking it by hand (as in "Optional Configuration" above) is a guess; `WHR.fit_w2()` picks it from your own data instead.

It works by temporal cross-validation: your games are cut into `n_splits` expanding-window folds by day (fold *i* trains on every game strictly before a cutoff day and tests on the games in the following window), so a candidate is always scored on games that happened *after* the ones it was trained on — there is no future leakage. For each candidate `w2`, a fresh model is trained on each fold's training games for `iterations` iterations and scored by predictive log-loss (lower is better) on that fold's held-out games, pooled across all folds; the candidate with the lowest pooled log-loss is `best_w2`. Test games where either player has no prior rated day (cold start) can't be scored and are skipped rather than counted against a candidate.

`fit_w2()` is a **pure query**: it builds its own throwaway models internally and never touches `self.config` or any rating already computed on this instance. Apply the result yourself:

```python
result = whr.fit_w2(candidates=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0], n_splits=5, iterations=50)
# result == {'best_w2': 100.0, 'log_loss': {10.0: 0.71, 30.0: 0.66, 100.0: 0.64, ...},
#            'n_splits': 5, 'n_test_scored': 812, 'n_test_skipped': 3}

whr = WHR({'w2': result['best_w2']})
whr.load_games([...])
whr.auto_iterate()
```

- `candidates`: the `w2` values to try (default `[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]`).
- `n_splits`: number of expanding-window folds (default `5`); raises `ValueError` if there aren't enough distinct days to form them.
- `iterations`: how many `iterate()` steps each fold's fresh model runs before scoring (default `50`).

**Cost caveat.** `fit_w2()` trains `len(candidates) × n_splits` separate models for `iterations` iterations each — a full model fit, not an incremental update. On large histories this gets expensive fast; until the ratings loop is vectorised, consider a smaller `candidates` list, fewer `n_splits`, or a subsample of your history when exploring interactively.
