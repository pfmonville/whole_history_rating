# API reference

The supported public entry point is `whr.WHR`. `whr.Base` remains a deprecated
compatibility alias.

## Constructing a model

### `WHR(config=None)`

Creates an empty rating model. Configuration keys include:

| Key | Default | Purpose |
|---|---:|---|
| `w2` | `300.0` | Rating random-walk variance per time unit, in elo² |
| `uncased` | `False` | Normalize player names to lowercase |
| `initial_prior_wins` | `0.5` | Strength of the first-day anchor |
| `pinned_handicap` | `{}` | Fixed handicap-category advantages |
| `pinned_komi` | `{}` | Fixed komi-category advantages |
| `pinned_draw` | `None` | Fixed Davidson draw tendency |
| `draw_rate` | `None` | Alternative draw-tendency declaration |
| `display_offset` | `0.0` | Constant applied only to displayed ratings |
| `display_uncertainty` | `"variance"` | Display variance or elo standard error |

See the [user guide](user-guide.md) for identifiability constraints and the
complete configuration discussion.

## Adding results

### `create_game(black, white, winner, time_step, handicap=0, komi=None)`

Adds one game and returns its `Game` object. `winner` is `"B"`, `"W"`, or
`"D"`. Handicap and komi values are learned category keys unless pinned.

### `load_games(games, separator=" ")`

Loads compact string records in the form
`black white result time_step [handicap]`.

Adding games invalidates the previous fit. Call `iterate()` or `auto_iterate()`
before reading updated ratings or predictions.

## Fitting and diagnostics

### `iterate(count)`

Runs a fixed number of Newton iterations.

### `auto_iterate(time_limit=None, precision=1e-3, batch_size=50)`

Iterates until the gradient infinity-norm reaches `precision` or the optional
time limit is reached.

### `max_gradient_norm()`

Returns the convergence quantity used by `auto_iterate()`.

### `log_likelihood()`

Returns the current total log-posterior density.

### `fit_w2(candidates, ...)`

Selects `w2` with forward temporal validation without mutating the model.

### `connected_components()` and `games_since_last_fit`

Expose disconnected rating pools and whether newly added games have not yet
been fitted.

## Ratings and uncertainty

### `ratings_for_player(name)`

Returns `(time_step, elo, uncertainty)` tuples for one player.

### `get_ordered_ratings(current=False, compact=False)`

Returns ratings ordered by time and strength.

### `rating_difference(name_a, name_b, day_a=None, day_b=None)`

Returns an elo difference, standard error, and approximate 95% interval.

### `rating_covariance(name)`

Returns the exact dense within-player covariance matrix in elo².

### `rating_change(name, day_from, day_to)`

Returns a covariance-aware change estimate and interval between two rated days.

### `display_offset_for(target, player=None, day=None)`

Derives a display-only offset from a field or player anchoring rule.

## Predictions

### `probability_future_match(player_a, player_b, ...)`

Returns two-way win probabilities. Set `account_for_uncertainty=True` to
integrate rating uncertainty into the prediction.

### `win_draw_loss_probabilities(player_a, player_b, ...)`

Returns `(win, draw, loss)` under the Davidson model and supports the same
uncertainty integration.

### `draw_tendency`, `draws_declared()`, `nu_from_draw_rate()`, and `draw_rate_from_nu()`

Inspect or convert the global Davidson draw parameter.

## Persistence

### `save_base(path)` and `WHR.load_base(path)`

Serialize and restore a fitted model. Only load pickle files from trusted
sources.

## Low-level objects

`Player`, `PlayerDay`, and `Game` remain importable for inspection and research,
but their lower-level shapes have weaker compatibility guarantees than the
`WHR` surface. Consult docstrings and the changelog before depending on them.
