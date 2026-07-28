# Whole-History Rating for Python

`whole-history-rating` estimates how competitor strength changes through time
from dated pairwise outcomes. It implements Rémi Coulom's Whole-History Rating
algorithm in pure Python and adds draws, uncertainty-aware predictions, learned
contextual advantages, diagnostics, and reproducible real-data benchmarks.

## Install

```bash
pip install whole-history-rating
```

## Minimal example

```python
from whr import WHR

model = WHR()
model.load_games(
    [
        "alice bob B 1",
        "alice bob W 2",
        "alice bob B 3",
    ]
)
model.auto_iterate()

print(model.ratings_for_player("alice"))
print(model.probability_future_match("alice", "bob"))
```

## Where to go next

- The [user guide](user-guide.md) covers games, convergence, uncertainty,
  draws, contextual advantages, persistence, and configuration.
- The [API reference](api.md) lists the supported high-level surface.
- The [benchmark guide](benchmarks.md) explains how to reproduce and interpret
  the real-data comparisons.

WHR is most useful when historical trajectories matter and later results should
be allowed to refine earlier ratings. It is a generic pairwise model: it does
not consume domain-specific features such as rosters, injuries, maps, prompts,
or evaluator identities.
