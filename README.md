# Whole-History Rating for Python

[![PyPI version](https://img.shields.io/pypi/v/whole-history-rating.svg)](https://pypi.org/project/whole-history-rating/)
[![Python versions](https://img.shields.io/pypi/pyversions/whole-history-rating.svg)](https://pypi.org/project/whole-history-rating/)
[![CI](https://github.com/pfmonville/whole_history_rating/actions/workflows/ci.yml/badge.svg)](https://github.com/pfmonville/whole_history_rating/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pfmonville/whole_history_rating/blob/master/LICENCE.txt)

A maintained, pure-Python implementation of Rémi Coulom's
Whole-History Rating (WHR) algorithm for estimating time-varying skills from
dated pairwise results.

WHR is useful when the history of a competitor's strength matters—not only the
latest leaderboard—and when later evidence should be allowed to refine earlier
ratings.

## Installation

```bash
pip install whole-history-rating
```

Python 3.11 or newer and NumPy 2.0 or newer are required.

## Quickstart

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

Games are formatted as `black_player white_player result time_step`, where the
result is `B`, `W`, or `D`. You can also add structured games with
`create_game()`.

## Why Whole-History Rating?

Unlike an online rating update, WHR jointly revisits each competitor's complete
trajectory when new results are added. A later result can therefore refine the
estimated rating at an earlier date.

This implementation provides:

- retrospective smoothing of time-varying skill;
- uncertainty for ratings, differences, and changes over time;
- uncertainty-aware match predictions;
- binary outcomes and Davidson-model draws;
- learned contextual advantages such as home advantage, handicap, and komi;
- convergence and data-connectivity diagnostics;
- a reproducible comparison with KickScore and TrueSkill Through Time.

The main reason to choose WHR is not that it wins every predictive benchmark—it
does not. Its strength is an interpretable, relatively lightweight model of
complete pairwise histories.

## Benchmark evidence

Lower log-loss is better. All systems below are trained, tuned, and evaluated
under the same temporal protocol.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pfmonville/whole_history_rating/master/benchmarks/results/bench_comparison_dark.png">
  <img alt="Predictive log-loss of WHR, KickScore and TrueSkill Through Time on NBA, ATP tennis and European football data." src="https://raw.githubusercontent.com/pfmonville/whole_history_rating/master/benchmarks/results/bench_comparison_light.png">
</picture>

| Benchmark | Test set | WHR | KickScore | TrueSkill Through Time |
|---|---:|---:|---:|---:|
| NBA | 2018–19, n=1,312 | 0.666 | **0.662** | 0.688 |
| ATP tennis | 2014, n=2,816 | 0.614 | 0.606 | **0.604** |
| Football, three outcomes | 2022–23, n=1,826 | **1.008** | 1.013 | 1.023 |

No system wins every dataset, and the gaps are small. WHR leads the
three-outcome football comparison, KickScore leads the NBA comparison, and
TrueSkill Through Time leads tennis. Domain-specific NBA models using rosters,
injuries, and travel still outperform all three generic rating systems.

The complete protocol, hyperparameter grids, limitations, data sources, and
reproduction commands are in the
[benchmark report](https://github.com/pfmonville/whole_history_rating/blob/master/benchmarks/REPORT.md)
and
[benchmark README](https://github.com/pfmonville/whole_history_rating/tree/master/benchmarks).

## Applications

WHR can be used for dated pairwise-comparison histories such as:

- board games including Go and chess;
- sports and esports rankings;
- historical leaderboards;
- matchmaking analysis;
- human-preference experiments;
- pairwise evaluation of models or systems.

The model observes outcomes, dates, and optional contextual effects. It does not
use domain-specific information such as rosters, injuries, maps, prompts, or
evaluator identities. For LLM evaluation, treat it as a rating component rather
than a complete evaluation methodology.

## Documentation

- [User guide](https://github.com/pfmonville/whole_history_rating/blob/master/docs/user-guide.md):
  complete examples, configuration, uncertainty, draws, handicap, and komi.
- [API reference](https://github.com/pfmonville/whole_history_rating/blob/master/docs/api.md):
  the supported high-level surface.
- [Benchmark report](https://github.com/pfmonville/whole_history_rating/blob/master/benchmarks/REPORT.md):
  protocol, results, and caveats.
- [Changelog](https://github.com/pfmonville/whole_history_rating/blob/master/CHANGELOG.md):
  release-by-release compatibility notes.

The documentation site can be built locally with:

```bash
uv sync --group docs
uv run --group docs mkdocs build --strict
```

## Implementation characteristics

- Pure Python with NumPy used where batching is beneficial.
- No compiled extension required.
- Binary and drawn outcomes.
- Learned contextual advantages.
- Reproducible real-data benchmark suite.
- High-level compatibility alias for the former `Base` class.

The project originated as a port of the
[GoShrine Ruby implementation](https://github.com/goshrine/whole_history_rating)
and has since added a modern API, diagnostics, draws, learned contextual
advantages, uncertainty-aware predictions, and cross-library benchmarks.

## Citation

If you use this package in research, cite both the software and Rémi Coulom's
original WHR paper. GitHub exposes the complete metadata from
[CITATION.cff](https://github.com/pfmonville/whole_history_rating/blob/master/CITATION.cff).

> Rémi Coulom. “Whole-History Rating: A Bayesian Rating System for Players of
> Time-Varying Strength.” *Computers and Games*, LNCS 5131, 113–124, 2008.
> DOI: [10.1007/978-3-540-87608-3_11](https://doi.org/10.1007/978-3-540-87608-3_11).

## Development

```bash
uv sync --dev
uv run ruff check whr tests benchmarks
uv run mypy
uv run pytest
```

The fast CI suite also runs a synthetic benchmark smoke test. Full real-data
benchmarks are intentionally manual because they download external datasets and
can take hours.

## License

MIT. See the
[license text](https://github.com/pfmonville/whole_history_rating/blob/master/LICENCE.txt).
