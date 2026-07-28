# Benchmark guide

The benchmark suite compares WHR, KickScore, and TrueSkill Through Time under a
shared temporal protocol on NBA, ATP tennis, and European football data.

The complete methodology and results live in the repository:

- [Benchmark README](https://github.com/pfmonville/whole_history_rating/tree/master/benchmarks)
- [Full report](https://github.com/pfmonville/whole_history_rating/blob/master/benchmarks/REPORT.md)
- [Committed results](https://github.com/pfmonville/whole_history_rating/tree/master/benchmarks/results)

## Quick smoke test

The deterministic smoke test uses only synthetic data and has no optional
dependencies or network access:

```bash
uv run python benchmarks/smoke.py
```

## Full reproduction

Download the three documented datasets:

```bash
uv run python benchmarks/download_data.py all
```

Then follow the commands in the benchmark README. Full runs are deliberately
manual: they download external data, sweep several model configurations, and
can take hours.

Every newly generated result records:

- the Git commit and dirty-tree state;
- Python, operating system, and machine details;
- versions of WHR and relevant third-party packages;
- a SHA-256 digest and size for every source data file;
- generation time and the random-seed policy.
