"""Validate that built distributions contain only intentional project files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


FORBIDDEN_SDIST_PARTS = (
    "/.agents/",
    "/.claude/",
    "/.codex/",
    "/benchmarks/data/",
    "/benchmarks/results/",
    "/docs/superpowers/",
)
REQUIRED_SDIST_SUFFIXES = (
    "/README.md",
    "/CITATION.cff",
    "/LICENCE.txt",
    "/pyproject.toml",
    "/whr/__init__.py",
)


def _single_match(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one {pattern!r} in {dist_dir}, found {matches}"
        )
    return matches[0]


def validate_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()

    for part in FORBIDDEN_SDIST_PARTS:
        offenders = [name for name in names if part in f"/{name}/"]
        if offenders:
            raise AssertionError(f"{path.name} contains forbidden {part}: {offenders[:5]}")

    for suffix in REQUIRED_SDIST_SUFFIXES:
        if not any(name.endswith(suffix) for name in names):
            raise AssertionError(f"{path.name} is missing required file {suffix}")


def validate_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

    unexpected = [
        name
        for name in names
        if not (name.startswith("whr/") or ".dist-info/" in name)
    ]
    if unexpected:
        raise AssertionError(f"{path.name} contains unexpected files: {unexpected[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()

    validate_sdist(_single_match(args.dist_dir, "*.tar.gz"))
    validate_wheel(_single_match(args.dist_dir, "*.whl"))


if __name__ == "__main__":
    main()
