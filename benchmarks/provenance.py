"""Machine-readable provenance for committed benchmark results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_is_dirty(repo_root: Path) -> bool:
    """Report source changes while ignoring generated benchmark outputs."""
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)benchmarks/results/**",
    )
    return bool(status)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    dataset_source: str,
    dataset_files: Iterable[str | Path],
    package_names: Iterable[str] = (),
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture enough context to audit and reproduce one benchmark result."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    paths = sorted((Path(path).resolve() for path in dataset_files), key=str)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark dataset files are missing: {missing}")

    requested_packages = dict.fromkeys(["whole-history-rating", *package_names])
    packages = {name: _package_version(name) for name in requested_packages}
    files = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "commit": _git(root, "rev-parse", "HEAD") or "unknown",
            "dirty": _source_is_dirty(root),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "dataset": {
            "source": dataset_source,
            "files": files,
        },
        "random_seed": None,
    }


def attach_provenance(
    result: dict[str, Any],
    *,
    dataset_source: str,
    dataset_files: Iterable[str | Path],
    package_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Attach provenance to a result dictionary and return that dictionary."""
    result["provenance"] = build_provenance(
        dataset_source=dataset_source,
        dataset_files=dataset_files,
        package_names=package_names,
    )
    return result
