from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.provenance import build_provenance


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def test_build_provenance_records_source_runtime_packages_and_dataset(tmp_path):
    dataset = tmp_path / "matches.csv"
    dataset.write_text("winner,loser\nalice,bob\n", encoding="utf-8")

    provenance = build_provenance(
        dataset_source="https://example.test/matches.csv",
        dataset_files=[dataset],
        package_names=["numpy"],
        repo_root=ROOT,
    )

    assert provenance["schema_version"] == 1
    assert provenance["source"]["commit"]
    assert isinstance(provenance["source"]["dirty"], bool)
    assert provenance["runtime"]["python"]
    assert provenance["runtime"]["platform"]
    assert provenance["packages"]["whole-history-rating"] == "3.6.1"
    assert provenance["packages"]["numpy"]
    assert provenance["dataset"]["source"] == "https://example.test/matches.csv"
    assert provenance["dataset"]["files"] == [
        {
            "path": dataset.name,
            "size": dataset.stat().st_size,
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        }
    ]
    assert provenance["random_seed"] is None
    assert provenance["generated_at"].endswith("Z")


def test_committed_result_files_have_provenance():
    paths = sorted(RESULTS.glob("*_results.json")) + sorted(
        RESULTS.glob("versus_*.json")
    )
    assert paths
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        provenance = result["provenance"]
        assert provenance["schema_version"] == 1
        assert "source" in provenance
        assert "packages" in provenance
        assert "dataset" in provenance
        assert "random_seed" in provenance


def test_synthetic_benchmark_smoke_script_runs():
    completed = subprocess.run(
        [sys.executable, "benchmarks/smoke.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "benchmark smoke: ok" in completed.stdout
