from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.download_data import download_plan
from benchmarks.provenance import build_provenance

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def test_download_plan_covers_every_benchmark_dataset():
    nba = download_plan("nba")
    tennis = download_plan("tennis")
    football = download_plan("football")

    assert len(nba) == 1
    assert nba[0].destination.name == "nba_elo.csv"
    assert len(tennis) == 16
    assert tennis[0].destination.name == "atp_matches_2000.csv"
    assert tennis[-1].destination.name == "atp_matches_2015.csv"
    assert len(football) == 50
    assert football[0].destination.name == "2014-15_de.1.json"
    assert football[-1].destination.name == "2023-24_it.1.json"


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


def test_generated_results_do_not_mark_source_as_dirty(tmp_path):
    repo = tmp_path / "repo"
    results = repo / "benchmarks" / "results"
    data = repo / "benchmarks" / "data"
    results.mkdir(parents=True)
    data.mkdir(parents=True)
    dataset = data / "matches.csv"
    dataset.write_text("winner,loser\nalice,bob\n", encoding="utf-8")
    result = results / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    for command in (
        ("init",),
        ("config", "user.email", "benchmark@example.test"),
        ("config", "user.name", "Benchmark Test"),
        ("add", "."),
        ("commit", "-m", "initial"),
    ):
        subprocess.run(
            ["git", *command],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    result.write_text('{"updated": true}\n', encoding="utf-8")
    provenance = build_provenance(
        dataset_source="https://example.test/matches.csv",
        dataset_files=[dataset],
        repo_root=repo,
    )
    assert provenance["source"]["dirty"] is False

    source.write_text("VALUE = 2\n", encoding="utf-8")
    provenance = build_provenance(
        dataset_source="https://example.test/matches.csv",
        dataset_files=[dataset],
        repo_root=repo,
    )
    assert provenance["source"]["dirty"] is True


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
