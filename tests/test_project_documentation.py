from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_onboards_before_presenting_benchmarks():
    headings = {
        heading: README.index(heading)
        for heading in (
            "## Installation",
            "## Quickstart",
            "## Why Whole-History Rating?",
            "## Benchmark evidence",
        )
    }
    assert list(headings.values()) == sorted(headings.values())
    assert "conversion from the original Ruby" not in README[:1000]


def test_pypi_readme_uses_portable_links_and_real_ci_badge():
    relative_links = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", README)
    assert relative_links == []
    assert "actions/workflows/ci.yml/badge.svg" in README
    assert "actions/workflows/tests.yml" not in README


def test_documentation_and_citation_files_are_configured():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in citation
    assert 'version: "3.6.1"' in citation
    assert "doi: 10.1007/978-3-540-87608-3_11" in citation
    assert "type: conference-paper" in citation

    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "User guide:" in mkdocs
    assert "API reference:" in mkdocs
    assert (ROOT / "docs" / "index.md").is_file()
    assert (ROOT / "docs" / "user-guide.md").is_file()
    assert (ROOT / "docs" / "api.md").is_file()


def test_sdist_selection_is_explicit():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/whr" in sdist["include"]
    assert "/README.md" in sdist["include"]
    assert "/.claude" in sdist["exclude"]
    assert "/benchmarks/data" in sdist["exclude"]
    assert "/benchmarks/results" in sdist["exclude"]


def test_ci_has_fast_smoke_and_manual_full_benchmarks():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    full = (
        ROOT / ".github" / "workflows" / "full-benchmarks.yml"
    ).read_text(encoding="utf-8")
    assert "benchmarks/smoke.py" in ci
    assert "workflow_dispatch:" in full
    assert "benchmarks/download_data.py all" in full
    for dataset in ("tennis", "nba", "football"):
        assert f"benchmarks/versus.py {dataset}" in full
