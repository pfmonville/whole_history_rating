"""Download the public datasets used by the real-data benchmark suite."""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NBA_URL = "https://lum-public.s3.eu-west-1.amazonaws.com/nba_elo.csv"
TENNIS_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    "atp_matches_{year}.csv"
)
FOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/"
    "{season}/{league}.json"
)


@dataclass(frozen=True)
class Download:
    url: str
    destination: Path


def download_plan(dataset: str) -> list[Download]:
    """Return the deterministic download plan for one dataset or for all."""
    plans = {
        "nba": [Download(NBA_URL, DATA / "nba_elo.csv")],
        "tennis": [
            Download(
                TENNIS_URL.format(year=year),
                DATA / "tennis" / f"atp_matches_{year}.csv",
            )
            for year in range(2000, 2016)
        ],
        "football": [
            Download(
                FOOTBALL_URL.format(season=season, league=league),
                DATA / "football" / f"{season}_{league}.json",
            )
            for season in (
                "2014-15",
                "2015-16",
                "2016-17",
                "2017-18",
                "2018-19",
                "2019-20",
                "2020-21",
                "2021-22",
                "2022-23",
                "2023-24",
            )
            for league in ("de.1", "en.1", "es.1", "fr.1", "it.1")
        ],
    }
    if dataset == "all":
        return plans["nba"] + plans["tennis"] + plans["football"]
    try:
        return plans[dataset]
    except KeyError as exc:
        choices = ", ".join([*plans, "all"])
        raise ValueError(f"unknown dataset {dataset!r}; choose {choices}") from exc


def download(spec: Download, *, force: bool = False) -> str:
    """Download one file atomically, or return ``cached`` when it exists."""
    if spec.destination.is_file() and not force:
        return "cached"

    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.destination.with_suffix(spec.destination.suffix + ".part")
    request = urllib.request.Request(
        spec.url, headers={"User-Agent": "whole-history-rating-benchmark/3.6.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if not payload:
            raise RuntimeError(f"downloaded an empty response from {spec.url}")
        temporary.write_bytes(payload)
        temporary.replace(spec.destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("nba", "tennis", "football", "all"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for spec in download_plan(args.dataset):
        status = download(spec, force=args.force)
        print(f"{status:10} {spec.destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
