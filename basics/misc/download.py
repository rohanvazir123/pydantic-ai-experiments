"""
Downloads three public-domain datasets to the datasets/ folder.

Datasets
--------
1. US Baby Names (SSA, 1880-present)
   - ZIP of ~140 year-files, each a 3-column CSV: Name, Sex, Count
   - ~25 MB extracted

2. Our World in Data - COVID-19 (OWID)
   - Single large CSV with 67+ columns and 300k+ rows
   - ~100 MB

3. World Bank - GDP (current USD) by Country
   - ZIP containing an XLS (wide format: one column per year)
   - ~3 MB extracted

Usage
-----
    python download.py

    # Force re-download (ignore cached files):
    python download.py --force
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests

DATASETS_DIR = Path(__file__).parent / "datasets"

SOURCES = {
    "baby_names": {
        "url": "https://www.ssa.gov/oact/babynames/names.zip",
        "dest": DATASETS_DIR / "baby_names.zip",
        "extract_to": DATASETS_DIR / "baby_names",
        "description": "US Baby Names 1880-present (SSA)",
        # SSA blocks programmatic downloads; manual fallback shown on 403
        "manual": "https://www.ssa.gov/oact/babynames/limits.html  ->  click 'National data'",
    },
    "owid_covid": {
        # GitHub raw mirror - same file, more reliable than the primary URL
        "url": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        "dest": DATASETS_DIR / "owid_covid.csv",
        "extract_to": None,
        "description": "OWID COVID-19 dataset (300k+ rows)",
    },
    "world_bank_gdp": {
        "url": "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=excel",
        "dest": DATASETS_DIR / "world_bank_gdp.zip",
        "extract_to": DATASETS_DIR / "world_bank_gdp",
        "description": "World Bank GDP per country (1960-present)",
    },
}


_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _download(url: str, dest: Path, extra_headers: dict | None = None,
              manual: str | None = None, force: bool = False) -> bool:
    """Stream-download url -> dest; returns True if actually downloaded.

    If the server returns 403 and manual instructions are provided, print them
    and return False instead of raising.
    """
    if dest.exists() and not force:
        print(f"  [cached] {dest.name} ({dest.stat().st_size // 1024:,} KB)")
        return False

    headers = {**_BASE_HEADERS, **(extra_headers or {})}
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            if r.status_code == 403 and manual:
                print(f"\r  [BLOCKED] Manual download required:")
                print(f"     {manual}")
                print(f"     Save the file to: {dest}")
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 17):  # 128 KB
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  Downloading {dest.name} ... {pct:3d}%", end="", flush=True)
    except requests.HTTPError as exc:
        print(f"\r  [FAILED] {exc}")
        raise
    size_kb = dest.stat().st_size // 1024
    print(f"\r  Downloaded  {dest.name} - {size_kb:,} KB")
    return True


def _extract_zip(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest_dir)
    count = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
    print(f"  Extracted {count} file(s) -> {dest_dir.relative_to(Path(__file__).parent)}/")


def main(force: bool = False) -> None:
    DATASETS_DIR.mkdir(exist_ok=True)

    for key, src in SOURCES.items():
        print(f"\n{'-' * 55}")
        print(f"  {src['description']}")

        # If the extract target already has files, treat as fully cached
        extract_to = src.get("extract_to")
        if extract_to and extract_to.exists() and any(extract_to.iterdir()):
            count = sum(1 for _ in extract_to.rglob("*") if _.is_file())
            print(f"  [cached] {extract_to.name}/ ({count} file(s) already extracted)")
            continue

        downloaded = _download(src["url"], src["dest"],
                               extra_headers=src.get("headers"),
                               manual=src.get("manual"),
                               force=force)
        if extract_to and (downloaded or not extract_to.exists()):
            _extract_zip(src["dest"], extract_to)

    print(f"\n{'-' * 55}")
    print("All datasets ready in datasets/\n")
    print("Next steps:")
    print("  python csv_crash_course.py    # python csv module walkthrough")
    print("  python pandas_crash_course.py # pandas walkthrough + challenges")
    print("  python load_to_postgres.py    # load data into local PostgreSQL")
    print("  psql -f sql_challenges.sql    # work through SQL challenges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    try:
        main(force=args.force)
    except requests.RequestException as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)
