"""
Python csv module crash course — using the US Baby Names dataset (SSA).

Run download.py first to fetch the data.

Each section shows a concept, runs it live, then poses a challenge.
Challenges have solutions hidden in the SOLUTIONS section at the bottom.

Usage:
    python csv_crash_course.py
    python csv_crash_course.py --section 3   # jump to a specific section
"""

import argparse
import csv
import io
import os
import sys
from pathlib import Path

# Force UTF-8 output so Unicode chars in print statements work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BABY_DIR = Path(__file__).parent / "datasets" / "baby_names"


def _check_data() -> None:
    if not BABY_DIR.exists() or not any(BABY_DIR.glob("yob*.txt")):
        raise SystemExit("Run download.py first to fetch baby names data.")


def _year_file(year: int) -> Path:
    return BABY_DIR / f"yob{year}.txt"


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  SECTION {n}: {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — csv.reader: raw row-by-row reading
# ─────────────────────────────────────────────────────────────
def s1_reader() -> None:
    section(1, "csv.reader — raw row iteration")
    f = _year_file(2023)
    print(f"Reading {f.name} with csv.reader …\n")

    with open(f, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)          # returns lists of strings
        rows = list(reader)

    print(f"Total rows: {len(rows):,}")
    print(f"First 5 rows (raw lists):")
    for row in rows[:5]:
        print(f"  {row}")

    # Each row is [name, sex, count_str]. All values are strings.
    print(f"\nDtype of count: {type(rows[0][2])!r}  ← always str, convert manually")

    # Aggregate manually
    total_births = sum(int(row[2]) for row in rows)
    print(f"Total recorded births in 2023: {total_births:,}")

    print("""
KEY POINTS
  csv.reader(fileobj)         → iterator of lists
  open(path, newline="")      → required; csv handles line endings internally
  All values are strings      → cast with int(), float() as needed
  dialect="excel"             → default (comma-separated, double-quote quoting)
""")

    print("""CHALLENGE 1
  How many UNIQUE names appear in the 2023 file (regardless of sex)?
  Hint: build a set from column 0.
  (Solution at bottom of file)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — csv.DictReader: named fields
# ─────────────────────────────────────────────────────────────
def s2_dictreader() -> None:
    section(2, "csv.DictReader — named-field access")

    # Baby names files have NO header row; we supply fieldnames ourselves.
    f = _year_file(2023)
    FIELDS = ["name", "sex", "count"]

    with open(f, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, fieldnames=FIELDS)
        rows = [{"name": r["name"], "sex": r["sex"], "count": int(r["count"])}
                for r in reader]

    print(f"Top 5 female names in 2023:")
    female = sorted([r for r in rows if r["sex"] == "F"], key=lambda r: -r["count"])
    for rank, r in enumerate(female[:5], 1):
        print(f"  {rank}. {r['name']:<15} {r['count']:>8,} births")

    print(f"\nTop 5 male names in 2023:")
    male = sorted([r for r in rows if r["sex"] == "M"], key=lambda r: -r["count"])
    for rank, r in enumerate(male[:5], 1):
        print(f"  {rank}. {r['name']:<15} {r['count']:>8,} births")

    print("""
KEY POINTS
  csv.DictReader(fh, fieldnames=[...])  → rows are dicts (or OrderedDicts in <3.7)
  If the file HAS a header row, omit fieldnames — it is used automatically.
  row["name"]  is cleaner than row[0] and survives column reordering.
""")

    print("""CHALLENGE 2
  Find the top-3 male names in 1950.
  Which name has changed the most in rank between 1950 and 2023?
  Hint: compare the 1950 rank-1 male name with its 2023 rank.
  (Solution at bottom of file)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — Filtering across multiple files
# ─────────────────────────────────────────────────────────────
def s3_multi_file() -> None:
    section(3, "Reading multiple files — tracking a name across decades")

    TARGET = "Emma"
    years = range(1980, 2024)
    FIELDS = ["name", "sex", "count"]
    results: list[dict] = []

    for year in years:
        f = _year_file(year)
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, fieldnames=FIELDS):
                if row["name"] == TARGET and row["sex"] == "F":
                    results.append({"year": year, "count": int(row["count"])})

    print(f"Popularity of '{TARGET}' (female) from 1980:")
    print(f"  {'Year':>4}  {'Births':>8}")
    print(f"  {'─' * 15}")
    for r in results:
        bar = "█" * (r["count"] // 2000)
        print(f"  {r['year']:>4}  {r['count']:>8,}  {bar}")

    print("""
KEY POINTS
  Iterate files in a loop; keep all logic in pure Python dicts / lists.
  For larger-scale multi-file work, pandas.concat is more ergonomic (see next script).
""")

    print("""CHALLENGE 3
  Find every name that was a top-10 female name in BOTH 2000 AND 2023.
  Hint: build two sets of names, then use set intersection.
  (Solution at bottom of file)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — csv.writer / csv.DictWriter: writing output
# ─────────────────────────────────────────────────────────────
def s4_writer() -> None:
    section(4, "csv.writer / csv.DictWriter — writing CSV output")

    # Build a summary: total births per name (female, 2020–2023)
    FIELDS = ["name", "sex", "count"]
    totals: dict[str, int] = {}

    for year in range(2020, 2024):
        f = _year_file(year)
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, fieldnames=FIELDS):
                if row["sex"] == "F":
                    totals[row["name"]] = totals.get(row["name"], 0) + int(row["count"])

    top20 = sorted(totals.items(), key=lambda kv: -kv[1])[:20]

    # ── csv.writer ───────────────────────────────────────────
    out_simple = Path(__file__).parent / "datasets" / "top_female_2020_2023.csv"
    with open(out_simple, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "name", "total_births"])          # header
        for rank, (name, count) in enumerate(top20, 1):
            w.writerow([rank, name, count])
    print(f"Written {out_simple.name} (csv.writer)\n")

    # ── csv.DictWriter ───────────────────────────────────────
    out_dict = Path(__file__).parent / "datasets" / "top_female_2020_2023_dict.csv"
    with open(out_dict, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["rank", "name", "total_births"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rank, (name, count) in enumerate(top20, 1):
            w.writerow({"rank": rank, "name": name, "total_births": count})
    print(f"Written {out_dict.name} (csv.DictWriter)\n")

    # Show first few lines of output
    with open(out_simple, newline="", encoding="utf-8") as fh:
        print("Preview:")
        for line in list(csv.reader(fh))[:6]:
            print(f"  {line}")

    print("""
KEY POINTS
  csv.writer(fh)                  → write rows as lists
  csv.DictWriter(fh, fieldnames)  → write rows as dicts (columns explicit)
  Always open with newline=""     → let csv control line endings (avoids blank lines on Windows)
  w.writeheader()                 → writes the fieldnames row
  quoting=csv.QUOTE_ALL           → force-quote every field
  quoting=csv.QUOTE_NONNUMERIC    → quote strings, leave numbers bare
""")

    print("""CHALLENGE 4
  Write a CSV of all names that ONLY appeared as female names in 2023
  (i.e., no male entries for that name in 2023).
  Columns: name, female_count
  Sort by female_count descending.
  (Solution at bottom of file)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — csv.Sniffer and StringIO
# ─────────────────────────────────────────────────────────────
def s5_sniffer() -> None:
    section(5, "csv.Sniffer — auto-detect dialect + StringIO for in-memory CSV")

    # Simulate a tab-separated "file" in memory
    tsv_data = "name\tsex\tcount\nOlivia\tF\t18735\nLiam\tM\t19954\n"
    fh = io.StringIO(tsv_data)

    dialect = csv.Sniffer().sniff(tsv_data[:128])
    print(f"Detected delimiter: {dialect.delimiter!r}")

    fh.seek(0)
    reader = csv.DictReader(fh, dialect=dialect)
    for row in reader:
        print(f"  {row}")

    print("""
KEY POINTS
  csv.Sniffer().sniff(sample)  → detects delimiter and quoting style
  io.StringIO(text)            → treat a string as a file-like object
  Use StringIO when CSV comes from an API response (response.text) instead of a file.
""")

    print("""CHALLENGE 5
  You receive a pipe-separated string:
      "Alice|F|4200\\nBob|M|3100\\nCarol|F|2800\\n"
  Parse it into a list of dicts using csv.DictReader with the correct delimiter.
  (Solution at bottom of file)
""")


# ─────────────────────────────────────────────────────────────
# SOLUTIONS
# ─────────────────────────────────────────────────────────────
SOLUTIONS = """
╔══════════════════════════════════════════════════════════╗
║                      SOLUTIONS                           ║
╚══════════════════════════════════════════════════════════╝

SOLUTION 1 — unique names in 2023
──────────────────────────────────
    FIELDS = ["name", "sex", "count"]
    with open(_year_file(2023), newline="") as fh:
        reader = csv.DictReader(fh, fieldnames=FIELDS)
        names = {row["name"] for row in reader}
    print(len(names))   # ~17,000+

SOLUTION 2 — top-3 male names in 1950 vs 2023
───────────────────────────────────────────────
    def top_names(year, sex, n=3):
        FIELDS = ["name", "sex", "count"]
        with open(_year_file(year), newline="") as fh:
            rows = [r for r in csv.DictReader(fh, fieldnames=FIELDS) if r["sex"] == sex]
        rows.sort(key=lambda r: -int(r["count"]))
        return [r["name"] for r in rows[:n]]

    top_1950 = top_names(1950, "M")    # e.g. ['James', 'Robert', 'John']
    top_2023 = top_names(2023, "M")
    for name in top_1950:
        rank_2023 = top_2023.index(name) + 1 if name in top_2023 else "outside top 3"
        print(f"{name}: rank 1950=1 → 2023={rank_2023}")

SOLUTION 3 — names in top-10 female in both 2000 and 2023
───────────────────────────────────────────────────────────
    def top10_female(year):
        FIELDS = ["name", "sex", "count"]
        with open(_year_file(year), newline="") as fh:
            rows = [r for r in csv.DictReader(fh, fieldnames=FIELDS) if r["sex"] == "F"]
        rows.sort(key=lambda r: -int(r["count"]))
        return {r["name"] for r in rows[:10]}

    overlap = top10_female(2000) & top10_female(2023)
    print(overlap)   # e.g. {'Emily', 'Hannah', 'Emma', ...}

SOLUTION 4 — female-only names in 2023
────────────────────────────────────────
    FIELDS = ["name", "sex", "count"]
    female, male = {}, set()
    with open(_year_file(2023), newline="") as fh:
        for row in csv.DictReader(fh, fieldnames=FIELDS):
            if row["sex"] == "F":
                female[row["name"]] = int(row["count"])
            else:
                male.add(row["name"])
    female_only = {n: c for n, c in female.items() if n not in male}
    with open("female_only_2023.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "female_count"])
        w.writeheader()
        for name, count in sorted(female_only.items(), key=lambda kv: -kv[1]):
            w.writerow({"name": name, "female_count": count})

SOLUTION 5 — pipe-separated string
─────────────────────────────────────
    import csv, io
    data = "Alice|F|4200\\nBob|M|3100\\nCarol|F|2800\\n"
    reader = csv.DictReader(io.StringIO(data),
                            fieldnames=["name", "sex", "count"],
                            delimiter="|")
    rows = [dict(r) for r in reader]
    print(rows)
"""


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
SECTIONS = {1: s1_reader, 2: s2_dictreader, 3: s3_multi_file, 4: s4_writer, 5: s5_sniffer}


def main() -> None:
    _check_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, choices=SECTIONS.keys(),
                        help="Run only this section (default: all)")
    parser.add_argument("--solutions", action="store_true", help="Print all solutions")
    args = parser.parse_args()

    if args.solutions:
        print(SOLUTIONS)
        return

    sections_to_run = [args.section] if args.section else list(SECTIONS.keys())
    for n in sections_to_run:
        SECTIONS[n]()

    print("\n" + "=" * 60)
    print("  Run with --solutions to see all challenge answers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
