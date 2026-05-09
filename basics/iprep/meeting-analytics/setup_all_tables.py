"""
One-shot setup: drop and reload all meeting_analytics tables for all three takes.

Target: rag_db @ localhost:5434 (rag_user:rag_pass) — pgvector required for Take C.

Order:
  [1] Take A  — generate_rule_based_taxonomy.py --reset
                Drops + recreates schema, loads 10 tables from raw dataset JSON.
  [2] Take B  — take_b/load_outputs_to_pg.py
                Creates 3 new tables, loads from take_b/outputs/ CSVs.
  [3] Take C  — take_c/load_outputs_to_pg.py
                Creates 3 new tables, loads from take_c/outputs/ JSON + CSVs.

Usage (from repo root):
  python basics/iprep/meeting-analytics/setup_all_tables.py
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def run(label: str, script: Path, extra_args: list[str] | None = None) -> None:
    args = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(args, cwd=Path(__file__).resolve().parent.parent.parent)
    if result.returncode != 0:
        print(f"\nERROR: {label} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    print("\nSetting up all meeting_analytics tables on rag_db @ localhost:5434\n")

    run(
        "Take A — rule-based taxonomy (reset + reload 10 tables)",
        BASE / "take_a" / "generate_rule_based_taxonomy.py",
        ["--reset"],
    )

    run(
        "Take B — KMeans clusters (3 tables from outputs/)",
        BASE / "take_b" / "load_outputs_to_pg.py",
    )

    run(
        "Take C — semantic clusters (3 tables from outputs/)",
        BASE / "take_c" / "load_outputs_to_pg.py",
    )

    print("\n" + "=" * 60)
    print("  All done — 16 tables loaded in meeting_analytics schema")
    print("  Run export_all_to_csv.py to snapshot to CSV.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
