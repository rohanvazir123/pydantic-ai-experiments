"""
One-shot setup: drop and reload all meeting_analytics tables.

Target: rag_db @ localhost:5434 (rag_user:rag_pass) — pgvector required for Final Version.

Order:
  [1] Take A  — generate_rule_based_taxonomy.py --reset
                Drops + recreates schema, loads base tables + Take A analytical tables.
  [2] Take B  — take_b/load_outputs_to_pg.py
                Creates 3 KMeans tables from take_b/outputs/ CSVs.
  [3] Final   — final_version/semantic_clustering.py --from-outputs --skip-base-load
                Loads 3 semantic tables from pre-computed outputs/ files; no re-embedding.

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
        "Final Version — semantic clusters (3 tables from outputs/)",
        BASE / "final_version" / "semantic_clustering.py",
        ["--from-outputs", "--skip-base-load"],
    )

    print("\n" + "=" * 60)
    print("  All done — tables loaded in meeting_analytics schema")
    print("  Run export_all_to_csv.py to snapshot to CSV.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
