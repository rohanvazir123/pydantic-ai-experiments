"""
Verify meeting_analytics tables after running the final_version loaders.

Sources sql/01_verify_tables.sql -- queries each table individually so missing
Take A / Take B tables are reported as SKIP rather than crashing the script.

Usage (from repo root):
  python basics/iprep/meeting-analytics/final_version/verify.py

Expected row counts:
  -- Final Version base tables (load_raw_jsons_to_db.py) --------------
  meetings                   100
  meeting_participants       311
  meeting_summaries          100
  key_moments                402
  action_items               397
  transcript_lines          4313
  -- Final Version semantic tables (load_output_csvs_to_db.py) --------
  semantic_clusters           26
  semantic_phrases           343
  semantic_meeting_themes    516
  -- Take A (optional -- generate_rule_based_taxonomy.py) --------------
  summary_topics             600
  meeting_themes             466
  call_types                 100
  sentiment_features         100
  -- Take B (optional -- take_b/load_outputs_to_pg.py) -----------------
  kmeans_clusters              8
  kmeans_cluster_terms        96
  kmeans_meeting_clusters    100
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg

SCHEMA = "meeting_analytics"
SCRIPT_DIR = Path(__file__).resolve().parent
SQL_FILE = SCRIPT_DIR.parent / "sql" / "01_verify_tables.sql"

# (expected_count, group_label)
_EXPECTED: dict[str, tuple[int, str]] = {
    # Final Version base tables
    "meetings":                (100,  "base"),
    "meeting_participants":    (311,  "base"),
    "meeting_summaries":       (100,  "base"),
    "key_moments":             (402,  "base"),
    "action_items":            (397,  "base"),
    "transcript_lines":        (4313, "base"),
    # Final Version semantic tables
    "semantic_clusters":       (26,   "semantic"),
    "semantic_phrases":        (343,  "semantic"),
    "semantic_meeting_themes": (516,  "semantic"),
    # Take A -- optional
    "summary_topics":          (600,  "take_a"),
    "meeting_themes":          (466,  "take_a"),
    "call_types":              (100,  "take_a"),
    "sentiment_features":      (100,  "take_a"),
    # Take B -- optional
    "kmeans_clusters":         (8,    "take_b"),
    "kmeans_cluster_terms":    (96,   "take_b"),
    "kmeans_meeting_clusters": (100,  "take_b"),
}

_SPOT_CHECKS: list[tuple[str, str, int]] = [
    (
        "semantic primary themes (one per meeting)",
        f"SELECT count(*) FROM {SCHEMA}.semantic_meeting_themes WHERE is_primary = true",
        100,
    ),
    (
        "action_items_by_theme view (action items with theme)",
        f"SELECT count(*) FROM {SCHEMA}.action_items_by_theme",
        397,
    ),
]


def _load_dotenv() -> None:
    env_file = SCRIPT_DIR.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _build_dsn() -> str:
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "postgres")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


async def _run(dsn: str) -> bool:
    conn = await asyncpg.connect(dsn)
    try:
        return await _verify(conn)
    finally:
        await conn.close()


async def _verify(conn: asyncpg.Connection) -> bool:
    print(f"\nSourcing: {SQL_FILE.relative_to(SCRIPT_DIR.parent.parent.parent)}\n")

    passed = 0
    failed = 0
    skipped = 0
    current_group = ""

    for table, (expected, group) in _EXPECTED.items():
        if group != current_group:
            current_group = group
            labels = {
                "base":     "Final Version base tables (load_raw_jsons_to_db.py)",
                "semantic": "Final Version semantic tables (load_output_csvs_to_db.py)",
                "take_a":   "Take A -- optional",
                "take_b":   "Take B -- optional",
            }
            print(f"  -- {labels[group]} {'-' * max(0, 52 - len(labels[group]))}")

        try:
            row = await conn.fetchrow(
                f"SELECT count(*) AS n FROM {SCHEMA}.{table}"
            )
            actual = int(row["n"])
            ok = actual == expected
            status = "PASS" if ok else "FAIL"
            marker = "*" if ok else "!"
            print(
                f"    {marker} {status}  {table:<28s}  {actual:>5}  (expected {expected})"
            )
            if ok:
                passed += 1
            else:
                failed += 1
        except asyncpg.exceptions.UndefinedTableError:
            print(f"    -  SKIP  {table:<28s}  table not found")
            skipped += 1

    print(f"\n  -- Spot checks {'-' * 42}")
    for label, sql, expected in _SPOT_CHECKS:
        try:
            val = await conn.fetchval(sql)
            actual = int(val)
            ok = actual == expected
            status = "PASS" if ok else "FAIL"
            marker = "*" if ok else "!"
            print(
                f"    {marker} {status}  {label:<38s}  {actual:>5}  (expected {expected})"
            )
            if ok:
                passed += 1
            else:
                failed += 1
        except asyncpg.exceptions.UndefinedTableError:
            print(f"    -  SKIP  {label:<38s}  table not found")
            skipped += 1

    all_ok = failed == 0
    verdict = "ALL PASS" if all_ok else f"{failed} FAILED"
    print(f"\n  {verdict}  --  {passed} passed, {failed} failed, {skipped} skipped\n")
    return all_ok


async def main() -> None:
    _load_dotenv()
    ok = await _run(_build_dsn())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
