"""
Export every meeting_analytics table to CSV.

Writes to meeting-analytics/outputs/csv/ so all pipeline data
has a flat-file fallback independent of Postgres.

Usage (from repo root):
  python basics/iprep/meeting-analytics/export_all_to_csv.py
"""

from __future__ import annotations

import asyncio
import csv
import os
from pathlib import Path

import asyncpg

SCHEMA = "meeting_analytics"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "csv"

TABLES = [
    # Take A tables
    "meetings",
    "meeting_participants",
    "meeting_summaries",
    "summary_topics",
    "action_items",
    "key_moments",
    "transcript_lines",
    "meeting_themes",
    "call_types",
    "sentiment_features",
    # Take C tables
    "semantic_clusters",
    "semantic_phrases",
    "semantic_meeting_themes",
]


def _load_dotenv() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
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


async def export_table(conn: asyncpg.Connection, table: str, out_dir: Path) -> int:
    rows = await conn.fetch(f"SELECT * FROM {SCHEMA}.{table}")
    if not rows:
        return 0
    out_path = out_dir / f"{table}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            # Convert non-string types (vectors, arrays) to strings
            writer.writerow({
                k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                for k, v in dict(row).items()
            })
    return len(rows)


async def main() -> None:
    _load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = await asyncpg.connect(_build_dsn())
    try:
        print(f"\nExporting to {OUT_DIR}\n")
        for table in TABLES:
            try:
                n = await export_table(conn, table, OUT_DIR)
                print(f"  {table:<35} {n:>5} rows")
            except Exception as e:
                print(f"  {table:<35} SKIP ({e})")
    finally:
        await conn.close()

    print(f"\nDone. All CSVs in {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
