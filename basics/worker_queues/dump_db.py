"""Dump the contents of a SQLite database to stdout.

A tiny inspection helper for the telemetry DB written by ``io_workloads_fixed.py``
(or any SQLite file). Lists every user table, its schema, row count, and rows.

Usage
-----
    python dump_db.py                      # dumps telemetry.db
    python dump_db.py path/to/other.db     # dumps a specific file
    python dump_db.py telemetry.db --table telemetry   # one table only
    python dump_db.py telemetry.db --limit 20          # cap rows per table

A plain synchronous ``sqlite3`` connection is fine here: this is a one-shot,
read-only CLI tool, not part of the async service.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """Return the names of user tables (excludes internal ``sqlite_*`` tables)."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def dump_table(conn: sqlite3.Connection, table: str, limit: int | None) -> None:
    """Print one table's schema, row count, and (up to ``limit``) rows."""
    # Schema (the original CREATE statement).
    schema_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()

    total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    print(f"\n=== table: {table} ({total} rows) ===")
    if schema_row and schema_row[0]:
        print(schema_row[0].strip())

    query = f'SELECT * FROM "{table}"'
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    cursor = conn.execute(query)

    columns = [d[0] for d in cursor.description]
    print(" | ".join(columns))
    print("-" * 40)

    shown = 0
    for row in cursor:
        print(" | ".join(str(value) for value in row))
        shown += 1

    if limit is not None and total > shown:
        print(f"... ({total - shown} more rows not shown; raise --limit to see them)")


def dump_database(db_path: str, table: str | None, limit: int | None) -> int:
    """Dump all tables (or one). Returns a process exit code."""
    if not Path(db_path).exists():
        print(f"error: database file not found: {db_path}")
        return 1

    # ``uri=True`` + ``mode=ro`` opens read-only so we never mutate the file.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = list_tables(conn)
        if table is not None:
            if table not in tables:
                print(f"error: table {table!r} not found. Available: {tables or '(none)'}")
                return 1
            tables = [table]

        print(f"database: {db_path}")
        if not tables:
            print("(no user tables)")
            return 0

        for name in tables:
            dump_table(conn, name, limit)
        return 0
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the contents of a SQLite database.")
    parser.add_argument(
        "db_path",
        nargs="?",
        default="telemetry.db",
        help="Path to the SQLite file (default: telemetry.db)",
    )
    parser.add_argument("--table", help="Only dump this table (default: all tables)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to print per table (default: no limit)",
    )
    args = parser.parse_args()

    raise SystemExit(dump_database(args.db_path, args.table, args.limit))


if __name__ == "__main__":
    main()
