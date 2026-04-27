"""
Runs each SELECT query from search.sql against the local PostgreSQL database
and pretty-prints the results.

search.sql contains two queries:
  1. phraseto_tsquery — matches the exact phrase "advanced phrase search"
  2. to_tsquery       — matches articles containing both "Jane" and "html"

Usage:
    python search.py

Environment variables (see config.py for defaults):
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

import asyncio
import textwrap
from pathlib import Path

import asyncpg

import config

SQL_FILE = Path(__file__).parent / "search.sql"


def _split_queries(sql: str) -> list[str]:
    """Split a SQL file into individual statements, dropping blanks."""
    return [q.strip() for q in sql.split(";") if q.strip()]


def _fmt_row(row: asyncpg.Record) -> str:
    html = (row["html_content"] or "")[:80].replace("\n", " ")
    meta = str(row["json_metadata"])[:60]
    vec = (row["search_vector"] or "")[:80]
    return (
        f"  id            : {row['id']}\n"
        f"  html_content  : {html!r}{'...' if len(row['html_content'] or '') > 80 else ''}\n"
        f"  json_metadata : {meta}{'...' if len(meta) >= 60 else ''}\n"
        f"  search_vector : {vec}..."
    )


async def main() -> None:
    print(f"Connecting to {config.HOST}:{config.PORT}/{config.DATABASE} as {config.USER}\n")
    conn: asyncpg.Connection = await asyncpg.connect(config.dsn())

    try:
        queries = _split_queries(SQL_FILE.read_text())
        print(f"Found {len(queries)} query/queries in {SQL_FILE.name}\n")
        print("=" * 60)

        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i} ---")
            print(textwrap.indent(query, "  "))
            print()

            rows = await conn.fetch(query)

            if rows:
                print(f"  {len(rows)} row(s) matched:\n")
                for row in rows:
                    print(_fmt_row(row))
                    print()
            else:
                print("  (no rows matched)\n")

            print("-" * 60)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
