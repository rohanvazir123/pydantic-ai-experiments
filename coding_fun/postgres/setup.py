"""
Runs create_table.sql against the local PostgreSQL database.

What the SQL does:
  1. Creates strip_html() — a helper that strips HTML tags via regex
  2. Drops + recreates the 'articles' table with a GENERATED tsvector column
     combining stripped HTML content and all JSONB string values
  3. Creates a GIN index on the search_vector column
  4. Inserts two sample articles

Usage:
    python setup.py

Environment variables (see config.py for defaults):
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

import asyncio
from pathlib import Path

import asyncpg

import config

SQL_FILE = Path(__file__).parent / "create_table.sql"


def _patch_sql(sql: str) -> str:
    # The file has a bare DROP TABLE which fails when the table doesn't exist yet.
    # Patch it to DROP TABLE IF EXISTS so setup is idempotent on the first run.
    return sql.replace("DROP TABLE articles", "DROP TABLE IF EXISTS articles", 1)


async def main() -> None:
    print(f"Connecting to {config.HOST}:{config.PORT}/{config.DATABASE} as {config.USER}")
    conn: asyncpg.Connection = await asyncpg.connect(config.dsn())

    try:
        sql = _patch_sql(SQL_FILE.read_text())
        print(f"\nExecuting {SQL_FILE.name} ...\n")
        await conn.execute(sql)

        count = await conn.fetchval("SELECT count(*) FROM articles")
        print(f"Done — articles table has {count} row(s).")

        # Show what the generated search_vector looks like for each row.
        print("\nGenerated search_vector values:")
        rows = await conn.fetch("SELECT id, search_vector::text FROM articles ORDER BY id")
        for row in rows:
            print(f"  id={row['id']}  {row['search_vector'][:120]}...")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
