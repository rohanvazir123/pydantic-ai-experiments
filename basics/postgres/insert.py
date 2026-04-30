"""
Inserts a new article into the 'articles' table and shows the auto-generated
search_vector to demonstrate the GENERATED ALWAYS AS tsvector column.

Usage:
    python insert.py

    # Pass optional HTML content and JSON metadata:
    python insert.py --html "<p>Custom content</p>" --meta '{"author": "Alice"}'

Environment variables (see config.py for defaults):
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""

import argparse
import asyncio
import json

import asyncpg

import config

DEFAULT_HTML = (
    "<article><h2>PostgreSQL tsvector</h2>"
    "<p>The <strong>GENERATED ALWAYS AS</strong> clause keeps your "
    "full-text <em>search vector</em> in sync automatically.</p></article>"
)

DEFAULT_META = {
    "author": "Alice Wonder",
    "tags": ["postgres", "full-text", "generated-column"],
    "year": 2026,
}


async def main(html: str, meta: dict) -> None:
    print(f"Connecting to {config.HOST}:{config.PORT}/{config.DATABASE} as {config.USER}\n")
    conn: asyncpg.Connection = await asyncpg.connect(config.dsn())

    try:
        meta_json = json.dumps(meta)

        row = await conn.fetchrow(
            """
            INSERT INTO articles (html_content, json_metadata)
            VALUES ($1, $2::jsonb)
            RETURNING id, html_content, json_metadata, search_vector::text
            """,
            html,
            meta_json,
        )

        print("Inserted article:")
        print(f"  id            : {row['id']}")
        print(f"  html_content  : {row['html_content'][:100]!r}")
        print(f"  json_metadata : {row['json_metadata']}")
        print()
        print("Auto-generated search_vector:")
        # Print each lexeme on its own line for readability
        lexemes = row["search_vector"].split(" ")
        for chunk in [lexemes[i : i + 6] for i in range(0, len(lexemes), 6)]:
            print("  " + " ".join(chunk))

        total = await conn.fetchval("SELECT count(*) FROM articles")
        print(f"\nTotal rows in articles: {total}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert a sample article into 'articles'.")
    parser.add_argument("--html", default=DEFAULT_HTML, help="HTML content to insert")
    parser.add_argument(
        "--meta",
        default=None,
        help="JSON metadata string (default: built-in sample)",
    )
    args = parser.parse_args()

    meta = json.loads(args.meta) if args.meta else DEFAULT_META
    asyncio.run(main(args.html, meta))
