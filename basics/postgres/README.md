# PostgreSQL Full-Text Search Demo

Demonstrates PostgreSQL's built-in full-text search using a `GENERATED ALWAYS AS` tsvector column, a custom HTML-stripping function, and JSONB metadata indexing — all driven by Python scripts using **asyncpg**.

## Files

| File | Purpose |
|------|---------|
| `create_table.sql` | Creates `strip_html()`, the `articles` table, a GIN index, and inserts 2 sample rows |
| `search.sql` | Two SELECT queries: phrase search and boolean AND search |
| `config.py` | PostgreSQL connection settings (reads `PG_*` env vars) |
| `setup.py` | Runs `create_table.sql` against the local database |
| `search.py` | Runs each query from `search.sql` and pretty-prints results |
| `insert.py` | Inserts a new article and shows the auto-generated `search_vector` |

## Prerequisites

- PostgreSQL running locally (tested on 18.3, default port 5432)
- `asyncpg` installed: `pip install asyncpg`

## Quick Start

```bash
cd coding_fun/postgres

# 1. Create the table and seed sample data
python setup.py

# 2. Run the full-text search queries
python search.py

# 3. Insert a new article and inspect the generated tsvector
python insert.py
```

## Configuration

Connection defaults to `localhost:5432`, user `postgres`, database `postgres` with no password. Override with environment variables:

```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=postgres
export PG_PASSWORD=secret
export PG_DATABASE=postgres
```

## How It Works

### Table Schema

```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    html_content TEXT,
    json_metadata JSONB,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(strip_html(html_content), '')) ||
        jsonb_to_tsvector('english', coalesce(json_metadata, '{}'), '["all"]')
    ) STORED
);
```

- **`strip_html()`** — strips HTML tags via `regexp_replace` before indexing, so `<b>html</b>` is indexed as `html`
- **`search_vector`** — maintained automatically by PostgreSQL; combines cleaned HTML text and all JSONB string values
- **GIN index** — speeds up `@@` full-text search queries

### Search Queries (`search.sql`)

```sql
-- Phrase search: words must appear adjacent and in order
SELECT * FROM articles
WHERE search_vector @@ phraseto_tsquery('english', 'advanced phrase search');

-- Boolean AND: both lexemes must appear anywhere in the document
SELECT * FROM articles
WHERE search_vector @@ to_tsquery('english', 'Jane & html');
```

### Custom Article Insertion (`insert.py`)

```bash
# Default sample article
python insert.py

# Custom content
python insert.py --html "<p>My article</p>" --meta '{"author": "Bob", "tags": ["demo"]}'
```

The script prints the auto-generated `search_vector` so you can see exactly which lexemes PostgreSQL extracted from your HTML and JSON.
