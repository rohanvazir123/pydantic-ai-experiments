# NL-to-SQL — Architecture

## Overview

Natural-language to SQL over the RAG PostgreSQL database.
A Pydantic AI agent translates user questions into SQL SELECT statements,
executes them via DuckDB's PostgreSQL scanner, and returns results.
A self-correcting retry loop feeds SQL errors back to the LLM for up to 3 attempts.

## Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (local) or any OpenAI-compatible API |
| SQL execution | DuckDB in-memory (attaches PostgreSQL via postgres scanner) |
| Agent framework | Pydantic AI |
| History persistence | asyncpg PostgreSQL (optional) |
| UI | Streamlit (`apps/nl2sql/streamlit_app.py`) |
| REST API | FastAPI (`apps/nl2sql/api.py`) |

## Components

```
apps/nl2sql/
├── streamlit_app.py   — Chat UI with SQL display and result tables
└── api.py             — FastAPI: /health, /v1/query, /v1/history, /v1/schema

nlp2sql/
├── nlp_sql_postgres_v2.py   — Core logic
│   ├── ConversationManager  — NL → SQL → execute → retry loop
│   ├── HistoryStore         — asyncpg-backed session persistence
│   ├── QueryResult          — Structured result (columns, rows, error, cached, attempts)
│   ├── UnifiedDataSource    — Multi-source: PostgreSQL + GCS Parquet via DuckDB
│   ├── _check_readonly()    — Guardrail: block non-SELECT statements
│   ├── _apply_row_cap()     — Guardrail: enforce LIMIT
│   └── _execute_with_timeout() — Guardrail: interrupt long queries
└── README.md
```

## Data Flow — NL Query

```
User question ("How many documents are stored?")
    │
    ▼
ConversationManager.run_query(nl_query)
    │
    ├── NL cache hit? → return cached QueryResult
    │
    ├── [loop: attempt 1..max_retries]
    │       │
    │       ├── Build prompt: schema_text + conversation history + question
    │       │   (attempt > 1: include failed SQL + error for self-correction)
    │       │
    │       ├── Pydantic AI agent.run(prompt) → raw SQL string
    │       │
    │       ├── Guardrail 1: _check_readonly() — block DROP/DELETE/INSERT/UPDATE/…
    │       │
    │       ├── Guardrail 2: _apply_row_cap() — append LIMIT if missing
    │       │
    │       ├── SQL hash cache hit? → return cached result
    │       │
    │       └── _execute_with_timeout(conn, sql, timeout)
    │               └── DuckDB → PostgreSQL (via postgres scanner attachment)
    │
    └── Return QueryResult(sql, columns, rows, cached, attempts, error)
```

## DuckDB ↔ PostgreSQL Bridge

```
duckdb.connect(":memory:")
    └── ATTACH 'postgresql://...' AS rag_db (TYPE postgres, READ_ONLY)
            └── exposes tables as: rag_db.main.<table>

Schema introspection:
    SELECT table_name FROM rag_db.information_schema.tables
    SELECT column_name, data_type FROM rag_db.information_schema.columns
```

DuckDB executes the LLM-generated SQL locally, pulling data from PostgreSQL
via the postgres scanner. This avoids running arbitrary SQL directly on PostgreSQL
and gives DuckDB's optimizer control over the query plan.

## Guardrails

| Guardrail | Implementation | Behavior |
|---|---|---|
| Read-only enforcement | `_check_readonly()` regex | Blocks DROP/DELETE/INSERT/UPDATE/TRUNCATE/ALTER/CREATE/GRANT/REVOKE |
| Result row cap | `_apply_row_cap()` | Appends `LIMIT {max_result_rows}` if none present |
| Query timeout | `threading.Timer` + `conn.interrupt()` | Kills query after `query_timeout` seconds |
| Self-correcting retry | `ConversationManager.run_query()` loop | Re-prompts LLM with error message on failure |

## Caching

Two LRU caches per `ConversationManager`:

- **NL cache** — keyed by `" ".join(nl.lower().split())` (normalized whitespace + case)
- **SQL hash cache** — keyed by MD5 of the SQL string (catches same SQL from different phrasing)

Cache size is configurable (`cache_size` param, default 20).
History-warmed: prior session turns pre-populate both caches on resume.

## Key Configuration (`.env`)

```
DATABASE_URL=postgresql://...          # RAG PostgreSQL (attached as rag_db)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB and LLM connectivity |
| POST | `/v1/query` | NL → SQL → execute → return results |
| GET | `/v1/history` | Recent conversation history |
| GET | `/v1/schema` | Database schema text used for SQL generation |

## Running

```bash
# UI
streamlit run apps/nl2sql/streamlit_app.py

# API
uvicorn apps.nl2sql.api:app --port 8001 --reload

# Example query
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many documents are stored?"}'
```
