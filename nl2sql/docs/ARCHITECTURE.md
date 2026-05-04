# NL-to-SQL — Architecture

## Detailed Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           NL-to-SQL System                                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌─────────────────────────┐    ┌──────────────────────────────────────────┐   ║
║   │       Entry Points      │    │              Data Sources                │   ║
║   │                         │    │                                          │   ║
║   │  ┌───────────────────┐  │    │  ┌──────────────────────────────────┐   │   ║
║   │  │  Streamlit UI     │  │    │  │       UnifiedDataSource           │   │   ║
║   │  │  (chat interface) │  │    │  │                                  │   │   ║
║   │  │  _build_manager() │  │    │  │  ┌────────────┐  ┌───────────┐  │   │   ║
║   │  │  [st.cache_resrc] │  │    │  │  │ GCS Parquet│  │PostgreSQL │  │   │   ║
║   │  └────────┬──────────┘  │    │  │  │  (gcsfs +  │  │  DB(s)    │  │   │   ║
║   │           │             │    │  │  │            │  │           │  │   │   ║
║   │  ┌───────────────────┐  │    │  │  │  datasets) │  │rag_db     │  │   │   ║
║   │  │  FastAPI REST     │  │    │  │  │            │  │local_pg   │  │   │   ║
║   │  │  POST /v1/query   │  │    │  │  │ lazy reads │  │(pgvector) │  │   │   ║
║   │  │  _get_manager()   │  │    │  │  └─────┬──────┘  └─────┬─────┘  │   │   ║
║   │  │  [module singleton│  │    │  │        │                │        │   │   ║
║   │  └────────┬──────────┘  │    │  │        ▼                ▼        │   │   ║
║   └───────────┼─────────────┘    │  │  ┌─────────────────────────────┐ │   │   ║
║               │                  │  │  │     DuckDB  (:memory:)      │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  conn.register('orders',    │ │   │   ║
║               │                  │  │  │    pyarrow.dataset) ← lazy  │ │   │   ║
║               │                  │  │  │  (predicate pushdown: only  │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │   needed row groups read)   │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  INSTALL postgres; LOAD pg  │ │   │   ║
║               │                  │  │  │  ATTACH '...' AS rag_db     │ │   │   ║
║               │                  │  │  │    (TYPE postgres, READ_ONLY)│ │   │   ║
║               │                  │  │  │  ATTACH '...' AS local_pg   │ │   │   ║
║               │                  │  │  │    (TYPE postgres, READ_ONLY)│ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  Schema introspection:      │ │   │   ║
║               │                  │  │  │  DESCRIBE <view>            │ │   │   ║
║               │                  │  │  │  information_schema.tables  │ │   │   ║
║               │                  │  │  │  information_schema.columns  │ │   │   ║
║               │                  │  │  └──────────────┬──────────────┘ │   │   ║
║               │                  │  │                 │ schema_text     │   │   ║
║               │                  │  └─────────────────┼────────────────┘   │   ║
║               │                  │                    │                     │   ║
║               │                  └────────────────────┼─────────────────────┘   ║
║               │                                       │                         ║
║               ▼                                       ▼                         ║
║   ┌───────────────────────────────────────────────────────────────────────────┐ ║
║   │                        ConversationManager                                │ ║
║   │                                                                           │ ║
║   │   nl_query                                                                │ ║
║   │      │                                                                    │ ║
║   │      ▼                                                                    │ ║
║   │   _normalize_nl()  ──►  NL Cache (LRU, size=20)                          │ ║
║   │   "how many docs?" ◄──  OrderedDict[normalized_nl → QueryResult]         │ ║
║   │      │ miss                                                               │ ║
║   │      ▼                                                                    │ ║
║   │   ┌─────────────────────────────────────────────────────────────────┐    │ ║
║   │   │               Retry Loop  (attempt 1 .. max_retries=3)          │    │ ║
║   │   │                                                                  │    │ ║
║   │   │  attempt == 1                   attempt > 1                      │    │ ║
║   │   │  _build_prompt()               _build_correction_prompt()        │    │ ║
║   │   │  ┌────────────────────┐        ┌────────────────────────────┐   │    │ ║
║   │   │  │ Schema:\n{schema}  │        │ Schema:\n{schema}          │   │    │ ║
║   │   │  │ History (last 3    │        │ Failed SQL: {bad_sql}      │   │    │ ║
║   │   │  │ successful turns)  │        │ Error: {error[:400]}       │   │    │ ║
║   │   │  │ Question: {nl}     │        │ Return ONLY corrected SQL. │   │    │ ║
║   │   │  └────────┬───────────┘        └───────────┬────────────────┘   │    │ ║
║   │   │           │                                │                     │    │ ║
║   │   │           └──────────────┬─────────────────┘                     │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │              ┌─────────────────────┐                             │    │ ║
║   │   │              │   Pydantic AI Agent  │                             │    │ ║
║   │   │              │   agent.run(prompt)  │                             │    │ ║
║   │   │              │                      │                             │    │ ║
║   │   │              │  OpenAI / Anthropic  │                             │    │ ║
║   │   │              │  or Ollama (local)   │                             │    │ ║
║   │   │              │  system_prompt:      │                             │    │ ║
║   │   │              │  "Return ONLY plain  │                             │    │ ║
║   │   │              │   SQL. No fences."   │                             │    │ ║
║   │   │              └──────────┬──────────┘                             │    │ ║
║   │   │                         │ raw SQL string                         │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │              strip_sql_fences()   (remove ```sql ... ```)        │    │ ║
║   │   │                         │                                        │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │  ┌──────────────────────────────────────────────────────────┐   │    │ ║
║   │   │  │                   Guardrails                              │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  G1: _check_readonly()                                    │   │    │ ║
║   │   │  │      regex: DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|...  │   │    │ ║
║   │   │  │      → error string if matched → retry with error         │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  G2: _apply_row_cap()                                     │   │    │ ║
║   │   │  │      if no LIMIT clause → append LIMIT {max_result_rows}  │   │    │ ║
║   │   │  │      safe_sql = sql + "\nLIMIT 10000"                     │   │    │ ║
║   │   │  └──────────────────────────────┬────────────────────────────┘   │    │ ║
║   │   │                                 │ safe_sql                       │    │ ║
║   │   │                                 ▼                                │    │ ║
║   │   │              _hash(safe_sql) ──► SQL Cache (LRU, size=20)        │    │ ║
║   │   │                             ◄── OrderedDict[md5 → QueryResult]   │    │ ║
║   │   │                                 │ miss                           │    │ ║
║   │   │                                 ▼                                │    │ ║
║   │   │  ┌──────────────────────────────────────────────────────────┐   │    │ ║
║   │   │  │  G3: _execute_with_timeout(conn, safe_sql, timeout=30s)  │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  threading.Timer(30, _cancel)                             │   │    │ ║
║   │   │  │      _cancel: timed_out.set() + conn.interrupt()          │   │    │ ║
║   │   │  │  conn.execute(safe_sql)   [DuckDB]                        │   │    │ ║
║   │   │  │      → pulls data from PostgreSQL via postgres scanner     │   │    │ ║
║   │   │  │      → or reads GCS Parquet via httpfs                    │   │    │ ║
║   │   │  │  cursor.description → columns                             │   │    │ ║
║   │   │  │  cursor.fetchall()  → rows                                │   │    │ ║
║   │   │  └──────────────────────────────┬────────────────────────────┘   │    │ ║
║   │   │                                 │                                │    │ ║
║   │   │              ┌──────────────────┴───────────────────┐           │    │ ║
║   │   │              │ Success                   Exception   │           │    │ ║
║   │   │              ▼                           ▼           │           │    │ ║
║   │   │     populate SQL cache           last_error = str(exc)          │    │ ║
║   │   │     populate NL cache            (or TimeoutError)              │    │ ║
║   │   │     → break loop                 → next attempt                 │    │ ║
║   │   └─────────────────────────────────────────────────────────────────┘    │ ║
║   │                                                                           │ ║
║   │      ▼  (after loop)                                                      │ ║
║   │   append turn to self.history                                             │ ║
║   │   if history_store: await history_store.save(session_id, ...)             │ ║
║   │      │                                                                    │ ║
║   │      ▼                                                                    │ ║
║   │   QueryResult(nl_query, sql, columns, rows, error, cached, attempts)      │ ║
║   └───────────────────────────────────────────────────────────────────────────┘ ║
║               │                                                                  ║
║               ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────────────────┐ ║
║   │                   HistoryStore  (optional)                                │ ║
║   │                                                                           │ ║
║   │   asyncpg pool → PostgreSQL table: conversation_history                   │ ║
║   │   ┌────────────────────────────────────────────────────────────────────┐  │ ║
║   │   │ id │ session_id │ nl_query │ sql │ columns │ rows │ error │ ts     │  │ ║
║   │   └────────────────────────────────────────────────────────────────────┘  │ ║
║   │   save(): INSERT on every successful/failed turn                           │ ║
║   │   load(): SELECT ORDER BY ts → warm caches on session resume              │ ║
║   │   sessions(): list all session_ids                                         │ ║
║   └───────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### Table naming by source

```
Query: "How many orders were placed in Q4?"

                      LLM sees schema text:
          ┌───────────────────────────────────────────┐
          │ === GCS Parquet tables (use bare name) === │
          │ Table: orders                              │
          │   - order_id (VARCHAR)                     │
          │   - order_date (DATE)                      │
          │                                            │
          │ === rag tables (prefix: rag.main.<table>) ││
          │ Table: rag.main.documents                  │
          │   - id (UUID)                              │
          │   - title (TEXT)                           │
          │                                            │
          │ === local_pg tables (prefix:              ││
          │        local_pg.main.<table>)              │
          │ Table: local_pg.main.baby_names            │
          │   - name (VARCHAR)                         │
          └───────────────────────────────────────────┘
                           │
                           ▼
          LLM generates: SELECT COUNT(*) FROM orders
                         WHERE order_date >= '2024-10-01'
                           │
                           ▼ DuckDB resolves:
                    orders → GCS Parquet view
                    rag.main.documents → postgres scanner
                    local_pg.main.* → postgres scanner
```

### Cache hit paths

```
NL Cache hit (same question, different casing/whitespace):
  "How many docs?" → normalize → "how many docs?"
  "HOW MANY DOCS?" → normalize → "how many docs?"  → cache hit, 0 LLM calls

SQL Cache hit (different question → identical SQL):
  "Count all docs" → LLM → SELECT COUNT(*) FROM rag.main.documents LIMIT 10000
  "Total documents?" → LLM → SELECT COUNT(*) FROM rag.main.documents LIMIT 10000
                                       → md5 match → cache hit, rows returned from memory

History context (last 3 successful turns fed into next prompt):
  Only successful turns included — failed SQL is excluded to avoid confusing the LLM.
  On session resume: HistoryStore.load() replays DB rows into self.history
                     then caches are warmed from those turns.
```

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
