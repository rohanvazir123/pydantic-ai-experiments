# NL-to-SQL — Call Graphs

Call graphs for the NL-to-SQL workflows.
Links jump directly to the relevant line in source code.

## Table of Contents

- [1. NL-to-SQL v1 — Sync MVP](#1-nl-to-sql-v1--sync-mvp)
- [2. NL-to-SQL v2 — Async + Retry + Guardrails](#2-nl-to-sql-v2--async--retry--guardrails)
- [3. SQL Discovery Agent](#3-sql-discovery-agent)
- [4. Key Files](#4-key-files)

---

## 1. NL-to-SQL v1 — Sync MVP

> `nl2sql/nlp_sql_postgres_v1.py`

**Entry point**: `python nl2sql/nlp_sql_postgres_v1.py`

```
main()  [sync]
  ├── duckdb.connect()            in-memory DuckDB
  ├── UnifiedDataSource(conn, gcs_bucket, gcs_prefix, postgres_dbs)  L222
  │     ├── load_gcs_tables()                                          L236
  │     │     ├── conn.execute("INSTALL httpfs / LOAD httpfs")
  │     │     ├── CREATE SECRET (GCS HMAC credentials)
  │     │     ├── list GCS prefixes  → table names
  │     │     └── CREATE VIEW tbl AS parquet_scan('gs://…')
  │     ├── attach_postgres_dbs()                                      L270
  │     │     ├── conn.execute("INSTALL/LOAD postgres_scanner")
  │     │     └── ATTACH 'dsn' AS alias (TYPE postgres, READ_ONLY)
  │     ├── generate_schema() → str                                    L281
  │     │     ├── information_schema.tables for each source
  │     │     ├── information_schema.columns per table
  │     │     └── format: "Table: alias.main.tbl\n  col type\n  …"
  │     ├── init_agent(model="gpt-4o") → Agent                        L324
  │     │     └── Pydantic AI Agent(OpenAIChatModel, system_prompt)
  │     └── conversation_manager(cache_size) → ConversationManager     L337
  │
  └── ConversationManager.run_query(nl_query)  [× 5 test queries]    L170

ConversationManager.run_query(nl_query)                               L170
  ├── 1. NL cache hit?  (exact key match, OrderedDict LRU)
  ├── 2. _build_prompt(nl_query)                                       L161
  │     └── schema_text + _history_context(last 3 success) + question
  ├── 3. agent.run_sync(prompt) → raw SQL string
  ├── 4. strip_sql_fences(sql)                                         L109
  ├── 5. SQL hash cache hit?  (_hash = MD5)                            L151
  ├── 6. conn.execute(sql).fetchall() → rows
  ├── 7. history.append((nl_query, sql, rows))
  └── return rows (or None on error)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L136) | `ConversationManager` | L136 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L170) | `run_query()` | L170 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L222) | `UnifiedDataSource` | L222 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L236) | `load_gcs_tables()` | L236 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L281) | `generate_schema()` | L281 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L324) | `init_agent()` | L324 |
| [`nl2sql/nlp_sql_postgres_v1.py`](../nlp_sql_postgres_v1.py#L109) | `strip_sql_fences()` | L109 |

---

## 2. NL-to-SQL v2 — Async + Retry + Guardrails

> `nl2sql/nlp_sql_postgres_v2.py`

**Entry point**: `python nl2sql/nlp_sql_postgres_v2.py`  
**Streamlit**: `streamlit run nl2sql/app/streamlit/streamlit_app.py`  
**REST API**: `uvicorn nl2sql.app.rest_api.api:app --port 8001`

### Full call graph

```
main()  [async]                                                       L634
  ├── UnifiedDataSource setup  (same as v1, but async)
  ├── optional: HistoryStore.create(dsn)                              L189
  └── ConversationManager via conversation_manager(…, session_id)    L589

── HistoryStore (asyncpg-backed persistence) ─────────────────────── L189
HistoryStore.create(dsn, **pool_kwargs)  [classmethod, async]
  ├── asyncpg.create_pool(dsn)
  └── _init_schema()
        └── CREATE TABLE conversation_history
              (session_id, nl_query, sql, rows JSONB, created_at)
              INDEX on session_id

save(session_id, nl_query, sql, qr: QueryResult)  [async]
  └── INSERT INTO conversation_history

load(session_id) → list[(nl_query, sql, QueryResult)]  [async]
  └── SELECT … ORDER BY created_at  → reconstruct QueryResult objects

sessions() → list[str]  [async]
  └── SELECT DISTINCT session_id ORDER BY MIN(created_at)

── ConversationManager (async + retry) ──────────────────────────── L303
__init__(conn, agent, schema_text, cache_size, max_retries,
         max_result_rows, query_timeout, history_store, session_id,
         _initial_history)
  └── warm caches from _initial_history (resumed sessions)

run_query(nl_query) → QueryResult  [async]                          L376
  ├── _normalize_nl(nl_query)                                        L340
  ├── 1. NL cache hit? (_nl_cache, OrderedDict LRU)
  ├── 2. for attempt in range(1, max_retries + 1):
  │     ├── [attempt == 1] _build_prompt(nl_query)                  L356
  │     │     └── schema + _history_context(last 3 successes) + question
  │     ├── [attempt > 1]  _build_correction_prompt(nl, bad_sql, error)  L365
  │     │     └── original prompt + "Previous SQL failed: …\nError: …"
  │     ├── await agent.run(prompt) → raw SQL
  │     ├── strip_sql_fences(sql)                                    L112
  │     ├── ── Guardrails ──────────────────────────────────────────
  │     ├── _check_readonly(sql)                                     L65
  │     │     └── regex for DROP/INSERT/UPDATE/DELETE/ALTER/CREATE/TRUNCATE
  │     │           → return error string or None
  │     ├── _apply_row_cap(sql, max_result_rows)                     L73
  │     │     └── append LIMIT if none present (regex)
  │     ├── _hash(sql) → MD5                                         L336
  │     ├── 3. SQL cache hit? (_sql_cache)
  │     ├── 4. _execute_with_timeout(conn, sql, query_timeout)       L81
  │     │     ├── threading.Timer(timeout, conn.interrupt)
  │     │     ├── conn.execute(sql) → cursor
  │     │     └── (columns from cursor.description, rows as list[tuple])
  │     ├── ── Success path ─────────────────────────────────────────
  │     ├── create QueryResult(nl, sql, columns, rows, attempts=attempt)
  │     ├── _cache_put(_nl_cache, key, qr)                          L343
  │     ├── _cache_put(_sql_cache, hash, qr)
  │     ├── history.append((nl, sql, qr))
  │     ├── await history_store.save(session_id, …)  [if provided]
  │     └── return qr
  │     ├── ── Error path ───────────────────────────────────────────
  │     └── last_error = str(exc); continue to next attempt
  └── return QueryResult(error=last_error, attempts=max_retries)

QueryResult                                                          L140
  ├── nl_query, sql, columns, rows, error, cached, attempts
  ├── .success → bool  (error is None)
  ├── .pretty_print(max_rows=20)  → tabulate to stdout
  └── .to_dataframe()             → pandas DataFrame
```

### Streamlit chat path

```
nl2sql/app/streamlit/streamlit_app.py:main()
    ├── _build_manager()                             [st.cache_resource — once per process]
    │       ├── duckdb.connect(":memory:")
    │       ├── conn.execute("ATTACH '...' AS rag_db (TYPE postgres, READ_ONLY)")
    │       ├── schema introspection queries         [information_schema.tables/columns]
    │       ├── OpenAIModel(settings.llm_model, ...)
    │       ├── Agent(model=llm, result_type=str, system_prompt=_SYSTEM_PROMPT)
    │       └── ConversationManager(conn, agent, schema_text, ...)
    │
    └── [on chat_input]
            └── asyncio.run(manager.run_query(prompt))
                    └── ConversationManager.run_query()      L376
                            ├── _normalize_nl(nl_query)      → NL cache lookup
                            ├── [loop attempt 1..max_retries]
                            │       ├── _build_prompt() / _build_correction_prompt()
                            │       ├── agent.run(prompt)    → Pydantic AI → LLM
                            │       ├── strip_sql_fences()
                            │       ├── _check_readonly()    → guardrail: write keywords
                            │       ├── _apply_row_cap()     → guardrail: LIMIT
                            │       ├── _hash(sql)           → SQL cache lookup
                            │       └── _execute_with_timeout(conn, sql, timeout)
                            │               ├── threading.Timer(_cancel)
                            │               └── conn.execute(sql) → DuckDB → PostgreSQL
                            └── return QueryResult
```

### FastAPI path

```
nl2sql/app/rest_api/api.py:query()
    └── _get_manager()                               [module-level singleton]
            └── [same setup as _build_manager above]
    └── manager.run_query(request.question)
            └── [same as above]
    └── → QueryResponse(sql, columns, rows, row_count, cached, attempts, error)
```

### Self-correction loop

```
attempt 1:
    _build_prompt(nl_query)
        → "Schema:\n{schema}\n\nQuestion: {nl_query}"
    agent.run(prompt) → sql_v1
    _execute_with_timeout(sql_v1) → Error: column X does not exist

attempt 2:
    _build_correction_prompt(nl_query, sql_v1, error)
        → "Schema:\n...\nThe following SQL failed:\n{sql_v1}\nError: {error}\nReturn ONLY corrected SQL."
    agent.run(correction_prompt) → sql_v2
    _execute_with_timeout(sql_v2) → OK, 5 rows
    → QueryResult(sql=sql_v2, rows=[...], attempts=2)
```

### Cache hit paths

```
NL cache hit (same question, different capitalisation):
    "How many docs?" == "how many docs?" == "HOW MANY DOCS?"
    → return cached QueryResult immediately (no LLM call)

SQL hash cache hit (different question → same SQL):
    "Count docs" → "SELECT COUNT(*) FROM rag_db.main.documents LIMIT 500"
    "How many documents?" → same SQL hash
    → return cached result with attempts=current_attempt
```

**v1 → v2 feature delta**:

| Feature | v1 | v2 |
|---------|----|----|
| Async | No (sync) | Yes (`await`) |
| Retry on SQL error | No | Yes (up to `max_retries`) |
| Correction prompt | No | Yes (feeds bad SQL + error back) |
| SELECT-only guardrail | No | Yes (`_check_readonly`) |
| Row cap | No | Yes (`_apply_row_cap`) |
| Query timeout | No | Yes (threading timer + `conn.interrupt()`) |
| Session persistence | No | Yes (`HistoryStore`, asyncpg) |
| Structured result | Raw rows | `QueryResult` dataclass |
| Provider support | OpenAI only | OpenAI + Anthropic |

---

## 3. SQL Discovery Agent

> `nl2sql/sql_discovery.py` — lightweight agent that discovers schema on-the-fly via tools rather than a pre-built schema string.

```
── Entry point ────────────────────────────────────────────────────────────────

UnifiedDataSource.discovery_query(prompt, pg_alias=None)             L616
  nl2sql/nlp_sql_postgres_v2.py
  ├── resolve pg_dsn from self.postgres_dbs by alias (default: first)
  ├── asyncpg.create_pool(pg_dsn)
  ├── MultiDBDeps(pg_pool, duck_conn=self.conn)
  └── sql_agent.run(prompt, deps=MultiDBDeps)  ──────────────────────┐
                                                                      │
── Agent ──────────────────────────────────────────────────────────── │ ──────

sql_agent = Agent(                                           L43  <───┘
  nl2sql/sql_discovery.py
  model=_make_model()     # reads LLM_MODEL/LLM_BASE_URL from .env
  deps_type=MultiDBDeps,
  result_type=SQLResponse,
  system_prompt="…Discover the schema first."
)

  No schema string injected — the LLM discovers schema by calling
  the tools below before generating SQL.

── Tools (registered via @sql_agent.tool) ─────────────────────────────────────

list_tables(ctx, db_type) → list[str]                                L53
  ├── [postgres]  SELECT table_name FROM information_schema.tables
  │               WHERE table_schema = 'public'  (via asyncpg pg_pool)
  └── [duckdb]    SHOW TABLES  (via duck_conn)

describe_table(ctx, db_type, table_name) → str                       L67
  ├── [postgres]  SELECT column_name, data_type FROM information_schema.columns
  └── [duckdb]    DESCRIBE {table_name}

── Agent inference loop ────────────────────────────────────────────────────────

  turn 1:  LLM calls list_tables("postgres") + list_tables("duckdb")
  turn 2:  LLM calls describe_table(db_type, tbl) for relevant tables only
  turn N:  LLM returns SQLResponse(database_type, sql, explanation)

── Execution & return ──────────────────────────────────────────────────────────

  result.output → SQLResponse
  ├── [postgres]  asyncpg pg_conn.fetch(sql)  → columns + rows as dicts
  └── [duckdb]    self.conn.execute(sql)       → columns + rows as tuples
  return (SQLResponse, columns, rows)

── Supporting types ────────────────────────────────────────────────────────────

MultiDBDeps                                                           L37
  ├── pg_pool: asyncpg.Pool
  └── duck_conn: duckdb.DuckDBPyConnection

SQLResponse                                                           L29
  ├── database_type: str   ("postgres" | "duckdb")
  ├── sql: str
  └── explanation: str
```

**Contrast with v1/v2**: v1/v2 call `generate_schema()` once up-front and inject the full schema into every prompt; discovery lets the LLM call tools to explore the schema dynamically — fewer prompt tokens for narrow queries, more LLM round-trips for broad ones.

---

## 4. Key Files

| Symbol | File | Line |
|--------|------|------|
| `ConversationManager` (v1) | `nl2sql/nlp_sql_postgres_v1.py` | L136 |
| `run_query()` (v1) | `nl2sql/nlp_sql_postgres_v1.py` | L170 |
| `UnifiedDataSource` (v1) | `nl2sql/nlp_sql_postgres_v1.py` | L222 |
| `QueryResult` | `nl2sql/nlp_sql_postgres_v2.py` | L140 |
| `HistoryStore` | `nl2sql/nlp_sql_postgres_v2.py` | L189 |
| `ConversationManager` (v2) | `nl2sql/nlp_sql_postgres_v2.py` | L303 |
| `run_query()` (v2) | `nl2sql/nlp_sql_postgres_v2.py` | L376 |
| `_check_readonly` | `nl2sql/nlp_sql_postgres_v2.py` | L65 |
| `_apply_row_cap` | `nl2sql/nlp_sql_postgres_v2.py` | L73 |
| `_execute_with_timeout` | `nl2sql/nlp_sql_postgres_v2.py` | L81 |
| `UnifiedDataSource` (v2) | `nl2sql/nlp_sql_postgres_v2.py` | L494 |
| `conversation_manager()` | `nl2sql/nlp_sql_postgres_v2.py` | L589 |
| `UnifiedDataSource.discovery_query` | `nl2sql/nlp_sql_postgres_v2.py` | L616 |
| `sql_agent` | `nl2sql/sql_discovery.py` | L43 |
| `SQLResponse` | `nl2sql/sql_discovery.py` | L29 |
| `MultiDBDeps` | `nl2sql/sql_discovery.py` | L37 |
| `list_tables` tool | `nl2sql/sql_discovery.py` | L53 |
| `describe_table` tool | `nl2sql/sql_discovery.py` | L67 |
| `run_query()` (discovery) | `nl2sql/sql_discovery.py` | L88 |
