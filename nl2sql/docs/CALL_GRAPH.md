# NL-to-SQL — Call Graph

## Streamlit Chat Path

```
apps/nl2sql/streamlit_app.py:main()
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
                    └── ConversationManager.run_query()      nlp_sql_postgres_v2.py:~376
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

## FastAPI Path

```
apps/nl2sql/api.py:query()
    └── _get_manager()                               [module-level singleton]
            └── [same setup as _build_manager above]
    └── manager.run_query(request.question)
            └── [same as above]
    └── → QueryResponse(sql, columns, rows, row_count, cached, attempts, error)
```

## Self-Correction Loop

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

## History Context Building

```
ConversationManager._history_context(n=3)
    └── last 3 SUCCESSFUL turns from self.history
            → "Q: {nl}\nSQL: {sql}\nResult preview: {rows[:3]}"
    (failed turns excluded — bad SQL confuses the model)
```

## Cache Hit Path

```
NL cache hit (same question, different capitalisation):
    "How many docs?" == "how many docs?" == "HOW MANY DOCS?"
    → return cached QueryResult immediately (no LLM call)

SQL hash cache hit (different question → same SQL):
    "Count docs" → "SELECT COUNT(*) FROM rag_db.main.documents LIMIT 500"
    "How many documents?" → same SQL hash
    → return cached result with attempts=current_attempt
```

## Schema Discovery Path

Schema is discovered at inference time — the LLM calls tools to explore the database rather than receiving a pre-built schema string.

```
── Entry point ──────────────────────────────────────────────────────────────────

UnifiedDataSource.discovery_query(prompt, pg_alias=None)             L616
  nl2sql/nlp_sql_postgres_v2.py
  ├── resolve pg_dsn from self.postgres_dbs by alias (default: first)
  ├── asyncpg.create_pool(pg_dsn)
  ├── MultiDBDeps(pg_pool, duck_conn=self.conn)
  └── sql_agent.run(prompt, deps=MultiDBDeps)  ──────────────────────┐
                                                                      │
── Agent ────────────────────────────────────────────────────────────│───────────

sql_agent = Agent(                                           L43  <───┘
  nl2sql/sql_discovery.py
  model=_make_model()     # reads LLM_MODEL/LLM_BASE_URL from .env
  deps_type=MultiDBDeps,
  result_type=SQLResponse,
  system_prompt="…Discover the schema first."
)

  No schema string injected — the LLM discovers schema by calling
  the tools below before generating SQL.

── Tools (registered via @sql_agent.tool) ───────────────────────────────────────

list_tables(ctx, db_type) → list[str]                                L53
  ├── [postgres]  SELECT table_name FROM information_schema.tables
  │               WHERE table_schema = 'public'  (via asyncpg pg_pool)
  └── [duckdb]    SHOW TABLES  (via duck_conn)

describe_table(ctx, db_type, table_name) → str                       L67
  ├── [postgres]  SELECT column_name, data_type FROM information_schema.columns
  └── [duckdb]    DESCRIBE {table_name}

── Agent inference loop ──────────────────────────────────────────────────────────

  turn 1:  LLM calls list_tables("postgres") + list_tables("duckdb")
  turn 2:  LLM calls describe_table(db_type, tbl) for relevant tables only
  turn N:  LLM returns SQLResponse(database_type, sql, explanation)

── Execution & return ────────────────────────────────────────────────────────────

  result.output → SQLResponse
  ├── [postgres]  asyncpg pg_conn.fetch(sql)  → columns + rows as dicts
  └── [duckdb]    self.conn.execute(sql)       → columns + rows as tuples
  return (SQLResponse, columns, rows)
```

---

## Key File Locations

| Symbol | File | Line |
|---|---|---|
| `UnifiedDataSource.discovery_query` | `nl2sql/nlp_sql_postgres_v2.py` | L616 |
| `sql_agent` | `nl2sql/sql_discovery.py` | L43 |
| `list_tables` tool | `nl2sql/sql_discovery.py` | L53 |
| `describe_table` tool | `nl2sql/sql_discovery.py` | L67 |
| `SQLResponse` | `nl2sql/sql_discovery.py` | L29 |
| `MultiDBDeps` | `nl2sql/sql_discovery.py` | L37 |
| `ConversationManager` | `nl2sql/nlp_sql_postgres_v2.py` | L305 |
| `ConversationManager.run_query` | `nl2sql/nlp_sql_postgres_v2.py` | L378 |
| `QueryResult` | `nl2sql/nlp_sql_postgres_v2.py` | L141 |
| `HistoryStore` | `nl2sql/nlp_sql_postgres_v2.py` | L191 |
| `_check_readonly` | `nl2sql/nlp_sql_postgres_v2.py` | L67 |
| `_apply_row_cap` | `nl2sql/nlp_sql_postgres_v2.py` | L76 |
| `_execute_with_timeout` | `nl2sql/nlp_sql_postgres_v2.py` | L83 |
