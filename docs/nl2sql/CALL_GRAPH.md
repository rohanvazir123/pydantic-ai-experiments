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

## Key File Locations

| Symbol | File | Line |
|---|---|---|
| `ConversationManager` | `nlp2sql/nlp_sql_postgres_v2.py` | ~303 |
| `ConversationManager.run_query` | `nlp2sql/nlp_sql_postgres_v2.py` | ~376 |
| `QueryResult` | `nlp2sql/nlp_sql_postgres_v2.py` | ~139 |
| `HistoryStore` | `nlp2sql/nlp_sql_postgres_v2.py` | ~189 |
| `_check_readonly` | `nlp2sql/nlp_sql_postgres_v2.py` | ~65 |
| `_apply_row_cap` | `nlp2sql/nlp_sql_postgres_v2.py` | ~74 |
| `_execute_with_timeout` | `nlp2sql/nlp_sql_postgres_v2.py` | ~81 |
| Streamlit `_build_manager` | `apps/nl2sql/streamlit_app.py` | ~54 |
| FastAPI `_get_manager` | `apps/nl2sql/api.py` | ~56 |
