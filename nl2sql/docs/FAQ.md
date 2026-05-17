# FAQ - NL-to-SQL System

Code references: line numbers point to files under `nl2sql/` in this repo.

---

## NLP-to-SQL System

<a id="q210"></a>
**Q210. Walk me through the end-to-end flow of the NLP-to-SQL system.**

`ConversationManager.run_query(nl_query)` is called:

1. NL query is normalized (lowercase + whitespace-collapsed) and checked against the NL cache — exact normalized match returns the cached `QueryResult` immediately.
2. Prompt is built: schema text + last 3 successful conversation turns (Q/SQL/result preview) + the new question.
3. `agent.run_sync(prompt)` calls GPT-4o (or Claude), which returns SQL.
4. `strip_sql_fences()` cleans any markdown wrapping.
5. SQL is MD5-hashed and checked against the SQL hash cache — if the same SQL was already generated for a different question, the cached result is returned.
6. `conn.execute(sql)` runs the SQL in DuckDB. Column names come from `cursor.description`.
7. On success: `QueryResult` (with `columns`, `rows`, `attempts`) is stored in both caches and history, then returned.
8. On failure: the error is fed back to the LLM in a correction prompt and step 3 retries (up to `max_retries`, default 3). If all retries fail, `QueryResult(error=...)` is returned.

---

<a id="q211"></a>
**Q211. How does schema discovery work?**

`UnifiedDataSource.generate_schema()` introspects all sources at startup:

**GCS Parquet:** Lists blob virtual prefixes with `delimiter="/"` — each subfolder becomes a table name. Creates a DuckDB `VIEW` over `parquet_scan('gs://...')`. Then `DESCRIBE view_name` gives column names and types.

**PostgreSQL:** After attaching via DuckDB's postgres extension, queries `{alias}.information_schema.tables` and `{alias}.information_schema.columns` through DuckDB's catalog.

Everything is serialized into a single schema string prepended to every LLM prompt. Schema is captured once at startup — changes require a restart.

---

<a id="q212"></a>
**Q212. Why DuckDB over Spark, Trino, pg_parquet, or duckdb_fdw?**

| Option | Problem |
|---|---|
| **Spark / Trino** | Cluster-based, heavy infrastructure. Overkill for single-analyst workloads. Adds 10–30s startup latency. |
| **pg_parquet** | PostgreSQL reads Parquet; limited SQL, no GCS HMAC auth, PostgreSQL is the bottleneck. |
| **duckdb_fdw** | Wrong direction — PostgreSQL queries DuckDB via FDW. Complex Windows setup, server-side changes required. |
| **DuckDB postgres_scanner** | DuckDB ATTACHes PostgreSQL as a catalog and JOINs it with GCS Parquets in a single in-process query. Zero server-side changes. Zero extra infrastructure. |

---

<a id="q213"></a>
**Q213. How do cross-source JOINs work?**

100% inside DuckDB's in-memory engine. GCS Parquets are read lazily via `httpfs` (predicate pushdown where possible). PostgreSQL tables are scanned via `postgres_scanner` (full table scan — no index pushdown). DuckDB handles the JOIN, aggregation, and projection internally. The user writes plain DuckDB SQL; the naming convention (`bare name` vs `alias.main.table`) tells DuckDB which catalog to use.

---

<a id="q214"></a>
**Q214. What are the limitations of this architecture?**

| Limitation | Detail |
|---|---|
| **PostgreSQL full scans** | `postgres_scanner` reads entire PG tables; no index pushdown. Large PG tables (>10M rows) are slow. |
| **In-memory result sets** | DuckDB defaults to in-memory. Very large results can OOM. No result pagination implemented. |
| **Static schema** | Captured at startup. Table changes require restart. |
| **GCS auth** | HMAC keys only. Service account JSON / Workload Identity not implemented. |
| **No timeout enforcement** | No per-query timeout. |
| **Semantically wrong SQL** | Syntactically valid but logically incorrect SQL returns wrong results silently. |

---

<a id="q215"></a>
**Q215. How is the model prompted to generate correct SQL?**

System prompt enforces DuckDB-specific table naming rules:
- GCS Parquet tables → bare table name (`FROM orders`)
- rag_db tables → `rag.main.<table>`
- local_pg tables → `local_pg.main.<table>`

And mandates plain SQL output (no markdown fences, no explanation, no comments).

The schema string (all tables and columns from all sources) is injected in the user-turn prompt. The last 3 successful conversation turns are included as history context for follow-up questions. Zero-shot — no hardcoded few-shot examples.

---

<a id="q216"></a>
**Q216. How are hallucinated table or column names handled?**

**v1:** No handling. DuckDB throws a `CatalogException`, the `except` block catches it, and `None` is returned.

**v2:** Self-correcting retry loop. The error message is sent back to the LLM:

```
The following SQL you generated failed:
Question: {original_nl_query}
SQL: {bad_sql}
Error: {duckdb_error_message}

Return ONLY the corrected SQL.
```

The model reads `"Table orders_2024 does not exist"` and corrects to `FROM orders`. Up to `max_retries` attempts (default 3).

---

<a id="q217"></a>
**Q217. What happens with semantically valid but semantically wrong SQL?**

Silent failure — the query executes and returns wrong results. No semantic validation layer. The conversation history partially mitigates this: a wrong answer in turn 1 can be corrected in turn 2 if the user notices. Real mitigation would require row count sanity checks, column type validation, or chain-of-thought reasoning before returning SQL.

---

<a id="q218"></a>
**Q218. How is ambiguous natural language handled?**

It isn't — the model guesses. *"Show me recent findings"* typically produces `ORDER BY created_at DESC LIMIT 10` or `WHERE date > NOW() - INTERVAL '7 days'` based on the model's priors. The fix is to inject current date/time and business-specific term definitions into the prompt, or ask a clarifying question before generating SQL.

---

<a id="q219"></a>
**Q219. How does ConversationManager maintain context across follow-ups?**

`history: list[tuple[str, str, QueryResult]]` stores every turn as `(nl_query, sql, result)`. `_history_context(n=3)` serializes the last 3 **successful** turns as:

```
Q: Revenue per customer?
SQL: SELECT c.name, SUM(s.revenue) ...
Result preview: [('Alice', 3000.0), ('Bob', 1400.0)]
```

This block is prepended to every new prompt as "Conversation so far:". Failed turns are recorded in `history` for audit but **excluded** from `_history_context()` so bad SQL examples don't confuse the model.

---

<a id="q220"></a>
**Q220. How is GCS authentication handled in DuckDB?**

HMAC keys (not service account JSON), stored in `.env` as `GCS_HMAC_ID` + `GCS_HMAC_SECRET`. Registered in DuckDB as:

```sql
CREATE OR REPLACE SECRET gcs_secret (
    TYPE gcs,
    KEY_ID  '{GCS_HMAC_ID}',
    SECRET  '{GCS_HMAC_SECRET}'
)
```

DuckDB's `httpfs` extension picks this up for all `gs://` paths automatically.

---

<a id="q221"></a>
**Q221. What did v2 improve over v1?**

| | v1 | v2 |
|---|---|---|
| **Return type** | Raw `Any` (list of tuples or `None` on error) | `QueryResult` with `.columns`, `.rows`, `.success`, `.error`, `.attempts` |
| **SQL errors** | Silent `None`, dead end | Self-correcting retry loop: error fed back to LLM, up to `max_retries` (default 3) |
| **NL cache matching** | Exact string equality | Normalized: lowercase + whitespace-collapsed |
| **Column names** | Anonymous tuples | Populated from `cursor.description` |
| **Provider** | OpenAI only, hardcoded Windows path | `provider="openai"` or `"anthropic"`, caller-supplied env paths |
| **History context** | Includes failed turns | Failed turns excluded from context shown to model |
| **Guardrails** | None | SELECT-only enforcement, result row cap, query timeout |

---

<a id="q222"></a>
**Q222. What guardrails are built into the NLP-to-SQL pipeline?**

Three execution-time guardrails run inside `ConversationManager.run_query()` after each SQL generation step but before DuckDB execution:

| Guardrail | When it fires | Effect |
|---|---|---|
| **SELECT-only** | Generated SQL contains a write/DDL keyword | Treated as an attempt error; self-correcting retry loop can request a new query from the LLM |
| **Result row cap** | Generated SQL has no `LIMIT` clause | `LIMIT N` appended automatically (`max_result_rows=10_000` by default) |
| **Query timeout** | DuckDB query exceeds wall-clock budget | `conn.interrupt()` cancels the running query; error surfaces as a retry-able attempt failure |

All three are applied in sequence on every attempt, so a query that passes the read-only check still gets capped and timed out.

---

<a id="q223"></a>
**Q223. How does the SELECT-only guardrail work?**

```python
_WRITE_PATTERN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def _check_readonly(sql: str) -> str | None:
    m = _WRITE_PATTERN.search(sql)
    if m:
        return f"Only SELECT queries are permitted. Detected keyword: {m.group(0).upper()}"
    return None
```

`run_query()` calls `_check_readonly(sql)` after stripping markdown fences. If it returns an error string, that string is stored as `last_error` and the attempt `continue`s — exactly like a DuckDB execution error. Because the self-correcting retry loop feeds `last_error` back to the LLM in the correction prompt, the model has a chance to rewrite the query as a SELECT. After `max_retries` failures the error is returned in the final `QueryResult`.

Why regex rather than SQL parsing? A full parser (e.g. `sqlglot`) would be more precise, but it is an optional dependency with its own versioning surface. The word-boundary regex catches the vast majority of LLM-generated write attempts and has zero false positives on SELECT/CTE queries.

---

<a id="q224"></a>
**Q224. How does the result row cap work?**

```python
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

def _apply_row_cap(sql: str, limit: int) -> str:
    if not _LIMIT_PATTERN.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql}\nLIMIT {limit}"
    return sql
```

If the LLM omits a `LIMIT` clause, `_apply_row_cap` appends one before the query reaches DuckDB. The cap defaults to `max_result_rows=10_000` (configurable per `ConversationManager`). If the LLM already includes a `LIMIT`, the original limit is left untouched — the guardrail only adds, never overrides.

The trailing semicolon is stripped before appending because `SELECT * FROM t;\nLIMIT 10` is invalid SQL.

---

<a id="q225"></a>
**Q225. How does the query timeout work?**

```python
def _execute_with_timeout(conn, sql, timeout):
    timed_out = threading.Event()

    def _cancel():
        timed_out.set()
        conn.interrupt()

    timer = threading.Timer(timeout, _cancel)
    timer.start()
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in (cursor.description or [])]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        timer.cancel()
```

`threading.Timer` fires `_cancel()` on a background thread after `timeout` seconds (default `query_timeout=30.0`). `conn.interrupt()` is DuckDB's thread-safe cancellation API — it signals the running query to abort, causing `conn.execute()` to raise an exception containing "Interrupted". `run_query()` catches that exception, labels it as a timeout error, and lets the retry loop handle it:

```python
except Exception as exc:
    err_str = str(exc)
    if "Interrupted" in err_str or "interrupted" in err_str:
        last_error = f"Query timed out after {self.query_timeout}s"
    else:
        last_error = err_str
```

`timer.cancel()` in the `finally` block prevents the timer from firing if the query completes normally, avoiding a spurious interrupt on the next query.
