# NLP-to-SQL

Natural language query interface over multiple data sources, powered by DuckDB + GPT-4o / Claude.

---

## Files

| File | Description |
|---|---|
| `nlp_sql_postgres_v1.py` | Original — GCS Parquets + PostgreSQL, returns raw tuples |
| `nlp_sql_postgres_v2.py` | **Current** — self-correcting retry, structured `QueryResult`, column names, normalized NL cache, execution guardrails |
| `test_nlp_sql_postgres_v2.py` | Test suite for v2 (87 tests, all offline) |
| `test_nlp_sql_postgres.py` | Test suite for v1 |

---

## Data sources

Single DuckDB in-memory session queries three sources with one SQL engine:

| Source | SQL access pattern | Example tables |
|---|---|---|
| GCS Parquet | `FROM orders` | orders, users, … |
| rag_db (PostgreSQL) | `FROM rag.main.documents` | documents, chunks, kg_entities |
| local_pg (PostgreSQL) | `FROM local_pg.main.baby_names` | baby_names, world_gdp, articles |

---

## End-to-end flow

`ConversationManager.run_query(nl_query)`:

1. NL query is normalized (lowercase + whitespace-collapsed) and checked against the NL cache — exact normalized match returns the cached `QueryResult` immediately.
2. Prompt is built: schema text + last 3 successful conversation turns (Q/SQL/result preview) + the new question.
3. `agent.run_sync(prompt)` calls GPT-4o (or Claude), which returns SQL.
4. `strip_sql_fences()` cleans any markdown wrapping.
5. **Guardrail — SELECT-only check**: SQL is rejected if it contains write/DDL keywords; error fed to correction prompt.
6. **Guardrail — row cap**: `LIMIT N` appended if the SQL has no LIMIT clause.
7. SQL is MD5-hashed and checked against the SQL hash cache — same SQL for a different question returns the cached result.
8. **Guardrail — query timeout**: DuckDB query runs under a `threading.Timer`; `conn.interrupt()` cancels if it exceeds budget.
9. On success: `QueryResult` (columns + rows + attempts) stored in both caches and history, returned.
10. On failure: error fed back to LLM in a correction prompt; retries up to `max_retries` (default 3). All retries failed → `QueryResult(error=...)`.

---

## `QueryResult` — structured return type

```python
@dataclass
class QueryResult:
    nl_query: str
    sql: str
    columns: list[str]   # from cursor.description — no more anonymous tuples
    rows: list[tuple]
    error: str | None    # set if all retries failed
    cached: bool
    attempts: int

    def pretty_print(self, max_rows=20) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    @property
    def success(self) -> bool: ...
```

---

## Self-correcting retry loop

When generated SQL fails, the error is fed back to the model with the original question and schema:

```
The following SQL you generated failed:
Question: {original_nl_query}
SQL: {bad_sql}
Error: {duckdb_error_message}

Return ONLY the corrected SQL.
```

```
Attempt 1: SELECT * FROM orders_2024   <- hallucinated table
           DuckDB error: Table "orders_2024" does not exist
Attempt 2: SELECT * FROM orders        <- model self-corrects
           OK — 1 423 rows, 6 cols
```

Failed turns are recorded in `history` for audit but **excluded** from the history context shown to the model on the next turn, so bad SQL examples don't confuse it.

---

## Guardrails

Three execution-time guardrails run on every attempt, before DuckDB executes:

| Guardrail | When it fires | Effect |
|---|---|---|
| **SELECT-only** | SQL contains DROP / DELETE / INSERT / UPDATE / TRUNCATE / ALTER / CREATE / GRANT / REVOKE | Treated as an attempt error; retry loop asks LLM to rewrite as SELECT |
| **Result row cap** | No `LIMIT` clause present | Appends `LIMIT N` automatically (`max_result_rows=10_000` by default) |
| **Query timeout** | Query exceeds wall-clock budget | `conn.interrupt()` cancels the running query; surfaces as a retry-able error |

### SELECT-only detail

```python
_WRITE_PATTERN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
```

Word-boundary regex, case-insensitive. Returns `None` on clean SELECT/CTE queries. The check fires before DuckDB sees the query — the table is never touched.

### Row cap detail

```python
def _apply_row_cap(sql: str, limit: int) -> str:
    if not _LIMIT_PATTERN.search(sql):
        sql = sql.rstrip().rstrip(";")   # trailing ; makes LIMIT syntax invalid
        return f"{sql}\nLIMIT {limit}"
    return sql
```

Only adds — never overrides an existing `LIMIT`.

### Timeout detail

```python
timer = threading.Timer(timeout, lambda: conn.interrupt())
timer.start()
try:
    cursor = conn.execute(sql)
    ...
finally:
    timer.cancel()   # no spurious interrupt if query completes in time
```

`conn.interrupt()` is DuckDB's thread-safe cancellation API. The raised exception is detected by checking for `"Interrupted"` in the message and re-surfaced as `"Query timed out after Xs"`.

---

## Conversation context

`history: list[tuple[str, str, QueryResult]]` stores every turn. `_history_context(n=3)` serializes the last 3 **successful** turns:

```
Q: Revenue per customer?
SQL: SELECT c.name, SUM(s.revenue) ...
Result preview: [('Alice', 3000.0), ('Bob', 1400.0)]
```

This block is prepended to every new prompt as "Conversation so far:", enabling follow-up questions like *"only US customers?"*.

---

## Schema discovery

`UnifiedDataSource.generate_schema()` introspects all sources at startup:

**GCS Parquet:** Lists blob virtual prefixes (`delimiter="/"`) — each subfolder becomes a table name. Creates a DuckDB `VIEW` over `parquet_scan('gs://...')`. `DESCRIBE view_name` gives columns and types.

**PostgreSQL:** After attaching via DuckDB's postgres extension, queries `{alias}.information_schema.tables` and `{alias}.information_schema.columns` through DuckDB's catalog.

Everything is serialized into a single schema string prepended to every LLM prompt. Schema is captured once at startup — table changes require a restart.

---

## Prompting

System prompt enforces DuckDB-specific table naming rules and mandates plain SQL (no markdown fences, no explanation, no comments). The schema string is injected in the user-turn prompt. The last 3 successful conversation turns are included as history context. Zero-shot — no hardcoded few-shot examples.

---

## Known limitations

| Limitation | Detail |
|---|---|
| **PostgreSQL full scans** | `postgres_scanner` reads entire PG tables; no index pushdown. Large tables (>10M rows) are slow. |
| **In-memory result sets** | Very large results can OOM. Row cap guardrail mitigates but doesn't eliminate this. |
| **Static schema** | Captured at startup. Table changes require restart. |
| **GCS auth** | HMAC keys only (`GCS_HMAC_ID` + `GCS_HMAC_SECRET`). Service account JSON / Workload Identity not supported. |
| **Semantically wrong SQL** | Syntactically valid but logically wrong SQL returns wrong results silently. No semantic validation layer. |
| **Ambiguous NL** | *"Show me recent findings"* — model guesses. No clarification step implemented. |

---

## Why DuckDB

| Option | Problem |
|---|---|
| **Spark / Trino** | Cluster-based, heavy infrastructure. Overkill for single-analyst workloads. 10–30s startup latency. |
| **pg_parquet** | PostgreSQL reads Parquet; limited SQL, no GCS HMAC auth, PostgreSQL is the bottleneck. |
| **duckdb_fdw** | Wrong direction — PostgreSQL queries DuckDB via FDW. Complex Windows setup, server-side changes required. |
| **DuckDB postgres_scanner** | DuckDB ATTACHes PostgreSQL as a catalog and JOINs it with GCS Parquets in a single in-process query. Zero server-side changes, zero extra infrastructure. |

Cross-source JOINs run 100% inside DuckDB's in-memory engine. GCS Parquets are read lazily via `httpfs` (predicate pushdown where possible). PostgreSQL tables are scanned via `postgres_scanner` (full table scan). DuckDB handles JOIN, aggregation, and projection internally.

---

## v1 vs v2

| | v1 | v2 |
|---|---|---|
| **Return type** | Raw `Any` (list of tuples or `None`) | `QueryResult` with `.columns`, `.rows`, `.success`, `.error`, `.attempts` |
| **SQL errors** | Silent `None`, dead end | Self-correcting retry loop: error fed back to LLM, up to `max_retries` (default 3) |
| **NL cache matching** | Exact string equality | Normalized: lowercase + whitespace-collapsed |
| **Column names** | Anonymous tuples | Populated from `cursor.description` |
| **Provider** | OpenAI only, hardcoded path | `provider="openai"` or `"anthropic"`, env-var paths |
| **History context** | Includes failed turns | Failed turns excluded from context shown to model |
| **Guardrails** | None | SELECT-only enforcement, result row cap, query timeout |

---

## Prerequisites

```bash
pip install duckdb pydantic-ai google-cloud-storage python-dotenv
```

DuckDB extensions (`httpfs`, `postgres`) install automatically on first run.

Required env vars:

| Variable | Used for |
|---|---|
| `GCS_BUCKET` | GCS bucket name |
| `GCS_USER_PROJECT` | GCS billing project |
| `GCS_HMAC_ID` | DuckDB httpfs GCS authentication |
| `GCS_HMAC_SECRET` | DuckDB httpfs GCS authentication |
| `RAG_DB_URL` | PostgreSQL connection string for rag_db |
| `LOCAL_PG_URL` | PostgreSQL connection string for local_pg |
| `OPENAI_API_KEY` | GPT-4o (default provider) |
| `ANTHROPIC_API_KEY` | Claude models (optional) |
| `EXTRA_ENV_PATH` | Optional second `.env` file path |

---

## Running

```bash
python nlp_sql_postgres_v2.py
```

Prints the full unified schema then runs five sample queries spanning all three sources.

## Running tests

```bash
pytest test_nlp_sql_postgres_v2.py -v
```

All 87 tests run offline — GCS and PostgreSQL connections are mocked.

---

## Extending

**Add another PostgreSQL database:**
```python
PostgresDB(alias="my_db", connection_string="postgresql://user:pass@host:port/dbname")
```

**Use Claude instead of GPT-4o:**
```python
source.init_agent(model="claude-sonnet-4-6", provider="anthropic")
```

**Tune guardrail parameters:**
```python
chat = source.conversation_manager(
    max_retries=5,
    max_result_rows=50_000,
    query_timeout=60.0,
)
```

**Use the result as a DataFrame:**
```python
result = chat.run_query("Top 10 customers by revenue?")
df = result.to_dataframe()
```
