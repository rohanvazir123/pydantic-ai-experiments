# NLP-to-SQL

Natural language query interface over multiple data sources, powered by DuckDB + GPT-4o / Claude.

---

## Files

| File | Description |
|---|---|
| `nlp_sql_postgres_v1.py` | Original — GCS Parquets + PostgreSQL, returns raw tuples |
| `nlp_sql_postgres_v2.py` | **Current** — self-correcting retry, structured `QueryResult`, column names, normalized NL cache, execution guardrails |
| `sql_discovery.py` | Schema-discovery agent — LLM calls `list_tables`/`describe_table` tools at inference time instead of receiving a pre-built schema string |
| `test_nlp_sql_postgres_v2.py` | Test suite for v2 (92 tests, all offline) |
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

Schema is discovered at inference time by the LLM via two Pydantic AI tools registered on `sql_agent` in `sql_discovery.py` — no schema string is pre-built or injected into the prompt.

| Tool | What the LLM calls | What it returns |
|---|---|---|
| `list_tables(db_type)` | `"postgres"` or `"duckdb"` | List of table names from `information_schema.tables` (Postgres) or `SHOW TABLES` (DuckDB) |
| `describe_table(db_type, table_name)` | db + a specific table | Column names and types from `information_schema.columns` (Postgres) or `DESCRIBE` (DuckDB) |

The system prompt instructs the LLM to call these tools before generating SQL. The LLM fetches only the tables/columns relevant to the question — narrower queries use fewer tokens than a full pre-built schema.

**How the LLM knows the tools exist:** the `@sql_agent.tool` decorator registers each function with the agent. When `sql_agent.run()` is called, Pydantic AI serialises every registered tool — its name, parameter names/types, and **docstring** — into the tool spec sent to the LLM in the API request. The docstrings (`"List tables for 'postgres' or 'duckdb'."` and `"Get columns and types for a specific table in either database."`) are the descriptions the LLM reads to decide when and how to call each tool. The system prompt `"Discover the schema first"` then nudges it to call them before generating SQL rather than guessing table names.

**Integrated via** `UnifiedDataSource.discovery_query(prompt, pg_alias=None)` in `nlp_sql_postgres_v2.py` — resolves the Postgres DSN from `postgres_dbs`, creates an `asyncpg` pool, runs the discovery agent, executes the returned SQL, and returns `(SQLResponse, columns, rows)`.

**Contrast with `generate_schema()` (commented out):** the old approach serialised the full schema once at startup and prepended it to every prompt. Table changes required a restart; the full schema was sent on every call regardless of query scope.

---

## How Pydantic AI manages conversation context

Within a single `sql_agent.run()` call, Pydantic AI manages the tool-calling loop internally as a growing message list:

```
user prompt
  → LLM response: call list_tables("postgres")
  → tool result appended as a message
  → LLM response: call describe_table("postgres", "documents")
  → tool result appended as a message
  → LLM response: SQLResponse(sql=..., explanation=...)   ← final structured output
```

Each tool result is sent back to the LLM as part of the same conversation so it can reason over accumulated schema information before generating SQL. This is handled entirely by Pydantic AI — no manual message threading required.

**Across calls** (`ConversationManager`), context is managed manually. `_history_context(n=3)` serialises the last 3 successful turns as a plain string block prepended to the next prompt:

```
Conversation so far:
Q: Revenue per customer?
SQL: SELECT c.name, SUM(s.revenue) ...
Result preview: [('Alice', 3000.0), ('Bob', 1400.0)]

Question: only US customers?
```

This is injected in the user-turn prompt, not via Pydantic AI's message history API — keeping it simple and giving full control over what context the model sees. Failed turns are excluded so bad SQL examples don't confuse the model.

---

## Prompting

System prompt enforces DuckDB-specific table naming rules and mandates plain SQL (no markdown fences, no explanation, no comments). Schema is not injected — the LLM discovers it by calling `list_tables`/`describe_table` tools. The last 3 successful conversation turns are included as history context. Zero-shot — no hardcoded few-shot examples.

---

## Known limitations

| Limitation | Detail |
|---|---|
| **PostgreSQL full scans** | `postgres_scanner` reads entire PG tables; no index pushdown. Large tables (>10M rows) are slow. |
| **In-memory result sets** | Very large results can OOM. Row cap guardrail mitigates but doesn't eliminate this. |
| **Static schema** | ~~Captured at startup.~~ Schema is now discovered dynamically via tools — no restart needed for table changes. |
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

Runs two discovery queries — the LLM calls `list_tables`/`describe_table` tools to explore schema, then generates and executes SQL.

## Running tests

```bash
pytest test_nlp_sql_postgres_v2.py -v
```

All 92 tests run offline — GCS and PostgreSQL connections are mocked.

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
