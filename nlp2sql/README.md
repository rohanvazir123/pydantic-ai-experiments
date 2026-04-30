# NLP-to-SQL

Natural language query interface over multiple data sources, powered by DuckDB + GPT-4o / Claude.

---

## Files

| File | Description |
|---|---|
| `nlp_sql_postgres_v1.py` | Original — GCS Parquets + PostgreSQL, returns raw tuples |
| `nlp_sql_postgres_v2.py` | **Current** — self-correcting retry, structured `QueryResult`, column names, normalized NL cache |
| `test_nlp_sql_postgres_v2.py` | Test suite for v2 |
| `test_nlp_sql_postgres.py` | Test suite for v1 |

---

## nlp_sql_postgres_v2.py

Single DuckDB in-memory session that queries three data sources with one SQL engine:

| Source | SQL access pattern | Tables |
|---|---|---|
| GCS Parquet (eagerbeaver-1) | `FROM orders` | orders, users, … |
| rag_db (Docker port 5434) | `FROM rag.main.documents` | documents, chunks, kg_entities, … |
| local_pg (PG port 5432) | `FROM local_pg.main.baby_names` | baby_names, world_gdp, articles |

---

## What v2 adds over v1

### `QueryResult` — structured return type

```python
@dataclass
class QueryResult:
    nl_query: str
    sql: str
    columns: list[str]   # from cursor.description -- no more anonymous tuples
    rows: list[tuple]
    error: str | None    # set if all retries failed
    cached: bool
    attempts: int

    def pretty_print(self, max_rows=20) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    @property
    def success(self) -> bool: ...
```

### Self-correcting retry loop

When the generated SQL fails, the error is fed back to the model along with the original question and the schema. The model corrects its own output, up to `max_retries` attempts (default 3).

```
Attempt 1: SELECT * FROM orders_2024   <- hallucinated table
           DuckDB error: Table "orders_2024" does not exist
Attempt 2: SELECT * FROM orders        <- model self-corrects
           OK — 1 423 rows, 6 cols
```

### Normalized NL cache

v1 used exact string match. v2 normalizes (lowercase + whitespace collapse) before cache lookup:

```python
"How many rows?"  ==  "how many rows?"  ==  "How  many  rows?"
```

### Multi-provider support

```python
# OpenAI (default)
source.init_agent(model="gpt-4o")

# Anthropic
source.init_agent(model="claude-sonnet-4-6", provider="anthropic")
```

### No hardcoded paths

```python
# v1 — hardcoded
_DELTALAKE_ENV = Path("C:/Users/rohan/Documents/deltalake-projects/.env")

# v2 — callers pass extra paths
load_env(Path("/your/path/to/.env"))
```

---

## Architecture

```
User NL query
    |
    v
ConversationManager
  - normalize NL -> NL cache lookup (case/whitespace insensitive)
  - build prompt: schema + last-3 successful turns + question
    |
    v
LLM (GPT-4o or Claude via Pydantic AI)
  - returns plain SQL
    |
  SQL hash cache lookup
    |   (miss)
    v
DuckDB engine
  - GCS views (httpfs + HMAC secret)
  - rag   catalog (postgres_scanner ATTACH)
  - local_pg catalog (postgres_scanner ATTACH)
    |
  success?
  yes -> QueryResult (columns + rows) -> cache -> history -> return
  no  -> correction prompt -> retry (up to max_retries)
         all retries failed -> QueryResult(error=...) -> history -> return
```

---

## Why DuckDB (not pg_parquet, duckdb_fdw, Spark, Trino)

- **Spark / Trino** — cluster-based, heavy infra, overkill for single-analyst workloads
- **pg_parquet** — PostgreSQL reads Parquet; limited SQL, no GCS auth story
- **duckdb_fdw** — wrong direction; PostgreSQL queries DuckDB via FDW, complex Windows setup
- **DuckDB postgres_scanner** — DuckDB ATTACHes PostgreSQL and JOINs it with GCS Parquets in a single in-process query; zero server-side changes needed

---

## Prerequisites

```bash
pip install duckdb pydantic-ai google-cloud-storage python-dotenv
```

DuckDB extensions (`httpfs`, `postgres`) install automatically on first run.

Required env vars:

| Variable | Used for |
|---|---|
| `GCS_HMAC_ID` | DuckDB httpfs GCS authentication |
| `GCS_HMAC_SECRET` | DuckDB httpfs GCS authentication |
| `OPENAI_API_KEY` | GPT-4o (default provider) |
| `ANTHROPIC_API_KEY` | Claude models (optional) |

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

All tests run offline — GCS and PostgreSQL connections are mocked.

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

**Increase retry attempts:**
```python
chat = source.conversation_manager(max_retries=5)
```

**Use the result as a DataFrame:**
```python
result = chat.run_query("Top 10 customers by revenue?")
df = result.to_dataframe()
```
