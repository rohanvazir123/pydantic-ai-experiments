# NLP-to-SQL

Natural language query interface over multiple data sources, powered by DuckDB + GPT-4o.

---

## Files

| File | Description |
|---|---|
| `nlp_sql_working_version6.py` | Original — GCS Parquets only, LangChain + DuckDB |
| `nlp_sql_postgres_v1.py` | Extended — GCS Parquets + PostgreSQL via DuckDB `postgres_scanner` |

---

## nlp_sql_postgres_v1.py

### What it does

Single DuckDB in-memory session that queries three data sources with one SQL engine:

| Source | Access pattern in SQL | Tables |
|---|---|---|
| GCS Parquet (eagerbeaver-1) | `FROM orders` (bare name) | orders, users, … |
| rag_db (Docker port 5434) | `FROM rag.main.documents` | documents, chunks, kg_entities, kg_relationships, pdf_documents, pdf_questions, pdf_chunks |
| local_pg (PG18 port 5432) | `FROM local_pg.main.baby_names` | baby_names, world_gdp, articles |

The LLM (GPT-4o) receives the unified schema and generates DuckDB SQL. A `ConversationManager` maintains history across turns and caches results at two levels: exact NL match and SQL hash.

### Why DuckDB (not pg_parquet or duckdb_fdw)

- **pg_parquet** — PostgreSQL reads Parquet; limited SQL, no GCS auth story, PG is the bottleneck
- **duckdb_fdw** — PostgreSQL queries DuckDB via FDW; wrong direction, complex Windows setup
- **DuckDB postgres_scanner** — DuckDB ATTACHes PostgreSQL and can JOIN GCS parquets with PG tables in a single query; zero PostgreSQL-side changes needed

### Architecture

```
User NL query
    |
    v
ConversationManager
  - history (last 3 turns in prompt)
  - NL-level cache (exact match skips LLM)
  - SQL hash cache (skip re-execution)
    |
    v
LLM (gpt-4o via LangChain)
  - prompt = unified schema + history + question
  - outputs plain SQL
    |
    v
DuckDB engine
  - GCS views (httpfs + HMAC secret)
  - rag   catalog (postgres_scanner ATTACH)
  - local_pg catalog (postgres_scanner ATTACH)
    |
    v
Result -> history
```

### Prerequisites

```bash
pip install duckdb langchain langchain-openai google-cloud-storage python-dotenv
```

DuckDB extensions (`httpfs`, `postgres`) are installed automatically on first run via `INSTALL`.

The GCS and OpenAI credentials are loaded from `../env` (i.e. `deltalake-projects/.env`).

Required env vars:

| Variable | Used for |
|---|---|
| `GCS_HMAC_ID` | DuckDB httpfs GCS authentication |
| `GCS_HMAC_SECRET` | DuckDB httpfs GCS authentication |
| `OPENAI_API_KEY` | GPT-4o via LangChain |

### Running

```bash
# From the nlp_sql directory
python nlp_sql_postgres_v1.py
```

The script prints the full unified schema then runs five sample queries spanning all three sources.

### Extending

**Add another PostgreSQL database:**
```python
PostgresDB(
    alias="my_db",
    connection_string="postgresql://user:pass@host:port/dbname",
)
```

**Add a custom query:**
```python
result = chat.run_query("Join orders with world GDP to find total sales by country GDP tier")
```

**Change the model:**
```python
source.init_sql_chain(model="gpt-4o-mini")  # cheaper, faster
```
