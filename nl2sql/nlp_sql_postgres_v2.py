"""
nlp_sql_postgres_v2.py

Improvements over v1:
- QueryResult dataclass: columns + rows + error + attempts + cached flag
- Self-correcting retry loop: on SQL error, error is fed back to the LLM (up to max_retries)
- Column names from cursor.description -- no more anonymous tuples
- Normalized NL cache: case-insensitive + whitespace-collapsed
  "How many rows?" == "how  many  rows?" == "HOW MANY ROWS?"
- Separate bounded LRU caches for NL hits and SQL hash hits
- Provider-agnostic model init: "openai" or "anthropic"
- No hardcoded filesystem paths -- load_env() accepts extra paths from callers
- HistoryStore: asyncpg-backed persistence so history survives process restarts
- Execution guardrails: SELECT-only enforcement, result row cap, query timeout
- GCS Parquet via PyArrow datasets (lazy reads, predicate pushdown) instead of DuckDB httpfs
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import asyncpg
import duckdb
import gcsfs
import pyarrow.dataset as ds
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def load_env(*extra_paths: Path) -> None:
    """Load .env files. Callers pass any paths beyond the default project root."""
    default = Path(__file__).parent.parent / ".env"
    for p in [default, *extra_paths]:
        if p.exists():
            load_dotenv(p, override=False)
            logger.info("Loaded .env: %s", p)
        else:
            logger.warning(".env not found: %s", p)


# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------
_WRITE_PATTERN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _check_readonly(sql: str) -> str | None:
    """Return an error string if SQL contains write/DDL keywords, else None."""
    m = _WRITE_PATTERN.search(sql)
    if m:
        return f"Only SELECT queries are permitted. Detected keyword: {m.group(0).upper()}"
    return None


def _apply_row_cap(sql: str, limit: int) -> str:
    """Append LIMIT if the SQL has none, capping result size."""
    if not _LIMIT_PATTERN.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql}\nLIMIT {limit}"
    return sql


def _execute_with_timeout(
    conn: "duckdb.DuckDBPyConnection",
    sql: str,
    timeout: float,
) -> tuple[list[str], list[tuple]]:
    """
    Execute *sql* on *conn*; raise TimeoutError if it takes longer than *timeout* seconds.
    Uses conn.interrupt() — the only thread-safe way to cancel a running DuckDB query.
    """
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
    # If interrupt fired before cancel(), DuckDB raises an exception that
    # propagates normally; we re-raise it as TimeoutError in run_query().


# ---------------------------------------------------------------------------
# SQL fence stripper
# ---------------------------------------------------------------------------
def strip_sql_fences(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```") and sql.endswith("```"):
        lines = sql.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql


_SYSTEM_PROMPT = """\
You are an expert SQL assistant working with DuckDB.

Table naming rules (IMPORTANT — always follow these):
- GCS Parquet tables  -> bare table name,          e.g.  FROM orders
- rag_db tables       -> rag.main.<table>,          e.g.  FROM rag.main.documents
- local_pg tables     -> local_pg.main.<table>,     e.g.  FROM local_pg.main.baby_names

Return ONLY plain SQL. No Markdown fences, no explanation, no comments.\
"""


# ---------------------------------------------------------------------------
# QueryResult -- structured return type replacing raw Any
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    nl_query: str
    sql: str
    columns: list[str]
    rows: list[tuple]
    error: str | None = None
    cached: bool = False
    attempts: int = 1

    @property
    def success(self) -> bool:
        return self.error is None

    def pretty_print(self, max_rows: int = 20) -> None:
        attempts_str = f"{self.attempts} attempt{'s' if self.attempts > 1 else ''}"
        status_str = "cached" if self.cached else "fresh"
        print(f"\nQ: {self.nl_query}")
        print(f"SQL ({attempts_str}, {status_str}):")
        print(f"  {self.sql}")
        if self.error:
            print(f"Error: {self.error}")
            return
        if not self.rows:
            print("(no rows)")
            return
        col_widths = [len(c) for c in self.columns]
        for row in self.rows[:max_rows]:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))
        header = "  ".join(c.ljust(w) for c, w in zip(self.columns, col_widths))
        sep    = "  ".join("-" * w for w in col_widths)
        print(header)
        print(sep)
        for row in self.rows[:max_rows]:
            print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
        if len(self.rows) > max_rows:
            print(f"... ({len(self.rows) - max_rows} more rows not shown)")

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Install pandas: pip install pandas") from None
        return pd.DataFrame(self.rows, columns=self.columns)


# ---------------------------------------------------------------------------
# HistoryStore -- asyncpg-backed persistence across process restarts
# ---------------------------------------------------------------------------
class HistoryStore:
    """
    Persists ConversationManager history to a PostgreSQL table using asyncpg.

    One HistoryStore instance (one pool) can serve multiple sessions — session_id
    is passed per-call so the same store can be shared across ConversationManagers.

    Usage:
        store = await HistoryStore.create(dsn)          # creates pool + table
        store = HistoryStore(existing_pool)             # reuse an existing pool
        sessions = await store.sessions()               # list all session IDs
        await store.close()                             # close the pool
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id         BIGSERIAL    PRIMARY KEY,
            session_id TEXT         NOT NULL,
            nl_query   TEXT         NOT NULL,
            sql        TEXT         NOT NULL,
            columns    JSONB        NOT NULL,
            rows       JSONB        NOT NULL,
            error      TEXT,
            cached     BOOLEAN      NOT NULL,
            attempts   INTEGER      NOT NULL,
            ts         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """
    _INDEX = """
        CREATE INDEX IF NOT EXISTS idx_ch_session
        ON conversation_history(session_id, ts)
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def create(cls, dsn: str, **pool_kwargs) -> "HistoryStore":
        """Create a pool, ensure the schema exists, and return a HistoryStore."""
        pool = await asyncpg.create_pool(dsn, **pool_kwargs)
        store = cls(pool)
        await store._init_schema()
        logger.info("HistoryStore ready (pool min=%s max=%s)",
                    pool_kwargs.get("min_size", 1), pool_kwargs.get("max_size", 5))
        return store

    async def _init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(self._CREATE)
            await conn.execute(self._INDEX)

    async def save(self, session_id: str, nl_query: str, sql: str, qr: "QueryResult") -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_history
                    (session_id, nl_query, sql, columns, rows, error, cached, attempts)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8)
                """,
                session_id,
                nl_query,
                sql,
                json.dumps(qr.columns),
                json.dumps([list(r) for r in qr.rows]),
                qr.error,
                qr.cached,
                qr.attempts,
            )

    async def load(self, session_id: str) -> list[tuple[str, str, "QueryResult"]]:
        async with self.pool.acquire() as conn:
            db_rows = await conn.fetch(
                """
                SELECT nl_query, sql, columns, rows, error, cached, attempts
                FROM conversation_history
                WHERE session_id = $1
                ORDER BY ts
                """,
                session_id,
            )
        history = []
        for row in db_rows:
            qr = QueryResult(
                nl_query=row["nl_query"],
                sql=row["sql"],
                columns=row["columns"],        # asyncpg decodes JSONB automatically
                rows=[tuple(r) for r in row["rows"]],
                error=row["error"],
                cached=row["cached"],
                attempts=row["attempts"],
            )
            history.append((row["nl_query"], row["sql"], qr))
        return history

    async def sessions(self) -> list[str]:
        """Return all session IDs, ordered by first-seen time."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id
                FROM conversation_history
                GROUP BY session_id
                ORDER BY MIN(ts)
                """
            )
        return [r["session_id"] for r in rows]

    async def close(self) -> None:
        await self.pool.close()


# ---------------------------------------------------------------------------
# ConversationManager -- history + self-correcting retry + two-level cache
# ---------------------------------------------------------------------------
class ConversationManager:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        agent: Agent,
        schema_text: str,
        cache_size: int = 20,
        max_retries: int = 3,
        max_result_rows: int = 10_000,
        query_timeout: float = 30.0,
        history_store: Optional[HistoryStore] = None,
        session_id: Optional[str] = None,
        _initial_history: Optional[list] = None,
    ):
        self.conn = conn
        self.agent = agent
        self.schema_text = schema_text
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.max_result_rows = max_result_rows
        self.query_timeout = query_timeout
        self.history_store = history_store
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[tuple[str, str, QueryResult]] = list(_initial_history or [])
        self._sql_cache: OrderedDict[str, QueryResult] = OrderedDict()
        self._nl_cache:  OrderedDict[str, QueryResult] = OrderedDict()
        # Warm caches from any pre-loaded history so resumed sessions get cache hits
        for nl, sql, qr in self.history:
            if qr.success:
                self._cache_put(self._nl_cache, self._normalize_nl(nl), qr)
                self._cache_put(self._sql_cache, self._hash(sql), qr)

    @staticmethod
    def _hash(sql: str) -> str:
        return hashlib.md5(sql.encode()).hexdigest()

    @staticmethod
    def _normalize_nl(nl: str) -> str:
        return " ".join(nl.lower().split())

    def _cache_put(self, cache: OrderedDict, key: str, value: QueryResult) -> None:
        cache[key] = value
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _history_context(self, n: int = 3) -> str:
        """Only successful turns are shown as examples -- failed SQL confuses the model."""
        parts = []
        for nl, sql, qr in self.history[-n:]:
            if qr.success:
                parts.append(f"Q: {nl}\nSQL: {sql}\nResult preview: {qr.rows[:3]}")
        return "\n\n".join(parts)

    def _build_prompt(self, nl_query: str) -> str:
        history = self._history_context()
        history_block = f"Conversation so far:\n{history}\n\n" if history else ""
        return (
            f"Schema:\n{self.schema_text}\n\n"
            f"{history_block}"
            f"Question: {nl_query}"
        )

    def _build_correction_prompt(self, nl_query: str, bad_sql: str, error: str) -> str:
        error_snippet = error[:400] if len(error) > 400 else error
        return (
            f"Schema:\n{self.schema_text}\n\n"
            f"The following SQL you generated failed:\n"
            f"Question: {nl_query}\n"
            f"SQL: {bad_sql}\n"
            f"Error: {error_snippet}\n\n"
            f"Return ONLY the corrected SQL."
        )

    async def run_query(self, nl_query: str) -> QueryResult:
        nl_key = self._normalize_nl(nl_query)

        # NL-level cache hit (normalized match) — no new turn, no persistence needed
        if nl_key in self._nl_cache:
            cached = self._nl_cache[nl_key]
            logger.info("NL cache hit -> %s", cached.sql)
            return QueryResult(
                nl_query=nl_query,
                sql=cached.sql,
                columns=cached.columns,
                rows=cached.rows,
                cached=True,
            )

        last_error: str | None = None
        sql = ""

        for attempt in range(1, self.max_retries + 1):
            prompt = (
                self._build_prompt(nl_query)
                if attempt == 1
                else self._build_correction_prompt(nl_query, sql, last_error)
            )
            raw = await self.agent.run(prompt)
            sql = strip_sql_fences(raw.output)
            logger.info("Attempt %d/%d — SQL: %s", attempt, self.max_retries, sql)

            # Guardrail 1: SELECT-only enforcement
            readonly_err = _check_readonly(sql)
            if readonly_err:
                last_error = readonly_err
                logger.warning("Guardrail (read-only): %s", readonly_err)
                continue

            # Guardrail 2: result row cap (append LIMIT if missing)
            safe_sql = _apply_row_cap(sql, self.max_result_rows)

            # SQL hash cache (same SQL generated for a different NL question)
            h = self._hash(safe_sql)
            if h in self._sql_cache:
                logger.info("SQL cache hit")
                cached_qr = self._sql_cache[h]
                qr = QueryResult(
                    nl_query=nl_query,
                    sql=safe_sql,
                    columns=cached_qr.columns,
                    rows=cached_qr.rows,
                    cached=True,
                    attempts=attempt,
                )
                self._cache_put(self._nl_cache, nl_key, qr)
                self.history.append((nl_query, safe_sql, qr))
                if self.history_store:
                    await self.history_store.save(self.session_id, nl_query, safe_sql, qr)
                return qr

            try:
                # Guardrail 3: query timeout
                columns, rows = _execute_with_timeout(
                    self.conn, safe_sql, self.query_timeout
                )
                qr = QueryResult(
                    nl_query=nl_query,
                    sql=safe_sql,
                    columns=columns,
                    rows=rows,
                    attempts=attempt,
                )
                self._cache_put(self._sql_cache, h, qr)
                self._cache_put(self._nl_cache, nl_key, qr)
                self.history.append((nl_query, safe_sql, qr))
                if self.history_store:
                    await self.history_store.save(self.session_id, nl_query, safe_sql, qr)
                logger.info("OK — %d row(s), %d col(s)", len(rows), len(columns))
                return qr
            except Exception as exc:
                err_str = str(exc)
                if "Interrupted" in err_str or "interrupted" in err_str:
                    last_error = f"Query timed out after {self.query_timeout}s"
                    logger.warning("Guardrail (timeout): %s", last_error)
                else:
                    last_error = err_str
                    logger.warning("Attempt %d/%d failed: %s", attempt, self.max_retries, last_error)

        # All retries exhausted -- return error result but still record in history
        qr = QueryResult(
            nl_query=nl_query,
            sql=sql,
            columns=[],
            rows=[],
            error=last_error,
            attempts=self.max_retries,
        )
        self.history.append((nl_query, sql, qr))
        if self.history_store:
            await self.history_store.save(self.session_id, nl_query, sql, qr)
        return qr

    def show_history(self) -> None:
        for i, (nl, sql, qr) in enumerate(self.history, 1):
            status = "OK" if qr.success else f"ERROR: {qr.error}"
            logger.info("%d. Q: %s\n   SQL: %s\n   Status: %s", i, nl, sql, status)


# ---------------------------------------------------------------------------
# Data-source descriptors
# ---------------------------------------------------------------------------
@dataclass
class PostgresDB:
    alias: str               # DuckDB catalog name, e.g. "rag" or "local_pg"
    connection_string: str   # postgresql://user:pass@host:port/db


# ---------------------------------------------------------------------------
# UnifiedDataSource -- wires everything into one DuckDB session
# ---------------------------------------------------------------------------
@dataclass
class UnifiedDataSource:
    conn: duckdb.DuckDBPyConnection
    gcs_bucket: str
    gcs_prefix: str           # e.g. "partitioned_data/"
    gcs_user_project: str
    postgres_dbs: list[PostgresDB] = field(default_factory=list)
    agent: Optional[Agent] = field(default=None, repr=False)
    schema_text: Optional[str] = None
    _gcs_views: dict[str, str] = field(default_factory=dict)

    def load_gcs_tables(self) -> None:
        # gcsfs uses ADC (gcloud auth application-default login) or GOOGLE_APPLICATION_CREDENTIALS.
        # Predicate pushdown and column projection are handled by PyArrow before any data transfer.
        fs = gcsfs.GCSFileSystem(project=self.gcs_user_project)
        bucket_prefix = f"{self.gcs_bucket}/{self.gcs_prefix.rstrip('/')}"
        try:
            entries = fs.ls(bucket_prefix, detail=False)
        except Exception as exc:
            logger.error("Failed to list GCS prefix gs://%s: %s", bucket_prefix, exc)
            return
        for entry in entries:
            if not fs.isdir(entry):
                continue
            name = entry.rstrip("/").split("/")[-1]
            view = name.lower()
            # ds.dataset() is lazy — only reads Parquet footers for schema discovery,
            # no row data is transferred until DuckDB executes a query against this view.
            arrow_ds = ds.dataset(entry, filesystem=fs, format="parquet")
            self.conn.register(view, arrow_ds)
            self._gcs_views[name] = view
            logger.info("PyArrow dataset (lazy): %-25s <- gs://%s", view, entry)
        logger.info("GCS tables: %s", sorted(self._gcs_views))

    def attach_postgres_dbs(self) -> None:
        self.conn.execute("INSTALL postgres; LOAD postgres;")
        for db in self.postgres_dbs:
            self.conn.execute(
                f"ATTACH '{db.connection_string}' AS {db.alias} (TYPE postgres, READ_ONLY)"
            )
            logger.info("Attached PostgreSQL: %s", db.alias)

    def generate_schema(self) -> str:
        lines: list[str] = []
        if self._gcs_views:
            lines.append("=== GCS Parquet tables (use bare name) ===")
            for view in sorted(self._gcs_views.values()):
                cols = self.conn.execute(f"DESCRIBE {view}").fetchall()
                lines.append(f"Table: {view}")
                for col in cols:
                    lines.append(f"  - {col[0]} ({col[1]})")
                lines.append("")
        for db in self.postgres_dbs:
            lines.append(
                f"=== {db.alias} tables (prefix: {db.alias}.main.<table>) ==="
            )
            tables = self.conn.execute(f"""
                SELECT table_name
                FROM {db.alias}.information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type   = 'BASE TABLE'
                ORDER BY table_name
            """).fetchall()
            for (table_name,) in tables:
                cols = self.conn.execute(f"""
                    SELECT column_name, data_type
                    FROM {db.alias}.information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = '{table_name}'
                    ORDER BY ordinal_position
                """).fetchall()
                lines.append(f"Table: {db.alias}.main.{table_name}")
                for col_name, col_type in cols:
                    lines.append(f"  - {col_name} ({col_type})")
                lines.append("")
        self.schema_text = "\n".join(lines).strip()
        return self.schema_text

    def init_agent(
        self,
        model: str = "gpt-4o",
        provider: Literal["openai", "anthropic"] = "openai",
    ) -> Agent:
        if provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            llm = AnthropicModel(model, api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            llm = OpenAIModel(model, api_key=os.environ["OPENAI_API_KEY"])
        self.agent = Agent(model=llm, result_type=str, system_prompt=_SYSTEM_PROMPT)
        logger.info("Agent ready: %s (%s)", model, provider)
        return self.agent

    async def conversation_manager(
        self,
        cache_size: int = 20,
        max_retries: int = 3,
        max_result_rows: int = 10_000,
        query_timeout: float = 30.0,
        history_store: Optional[HistoryStore] = None,
        session_id: Optional[str] = None,
    ) -> ConversationManager:
        if self.agent is None or self.schema_text is None:
            raise ValueError("Call generate_schema() and init_agent() first.")
        sid = session_id or str(uuid.uuid4())
        initial_history: list = []
        if history_store is not None:
            initial_history = await history_store.load(sid)
            logger.info("Resumed session %s — %d prior turn(s)", sid, len(initial_history))
        return ConversationManager(
            self.conn,
            self.agent,
            self.schema_text,
            cache_size=cache_size,
            max_retries=max_retries,
            max_result_rows=max_result_rows,
            query_timeout=query_timeout,
            history_store=history_store,
            session_id=sid,
            _initial_history=initial_history,
        )


# ---------------------------------------------------------------------------
# Entry point
#
# Required env vars (add to .env):
#   GCS_BUCKET, GCS_PREFIX, GCS_USER_PROJECT
#   GCS auth: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
#             or run: gcloud auth application-default login
#   RAG_DB_URL        e.g. postgresql://user:pass@host:port/db
#   LOCAL_PG_URL      e.g. postgresql://user:pass@host:port/db
#   OPENAI_API_KEY    (or ANTHROPIC_API_KEY if using provider="anthropic")
#
# Optional:
#   EXTRA_ENV_PATH    absolute path to a second .env file
#   LLM_MODEL         model name (default: gpt-4o)
#   LLM_PROVIDER      openai | anthropic (default: openai)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def main() -> None:
        extra = os.environ.get("EXTRA_ENV_PATH")
        load_env(*([Path(extra)] if extra else []))

        conn = duckdb.connect(database=":memory:")
        source = UnifiedDataSource(
            conn=conn,
            gcs_bucket=os.environ["GCS_BUCKET"],
            gcs_prefix=os.environ.get("GCS_PREFIX", "partitioned_data/"),
            gcs_user_project=os.environ["GCS_USER_PROJECT"],
            postgres_dbs=[
                PostgresDB("rag",      os.environ["RAG_DB_URL"]),
                PostgresDB("local_pg", os.environ["LOCAL_PG_URL"]),
            ],
        )

        source.load_gcs_tables()
        source.attach_postgres_dbs()

        schema = source.generate_schema()
        print("\n--- UNIFIED SCHEMA ---")
        print(schema)
        print("--- END SCHEMA ---\n")

        source.init_agent(
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            provider=os.environ.get("LLM_PROVIDER", "openai"),
        )

        # Optional: persist history across runs.
        # history_store = await HistoryStore.create(os.environ["LOCAL_PG_URL"])
        # session_id = os.environ.get("SESSION_ID")  # pass to resume a prior session
        chat = await source.conversation_manager(max_retries=3)

        queries = [
            "What was the total items sold by user?",
            "Which products have sale numbers above 200?",
            "How many baby names were registered in 1990?",
            "Which countries had the highest GDP in 2020?",
            "How many documents are in the RAG knowledge base?",
        ]

        for q in queries:
            result = await chat.run_query(q)
            result.pretty_print()

        print("\n--- Conversation history ---")
        chat.show_history()

    asyncio.run(main())
