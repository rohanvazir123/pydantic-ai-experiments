"""
nlp_sql_postgres_v1.py

NL -> SQL over a unified schema combining:
  - GCS Parquet tables  (eagerbeaver-1 / partitioned_data/)
  - rag_db PostgreSQL   (localhost:5434 -- documents, chunks, kg_entities, ...)
  - local_pg PostgreSQL (localhost:5432 -- baby_names, world_gdp, articles)

DuckDB is the single query engine.  PostgreSQL tables are attached via
DuckDB's native postgres_scanner extension -- no FDW, no pg_parquet needed.
GCS Parquets use httpfs, same as nlp_sql_working_version6.py.

Architecture:
  User NL -> ConversationManager -> Pydantic AI Agent (gpt-4o) -> SQL -> DuckDB -> result

Table naming in generated SQL:
  GCS views  ->  bare name              e.g.  FROM orders
  rag_db     ->  rag.main.<table>       e.g.  FROM rag.main.documents
  local_pg   ->  local_pg.main.<table>  e.g.  FROM local_pg.main.baby_names

                ┌───────────────────────────┐
                │          User             │
                │ (asks NL query in chat)   │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Conversation Manager      │
                │ - Maintains chat history    │
                │ - Caches prev queries/res   │
                │ - Builds prompt w/ context  │
                └─────────────┬───────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Pydantic AI Agent         │
                │ - Prompt = Schema           │
                │   + NL Query + History      │
                │ - result_type=str (SQL)     │
                │ - model: gpt-4o             │
                └─────────────┬───────────────┘
                              │
                           SQL query
                              │
                              ▼
                ┌─────────────────────────────┐
                │   DuckDB Engine             │
                │ - Lazy views on GCS Parquet │
                │   (httpfs + HMAC secret)    │
                │ - ATTACH rag_db  (port 5434)│
                │ - ATTACH local_pg(port 5432)│
                │ - In-memory JOIN across all │
                └─────────────┬───────────────┘
                              │
                         Query results
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Response Formatter        │
                │ - Pretty-prints results     │
                │ - Feeds back into history   │
                └─────────────┬───────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │          User             │
                │ (sees answer, asks follow │
                │  up: "only US customers?" │
                └───────────────────────────┘
"""

import hashlib
import logging
import os
import pprint
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb
from dotenv import load_dotenv
from google.cloud import storage
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load credentials from both .env files:
#   pydantic-ai-experiments/.env  -> OPENAI_API_KEY
#   deltalake-projects/.env       -> GCS_HMAC_ID, GCS_HMAC_SECRET
# ---------------------------------------------------------------------------
_DELTALAKE_ENV = Path("C:/Users/rohan/Documents/deltalake-projects/.env")


def _load_env() -> None:
    for env_path in [Path(__file__).parent.parent / ".env", _DELTALAKE_ENV]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.info("Loaded .env from %s", env_path)
        else:
            logger.warning(".env not found at %s", env_path)


# ---------------------------------------------------------------------------
# Strip markdown fences the LLM sometimes adds
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
# ConversationManager -- history + two-level cache (NL match + SQL hash)
# ---------------------------------------------------------------------------
class ConversationManager:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        agent: Agent,
        schema_text: str,
        cache_size: int = 20,
    ):
        self.conn = conn
        self.agent = agent
        self.schema_text = schema_text
        self.history: list[tuple[str, str, Any]] = []
        self.query_cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_size = cache_size

    def _hash(self, sql: str) -> str:
        return hashlib.md5(sql.encode()).hexdigest()

    def _history_context(self, n: int = 3) -> str:
        parts = [
            f"Q: {q}\nSQL: {sql}\nResult: {res}"
            for q, sql, res in self.history[-n:]
        ]
        return "\n\n".join(parts)

    def _build_prompt(self, nl_query: str) -> str:
        history = self._history_context()
        history_block = f"Conversation so far:\n{history}\n\n" if history else ""
        return (
            f"Schema:\n{self.schema_text}\n\n"
            f"{history_block}"
            f"Latest Question: {nl_query}"
        )

    def run_query(self, nl_query: str) -> Any:
        # NL-level cache hit
        for prev_nl, prev_sql, prev_res in self.history:
            if nl_query.strip().lower() == prev_nl.strip().lower():
                logger.info("NL cache hit -> %s", prev_sql)
                return prev_res

        # Generate SQL via Pydantic AI agent
        prompt = self._build_prompt(nl_query)
        result = self.agent.run_sync(prompt)
        sql = strip_sql_fences(result.output)

        logger.info("NL  : %s", nl_query)
        logger.info("SQL : %s", pprint.pformat(sql))

        # SQL hash cache
        h = self._hash(sql)
        if h in self.query_cache:
            logger.info("SQL cache hit")
            result_data = self.query_cache[h]
        else:
            try:
                result_data = self.conn.execute(sql).fetchall()
                self.query_cache[h] = result_data
                if len(self.query_cache) > self.cache_size:
                    self.query_cache.popitem(last=False)
            except Exception as exc:
                logger.error("SQL error: %s", exc)
                result_data = None

        self.history.append((nl_query, sql, result_data))
        logger.info("Result: %s\n", result_data)
        return result_data

    def show_history(self) -> None:
        for i, (nl, sql, res) in enumerate(self.history, 1):
            logger.info("%d. Q: %s\n   SQL: %s\n   Result: %s", i, nl, sql, res)


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

    # ------------------------------------------------------------------
    # Step 1: register GCS Parquet files as lazy DuckDB views
    # ------------------------------------------------------------------
    def load_gcs_tables(self) -> None:
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        self.conn.execute(f"""
            CREATE OR REPLACE SECRET gcs_secret (
                TYPE gcs,
                KEY_ID  '{os.environ["GCS_HMAC_ID"]}',
                SECRET  '{os.environ["GCS_HMAC_SECRET"]}'
            )
        """)

        client = storage.Client(project=self.gcs_user_project)
        bucket = client.bucket(self.gcs_bucket, user_project=self.gcs_user_project)
        blobs = bucket.list_blobs(prefix=self.gcs_prefix, delimiter="/")

        table_names: set[str] = set()
        for page in blobs.pages:
            for prefix in page.prefixes:
                table_names.add(prefix.strip("/").split("/")[-1])

        for name in table_names:
            prefix = f"{self.gcs_prefix}{name}/"
            path = f"gs://{self.gcs_bucket}/{prefix}*/*.parquet"
            view = name.lower()
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM parquet_scan('{path}')"
            )
            self._gcs_views[name] = view
            logger.info("GCS view: %-25s <- %s", view, path)

        logger.info("GCS tables registered: %s", sorted(self._gcs_views))

    # ------------------------------------------------------------------
    # Step 2: attach PostgreSQL databases
    # ------------------------------------------------------------------
    def attach_postgres_dbs(self) -> None:
        self.conn.execute("INSTALL postgres; LOAD postgres;")
        for db in self.postgres_dbs:
            self.conn.execute(
                f"ATTACH '{db.connection_string}' AS {db.alias} (TYPE postgres, READ_ONLY)"
            )
            logger.info("Attached PostgreSQL: %s", db.alias)

    # ------------------------------------------------------------------
    # Step 3: generate unified schema string for the LLM prompt
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 4: initialise Pydantic AI agent
    # ------------------------------------------------------------------
    def init_agent(self, model: str = "gpt-4o") -> Agent:
        openai_model = OpenAIModel(model, api_key=os.environ["OPENAI_API_KEY"])
        self.agent = Agent(
            model=openai_model,
            result_type=str,
            system_prompt=_SYSTEM_PROMPT,
        )
        logger.info("Pydantic AI agent ready: %s", model)
        return self.agent

    # ------------------------------------------------------------------
    # Convenience: build a ConversationManager
    # ------------------------------------------------------------------
    def conversation_manager(self, cache_size: int = 20) -> ConversationManager:
        if self.agent is None or self.schema_text is None:
            raise ValueError("Call generate_schema() and init_agent() first.")
        return ConversationManager(
            self.conn, self.agent, self.schema_text, cache_size=cache_size
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _load_env()

    conn = duckdb.connect(database=":memory:")

    source = UnifiedDataSource(
        conn=conn,
        gcs_bucket="eagerbeaver-1",
        gcs_prefix="partitioned_data/",
        gcs_user_project="heroic-rain-447418-j7",
        postgres_dbs=[
            PostgresDB(
                alias="rag",
                connection_string="postgresql://rag_user:rag_pass@localhost:5434/rag_db",
            ),
            PostgresDB(
                alias="local_pg",
                connection_string="postgresql://postgres:postgres@localhost:5432/postgres",
            ),
        ],
    )

    source.load_gcs_tables()
    source.attach_postgres_dbs()

    schema = source.generate_schema()
    print("\n--- UNIFIED SCHEMA ---")
    print(schema)
    print("--- END SCHEMA ---\n")

    source.init_agent(model="gpt-4o")
    chat = source.conversation_manager()

    queries = [
        "What was the total items sold by user?",             # GCS parquet
        "Which products have sale numbers above 200?",        # GCS parquet
        "How many baby names were registered in 1990?",       # local_pg
        "Which countries had the highest GDP in 2020?",       # local_pg
        "How many documents are in the RAG knowledge base?",  # rag_db
    ]

    for q in queries:
        print(f"\n>> {q}")
        result = chat.run_query(q)
        pprint.pprint(result)

    print("\n--- Conversation history ---")
    chat.show_history()
