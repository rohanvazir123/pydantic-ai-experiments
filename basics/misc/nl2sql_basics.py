import asyncio
import duckdb
import pyarrow.dataset as ds
import sqlglot
from sqlglot import exp, parse_one
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel

# --- 1. STRUCTURED MODELS ---


class SQLResponse(BaseModel):
    sql: str = Field(
        description="SQL query for DuckDB. Use 'pg.' for Postgres and 'gcs.' for Parquet."
    )
    explanation: str
    is_cross_source: bool = Field(description="True if joining Postgres and GCS")


class DBContext:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        # List of columns that must never be selected raw
        self.pii_columns = ["email", "phone", "ssn", "credit_card", "address"]


# --- 2. ASYNC DATABASE & SAFETY UTILS ---


async def validate_sql_safety(sql: str, pii_columns: List[str]) -> bool:
    """AST validation to block raw PII selection."""
    try:
        parsed = parse_one(sql, read="duckdb")
        for expression in parsed.find_all(exp.Column):
            if expression.name.lower() in pii_columns:
                # Allow PII only if wrapped in an aggregate (COUNT, etc)
                if not isinstance(
                    expression.parent,
                    (exp.Count, exp.Max, exp.Min, exp.Sum, exp.ApproxDistinct),
                ):
                    return False
        return True
    except Exception:
        return False


async def setup_federated_conn():
    """Initializes the federated engine with Postgres and GCS links."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL postgres; LOAD postgres; INSTALL httpfs; LOAD httpfs;")

    # Mocking GCS Lazy Load (Millions of rows, zero-copy)
    # gcs_ds = ds.dataset("gs://your-bucket/logs.parquet", format="parquet")
    # conn.register("gcs_logs", gcs_ds)

    # Mocking Postgres Attachment
    # conn.execute("ATTACH 'dbname=pii_prod user=admin' AS pg (TYPE POSTGRES, READ_ONLY);")

    return conn


# --- 3. THE ASYNC AGENT ---

model = OllamaModel(model_name="llama3.2")

agent = Agent(
    model,
    deps_type=DBContext,
    result_type=SQLResponse,
    system_prompt=(
        "You are a secure data assistant. You have two sources: Postgres (pg) and GCS (gcs). "
        "Rule 1: Use the 'get_schema' tool to identify tables. "
        "Rule 2: Never select PII columns directly. You may only COUNT them. "
        "Rule 3: Use DuckDB SQL dialect."
    ),
)


@agent.tool
async def get_schema(ctx: RunContext[DBContext]) -> str:
    """Async tool to fetch metadata for the 50+ tables."""
    # In practice, query your DuckDB metadata table here
    return """
    - pg.customers: [id, email (PII), phone (PII), country]
    - gcs.web_events: [user_id, event_name, event_time]
    """


# --- 4. THE MAIN ASYNC PIPELINE ---


async def run_analytics_task(user_query: str):
    # 1. Initialize Engine
    conn = await setup_federated_conn()
    deps = DBContext(conn)

    print(f"🌀 Processing: {user_query}")

    # 2. Run Agent (Non-blocking LLM call)
    try:
        result = await agent.run(user_query, deps=deps)
        sql = result.data.sql
        print(f"🤖 AI suggested SQL: {sql}")
    except Exception as e:
        print(f"LLM Error: {e}")
        return

    # 3. Apply PII Guardrails
    if await validate_sql_safety(sql, deps.pii_columns):
        print("✅ PII Safety Check: PASSED")

        # 4. Execute Query (Using thread pool to keep loop free)
        loop = asyncio.get_running_loop()
        try:
            # We wrap the blocking DuckDB call in an executor
            df = await loop.run_in_executor(None, lambda: conn.execute(sql).df())
            print("\n📊 Results:\n", df)
        except Exception as e:
            print(f"❌ Execution Error: {e}")
    else:
        print("🚨 SECURITY ALERT: LLM attempted to select raw PII. Query blocked.")


# --- 5. ENTRY POINT ---

if __name__ == "__main__":
    # Run the top-level async loop
    asyncio.run(
        run_analytics_task("How many customers from Postgres have events in GCS?")
    )
