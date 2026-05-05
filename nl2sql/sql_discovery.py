import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import asyncio
import duckdb
import asyncpg
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config.settings import load_settings


def _make_model() -> OpenAIModel:
    settings = load_settings()
    return OpenAIModel(
        settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )


# 1. Structured Response Model
class SQLResponse(BaseModel):
    database_type: str = Field(description="'postgres' or 'duckdb'")
    sql: str = Field(description="The executable SQL query")
    explanation: str = Field(description="Reasoning for the table/column choice")


# 2. Multi-DB Dependencies
@dataclass
class MultiDBDeps:
    pg_pool: asyncpg.Pool
    duck_conn: duckdb.DuckDBPyConnection


# 3. Agent Setup — uses Ollama (or whatever LLM_PROVIDER is set to in .env)
sql_agent = Agent(
    _make_model(),
    deps_type=MultiDBDeps,
    result_type=SQLResponse,
    system_prompt="You are a data expert with access to Postgres and DuckDB. Discover the schema first.",
)


# 4. Schema Discovery Tools
@sql_agent.tool
async def list_tables(ctx: RunContext[MultiDBDeps], db_type: str) -> list[str]:
    """List tables for 'postgres' or 'duckdb'."""
    if db_type == "postgres":
        async with ctx.deps.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            return [r["table_name"] for r in rows]
    else:
        # DuckDB metadata query
        return [row[0] for row in ctx.deps.duck_conn.execute("SHOW TABLES").fetchall()]


@sql_agent.tool
async def describe_table(
    ctx: RunContext[MultiDBDeps], db_type: str, table_name: str
) -> str:
    """Get columns and types for a specific table in either database."""
    if db_type == "postgres":
        async with ctx.deps.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1",
                table_name,
            )
            cols = [f"{r['column_name']} ({r['data_type']})" for r in rows]
            return f"Postgres table {table_name}: {', '.join(cols)}"
    else:
        # DuckDB DESCRIBE statement
        res = ctx.deps.duck_conn.execute(f"DESCRIBE {table_name}").fetchall()
        cols = [f"{r[0]} ({r[1]})" for r in res]
        return f"DuckDB table {table_name}: {', '.join(cols)}"


# 5. Execute the SQL returned by the agent on the appropriate database.
#    Returns (SQLResponse, columns, rows).
async def run_query(
    user_prompt: str,
    duck_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[SQLResponse, list[str], list]:
    settings = load_settings()
    pg_pool = await asyncpg.create_pool(settings.database_url)
    conn = duck_conn or duckdb.connect(":memory:")
    try:
        deps = MultiDBDeps(pg_pool=pg_pool, duck_conn=conn)
        result = await sql_agent.run(user_prompt, deps=deps)
        response: SQLResponse = result.output

        if response.database_type == "postgres":
            async with pg_pool.acquire() as pg_conn:
                rows = await pg_conn.fetch(response.sql)
                columns = list(rows[0].keys()) if rows else []
                data: list = [dict(r) for r in rows]
        else:
            cursor = conn.execute(response.sql)
            columns = [d[0] for d in (cursor.description or [])]
            data = cursor.fetchall()

        return response, columns, data
    finally:
        await pg_pool.close()
