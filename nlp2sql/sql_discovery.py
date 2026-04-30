from dataclasses import dataclass
from typing import Union
import asyncio
import duckdb
import asyncpg
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


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


# 3. Agent Setup
sql_agent = Agent(
    "openai:gpt-4o",
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


# 5. Execution Example
async def run_query(user_prompt: str):
    # Initialize connections
    pg_pool = await asyncpg.create_pool("postgresql://user:pass@localhost/db")
    duck_conn = duckdb.connect(":memory:")

    deps = MultiDBDeps(pg_pool=pg_pool, duck_conn=duck_conn)
    result = await sql_agent.run(user_prompt, deps=deps)
    return result.data
