"""
NL-to-SQL — FastAPI server.

Endpoints
---------
GET  /health      — DB and LLM connectivity check
POST /v1/query    — Translate natural language to SQL and execute
GET  /v1/history  — Recent query history for the session

Usage:
    uvicorn apps.nl2sql.api:app --host 0.0.0.0 --port 8001 --reload

Example:
    curl -X POST http://localhost:8001/v1/query \\
      -H "Content-Type: application/json" \\
      -d '{"question": "How many documents are stored?"}'
"""

import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from nlp2sql.nlp_sql_postgres_v2 import ConversationManager, QueryResult
from rag.config.settings import load_settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NL-to-SQL API",
    description="Natural-language to SQL over PostgreSQL via DuckDB + Pydantic AI",
    version="1.0.0",
)

_SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural-language question, "
    "return ONLY a valid SQL SELECT statement with no explanation, no markdown fences. "
    "Use table prefixes shown in the schema (e.g. rag_db.main.chunks). "
    "Always include a LIMIT clause (max 500 rows)."
)

# Module-level singletons initialized on first request
_manager: ConversationManager | None = None
_schema_text: str = ""


async def _get_manager() -> tuple[ConversationManager, str]:
    global _manager, _schema_text
    if _manager is not None:
        return _manager, _schema_text

    settings = load_settings()

    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL postgres; LOAD postgres;")
    conn.execute(
        f"ATTACH '{settings.database_url}' AS rag_db (TYPE postgres, READ_ONLY)"
    )

    lines: list[str] = ["=== rag_db tables (prefix: rag_db.main.<table>) ==="]
    tables = conn.execute(
        "SELECT table_name FROM rag_db.information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
        "ORDER BY table_name"
    ).fetchall()
    for (tbl,) in tables:
        cols = conn.execute(
            f"SELECT column_name, data_type "
            f"FROM rag_db.information_schema.columns "
            f"WHERE table_schema = 'public' AND table_name = '{tbl}' "
            f"ORDER BY ordinal_position"
        ).fetchall()
        lines.append(f"Table: rag_db.main.{tbl}")
        for col_name, col_type in cols:
            lines.append(f"  - {col_name} ({col_type})")
        lines.append("")
    _schema_text = "\n".join(lines).strip()

    llm = OpenAIModel(
        settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    pydantic_agent = Agent(model=llm, result_type=str, system_prompt=_SYSTEM_PROMPT)

    _manager = ConversationManager(
        conn=conn,
        agent=pydantic_agent,
        schema_text=_schema_text,
        cache_size=50,
        max_retries=3,
        max_result_rows=500,
        query_timeout=30.0,
    )
    return _manager, _schema_text


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the database")


class QueryResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    cached: bool
    attempts: int
    error: str | None = None
    success: bool


class HistoryEntry(BaseModel):
    question: str
    sql: str
    success: bool
    row_count: int


class HealthResponse(BaseModel):
    status: str
    db: bool
    llm: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    checks = {"db": False, "llm": False}

    try:
        settings = load_settings()
        conn = duckdb.connect(database=":memory:")
        conn.execute("INSTALL postgres; LOAD postgres;")
        conn.execute(
            f"ATTACH '{settings.database_url}' AS rag_db (TYPE postgres, READ_ONLY)"
        )
        conn.execute("SELECT 1 FROM rag_db.information_schema.tables LIMIT 1").fetchone()
        conn.close()
        checks["db"] = True
    except Exception as exc:
        logger.warning("Health: DB check failed: %s", exc)

    try:
        import httpx
        settings = load_settings()
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = settings.llm_base_url.rstrip("/")
            resp = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            checks["llm"] = resp.status_code < 500
    except Exception as exc:
        logger.warning("Health: LLM check failed: %s", exc)

    all_ok = all(checks.values())
    any_ok = any(checks.values())
    status = "ok" if all_ok else ("degraded" if any_ok else "unhealthy")
    response = HealthResponse(status=status, **checks)
    if not all_ok:
        raise HTTPException(status_code=503, detail=response.model_dump())
    return response


@app.post("/v1/query", response_model=QueryResponse, tags=["query"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Translate a natural-language question to SQL, execute it, and return results.

    The self-correcting retry loop re-prompts the LLM with the error message
    on failure (up to 3 attempts). Read-only guardrails block any write/DDL.
    """
    try:
        manager, _ = await _get_manager()
        qr: QueryResult = await manager.run_query(request.question)
        return QueryResponse(
            question=qr.nl_query,
            sql=qr.sql,
            columns=qr.columns,
            rows=[list(row) for row in qr.rows],
            row_count=len(qr.rows),
            cached=qr.cached,
            attempts=qr.attempts,
            error=qr.error,
            success=qr.success,
        )
    except Exception as exc:
        logger.exception("Query endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/history", response_model=list[HistoryEntry], tags=["query"])
async def history() -> list[HistoryEntry]:
    """Return recent query history from the in-memory conversation manager."""
    try:
        manager, _ = await _get_manager()
        return [
            HistoryEntry(
                question=nl,
                sql=sql,
                success=qr.success,
                row_count=len(qr.rows),
            )
            for nl, sql, qr in manager.history
        ]
    except Exception as exc:
        logger.exception("History endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/schema", tags=["query"])
async def schema() -> dict[str, str]:
    """Return the database schema text used for SQL generation."""
    try:
        _, schema_text = await _get_manager()
        return {"schema": schema_text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.nl2sql.api:app", host="0.0.0.0", port=8001, reload=True)
