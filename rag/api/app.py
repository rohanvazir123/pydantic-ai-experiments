"""
FastAPI HTTP layer for the RAG agent.

Module: rag.api.app
===================

Exposes the RAG system over HTTP with versioned endpoints.

Endpoints
---------
GET  /health          — DB, embedding API, and LLM API connectivity checks
POST /v1/chat         — Non-streaming query via traced_agent_run
POST /v1/chat/stream  — Streaming query via agent.run_stream() → SSE
POST /v1/ingest       — Trigger document ingestion pipeline

Usage
-----
    # Run with uvicorn
    uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload

    # Or via Python
    python -m rag.api.app
"""

import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.agent.rag_agent import RAGState, agent, traced_agent_run
from rag.config.settings import load_settings
from rag.ingestion.pipeline import create_pipeline
from rag.storage.vector_store.postgres import PostgresHybridStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent API",
    description="Agentic RAG system — PostgreSQL/pgvector + Pydantic AI",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    user_id: str | None = Field(
        default=None, description="User identifier for Mem0 personalisation"
    )
    session_id: str | None = Field(
        default=None, description="Session identifier for Langfuse tracing"
    )
    message_history: list[dict[str, Any]] | None = Field(
        default=None, description="Previous messages for multi-turn context"
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str | None = None


class IngestRequest(BaseModel):
    documents_folder: str = Field(
        default="rag/documents", description="Path to documents folder"
    )
    clean: bool = Field(
        default=True, description="Clear existing data before ingesting"
    )
    chunk_size: int = Field(
        default=1000, ge=100, description="Target chunk size in characters"
    )
    max_tokens: int = Field(default=512, ge=64, description="Maximum tokens per chunk")


class IngestResult(BaseModel):
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: list[str]


class IngestResponse(BaseModel):
    results: list[IngestResult]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "unhealthy"
    db: bool
    embedding_api: bool
    llm_api: bool


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Check connectivity to database, embedding API, and LLM API.
    Returns HTTP 200 when all healthy, 503 when any component is down.
    """
    settings = load_settings()
    checks: dict[str, bool] = {"db": False, "embedding_api": False, "llm_api": False}

    # --- DB check ---
    try:
        store = PostgresHybridStore()
        await asyncio.wait_for(store.initialize(), timeout=5.0)
        await store.close()
        checks["db"] = True
    except Exception as exc:
        logger.warning("Health: DB check failed: %s", exc)

    # --- Embedding API check ---
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = settings.embedding_base_url.rstrip("/")
            resp = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            )
            checks["embedding_api"] = resp.status_code < 500
    except Exception as exc:
        logger.warning("Health: Embedding API check failed: %s", exc)

    # --- LLM API check ---
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base = settings.llm_base_url.rstrip("/")
            resp = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            checks["llm_api"] = resp.status_code < 500
    except Exception as exc:
        logger.warning("Health: LLM API check failed: %s", exc)

    all_ok = all(checks.values())
    any_ok = any(checks.values())
    status = "ok" if all_ok else ("degraded" if any_ok else "unhealthy")

    response = HealthResponse(status=status, **checks)
    if not all_ok:
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


# ---------------------------------------------------------------------------
# Chat endpoints (v1)
# ---------------------------------------------------------------------------


@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the RAG agent and return the full answer.

    Wraps ``traced_agent_run`` which attaches Langfuse tracing automatically.
    """
    try:
        result = await traced_agent_run(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            message_history=request.message_history,
        )
        return ChatResponse(answer=str(result.output), session_id=request.session_id)
    except Exception as exc:
        logger.exception("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/chat/stream", tags=["chat"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Stream the RAG agent response as Server-Sent Events (SSE).

    Each event is a JSON object: ``{"delta": "<text>"}``
    A final ``{"done": true}`` event signals end of stream.
    """

    async def _generate():
        state = RAGState(user_id=request.user_id)
        try:
            async with agent.run_stream(
                request.query,
                deps=state,
                message_history=request.message_history or [],
            ) as streamed:
                async for delta in streamed.stream_text(delta=True):
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            await state.close()

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Ingest endpoint (v1)
# ---------------------------------------------------------------------------


@app.post("/v1/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Trigger document ingestion from a folder on the server.

    This runs synchronously within the request. For large corpora consider
    moving this to a background job queue (see FAQ §4).
    """
    pipeline = create_pipeline(
        documents_folder=request.documents_folder,
        clean=request.clean,
        chunk_size=request.chunk_size,
        max_tokens=request.max_tokens,
    )
    try:
        await pipeline.initialize()
        raw_results = await pipeline.ingest_documents()
    except Exception as exc:
        logger.exception("Ingest endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await pipeline.close()

    results = [
        IngestResult(
            document_id=r.document_id,
            title=r.title,
            chunks_created=r.chunks_created,
            processing_time_ms=r.processing_time_ms,
            errors=r.errors,
        )
        for r in raw_results
    ]

    return IngestResponse(
        results=results,
        total_documents=len(results),
        total_chunks=sum(r.chunks_created for r in results),
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.api.app:app", host="0.0.0.0", port=8000, reload=True)
