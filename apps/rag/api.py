"""
RAG Agent — FastAPI server.

Endpoints
---------
GET  /health           — DB, embedding API, and LLM connectivity
POST /v1/chat          — Full agent run (with tool calls + synthesis)
POST /v1/chat/stream   — SSE-streamed agent response
POST /v1/retrieve      — Raw retrieval (no LLM synthesis)
POST /v1/ingest        — Trigger document ingestion pipeline

Usage:
    uvicorn apps.rag.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    description="Agentic RAG — PostgreSQL/pgvector + Apache AGE + Pydantic AI",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    user_id: str | None = None
    session_id: str | None = None
    message_history: list[dict[str, Any]] | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: str = Field(default="hybrid", description="hybrid | semantic | text")
    match_count: int = Field(default=5, ge=1, le=50)


class RetrieveResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    search_type: str


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
    total: int


class IngestRequest(BaseModel):
    documents_folder: str = Field(default="rag/documents")
    clean: bool = Field(default=True)
    chunk_size: int = Field(default=1000, ge=100)
    max_tokens: int = Field(default=512, ge=64)


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
    status: str
    db: bool
    embedding_api: bool
    llm_api: bool


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    settings = load_settings()
    checks: dict[str, bool] = {"db": False, "embedding_api": False, "llm_api": False}

    try:
        store = PostgresHybridStore()
        await asyncio.wait_for(store.initialize(), timeout=5.0)
        await store.close()
        checks["db"] = True
    except Exception as exc:
        logger.warning("Health: DB check failed: %s", exc)

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
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Full agent run with tool calls and LLM synthesis."""
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
    """Stream the agent response as Server-Sent Events. Each event: ``{"delta": "<text>"}``"""

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
# Retrieve endpoint (no LLM synthesis)
# ---------------------------------------------------------------------------

@app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["retrieve"])
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Run retrieval directly — returns ranked chunks without LLM synthesis.

    Use this when you want to embed retrieval into your own pipeline or
    display raw source passages to users.
    """
    state = RAGState()
    try:
        retriever = await state.get_retriever()
        chunks = await retriever.retrieve(
            query=request.query,
            search_type=request.search_type,
            match_count=request.match_count,
        )
        results = [
            RetrieveResult(
                id=c.chunk_id,
                title=c.title,
                content=c.content,
                score=float(c.similarity),
                search_type=request.search_type,
            )
            for c in chunks
        ]
        return RetrieveResponse(results=results, total=len(results))
    except Exception as exc:
        logger.exception("Retrieve endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await state.close()


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """Trigger document ingestion from a server-side folder."""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.rag.api:app", host="0.0.0.0", port=8000, reload=True)
