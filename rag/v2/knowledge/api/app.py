"""FastAPI application factory.

Lifespan (startup):
  1. Connect PostgreSQL pool (vector store)
  2. Connect Apache AGE pool (graph store)
  3. Connect Redis client
  4. Connect embedding + retrieval pipeline
  5. Build ConfidenceAwarePipeline
  6. Register built-in placeholder hooks
  7. Start APScheduler (periodic ingest)

Lifespan (shutdown):
  8. Stop APScheduler
  9. Close all DB pools and Redis

Middleware stack (outermost first):
  CorrelationID → StructuredLog → CORS → RateLimiter (slowapi)
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge.api.middleware import CorrelationIDMiddleware, StructuredLogMiddleware
from knowledge.api.routes import (
    auth,
    chat,
    corpus,
    evaluate,
    feedback,
    health,
    ingest,
    logs,
    memory,
    scheduler,
    search,
)
from knowledge.bus.publisher import Publisher
from knowledge.config.settings import Settings, load_settings
from knowledge.hooks.builtins import register_builtin_hooks
from knowledge.ingestion.embedder import Embedder
from knowledge.retrieval.retriever import Retriever
from knowledge.store.cache import RedisCache
from knowledge.store.vector import PostgresHybridStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Connect all stores on startup; close them on shutdown."""
    settings: Settings = app.state.settings

    # Redis
    redis_client = aioredis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=False,
    )
    app.state.redis = redis_client

    # L2 cache (wraps Redis)
    cache = RedisCache(settings=settings)
    await cache.connect()
    app.state.cache = cache

    # Publisher
    app.state.publisher = Publisher(redis_client)

    # Vector store (PostgreSQL + pgvector)
    vector_store = PostgresHybridStore(settings=settings, cache=cache)
    await vector_store.initialize()
    app.state.vector_store = vector_store

    # Embedder
    embedder = Embedder(settings=settings)
    app.state.embedder = embedder

    # Retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        cache=cache,
        settings=settings,
    )
    app.state.retriever = retriever

    # Confidence-Aware Pipeline
    from knowledge.agent.pipeline import ConfidenceAwarePipeline
    app.state.pipeline = ConfidenceAwarePipeline(
        retriever=retriever,
        settings=settings,
    )

    # Register built-in placeholder hooks
    register_builtin_hooks()

    logger.info("knowledge API started")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await vector_store.close()
    await cache.close()
    await redis_client.aclose()  # type: ignore[attr-defined]
    logger.info("knowledge API shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI factory. Call once per process."""
    _settings = settings or load_settings()

    app = FastAPI(
        title="knowledge API",
        description="RAG v2 — multi-corpus knowledge system",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = _settings

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(StructuredLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # tighten in Phase 9 (Security)
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    prefix = "/api/v2"
    app.include_router(auth.router,      prefix=prefix)
    app.include_router(chat.router,      prefix=prefix)
    app.include_router(search.router,    prefix=prefix)
    app.include_router(ingest.router,    prefix=prefix)
    app.include_router(corpus.router,    prefix=prefix)
    app.include_router(memory.router,    prefix=prefix)
    app.include_router(evaluate.router,  prefix=prefix)
    app.include_router(feedback.router,  prefix=prefix)
    app.include_router(scheduler.router, prefix=prefix)
    app.include_router(logs.router,      prefix=prefix)
    app.include_router(health.router)    # /health + /metrics at root

    return app


# Module-level app instance (used by Gunicorn CMD)
app = create_app()
