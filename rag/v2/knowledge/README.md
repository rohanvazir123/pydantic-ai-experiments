# knowledge/

## Table of Contents

- [What This Is](#what-this-is)
- [Sub-packages](#sub-packages)
- [Conventions](#conventions)

---

## What This Is

The `knowledge` Python package is the entire backend service. It is imported as `knowledge.*` throughout the codebase. The FastAPI application (`knowledge.api.app`) is the entry point; workers (`knowledge.ingestion.worker`, `knowledge.retrieval.worker`) are separate processes.

---

## Sub-packages

| Package | Responsibility |
|---------|---------------|
| `config/` | Pydantic-settings: all runtime configuration in one place |
| `api/` | FastAPI app factory, middleware, auth, schemas, route handlers |
| `bus/` | Redis Streams message bus: publisher, consumer, circuit breaker, backoff |
| `ingestion/` | Ingestion pipeline: Docling → chunks → embeddings → vector store + graph |
| `store/` | Storage layer: PostgreSQL (vector), Apache AGE (graph), Redis (cache) |
| `retrieval/` | Retrieval pipeline: hybrid search, reranking, semantic cache |
| `agent/` | Pydantic AI agent, confidence-aware pipeline, model router, judge |
| `corpus/` | Corpus registry, RBAC, ontology loader |
| `scheduler/` | APScheduler-based periodic ingestion jobs |
| `hooks/` | Hook system: lifecycle points, registry, built-in hooks |
| `validation/` | Request validation pipeline (V1–V6) |
| `memory/` | All five memory tiers: conversation store, Mem0, working memory, pruning |
| `billing/` | Tenant provisioning, quota enforcement, Stripe metering |
| `evaluation/` | Offline eval harness, metrics, gold datasets, regression reporter |
| `observability/` | Prometheus metrics, Langfuse tracing, SMTP alert sender |

---

## Conventions

- **All I/O is async** — `asyncpg`, `redis.asyncio`, `AsyncOpenAI`, Pydantic AI
- **CPU-bound sync work** (Docling, tokenizers) uses `asyncio.to_thread()`
- **Python 3.13** — `X | None` not `Optional[X]`, `list[str]` not `List[str]`, no `from __future__ import annotations`
- **Absolute imports only** — `from knowledge.config.settings import load_settings`
- **No re-exports** — import from the canonical module, not from `__init__.py`
- **Type annotations required** on all public functions and class attributes
