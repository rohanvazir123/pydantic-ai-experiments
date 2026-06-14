---
name: rag-v2
description: RAG v2 development guide for the knowledge/ package in rag/v2/. Use when working on the multi-corpus RAG backend — adding routes, ingestion pipeline changes, retrieval tuning, store layer, Redis bus, scheduler, auth, or running/debugging the API and tests.
metadata:
  version: "1.0.0"
---

# RAG v2 — Development Guide

## When to Use This Skill

- Working in `rag/v2/` on the `knowledge/` Python package
- Adding or modifying API routes, ingestion pipeline, retrieval logic, or store layer
- Debugging the API server, Redis workers, or scheduler
- Running seeds, migrations, or the test suite
- Understanding the multi-corpus/multi-tenant model or JWT auth flow

## Key Commands

```bash
cd rag/v2

# Start API (dev, auto-reload)
uv run uvicorn knowledge.api.app:app --reload --port 8001

# Apply DB schemas (idempotent)
make databaseschemas          # requires DATABASE_URL env var

# Seed default tenant + corpus + sample docs
make seed

# Unit tests (no services needed)
make test-unit

# All tests
make test

# Lint + format
make ruff

# Type-check
make mypy
```

## File Map

| Path | What it is |
|------|-----------|
| `knowledge/config/settings.py` | All settings (pydantic-settings, reads `.env`) |
| `knowledge/api/app.py` | FastAPI app — middleware, route registration |
| `knowledge/api/routes/` | One file per route group: `chat`, `ingest`, `search`, `corpus`, `auth`, `health`, `evaluate`, `feedback`, `memory`, `scheduler`, `logs` |
| `knowledge/api/schemas.py` | Request/response Pydantic models |
| `knowledge/api/auth.py` | JWT RS256 verification + RBAC |
| `knowledge/ingestion/pipeline.py` | `IngestionPipeline` — orchestrates chunker → embedder → graph extractor → store |
| `knowledge/ingestion/worker.py` | Redis Stream consumer for async ingestion jobs |
| `knowledge/retrieval/retriever.py` | `Retriever` — hybrid search (vector + text + RRF) with semantic cache check |
| `knowledge/retrieval/fusion.py` | RRF implementation |
| `knowledge/retrieval/semantic_cache.py` | L3 pgvector semantic cache |
| `knowledge/retrieval/worker.py` | Redis Stream consumer for async retrieval jobs |
| `knowledge/store/vector.py` | `VectorStore` — asyncpg pool, pgvector HNSW search |
| `knowledge/store/cache.py` | L1 (in-process LRU) + L2 (Redis msgpack) cache |
| `knowledge/store/graph.py` | Apache AGE graph store wrapper |
| `knowledge/store/entity_index.py` | Entity shadow table (tsvector GIN + IVFFlat) |
| `knowledge/bus/` | Redis Streams publish/consume helpers |
| `knowledge/hooks/` | `HookPoint` enum + async `HookRegistry` |
| `knowledge/validation/` | 6-stage query validation pipeline |
| `knowledge/scheduler/` | APScheduler-backed job scheduler |
| `knowledge/billing/` | Cost tracking + circuit breaker |
| `knowledge/memory/` | Mem0-backed user memory (optional) |
| `knowledge/evaluation/` | Retrieval metrics (Hit Rate, MRR, NDCG) |
| `schema/*.sql` | DB migrations — run in filename order |
| `scripts/seed.py` | Seeds default tenant, corpus, sample docs |
| `scripts/purge.py` | Drops corpus content for re-ingestion |
| `tests/unit/` | No-service tests (fakeredis, mocked asyncpg) |
| `tests/integration/` | Needs PostgreSQL + Redis |
| `tests/retrieval/` | Needs PostgreSQL + Redis + Ollama + ingested data |
| `infra/keys/` | RSA key pair for JWT (git-ignored) |
| `.env` | Local env — copy from `.env.example` |

## Architecture at a Glance

```
API (FastAPI)
  └─ auth middleware (JWT RS256)
  └─ rate limiting (slowapi)
  └─ routes → publish job to Redis Stream
        │
        ├─ ingest-worker (consumes stream)
        │     Docling → chunk → embed → graph extract → VectorStore + AGE
        │
        └─ retrieval-worker (consumes stream)
              L1 LRU → L2 Redis → L3 semantic cache → hybrid search (RRF)
```

**Three-layer cache:** L1 in-process LRU → L2 Redis msgpack → L3 pgvector cosine-sim (stored as JWE).

**Model tiers:** nano (`qwen2.5:0.5b`) for routing/classification, small (`llama3.2:3b`) for standard RAG, large (`llama3.1:70b`) for complex reasoning. Controlled by `MODEL_ROUTING_ENABLED`.

**Multi-tenant:** every request carries a `tenant_id` extracted from the JWT. Each tenant has one or more named corpora; retrieval is scoped to the corpus specified in the request.

## Common Patterns

### Add a new route

1. Create `knowledge/api/routes/my_route.py` with an `APIRouter`
2. Add request/response models to `knowledge/api/schemas.py`
3. Register the router in `knowledge/api/app.py`

### Add a new setting

Add to `knowledge/config/settings.py` (pydantic-settings reads from `.env` automatically).

### Add a DB migration

Create `schema/NNN_description.sql` (next number in sequence). Run `make databaseschemas`.

### Run the full stack locally

```bash
# Start services
docker compose up -d postgres age redis

# Apply schema + seed
make seed

# Pull models (one-time, slow for 70b)
make pull-models

# Start API
uv run uvicorn knowledge.api.app:app --reload --port 8001

# Health check
curl http://localhost:8001/health
```

### Auth for manual testing

JWT RS256 keys are generated by `INSTALL.ps1` into `infra/keys/`. For Postman use the collection at `postman/RAG_v2.postman_collection.json` — it handles token generation automatically.

## Key Design Constraints

- All DB I/O is async (`asyncpg` pools, never psycopg2 or sync drivers)
- Workers consume from Redis Streams (not plain pub/sub) for durability
- Confidence scores gate retrieval: results below `MIN_CONFIDENCE_SCORE` are dropped
- Graph extraction timeout is separate (`GRAPH_EXTRACTION_TIMEOUT_S`) — don't conflate with `JOB_TIMEOUT_S`
- `VLM_ENABLED=false` by default — enabling requires Ollama + a VLM model (adds latency to PDF ingestion)
