# RAG v2 Development Instructions

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Core Principles](#core-principles)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Key Commands](#key-commands)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Architecture](#architecture)
- [Common Issues](#common-issues)
- [Development Workflow](#development-workflow)
- [Quick Reference](#quick-reference)

---

## Project Overview

**Active system: `rag/v2/`** — multi-tenant, multi-corpus RAG backend built on the `knowledge/` Python package. Uses FastAPI, asyncpg/pgvector, Apache AGE, Redis Streams, Docling, and Pydantic AI. Supports hybrid search (vector + full-text + RRF), a 3-layer cache (in-process LRU → Redis → pgvector semantic), JWT RS256 auth, APScheduler-based ingestion, and Prometheus observability.

**Legacy system: `rag/`** — the original v1 single-tenant RAG agent. Still present but not actively developed. Imports from the now-moved `kg/` module will be broken.

---

## Quick Start

```bash
cd rag/v2

# Copy env and fill in values
cp .env.example .env

# Install deps (uv)
uv sync

# Start services (PostgreSQL + Apache AGE + Redis)
docker compose up -d postgres age redis

# Apply DB schemas (idempotent)
make databaseschemas

# Seed default tenant, corpus, and sample docs
make seed

# Pull Ollama models
make pull-models

# Start the API (auto-reload)
uv run uvicorn knowledge.api.app:app --reload --port 8001

# Health check
curl http://localhost:8001/health

# Run smoke tests (no services needed, <1 s)
make test-unit
```

## Core Principles

1. **TYPE SAFETY IS NON-NEGOTIABLE**
   - All functions, methods, and variables MUST have type annotations
   - Use Pydantic models for all data structures
   - Import `Callable` from `collections.abc`, not lowercase `callable`

2. **KISS** (Keep It Simple, Stupid)
   - Prefer simple, readable solutions over clever abstractions
   - Trust PostgreSQL RRF — no manual score combination

3. **ASYNC ALL THE WAY**
   - All I/O operations MUST be async (PostgreSQL, Redis, embeddings, LLM calls)
   - Use `asyncio` for concurrent operations; CPU-bound libs via `asyncio.to_thread()`

---

## Project Structure

```
rag/
├── v2/                               # PRIMARY — all new development here
│   ├── knowledge/                    # Main Python package
│   │   ├── agent/                    # Pydantic AI agent (pipeline, judge, model router)
│   │   ├── api/                      # FastAPI app + middleware
│   │   │   └── routes/               # chat, ingest, search, corpus, auth, health, evaluate,
│   │   │                             #   feedback, memory, scheduler, logs
│   │   ├── billing/                  # Cost tracking + quota
│   │   ├── bus/                      # Redis Streams publisher/consumer + circuit breaker
│   │   ├── config/                   # settings.py (pydantic-settings, reads .env)
│   │   ├── corpus/                   # Corpus ontologies
│   │   ├── evaluation/               # Retrieval metrics (Hit Rate, MRR, NDCG)
│   │   ├── hooks/                    # HookPoint enum + async HookRegistry
│   │   ├── ingestion/                # pipeline.py, chunker, embedder, graph_extractor, worker
│   │   ├── memory/                   # Working memory, conversation store, summarizer
│   │   ├── observability/            # Prometheus metrics + alert helpers
│   │   ├── retrieval/                # Retriever, RRF fusion, semantic cache, worker
│   │   ├── scheduler/                # APScheduler job runner + job store
│   │   ├── store/                    # vector.py, graph.py, cache.py, entity_index.py
│   │   └── validation/               # 6-stage query validation pipeline
│   ├── kg/                           # Knowledge graph (moved from top-level kg/)
│   │   ├── __init__.py               # create_kg_store() factory
│   │   ├── age_graph_store.py        # AgeGraphStore: Apache AGE / Cypher (port 5433)
│   │   ├── entity_index.py           # Entity indexing utilities
│   │   ├── app/                      # CLI, REST API, Streamlit apps for KG
│   │   ├── docs/                     # KG-specific documentation
│   │   ├── evals/                    # KG evaluation data
│   │   └── tests/                    # KG tests
│   ├── docs/                         # Architecture + pipeline docs (moved from rag/docs/)
│   │   ├── ARCHITECTURE.md
│   │   ├── ARCHITECTURE_SUMMARY.md
│   │   ├── CALL_GRAPH.md
│   │   ├── DATASTORE_GUIDE.md
│   │   ├── FAQ.md
│   │   ├── INGESTION_PIPELINE.md
│   │   ├── LOCAL_LLM_GUIDE.md
│   │   ├── PROMPTS.md
│   │   ├── RAG.md
│   │   └── RETRIEVAL_PIPELINE.md
│   ├── documents/                    # Sample corpus (moved from rag/documents/)
│   │   ├── company-overview.md
│   │   ├── team-handbook.md
│   │   ├── *.pdf  *.docx  *.mp3     # PDFs, DOCX, audio recordings
│   │   └── ...
│   ├── frontend/                     # Next.js UI
│   ├── infra/                        # nginx, Grafana dashboards, Prometheus config
│   ├── postman/                      # Postman collection + environment
│   ├── schema/                       # SQL migrations (001_initial_schema.sql … 008_memory.sql)
│   ├── scripts/                      # seed.py, purge.py
│   ├── tests/
│   │   ├── unit/                     # No-service tests — fakeredis, mocked asyncpg
│   │   ├── integration/              # Needs PostgreSQL + Redis
│   │   ├── retrieval/                # Needs PostgreSQL + Redis + Ollama + ingested data
│   │   ├── api/                      # API surface tests
│   │   ├── agent/                    # Agent integration tests
│   │   ├── ingestion/                # Ingestion pipeline tests
│   │   ├── load/                     # Locust load tests
│   │   └── chaos/                    # Chaos / fault injection tests
│   ├── docker-compose.yml            # PostgreSQL + Apache AGE + Redis + API
│   ├── docker-compose.observability.yml  # Prometheus + Grafana
│   ├── Dockerfile                    # API container
│   ├── Makefile                      # make databaseschemas | seed | test | ruff | mypy
│   ├── pyproject.toml                # uv-managed deps
│   ├── .env.example                  # Copy to .env and fill in values
│   ├── RAGV2_DESIGN.md               # Full design doc
│   ├── DATASTORE.md                  # DB schema reference
│   ├── PROMPTS.md                    # Prompt engineering guide
│   └── README.md                     # v2 entry point
│
└── (v1 — legacy, not actively developed)
    ├── agent/    rag_agent.py, kg_agent.py, prompts.py
    ├── api/      app.py (FastAPI, single-tenant)
    ├── app/      cli/, rest_api/, streamlit/
    ├── config/   settings.py
    ├── ingestion/ pipeline.py, embedder.py, models.py, chunkers/, processors/
    ├── mcp/      server.py (FastMCP)
    ├── memory/   mem0_store.py
    ├── retrieval/ retriever.py, hybrid_kg_retriever.py, rerankers.py
    ├── storage/  vector_store/postgres.py
    ├── tests/    (see rag/TESTS.md)
    └── main.py   CLI entry point

misc/                                 # Archived experiments / reference code
└── kg_legal_cuad/                    # CUAD legal corpus use-case (archived)
    ├── kg_legal/                     # Legal KG ingestion + retrieval
    └── tests/                        # Legal KG tests

docker-compose.yml                    # Root-level AGE container shortcut (port 5433)
```

---

## Configuration

### Environment Variables (`rag/v2/.env`)

Copy `rag/v2/.env.example` to `rag/v2/.env` and fill in:

```bash
# PostgreSQL (pgvector)
DATABASE_URL=postgresql://ragv2:pass@localhost:5432/ragv2

# Apache AGE (graph store)
AGE_DATABASE_URL=postgresql://age:pass@localhost:5433/age_graph

# Redis
REDIS_URL=redis://localhost:6379

# LLM (Ollama local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Embeddings (Ollama local)
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_DIMENSION=768

# Auth (JWT RS256 — keys generated by INSTALL.ps1 / INSTALL.sh into infra/keys/)
JWT_ALGORITHM=RS256
JWT_PUBLIC_KEY_PATH=infra/keys/public.pem
JWT_PRIVATE_KEY_PATH=infra/keys/private.pem
```

### PostgreSQL / AGE Setup

Run `make databaseschemas` to apply all migrations in `schema/` order. Or manually:

```bash
# In rag/v2/
uv run python -c "
import asyncio, asyncpg
async def run():
    conn = await asyncpg.connect(DATABASE_URL)
    with open('schema/001_initial_schema.sql') as f:
        await conn.execute(f.read())
asyncio.run(run())
"
```

---

## Key Commands

All commands run from `rag/v2/`:

```bash
# Start API (dev, auto-reload)
uv run uvicorn knowledge.api.app:app --reload --port 8001

# Apply DB schemas (idempotent)
make databaseschemas

# Seed default tenant + corpus + sample docs
make seed

# Pull Ollama models
make pull-models

# Lint + format
make ruff

# Type-check
make mypy

# Unit tests (no services)
make test-unit

# All tests
make test
```

---

## Testing

### Run Tests

```bash
cd rag/v2

# Smoke tests — fastest, no services (<1 s)
python -m pytest tests/unit/test_smoke.py -v

# All unit tests — no services
make test-unit
# or: python -m pytest tests/unit/ -v

# Integration tests — needs PostgreSQL + Redis
python -m pytest tests/integration/ -v

# Retrieval quality — needs PostgreSQL + Redis + Ollama + ingested data
python -m pytest tests/retrieval/ -v --log-cli-level=INFO --tb=short

# Full suite
make test
```

### Test Categories

| Folder | What It Tests | Requirements |
|--------|--------------|--------------|
| `tests/unit/test_smoke.py` | Package imports, settings, models, RRF, hooks, API factory | None |
| `tests/unit/test_settings.py` | Settings loading, validation, credential masking | None |
| `tests/unit/test_agent.py` | Agent pipeline logic | None (mocked) |
| `tests/unit/test_api.py` | FastAPI routes (mocked) | None |
| `tests/unit/test_cache.py` | L1/L2 cache logic | None (fakeredis) |
| `tests/unit/test_circuit_breaker.py` | Circuit breaker state machine | None (fakeredis) |
| `tests/unit/test_backoff.py` | Exponential backoff | None |
| `tests/unit/test_consumer.py` | Redis Stream consumer | None (fakeredis) |
| `tests/unit/test_ingestion.py` | Chunker + pipeline models | None |
| `tests/unit/test_retrieval.py` | RRF fusion, confidence filter | None |
| `tests/unit/test_store.py` | VectorStore SQL generation | None (mocked asyncpg) |
| `tests/unit/test_hooks_and_validation.py` | Hook registry + validation pipeline | None |
| `tests/integration/` | Full stack — ingest → retrieve → chat | PostgreSQL + Redis |
| `tests/retrieval/` | Hit Rate / MRR / NDCG on sample corpus | PostgreSQL + Redis + Ollama |
| `tests/load/` | Locust throughput tests | Full stack |

### Sample Corpus

Sample documents live in `rag/v2/documents/`. Run `make seed` to ingest them into the default corpus. Test queries:

```python
"What does NeuralFlow AI do?"
"What is the PTO policy?"
"What technologies does the company use?"
```

---

## Code Quality

```bash
cd rag/v2

# Ruff lint + format (auto-fix)
make ruff
# or: ruff check --fix knowledge/ && ruff format knowledge/

# Type checking
make mypy
# or: mypy knowledge/
```

---

## Architecture

### System at a Glance

```
API (FastAPI, port 8001)
  └─ JWT RS256 auth middleware
  └─ rate limiting (slowapi)
  └─ routes → publish job to Redis Stream
        │
        ├─ ingest-worker  (Redis consumer)
        │     Docling → chunk → embed → graph extract → VectorStore + AGE
        │
        └─ retrieval-worker  (Redis consumer)
              L1 LRU → L2 Redis → L3 semantic cache → hybrid search (RRF) → CrossEncoder
```

### Three-Layer Cache

| Layer | Store | Mechanism |
|-------|-------|-----------|
| L1 | In-process | LRU dict |
| L2 | Redis | msgpack serialisation |
| L3 | PostgreSQL | pgvector cosine-sim (stored as JWE) |

### Model Tiers

| Tier | Model | Used for |
|------|-------|---------|
| nano | `qwen2.5:0.5b` | routing, classification |
| small | `llama3.2:3b` | standard RAG responses |
| large | `llama3.1:70b` | complex reasoning |

Tier selection is controlled by `MODEL_ROUTING_ENABLED`.

### Multi-Tenancy

Every request carries a `tenant_id` from the JWT. Each tenant has one or more named corpora; retrieval is scoped to the corpus in the request.

### Corpus-Scoped KG

`knowledge/store/graph.py` wraps Apache AGE. Graph names are namespaced per tenant + corpus via `settings.age_graph_name(tenant_id, corpus_id)`.

### DB Schema (migrations in `schema/`)

| File | What it creates |
|------|----------------|
| `001_initial_schema.sql` | tenants, corpora, documents, chunks (pgvector) |
| `002_corpus_tenant.sql` | corpus membership + RBAC |
| `003_semantic_cache.sql` | L3 semantic cache table |
| `004_evaluation.sql` | eval runs + result rows |
| `005_feedback.sql` | user thumbs up/down |
| `006_billing.sql` | token + cost tracking |
| `007_scheduler.sql` | scheduled job store |
| `008_memory.sql` | conversation + working memory |

---

## Common Issues

### 1. `DATABASE_URL` or `AGE_DATABASE_URL` not set
Both are required. Copy `.env.example` → `.env` and fill in the values.

### 2. "pgvector extension not found"
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. JWT public key missing
Run `INSTALL.ps1` (Windows) or `INSTALL.sh` (macOS/Linux) to generate RSA keys into `infra/keys/`. Or manually:
```bash
mkdir -p infra/keys
openssl genrsa -out infra/keys/private.pem 2048
openssl rsa -in infra/keys/private.pem -pubout -out infra/keys/public.pem
```

### 4. "callable is not subscriptable"
Use `Callable` from `collections.abc`:
```python
from collections.abc import Callable
def func(callback: Callable | None = None): ...
```

### 5. Ollama connection refused
```bash
ollama serve
```

### 6. Embedding dimension mismatch
Ensure `EMBEDDING_DIMENSION` matches your model:
- `nomic-embed-text`: 768
- `text-embedding-3-small` / `ada-002`: 1536

### 7. Audio transcription fails
Requires FFmpeg in PATH + `openai-whisper`:
```bash
brew install ffmpeg        # macOS
pip install openai-whisper
```

### 8. `basics/frontend/day1_exercises` — `npm install` fails with ENOTFOUND
`package-lock.json` must never be committed for this project — it can embed a private registry URL that is unreachable outside that network. If install fails, delete the lockfile and reinstall:
```bash
cd basics/frontend/day1_exercises
rm -f package-lock.json
npm install
```
`node_modules/` and `package-lock.json` are in `.gitignore`. Do not commit either.

---

## Development Workflow

1. **Work in `rag/v2/`** — all active development lives here
2. **Smoke test first**: `python -m pytest tests/unit/test_smoke.py -v` (< 1 s, no services)
3. **Lint**: `make ruff`
4. **Type-check**: `make mypy`
5. **Full unit suite**: `make test-unit`
6. **Integration tests** when touching DB/Redis layer: `python -m pytest tests/integration/ -v`

### Pre-commit gate (REQUIRED before every commit)

Run all three in order from `rag/v2/`. All must be green before committing:

```bash
uv run ruff check knowledge/ tests/   # lint — catches import order, naming, style
uv run mypy knowledge/                # type check — catches type errors
uv run pytest tests/unit/ -q          # unit tests — 279 tests, ~15s, no services needed
```

Or as a single command:

```bash
uv run ruff check knowledge/ tests/ && uv run mypy knowledge/ && uv run pytest tests/unit/ -q
```

**Never commit if any of these fail.** CI runs the same checks and will block the push.

### Commit message style

Keep commit messages minimal — one short line, no body:

```
fix: seed script DATABASE_URL default
feat(api): add corpus cache invalidation endpoint
docs: update REST_API endpoint list
chore: gitignore package-lock.json
```

No bullet lists, no "Co-Authored-By" unless explicitly asked, no references to internal tools or registries. Scope is optional but useful for large repos (`feat(ui):`, `fix(ci):`, `docs:`). The message should say what changed, not why — the code is the why.

### Add a New Route

1. Create `knowledge/api/routes/my_route.py` with an `APIRouter`
2. Add request/response models to `knowledge/api/schemas.py`
3. Register the router in `knowledge/api/app.py`

### Add a DB Migration

Create `schema/NNN_description.sql` (next number). Run `make databaseschemas`.

### Add a New Setting

Add to `knowledge/config/settings.py` — pydantic-settings reads from `.env` automatically.

---

## Quick Reference

### Health Check
```bash
curl http://localhost:8001/health
```

### Run the API Programmatically
```python
import asyncio
import httpx

async def chat(question: str, token: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://localhost:8001/v1/chat",
            json={"query": question, "corpus_id": "default"},
            headers={"Authorization": f"Bearer {token}"},
        )
        return r.json()["answer"]

asyncio.run(chat("What does NeuralFlow AI do?", token="..."))
```

### Search via Retriever Directly
```python
import asyncio, os
from knowledge.config.settings import load_settings
from knowledge.store.vector import PostgresHybridStore
from knowledge.retrieval.retriever import Retriever

async def search(query: str) -> None:
    settings = load_settings()
    store = PostgresHybridStore(settings)
    await store.connect()
    retriever = Retriever(store=store, settings=settings)
    results = await retriever.retrieve(query, corpus_id="default", tenant_id="default")
    for r in results:
        print(f"[{r.raw_score:.3f}] {r.document_title}: {r.content[:120]}...")
    await store.close()

asyncio.run(search("employee PTO policy"))
```

### Ingest Documents
```bash
cd rag/v2
make seed                              # seed default corpus from rag/v2/documents/

# Or call the API directly
curl -X POST http://localhost:8001/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@documents/team-handbook.md" \
  -F "corpus_id=default"
```
