# pydantic-ai-experiments

## Table of Contents

- [What This Repository Is](#what-this-repository-is)
- [RAG v1](#rag-v1--ragrag)
- [RAG v2](#rag-v2--ragv2)
- [Other Modules](#other-modules)
- [Quick Start](#quick-start)
- [E2E Manual Testing](#e2e-manual-testing)
  - [v1 E2E](#v1-e2e)
  - [v2 E2E](#v2-e2e)

---

## What This Repository Is

Experiments and production systems built on top of [Pydantic AI](https://ai.pydantic.dev). The main deliverable is a multi-corpus RAG system in two generations:

| Generation | Directory | Stack |
|------------|-----------|-------|
| **RAG v1** | [`rag/`](rag/) | pgvector · Pydantic AI · Ollama · FastAPI |
| **RAG v2** | [`rag/v2/`](rag/v2/) | pgvector · FastAPI · Gunicorn · Pydantic AI · Apache AGE · Redis Streams · docling-graph · Next.js |

---

## RAG v1 — [`rag/`](rag/)

The original production RAG system. Answers questions over a document corpus using hybrid vector + full-text search and a Pydantic AI agent.

**Key docs:**
- [`rag/README.md`](rag/README.md) — setup, configuration, CLI usage
- [`rag/TESTS.md`](rag/TESTS.md) — test plan and quality metrics
- [`rag/TEST_QA_REFERENCE.md`](rag/TEST_QA_REFERENCE.md) — IR metric formulas and thresholds

**Quick run:**
```bash
# Install
pip install -e ".[all]"

# Ingest sample documents
python -m rag.main --ingest --documents rag/documents

# Start API
uvicorn rag.api.app:app --reload --port 8000

# Ask a question
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the PTO policy?"}'
```

---

## RAG v2 — [`rag/v2/`](rag/v2/)

Enterprise rewrite of RAG v1. Adds multi-corpus support, knowledge graphs (Apache AGE + docling-graph), a five-tier memory system, Redis Streams async workers, a confidence-aware 3-layer pipeline, and a Next.js frontend.

**Key docs:**
- [`rag/v2/RAGV2_DESIGN.md`](rag/v2/RAGV2_DESIGN.md) — full architecture: system diagram, data flow, all design decisions
- [`rag/v2/TODO_implementation.md`](rag/v2/TODO_implementation.md) — phased build plan (Phases 0–16)
- [`rag/v2/DATASTORE.md`](rag/v2/DATASTORE.md) — complete database reference (tables, TTLs, key patterns)
- [`rag/v2/PROMPTS.md`](rag/v2/PROMPTS.md) — all LLM prompts with design rationale
- [`rag/v2/TESTS.md`](rag/v2/TESTS.md) — test plan
- [`rag/v2/TEST_QA_REFERENCE.md`](rag/v2/TEST_QA_REFERENCE.md) — metric formulas, thresholds, load model
- [`basics/rag/memory/MEMORY_DESIGN.md`](basics/rag/memory/MEMORY_DESIGN.md) — five-tier memory architecture

**Quick start:**
```bash
cd rag/v2

# 1. Install Python deps
uv sync --extra all

# 2. Copy and edit environment
cp .env.example .env
# Set DATABASE_URL, AGE_DATABASE_URL, LLM_* in .env

# 3. Start services
docker compose up -d postgres age redis ollama

# 4. Pull Ollama models
make pull-models

# 5. Set up schema and ingest sample docs
make seed

# 6. Run unit tests
make test-unit

# 7. Start the API
uv run uvicorn knowledge.api.app:app --reload

# 8. (Optional) Start the frontend
cd frontend && npm install && npm run dev
```

**Directory layout:**
```
rag/v2/
├── knowledge/          Python backend package
├── schema/             PostgreSQL schema files (run with make databaseschemas / make seed)
├── tests/              Pytest test suite (251 unit tests)
├── frontend/           Next.js 15 + Tailwind CSS chat UI
├── postman/            Postman collection (import to test APIs manually)
├── scripts/            seed.py, purge.py for dev bootstrapping
├── infra/              Nginx config, Grafana dashboards, Prometheus config
├── Dockerfile          Multi-stage image (api / ingest-worker / retrieval-worker)
├── docker-compose.yml  Full local stack
└── Makefile            All common commands
```

**Postman:** import `rag/v2/postman/RAG_v2.postman_collection.json` + `RAG_v2_local.postman_environment.json` to test all 30+ API endpoints without the frontend.

---

## What Changed from v1 to v2

| Capability | RAG v1 | RAG v2 |
|------------|--------|--------|
| **Corpus support** | Single corpus, single tenant | Multi-corpus, multi-tenant with RLS isolation |
| **Knowledge graph** | Apache AGE (legal CUAD only) | Per-corpus AGE graphs; docling-graph ontology extraction for any domain |
| **Graph schema** | Hardcoded CUAD label allowlist | User-uploaded Pydantic ontology templates; generic fallback |
| **Ingestion** | Sync, blocking in API process | Async Redis Streams workers; `asyncio.gather` chunk + graph in parallel |
| **Retrieval** | Vector + text search | Hybrid RRF + CrossEncoder reranker → calibrated `confidence` score |
| **Answer quality** | Single LLM call | 3-layer confidence-aware pipeline: retrieval gate → citation gate → LLM judge |
| **Caching** | None | L1 in-process LRU + L2 Redis (embed/search/fingerprint) + L3 pgvector semantic cache (JWE) |
| **Memory** | `message_history` in request body | 5-tier memory: working / episodic (server-side) / semantic-user (Mem0) / semantic-world / procedural |
| **Conversation history** | Client-side (lost on refresh) | Server-side PostgreSQL; loaded by `session_id`; auto-summarised at 20 turns |
| **Scheduler** | Manual `--ingest` CLI | APScheduler periodic jobs with cron + incremental SHA-256 dedup |
| **Security** | No auth | JWT RS256 + RBAC + JWE payload encryption + rate limiting |
| **API** | 3 endpoints (chat, ingest, health) | 30+ endpoints across auth / chat / search / ingest / corpus / scheduler / eval / memory / logs |
| **Frontend** | Streamlit (dev only) | Next.js 15 + Tailwind CSS; SSE streaming; cost badge; per-stage latency debug panel |
| **Observability** | Basic logging | Prometheus metrics + Langfuse LLM traces + Redis log ring buffer + SMTP alerts |
| **Deployment** | uvicorn + `.env` | Gunicorn + UvicornWorker; Docker Compose (Nginx + API + 3 workers + PG + AGE + Redis + Ollama) |
| **Testing** | ~100 unit tests | 251 unit tests; load tests (Locust); chaos tests (Makefile targets) |

**In short:** v2 is v1 with production hardening, multi-tenancy, graph extraction, a confidence-aware pipeline, full memory system, and a proper frontend. v1 remains the simpler, dependency-lighter system for use cases that don't need the full stack.

---

## Other Modules

| Directory | Purpose |
|-----------|---------|
| [`basics/`](basics/) | Experiments: algorithms, Pydantic AI patterns, docling, LightRAG, RAG-Anything |
| [`basics/pydantic_ai/`](basics/pydantic_ai/) | Pydantic AI usage: single-agent modes, multi-agent pipelines |
| [`basics/rag/`](basics/rag/) | RAG research notes: docling FAQ, docling-graph FAQ, memory design |
| [`kg/`](kg/) | Apache AGE knowledge graph store (v1) — absorbed into `rag/v2/knowledge/store/graph.py` |
| [`nl2sql/`](nl2sql/) | NL-to-SQL experiments |
| [`misc/`](misc/) | Archived: CUAD legal KG experiments, graphiti |
| [`notebook/`](notebook/) | Jupyter notebooks for one-off analysis |

---

## Quick Start

For first-time setup of the full v2 stack:

```bash
# Prerequisites: Python 3.13, Docker, Ollama, uv
cd rag/v2
make seed        # set up database schema + ingest sample NeuralFlow AI docs
make test-unit   # verify 251 unit tests pass
make dev         # start full Docker Compose stack
```

For the v1 system (simpler, no Docker required):
```bash
pip install -e ".[all]"
ollama serve && ollama pull llama3.1:8b nomic-embed-text
python -m rag.main --validate
python -m rag.main --ingest --documents rag/documents
uvicorn rag.api.app:app --reload
```

---

## E2E Manual Testing

Step-by-step setup → ingest → tests → manual smoke for both generations.

---

### v1 E2E

**Prerequisites:** Python 3.13, PostgreSQL with pgvector extension, Ollama

#### 1. Install

```bash
pip install -e ".[all]"
```

#### 2. Configure

```bash
cp .env.example .env   # if present, or create manually
# Set in .env:
#   DATABASE_URL=postgresql://user:pass@localhost/ragdb
#   LLM_MODEL=llama3.1:8b
#   EMBEDDING_MODEL=nomic-embed-text:latest
```

#### 3. Start Ollama and pull models

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
```

#### 4. Validate config

```bash
python -m rag.main --validate
# Expected: "Configuration is valid" with no errors
```

#### 5. Ingest sample documents

```bash
python -m rag.main --ingest --documents rag/documents
# Expected: chunks created and embeddings stored in PostgreSQL
```

#### 6. Run automated tests

```bash
# Fast unit tests — no external deps (~12 + ~34 tests, <10 s)
python -m pytest rag/tests/core/ -v

# DB-layer tests — requires PostgreSQL (~46 tests)
python -m pytest rag/tests/storage/test_postgres_store.py -v

# Retrieval quality + agent — requires PostgreSQL + Ollama (~28 + 25 tests, 2-5 min)
python -m pytest rag/tests/retrieval/ rag/tests/agent/test_rag_agent.py -v --tb=short

# API + MCP mocked tests — no external deps (14 + 21 tests)
python -m pytest rag/tests/agent/test_api.py rag/tests/agent/test_mcp_server.py -v
```

#### 7. Start the API

```bash
uvicorn rag.api.app:app --reload --port 8000
```

#### 8. Manual smoke tests

```bash
# Health check
curl http://localhost:8000/health

# Chat query
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the PTO policy?"}'

# Streaming chat
curl -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What technologies does the company use?"}'
```

---

### v2 E2E

**Prerequisites:** Python 3.13, Docker, [uv](https://docs.astral.sh/uv/), Ollama

```bash
cd rag/v2
```

#### 1. Install Python deps

```bash
uv sync --extra all
```

#### 2. Configure

```bash
cp .env.example .env
# Set in .env:
#   DATABASE_URL=postgresql://user:pass@localhost:5432/ragv2
#   AGE_DATABASE_URL=postgresql://user:pass@localhost:5433/age_db
#   REDIS_URL=redis://localhost:6379
#   LLM_MODEL=llama3.2:3b
#   EMBEDDING_MODEL=nomic-embed-text:latest
```

#### 3. Start infrastructure

```bash
docker compose up -d postgres age redis ollama
# Wait ~10 s for services to be healthy
docker compose ps   # all services should show "healthy" or "running"
```

#### 4. Pull Ollama models

```bash
make pull-models
# Pulls: llama3.2:3b, nomic-embed-text, qwen2.5:0.5b
```

#### 5. Set up schema and ingest sample docs

```bash
make seed
# Runs migrations, creates default tenant/corpus, ingests NeuralFlow AI docs
```

#### 6. Run automated tests

```bash
# Unit tests — no external deps (251 tests, <30 s)
make test-unit

# Integration + retrieval + agent + API (requires full stack, 5-15 min)
make test-integration

# All tests
make test
```

#### 7. Start the API

```bash
uv run uvicorn knowledge.api.app:app --reload --port 8001
```

#### 8. Manual smoke tests

```bash
# Health check
curl http://localhost:8001/health

# Chat query (requires JWT — see rag/v2/README.md for token setup)
curl -X POST http://localhost:8001/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "What is the PTO policy?", "session_id": "test-session"}'

# Search endpoint
curl -X POST http://localhost:8001/v1/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "employee benefits", "corpus_id": "default"}'
```

**Postman:** import `rag/v2/postman/RAG_v2.postman_collection.json` + `RAG_v2_local.postman_environment.json` to hit all 30+ endpoints with pre-configured auth.

#### 9. Optional: chaos and load tests

```bash
# Chaos — kill a dependency and verify graceful degradation
make chaos-kill-redis       # Redis gone; should fall back to in-process cache
make chaos-kill-ollama      # Ollama gone; should return 503 with circuit-breaker message
make chaos-kill-postgres    # DB gone; should return 503

# Load — baseline at 10 RPS for 60 s
make load-baseline          # requires locust: pip install locust
```
