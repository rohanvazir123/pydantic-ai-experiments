# rovaz — pydantic-ai-experiments

## Table of Contents

- [What This Repository Is](#what-this-repository-is)
- [RAG v1](#rag-v1--ragrag)
- [RAG v2](#rag-v2--ragv2)
- [Other Modules](#other-modules)
- [Quick Start](#quick-start)

---

## What This Repository Is

Experiments and production systems built on top of [Pydantic AI](https://ai.pydantic.dev). The main deliverable is a multi-corpus RAG system in two generations:

| Generation | Directory | Status | Stack |
|------------|-----------|--------|-------|
| **RAG v1** | [`rag/`](rag/) | Production-ready (used in prod) | pgvector · Pydantic AI · Ollama · FastAPI |
| **RAG v2** | [`rag/v2/`](rag/v2/) | In active development | v1 + Apache AGE · Redis Streams · docling-graph · Next.js |

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
├── schema/             PostgreSQL schema files (run with make migrate / make seed)
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
