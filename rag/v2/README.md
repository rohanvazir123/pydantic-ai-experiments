# RAG v2

## Table of Contents

- [What This Is](#what-this-is)
- [Directory Layout](#directory-layout)
- [Quick Start](#quick-start)
- [Where to Read More](#where-to-read-more)

---

## What This Is

RAG v2 is the production rewrite of the RAG system. It adds multi-corpus support, Redis-backed async workers, a confidence-aware pipeline, knowledge graph extraction via docling-graph and Apache AGE, a full memory system, and a Next.js frontend.

This directory contains both the **design documents** and the **implementation** for the backend Python service.

---

## Directory Layout

| Path | What it is |
|------|-----------|
| `RAGV2_DESIGN.md` | Architecture reference — system design, data schemas, SLAs |
| `TODO_implementation.md` | Phased build plan — what to implement and in what order |
| `TESTS.md` | Test plan — categories, requirements, per-phase gates |
| `TEST_QA_REFERENCE.md` | QA reference — metric formulas, thresholds, load model |
| `knowledge/` | Main Python package — all backend logic |
| `migrations/` | SQL files that build the PostgreSQL schema |
| `tests/` | Pytest test suite |
| `postman/` | Postman collection for manual API testing |
| `infra/` | Nginx config, Grafana dashboards |
| `pyproject.toml` | Python project config, dependencies, tool settings |
| `Makefile` | Common commands: dev, migrate, test, lint, typecheck |
| `.env.example` | All required environment variables with defaults |

---

## Quick Start

```bash
cd rag/v2

# 1. Install dependencies
uv sync --extra all

# 2. Copy and edit env
cp .env.example .env
# Set DATABASE_URL, AGE_DATABASE_URL, and LLM settings in .env

# 3. Start services
docker compose up -d postgres age redis ollama

# 4. Run migrations
make migrate

# 5. Pull Ollama models
make pull-models

# 6. Run unit tests (no services needed)
make test-unit

# 7. Start the API
uv run uvicorn knowledge.api.app:app --reload
```

---

## Where to Read More

- **System design**: `RAGV2_DESIGN.md`
- **Build plan**: `TODO_implementation.md`
- **Memory design**: `../../basics/rag/memory/MEMORY_DESIGN.md`
- **docling-graph**: `../../basics/rag/docling-graph/faq.md`
