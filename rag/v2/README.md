# RAG v2

## Table of Contents

- [What This Is](#what-this-is)
- [Directory Layout](#directory-layout)
- [Quick Start](#quick-start)
- [Key Commands](#key-commands)
- [Testing](#testing)
- [Reproducing CI Locally](#reproducing-ci-locally)
- [Where to Read More](#where-to-read-more)

---

## What This Is

RAG v2 is the production rewrite of the RAG system. Multi-corpus, multi-tenant, Redis-backed async workers, confidence-aware pipeline, knowledge-graph extraction via docling-graph + Apache AGE, full memory system, and a Next.js frontend.

**`knowledge/`** is the main Python package. Everything else in this directory is configuration, infrastructure, or documentation.

---

## Directory Layout

| Path | What it is |
|------|-----------|
| `knowledge/` | Main Python package — all backend logic |
| `kg/` | Knowledge graph module (Apache AGE / Cypher) |
| `documents/` | Sample corpus for seeding / testing |
| `docs/` | All documentation — see [docs/README.md](docs/README.md) |
| `schema/` | SQL migrations (001 → 008, apply in order) |
| `scripts/` | `seed.py` (ingest sample docs), `purge.py` (reset corpus) |
| `tests/` | Pytest suite: `unit/`, `integration/`, `retrieval/` |
| `frontend/` | Next.js UI |
| `infra/` | Nginx, Grafana dashboards, Prometheus config |
| `postman/` | Postman collection + environment for manual API testing |
| `docker-compose.yml` | PostgreSQL + Apache AGE + Redis + API |
| `docker-compose.observability.yml` | Prometheus + Grafana stack |
| `Dockerfile` | Multi-stage image: `api`, `ingest-worker`, `retrieval-worker` |
| `Makefile` | Common commands — see [Key Commands](#key-commands) |
| `pyproject.toml` | Python project config, deps, tool settings |
| `.env.example` | All required environment variables with defaults |

---

## Quick Start

> **Windows prerequisite:** `INSTALL.ps1` requires PowerShell 7.1+ (`pwsh`). Install from
> <https://aka.ms/powershell>, then: `pwsh -ExecutionPolicy Bypass -File INSTALL.ps1`

```bash
cd rag/v2

# 1. Install deps (uv)
uv sync --extra all

# 2. Configure
cp .env.example .env
# Edit: DATABASE_URL, AGE_DATABASE_URL, REDIS_URL, LLM settings

# 3. Start services
docker compose up -d postgres age redis

# 4. Apply DB schema + seed sample docs
make seed           # runs databaseschemas then seed.py

# 5. Pull Ollama models
make pull-models    # llama3.2:3b, nomic-embed-text, qwen2.5:0.5b

# 6. Start the API
uv run uvicorn knowledge.api.app:app --reload --port 8001

# Health check
curl http://localhost:8001/health
```

---

## Key Commands

```bash
make databaseschemas   # apply SQL migrations (idempotent)
make seed              # schema + ingest rag/v2/documents/ into default corpus
make test-unit         # unit + retrieval metric tests — no services needed
make test              # full suite
make lint              # ruff check knowledge/ tests/
make typecheck         # mypy knowledge/
make ruff              # ruff --fix + format
```

---

## Testing

### Quick smoke test (no services, <2 s)
```bash
pytest tests/unit/test_smoke.py -v
```

### Unit tests (no services, ~15 s)
```bash
make test-unit
# = pytest tests/unit/ tests/retrieval/test_retrieval_metrics.py::TestMetricFunctions
```

### Integration tests (PostgreSQL + Redis required)
```bash
DATABASE_URL=postgresql://ragv2:test@localhost:5432/ragv2_test \
REDIS_URL=redis://localhost:6379/1 \
pytest tests/integration/ -v
```

### Retrieval quality tests (PostgreSQL + Redis + Ollama + ingested data)
```bash
# Requires make seed first
DATABASE_URL=... REDIS_URL=... pytest tests/retrieval/ -v --log-cli-level=INFO
```

### Test categories

| Folder | What | Services |
|--------|------|---------|
| `tests/unit/` | Settings, models, RRF, hooks, API factory | None |
| `tests/retrieval/::TestMetricFunctions` | IR metric math | None |
| `tests/integration/test_vector_store.py` | Document CRUD, vector/text/hybrid search, corpus isolation | PostgreSQL |
| `tests/integration/test_cache.py` | Embedding/search/fingerprint/health cache | Redis |
| `tests/retrieval/::TestRetrievalMetrics` | Hit Rate/MRR/NDCG against gold dataset (auto-skips if corpus empty) | PostgreSQL + Ollama |

---

## Reproducing CI Locally

CI uses `ragv2:test` on port 5432 (postgres Docker service). To replicate exactly:

```bash
# 1. Create the CI database (one-time setup)
python - <<'EOF'
import asyncio, asyncpg

async def main():
    # Connect as your admin user
    conn = await asyncpg.connect("postgresql://<admin>:<pass>@localhost:<port>/postgres")
    await conn.execute("CREATE USER ragv2 WITH PASSWORD 'test' SUPERUSER")
    await conn.execute("CREATE DATABASE ragv2_test OWNER ragv2")
    await conn.close()
    # Enable vector extension
    conn2 = await asyncpg.connect("postgresql://ragv2:test@localhost:<port>/ragv2_test")
    await conn2.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn2.close()
    print("Done")

asyncio.run(main())
EOF

# 2. Apply migrations (same as CI)
DATABASE_URL=postgresql://ragv2:test@localhost:<port>/ragv2_test \
python - <<'PYEOF'
import asyncio, asyncpg, glob, os, sys

async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    for f in sorted(glob.glob("schema/*.sql")):
        await conn.execute(open(f).read())
        print(f"  applied {f}")
    await conn.close()

asyncio.run(main())
PYEOF

# 3. Run all CI steps in order
export DATABASE_URL=postgresql://ragv2:test@localhost:<port>/ragv2_test
export AGE_DATABASE_URL=$DATABASE_URL
export REDIS_URL=redis://localhost:6379/1

make lint
make typecheck
pytest tests/unit/test_smoke.py -v
pytest tests/unit/ tests/retrieval/test_retrieval_metrics.py::TestMetricFunctions -v
pytest tests/integration/ -v
```

**Important:** `ragv2` must be created as a SUPERUSER so it owns its tables and
bypasses Row-Level Security. In CI, `POSTGRES_USER=ragv2` in the service container
automatically grants superuser. Locally, ensure `ragv2` owns all tables in
`ragv2_test` or has `SUPERUSER` privilege, otherwise RLS policies will block
`INSERT` statements (the `SET LOCAL` in `_conn()` only works for the session owner).

---

## Troubleshooting

### "No results found" / "No relevant information found"
Database is empty. Apply schemas and seed:
```bash
make databaseschemas
uv run python scripts/seed.py
```

### API won't start — Redis / Postgres connection refused
Port mismatch between `.env` and `docker-compose.yml`. Check:
```bash
docker compose ps                    # shows actual host ports
grep "DATABASE_URL\|REDIS_URL" .env  # shows what the app expects
```
Ports in `.env` must match the left side of `host:container` in docker-compose:
- `postgres` → `7300:5432` → use `localhost:7300` in `DATABASE_URL`
- `redis`    → `7500:6379` → use `localhost:7500` in `REDIS_URL`

### Login page proxy errors (`connect ECONNREFUSED 127.0.0.1:7100`)
Frontend Vite proxy points to the wrong API port. Check `frontend/vite.config.ts`:
```ts
proxy: { '/api/v2': { target: 'http://127.0.0.1:8001' } }  // must match API port
```

### `make databaseschemas` fails — `psql: command not found`
`psql` is not required. `make databaseschemas` uses Python/asyncpg directly.

### Audio transcription fails — whisper not installed
```bash
uv sync --extra audio   # installs openai-whisper
brew install ffmpeg     # required by whisper
```

### Token sequence too long warnings from reranker
Already handled — CrossEncoder is initialized with `max_length=512`. These warnings are suppressed.

### `npm` not found in terminal
nvm not loaded. Either open a new terminal (`.bash_profile` sources nvm automatically) or:
```bash
source ~/.bash_profile
```

---

## Where to Read More

All documentation lives in [`docs/`](docs/README.md):

| Doc | What it covers |
|-----|---------------|
| [`docs/REST_API.md`](docs/REST_API.md) | Every endpoint: method, path, request/response shapes |
| [`docs/RAGV2_DESIGN.md`](docs/RAGV2_DESIGN.md) | Full system design: architecture, Redis Streams, SLAs, cost model |
| [`docs/DATASTORE.md`](docs/DATASTORE.md) | PostgreSQL, AGE, Redis — schema and access patterns |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | All agent system prompts and structured output schemas |
| [`docs/LOCAL_LLM_GUIDE.md`](docs/LOCAL_LLM_GUIDE.md) | Ollama setup, model tiers, VRAM requirements |
| [`docs/TESTS.md`](docs/TESTS.md) | Test plan — categories, requirements, per-phase gates |
| [`docs/TEST_QA_REFERENCE.md`](docs/TEST_QA_REFERENCE.md) | Metric formulas, thresholds, acceptance criteria |
| [`docs/TODO_implementation.md`](docs/TODO_implementation.md) | Bottom-up build plan: phases and test gates |
