# RAG v2 — Multi-Corpus Knowledge Assistant

A production-grade Retrieval-Augmented Generation system with a multi-tenant, multi-corpus backend and a modern React SPA frontend. Ask questions across PDFs, DOCX, Markdown, audio transcripts, and more — answers are grounded in your documents with inline citations.

---

## Quick Start

```bash
# First time — installs everything and opens the UI
bash install.sh

# Every subsequent run
bash start.sh
```

UI → **http://localhost:7200** · Sign in with `dev@neuralflow.ai` / `devpass`

---

## Tech Stack

### Backend — `rag/v2/knowledge/`

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | **FastAPI** (Python 3.13) | REST + SSE streaming endpoints, JWT RS256 auth, rate limiting |
| ORM / DB driver | **asyncpg** | Async PostgreSQL — all queries non-blocking |
| Vector store | **pgvector** (PostgreSQL) | HNSW index for embedding search |
| Hybrid search | **RRF** (Reciprocal Rank Fusion) | Combines vector + full-text scores |
| Graph store | **Apache AGE** | Entity + relationship graph (Cypher over PostgreSQL) |
| Ingestion | **Docling** | PDF, DOCX, PPTX → structured text + tables |
| Audio | **Whisper** (OpenAI) | MP3/WAV → transcript |
| Embeddings | **Ollama** (`nomic-embed-text`) | Local embedding model, 768-dim |
| LLM | **Ollama** (`llama3.2:3b` default) | Local inference; pluggable (any OpenAI-compatible API) |
| Agent framework | **Pydantic AI** | Structured LLM outputs, tool calls, streaming |
| Message bus | **Redis Streams** | Async job queue for ingestion + retrieval workers |
| Cache — L1 | In-process LRU dict | Sub-millisecond hits within a process |
| Cache — L2 | **Redis** (msgpack) | Cross-process shared cache |
| Cache — L3 | pgvector semantic cache | Cosine-similarity match on past queries |
| Scheduler | **APScheduler** | Periodic re-ingestion and maintenance jobs |
| Observability | **Prometheus** + **structlog** | Metrics, request tracing, Redis log ring buffer |
| Auth | JWT RS256 + JWE | Stateless auth, encrypted payloads |
| Config | **pydantic-settings** | `.env`-driven, validated at startup |

### Frontend — `rag/v2/frontend/`

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Build tool | **Vite 8** | Sub-second HMR, esbuild bundling |
| UI framework | **React 19** | Component model, concurrent features |
| Language | **TypeScript** (strict) | End-to-end type safety |
| Routing | **React Router v7** | Client-side SPA routing, `PrivateRoute` auth guard |
| State | **Zustand** | Global chat store; `getState()` for async non-reactive reads |
| Styling | **Tailwind CSS v4** | Utility-first, `@tailwindcss/vite` plugin |
| Streaming | Custom `streamSSE` hook | `ReadableStream` + `TextDecoder` for token-by-token chat |
| Session cache | `sessionStorage` + TTL | Q&A cache — cleared on reload, 30-min expiry |
| Icons | **Lucide React** | Consistent icon set |
| Markdown | **react-markdown** + `remark-gfm` | Renders assistant answers with tables, code blocks |
| Toast | **react-hot-toast** | Non-blocking notifications |
| E2E tests | **Playwright** | Logs page smoke tests; runs on port 7200 |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| Databases | PostgreSQL 16 + pgvector, Apache AGE (separate container) |
| Cache / broker | Redis 7 |
| Local LLM | Ollama (llama3.2:3b, nomic-embed-text, qwen2.5 variants) |
| Containers | Docker Compose |
| Reverse proxy | Nginx (production) |
| Package manager (Python) | **uv** |
| Package manager (Node) | **npm** |

---

## Commands

```bash
# From repo root
bash start.sh           # start API (:7100) + frontend (:7200)
bash install.sh         # one-shot setup: deps, DB, models, seed, launch

# From rag/v2/
make databaseschemas    # apply SQL migrations (idempotent)
make seed               # ingest sample documents into default corpus
make pull-models        # pull Ollama models
make ruff               # lint + format
make mypy               # type check
make test-unit          # unit tests (~15s, no services)
make test               # full suite (321 tests)

# Frontend (from rag/v2/frontend/)
npm run dev             # Vite dev server on :7200
npm run build           # production build → dist/
npm run test:e2e        # Playwright tests
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13 | https://www.python.org |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 20 LTS | https://nodejs.org |
| Docker Desktop | running | https://www.docker.com/products/docker-desktop |
| Ollama | latest | https://ollama.com |

---

## Repository Layout

```
rag/v2/                     Active system
├── knowledge/              Python package (API, agent, retrieval, ingestion…)
├── frontend/               Vite + React SPA
├── documents/              Sample corpus (PDF, DOCX, MD, MP3)
├── schema/                 SQL migrations (001–008)
├── scripts/                seed.py, purge.py
├── tests/                  279 unit + integration + retrieval tests
├── docs/                   Architecture, pipeline, API docs
└── docker-compose.yml

basics/frontend/            Frontend engineering reference
├── js/javascript.md        Modern JS — basics + advanced
├── ts/typescript.md        TypeScript — types, generics, project patterns
├── react/react.md          React 19 — core + production patterns
├── react/hooks.md          All React hooks with TypeScript
└── tsx/tsx.md              TSX — prop types, events, refs, build pipeline

rag/                        Legacy v1 (not actively developed)
misc/                       Archived experiments
```
