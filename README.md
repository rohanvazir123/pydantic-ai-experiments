# pydantic-ai-experiments

Multi-corpus RAG system built on Pydantic AI. All commands run from the repo root.

---

## RAG v1 — [`rag/`](rag/)

```bash
# First time — installs everything and opens the UI
bash install.sh

# Every subsequent run
bash start.sh
```

UI opens at **http://localhost:3000** — sign in with `dev@neuralflow.ai` / `devpass`.

---

## Commands

```bash
make start          # start API (:8001) + frontend (:3000)
make install        # one-shot setup: deps, DB, models, seed, launch
make pull-models    # pull Ollama models (nomic + qwen2.5:0.5b + llama3.2:3b)
make seed           # apply DB migrations + ingest sample docs
make lint           # ruff check
make typecheck      # mypy
make test           # unit tests (~15s, no services needed)
make check          # lint + typecheck + test (pre-commit gate)
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.13 | https://www.python.org/downloads/ |
| Docker Desktop (running) | https://www.docker.com/products/docker-desktop/ |
| Node.js 20 LTS | https://nodejs.org |
| Ollama | https://ollama.com |

---

## What's in here

| Directory | Purpose |
|-----------|---------|
| [`rag/v2/`](rag/v2/) | **Active** — multi-corpus RAG: FastAPI + pgvector + AGE + Redis Streams + Next.js |
| [`rag/v2/docs/`](rag/v2/docs/) | Architecture, API reference, design deep-dives |
| [`rag/`](rag/) | Legacy v1 — single-corpus RAG, not actively developed |
| [`basics/`](basics/) | Experiments: Pydantic AI patterns, docling, LightRAG |
| [`misc/`](misc/) | Archived: CUAD legal KG |

The active system is documented in [`rag/v2/docs/`](rag/v2/docs/) — start with [`RAGV2_DESIGN.md`](rag/v2/docs/RAGV2_DESIGN.md).
