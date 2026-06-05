# RAG Agent

Agentic RAG system combining PostgreSQL/pgvector with Pydantic AI for intelligent document retrieval. Documents are ingested, chunked, embedded, and stored for hybrid (vector + full-text) retrieval. An optional knowledge-graph pipeline (Apache AGE) extracts entities and relationships from the same documents.

---

## Table of Contents

- [Stack](#stack)
- [Prerequisites](#prerequisites)
- [Install](#install)
  - [One-command install](#one-command-install)
  - [Interactive install](#interactive-install)
  - [Manual install](#manual-install)
  - [Optional extras](#optional-extras)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Running the system](#running-the-system)
- [Knowledge Graph](#knowledge-graph)
- [Development](#development)
- [Architecture docs](#architecture-docs)

---

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| AI framework | Pydantic AI |
| Database | PostgreSQL 17 + pgvector (vector + full-text hybrid search) |
| Knowledge graph | Apache AGE on PostgreSQL 16 (openCypher, optional) |
| LLM / embeddings | Ollama — `llama3.1:8b`, `nomic-embed-text` (local, no API key required) |
| Ingestion | Docling (PDF, DOCX, HTML, PPTX, audio via Whisper) |
| Reranker | sentence-transformers CrossEncoder (`BAAI/bge-reranker-base`) |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Observability | Langfuse |
| Memory | Mem0 (pgvector-backed user memory) |
| Package manager | uv |

---

## Prerequisites

Install these before running the install script:

| Tool | macOS | Linux | Windows | Purpose |
|------|-------|-------|---------|---------|
| **Git** | pre-installed | `sudo apt install git` / `sudo dnf install git` | [git-scm.com](https://git-scm.com) | Clone the repo |
| **Docker** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | `curl -fsSL https://get.docker.com \| sh` | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Runs PostgreSQL containers |
| **Ollama** | `curl -fsSL https://ollama.com/install.sh \| sh` | `curl -fsSL https://ollama.com/install.sh \| sh` | [ollama.com](https://ollama.com) | Local LLM + embeddings |
| **uv** | auto-installed by script | auto-installed by script | auto-installed by script | Python package manager |

> **Python 3.13** is **not** required — uv downloads and manages the correct Python version automatically.

> **Linux Docker note:** after installing Docker Engine, add your user to the docker group so you can run containers without `sudo`:
> ```bash
> sudo usermod -aG docker $USER && newgrp docker
> ```

---

## Install

### One-command install

Clone the repo, then run the install script for your platform. Each script:
1. Installs **uv** (if not already present)
2. Copies `.env.sample` → `.env` (if `.env` does not exist)
3. Runs `uv sync --extra all` — creates `.venv/` with every feature installed
4. Starts the **pgvector** container (`docker compose up -d pgvector`) if Docker is running
5. Pulls Ollama models (`llama3.1:8b`, `nomic-embed-text`) if Ollama is running
6. Pre-downloads the **CrossEncoder reranker model** (`BAAI/bge-reranker-base`, ~1.1 GB) if sentence-transformers was installed

**Windows (PowerShell):**
```powershell
.\install.ps1
```

> If you see a script-execution error, run this once in an elevated PowerShell, then retry:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Or bypass per-session: `powershell -ExecutionPolicy Bypass -File install.ps1`

**macOS / Linux (Bash):**
```bash
git clone <repo-url> && cd pydantic-ai-experiments
chmod +x install.sh && ./install.sh
```

---

### Interactive install

Add `--interactive` (Bash) or `-Interactive` (PowerShell) to step through each stage with prompts — choose which extras to install, whether to pull models, and whether to pre-download the reranker.

```powershell
# Windows
.\install.ps1 -Interactive

# macOS / Linux
./install.sh --interactive
```

---

### Manual install

If you prefer full control, follow the steps for your platform.

**macOS / Linux:**
```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env   # or open a new terminal

# 2. Clone the repo
git clone <repo-url> && cd pydantic-ai-experiments

# 3. Create and edit config
cp .env.sample .env
# edit .env — set DATABASE_URL, LLM_*, EMBEDDING_*

# 4. Install Python packages
uv sync --extra all

# 5. Start PostgreSQL (pgvector)
docker compose up -d pgvector

# 6. Pull Ollama models (start Ollama first: ollama serve)
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

# 7. Pre-download reranker model (optional, ~1.1 GB)
uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# 8. Validate the setup
uv run python -m rag.main --validate
```

**Windows (PowerShell):**
```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Open a new PowerShell window after this step

# 2. Clone the repo
git clone <repo-url>; cd pydantic-ai-experiments

# 3. Create and edit config
Copy-Item .env.sample .env
# Edit .env — set DATABASE_URL, LLM_*, EMBEDDING_*

# 4. Install Python packages
uv sync --extra all

# 5. Start PostgreSQL (pgvector)
docker compose up -d pgvector

# 6. Pull Ollama models (start Ollama first: ollama serve)
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

# 7. Pre-download reranker model (optional, ~1.1 GB)
uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# 8. Validate the setup
uv run python -m rag.main --validate
```

---

### Optional extras

The package is split into a small core and named optional extras. Install only what you need:

```bash
uv sync --extra ingestion     # core + Docling document pipeline (required for --ingest)
uv sync --extra audio         # Whisper ASR — also requires FFmpeg in PATH (see below)
uv sync --extra reranker      # CrossEncoder reranking (sentence-transformers)
uv sync --extra ui            # Streamlit chat interface
uv sync --extra observability # Langfuse tracing
uv sync --extra mcp           # MCP server (FastMCP, stdio transport)
uv sync --extra mem0          # Per-user memory layer (mem0ai)
uv sync --extra nl2sql        # NL-to-SQL query parsing (sqlglot)
uv sync --extra all           # Everything — recommended for development (default)
```

> Extras can be re-run at any time to add a feature to an existing environment. uv only installs what is missing.

**FFmpeg** is required by the `audio` extra (Whisper cannot transcribe without it):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Fedora / RHEL
sudo dnf install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Windows (winget)
winget install ffmpeg
```

Verify it is in PATH: `ffmpeg -version`

---

## Configuration

Edit `.env` after install. The minimum required fields are:

```bash
# PostgreSQL — created by docker compose up -d pgvector
DATABASE_URL=postgresql://rag_user:rag_pass@localhost:5434/rag_db

# LLM — Ollama (local, no API key needed)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Embeddings — Ollama (768 dimensions, matches nomic-embed-text)
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_DIMENSION=768
```

See `.env.sample` for all settings: Apache AGE connection, Langfuse keys, Mem0, reranker, VLM pipeline, rate-limiting, and timeout knobs.

**Feature flags** — all off by default, enable in `.env`:

| Setting | Default | What it enables |
|---------|---------|----------------|
| `LANGFUSE_ENABLED=true` | false | Langfuse tracing |
| `RERANKER_ENABLED=true` | false | CrossEncoder reranking |
| `MEM0_ENABLED=true` | false | Per-user memory |
| `VLM_ENABLED=true` | false | Vision-language model for PDF images |

---

## Quick start

```bash
# 1. Validate the config and database connection
uv run python -m rag.main --validate

# 2. Ingest the bundled sample documents
uv run python -m rag.main --ingest --documents rag/documents

# 3. Run fast unit tests (no external dependencies, < 10 s)
uv run pytest rag/tests/core/ -v

# 4. Run the full test suite
uv run pytest rag/tests/ -v
```

---

## Running the system

```bash
# REST API  →  http://localhost:8000/docs
uv run uvicorn rag.api.app:app --reload

# Streamlit chat UI
uv run streamlit run rag/app/streamlit/streamlit_app.py

# Streamlit UI with Mem0 user memory
uv run streamlit run rag/app/streamlit/streamlit_mem0_app.py

# MCP server (stdio transport — connect from Claude Desktop or any MCP client)
uv run python -m rag.mcp.server
```

Activate the venv once to drop `uv run` from every command:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1
```

---

## Knowledge Graph

Apache AGE is optional. Start it alongside the RAG database:

```bash
docker compose up -d age age-viewer
```

Add to `.env`:
```bash
KG_BACKEND=age
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
AGE_GRAPH_NAME=legal_graph
```

Open the graph viewer at **http://localhost:3001** (use Chrome).

Connection settings for the viewer:
- Host: `age` · Port: `5432`
- Database: `legal_graph`
- User: `age_user` · Password: `age_pass`
- Graph Path: `legal_graph`

See [`kg/docs/GRAPH_VIEWER.md`](kg/docs/GRAPH_VIEWER.md) for Cypher query examples.

---

## Development

```bash
# Lint and auto-fix
uv run ruff check --fix rag/ && uv run ruff format rag/

# Type check
uv run mypy rag/

# Unit tests only (no external deps)
uv run pytest rag/tests/ -m "not integration" -v

# Integration tests (requires PostgreSQL + Ollama)
uv run pytest rag/tests/ -v

# Specific category
uv run pytest rag/tests/core/ -v          # config + model tests
uv run pytest rag/tests/storage/ -v       # DB layer (needs PostgreSQL)
uv run pytest rag/tests/agent/ -v         # RAG agent + API + MCP tests
uv run pytest rag/tests/retrieval/ -v     # retrieval quality metrics
```

---

## Architecture docs

| File | Contents |
|------|---------|
| `rag/docs/ARCHITECTURE_SUMMARY.md` | Single-page system overview — start here |
| `rag/docs/ARCHITECTURE.md` | Ingestion + retrieval pipeline diagrams, DB schema, API endpoints |
| `rag/docs/RETRIEVAL_PIPELINE.md` | Hybrid search (vector + text + RRF), caching, reranking |
| `rag/docs/RAG.md` | Deep dive on RAG techniques |
| `kg/docs/KG_INGESTION_PIPELINE.md` | KG entity extraction pipeline |
| `kg/docs/KG_RETRIEVAL_PIPELINE.md` | NL→Cypher retrieval pipeline |
| `kg/docs/GRAPH_VIEWER.md` | AGE Viewer setup and Cypher queries |
| `nl2sql/docs/ARCHITECTURE.md` | NL→SQL pipeline design |
| `RAGV2_DESIGN.md` | Enterprise RAG v2 architecture proposal |
| `CLAUDE.md` | Dev conventions, project structure, test guide |
| `TESTS.md` | Full test suite documentation |
