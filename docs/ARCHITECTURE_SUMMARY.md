# Architecture Summary

One-stop reference for the entire RAG system. Read this first; dive into the other docs for deeper detail.

## Table of Contents

- [1. What This System Does](#1-what-this-system-does)
- [2. Top-Level Architecture](#2-top-level-architecture)
  - [RAG Agent Workflow](#rag-agent-workflow)
  - [PDF Question Generator Workflow](#pdf-question-generator-workflow)
- [3. Data Schema](#3-data-schema)
- [4. Ingestion Pipeline](#4-ingestion-pipeline)
- [5. Retrieval Pipeline](#5-retrieval-pipeline)
- [6. RAG Agent](#6-rag-agent)
- [7. Mem0 Memory Layer](#7-mem0-memory-layer)
- [8. Langfuse Observability](#8-langfuse-observability)
- [9. Streamlit Apps](#9-streamlit-apps)
- [10. Configuration (.env Quick Reference)](#10-configuration-env-quick-reference)
- [11. Key File Map](#11-key-file-map)
- [12. How to Run](#12-how-to-run)
- [13. Further Reading](#13-further-reading)

---

## 1. What This System Does

An **agentic RAG (Retrieval-Augmented Generation)** system that:
1. **Ingests** documents (PDF, DOCX, audio, Markdown, …) into PostgreSQL/pgvector
2. **Retrieves** relevant chunks via hybrid search (vector + full-text, fused with RRF)
3. **Answers** questions through a Pydantic AI agent that calls the retrieval tool
4. **Remembers** users across sessions via Mem0 (pgvector-backed user memory)
5. **Observes** itself with optional Langfuse tracing
6. **Deep-processes PDFs** via a separate PDF Question Generator workflow — uses RAGAnything/MinerU to extract multimodal content (tables, images, equations), processes each with specialised LLM/vision processors, then generates structured Q&A pairs stored in their own PostgreSQL tables

---

## 2. Top-Level Architecture

### RAG Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Interfaces                                                         │
│  CLI agent · Streamlit RAG app · Streamlit Mem0 app                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────┐
                 │  Pydantic AI Agent               │
                 │  rag/agent/rag_agent.py           │
                 │  search_knowledge_base()          │
                 │  remember_user_context()          │
                 └──────────┬──────────────┬─────────┘
                            │              │
                            ▼              ▼
         ┌──────────────────────┐   ┌──────────────────────────────┐
         │  Retriever           │   │  Mem0Store                   │
         │  rag/retrieval/      │   │  memory/mem0_store.py        │
         │                      │   │                              │
         │  1. HyDE (opt.)      │   │  pgvector similarity search  │
         │  2. Embed query      │   │  over user memory facts      │
         │  3. Hybrid search    │   │                              │
         │     (RRF)            │   │  · fact extraction           │
         │  4. Rerank (opt.)    │   │  · embed memories            │
         │     LLM or CrossEnc  │   │                              │
         └──────────┬───────────┘   └──────────────┬───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────────┐
         │  PostgreSQL / Neon  (pgvector extension)                 │
         │  ├── documents        ← metadata + file hash             │
         │  ├── chunks           ← vector(768) + tsvector           │
         │  └── mem0_memories    ← user memory facts                │
         └──────────────────────────────────────────────────────────┘
                                    ▲
                                    │  (ingest)
         ┌──────────────────────────┴───────────────────────────────┐
         │  Ingestion Pipeline  (rag/ingestion/pipeline.py)         │
         │  Docling (PDF/DOCX/audio) → chunk → embed → store       │
         │  Note: MinerU/RAGAnything is PDF Question Generator only │
         └──────────────────────────────────────────────────────────┘
```

**AI service calls** — which component calls what and when:

| Caller | Service | Condition |
|--------|---------|-----------|
| Agent | LLM | Every query (inference) |
| Retriever – HyDE | LLM | `HYDE_ENABLED=true` |
| Retriever – Embed query | Embeddings | Always |
| Retriever – Rerank | LLM | `RERANKER_ENABLED=true` + `RERANKER_TYPE=llm` (default) |
| Retriever – Rerank | CrossEncoder (local) | `RERANKER_ENABLED=true` + `RERANKER_TYPE=cross_encoder` |
| Mem0Store – fact extraction | LLM | Every `add()` call with `infer=True` |
| Mem0Store – embed memories | Embeddings | Every `add()` call |
| Ingestion Pipeline | Embeddings | Every chunk during ingest |

**CrossEncoder** runs entirely locally via `sentence-transformers` — no API call.

**LLM default**: Ollama `llama3.1:8b` (swap: OpenAI, Anthropic, any OpenAI-compatible endpoint)

**Embeddings default**: Ollama `nomic-embed-text` 768-dim (swap: OpenAI `text-embedding-3-*`)

### PDF Question Generator Workflow

A separate ingestion workflow that deep-processes PDFs into structured Q&A pairs using multimodal content understanding.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Entry: python -m rag.ingestion.processors.pdf_question_generator    │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RAGAnything  (primary path)                                         │
│  MinerU parser → extracts structured content from PDF               │
│  ├── text blocks                                                     │
│  ├── tables          ──► TableProcessor                              │
│  ├── images          ──► ImageProcessor                              │
│  └── equations       ──► EquationProcessor                           │
└──────────────────┬────────────┬────────────────┬─────────────────────┘
                   │            │                │
                   ▼            ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Table        │ │ Image        │ │ Equation     │
        │ Processor    │ │ Processor    │ │ Processor    │
        │              │ │              │ │              │
        │ LLM call     │ │ Vision model │ │ LLM call     │
        │ → summary    │ │ → describe   │ │ → LaTeX +    │
        │ → key facts  │ │ → objects    │ │   meaning    │
        │ → insights   │ │ → context    │ │              │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LightRAG chunk store  →  extract_chunks_from_lightrag()             │
│  format_chunks_as_context()  →  combined context string              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LLM  (QUESTION_GENERATION_SYSTEM_PROMPT)                            │
│  Input: full multimodal context                                      │
│  Output JSON: { questions[], entities[], summary }                   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PDFQuestionStore  (PostgreSQL)                                      │
│  ├── pdf_documents   ← document metadata                            │
│  ├── pdf_questions   ← Q&A pairs + embeddings (searchable)          │
│  └── pdf_chunks      ← raw chunks + embeddings (searchable)         │
└──────────────────────────────────────────────────────────────────────┘

Fallback (if RAGAnything / MinerU unavailable):
  Docling DocumentConverter (text-only) → same LLM call → same store
```

**External AI services used by PDF Question Generator**:

| Step | Service |
|------|---------|
| TableProcessor, EquationProcessor | LLM (same `LLM_MODEL` / `LLM_BASE_URL`) |
| ImageProcessor | Vision model (`VISION_MODEL` — default `gpt-4o`) |
| Q&A generation | LLM |
| Embedding questions & chunks | Embeddings (`EMBEDDING_MODEL`) |

---

## 3. Data Schema

### `documents` table
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID PK | `gen_random_uuid()` |
| `title` | TEXT | Extracted from doc heading or filename |
| `source` | TEXT UNIQUE | File path (used for incremental dedup) |
| `content` | TEXT | Full document text |
| `metadata` | JSONB | File hash, format, extra fields |
| `created_at` | TIMESTAMPTZ | |

### `chunks` table
| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID PK | |
| `document_id` | UUID FK → documents(id) CASCADE | |
| `content` | TEXT | Chunk text (≤512 tokens) |
| `embedding` | vector(768) | nomic-embed-text; 1536 for OpenAI models |
| `chunk_index` | INTEGER | Order within document |
| `metadata` | JSONB | heading context, page, etc. |
| `token_count` | INTEGER | |
| `content_tsv` | tsvector GENERATED | Full-text index column (auto-updated) |

**Indexes**:
- `IVFFlat` on `embedding` (lists=100, probes=10) — approximate cosine similarity
- `GIN` on `content_tsv` — full-text keyword search
- `B-tree` on `document_id`, `source`

---

## 4. Ingestion Pipeline

**Entry**: `python -m rag.main --ingest --documents rag/documents`

**Key file**: `rag/ingestion/pipeline.py` → `DocumentIngestionPipeline`

```
For each document file:
  1. _compute_file_hash()          MD5 of file bytes
  2. [incremental] check hash      skip if unchanged; delete+re-ingest if changed
  3. _read_document()
       PDF/DOCX/…  → Docling DocumentConverter  (ML model, cached per pipeline)
       Audio        → Docling ASR pipeline + Whisper
       MD/TXT       → direct read
  4. _extract_title()              first "# " heading or filename
  5. chunker.chunk_document()      Docling HybridChunker (token-aware, heading-context)
                                   fallback: sliding-window _simple_fallback_chunk()
  6. embedder.embed_chunks()       batch OpenAI-compatible API call
  7. store.save_document()         INSERT INTO documents
  8. store.add(chunks)             executemany INSERT INTO chunks  (single batch)
```

**Modes**:
- `--ingest` (full): TRUNCATE both tables, re-ingest everything
- `--ingest --no-clean` (incremental): hash-based skip/update/delete

---

## 5. Retrieval Pipeline

**Key file**: `rag/retrieval/retriever.py` → `Retriever.retrieve()`

```
1. Cache check         ResultCache (LRU, TTL=5min, max=100 entries)
                       key = sha256(query:search_type:match_count)[:24]
2. Query embedding     [HyDE off] embed_query(query)  → async_lru cache
                       [HyDE on]  LLM generates hypothetical answer doc
                                  → embed that doc instead for better recall
3. Over-fetch          fetch_count = match_count × overfetch_factor (if reranking)
4. Search
   "semantic"  → ORDER BY embedding <=> $1::vector LIMIT N
   "text"      → WHERE content_tsv @@ plainto_tsquery('english', $1)
   "hybrid"    → asyncio.gather(semantic, text) → RRF merge (k=60)
5. Rerank (optional)
   "llm"           → parallel LLM scoring with asyncio.gather, then sort & trim
   "cross_encoder" → sentence-transformers CrossEncoder, then sort & trim
6. Cache + return list[SearchResult]
```

### Feature flags (`.env`)
| Flag | Default | Effect |
|------|---------|--------|
| `HYDE_ENABLED` | `false` | Embed hypothetical answer instead of raw query |
| `RERANKER_ENABLED` | `false` | Over-fetch then rerank results |
| `RERANKER_TYPE` | `llm` | `llm` or `cross_encoder` |
| `RERANKER_MODEL` | `` | Defaults to `LLM_MODEL` / `BAAI/bge-reranker-base` |
| `RERANKER_OVERFETCH_FACTOR` | `3` | Fetch 3× before reranking |

### Caching layers
| Cache | Mechanism | Key | TTL | Capacity |
|-------|-----------|-----|-----|----------|
| Embedding | `@alru_cache` | `(text, model)` | None | 1000 entries |
| Search results | `ResultCache` (OrderedDict LRU) | `(query, type, count)` | 5 min | 100 entries |

---

## 6. RAG Agent

**Key file**: `rag/agent/rag_agent.py`

**`RAGState`** — Pydantic BaseModel passed as `deps` to the agent; holds lazy-initialized resources:
```python
_store:       PostgresHybridStore  (PrivateAttr)
_retriever:   Retriever            (PrivateAttr)
_mem0:        Mem0Store            (PrivateAttr, if mem0_enabled)
_initialized: bool                 (PrivateAttr)
_init_lock:   asyncio.Lock         (PrivateAttr)
```

**`search_knowledge_base` tool** — the agent's only retrieval tool:
1. `RAGState.get_retriever()` — lazy-init store + retriever (thread-safe via asyncio.Lock)
2. `retriever.retrieve_as_context(query)` — hybrid search → formatted chunk string
3. `mem0_store.get_context_string(query, user_id)` — user memory facts
4. Return combined context to LLM

**`traced_agent_run(query, user_id, session_id, message_history)`**:
- Creates `RAGState(user_id=user_id)`, passes as `deps=state`
- Optional Langfuse trace (via `_trace_context` ContextVar for coroutine isolation)
- `finally`: closes state, flushes Langfuse

---

## 7. Mem0 Memory Layer

**Key file**: `rag/memory/mem0_store.py` → `Mem0Store`, `create_mem0_store()`

- Backed by **same PostgreSQL database**, table `mem0_memories` (auto-created)
- LLM and embedder providers are **dynamic** — reads `settings.llm_provider` / `settings.embedding_provider` to build `mem0ai.Memory` config (ollama or openai-compatible)

**Key methods**:
| Method | What it does |
|--------|-------------|
| `add(text, user_id, infer=True)` | LLM extracts facts, stores as pgvector |
| `search(query, user_id, limit)` | pgvector similarity search |
| `get_context_string(query, user_id)` | Search + format as `"## User Context\n- fact..."` |
| `get_all(user_id)` | All memories for user |
| `delete_all(user_id)` | Clear all memories for user |

**Enable**: `MEM0_ENABLED=true` in `.env`

---

## 8. Langfuse Observability

**Controlled by**: `LANGFUSE_ENABLED=true` (+ public/secret keys in `.env`)

- Traces are attached to the running coroutine via `_trace_context: ContextVar` — each concurrent agent run gets its own trace without thread-local collision
- `traced_agent_run` creates a trace per call; `search_knowledge_base` appends tool call spans
- `langfuse.flush()` called in `finally` block to ensure delivery

---

## 9. Streamlit Apps

### `streamlit_mem0_app.py` — Chat with Memory (simple)
- `@st.cache_resource`: `get_mem0_store()`, `get_agent()`
- Uses `Mem0Store.get_context_string()` to prepend user context to every prompt
- Sidebar: Clear Chat / Clear Memories (`mem0_store.delete_all`) / Show Memories (`mem0_store.get_all`)
- Run: `streamlit run streamlit_mem0_app.py`

### `rag/agent/streamlit_app.py` — Full RAG Chat (streaming)
- Streaming agent responses with live tool-call visibility
- Backed by the full Pydantic AI RAG agent (`rag/agent/rag_agent.py`)
- Run: `streamlit run rag/agent/streamlit_app.py`

---

## 10. Configuration (`.env` Quick Reference)

```bash
# Database
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

# LLM
LLM_PROVIDER=ollama          # or openai, anthropic, etc.
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_DIMENSION=768      # 768 for nomic, 1536 for OpenAI

# Search
DEFAULT_MATCH_COUNT=10
MAX_MATCH_COUNT=50

# HyDE
HYDE_ENABLED=false

# Reranker
RERANKER_ENABLED=false
RERANKER_TYPE=llm            # or cross_encoder
RERANKER_OVERFETCH_FACTOR=3

# Mem0
MEM0_ENABLED=false

# Langfuse
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

---

## 11. Key File Map

| Purpose | File |
|---------|------|
| Settings / env vars | `rag/config/settings.py` |
| Ingestion pipeline | `rag/ingestion/pipeline.py` |
| Docling chunker | `rag/ingestion/chunkers/docling.py` |
| Embedding generator | `rag/ingestion/embedder.py` |
| Data models | `rag/ingestion/models.py` |
| PostgreSQL store | `rag/storage/vector_store/postgres.py` |
| Retriever (search orchestrator) | `rag/retrieval/retriever.py` |
| HyDE processor | `rag/retrieval/query_processors.py` |
| Rerankers (LLM + CrossEncoder) | `rag/retrieval/rerankers.py` |
| RAG agent + RAGState | `rag/agent/rag_agent.py` |
| Agent system prompts | `rag/agent/prompts.py` |
| CLI entry point | `rag/main.py` |
| Mem0 memory layer | `rag/memory/mem0_store.py` |
| Streamlit (memory chat) | `streamlit_mem0_app.py` |
| Streamlit (RAG chat) | `rag/agent/streamlit_app.py` |
| CLI chat | `rag/agent/agent_main.py` |
| Tests | `rag/tests/` |

---

## 12. How to Run

```bash
# 1. Install
pip install -e .

# 2. Set up .env (copy from above template)

# 3. Ingest documents
python -m rag.main --ingest --documents rag/documents

# 4. Incremental re-ingest (only changed files)
python -m rag.main --ingest --documents rag/documents --no-clean

# 5. Run CLI agent
python -m rag.agent.agent_main

# 6. Run Streamlit (simple chat with memory)
streamlit run streamlit_mem0_app.py

# 7. Run Streamlit (full RAG agent)
streamlit run rag/agent/streamlit_app.py

# 8. Run tests
python -m pytest rag/tests/ -v

# 9. Lint
ruff check --fix rag/ && ruff format rag/
```

---

## 13. Further Reading

| Doc | What's in it |
|-----|-------------|
| `docs/RAG.md` | Deep-dive on every technique: chunking strategies, reranking, HyDE, Mem0, Langfuse, Knowledge Graph RAG, performance tuning, caching |
| `docs/DATASTORE_GUIDE.md` | PostgreSQL schema, indexes, SQL examples, pgvector setup |
| `docs/CALL_GRAPH.md` | Step-by-step call graphs for ingestion, retrieval, agent, Mem0, Streamlit |
| `docs/TESTS.md` | Test suite overview, what each test covers, how to run |
| `docs/CLAUDE.md` | Development conventions, code quality rules, common issues |
