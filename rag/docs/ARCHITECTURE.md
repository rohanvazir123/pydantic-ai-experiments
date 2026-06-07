# RAG System — Architecture

Agentic RAG over PostgreSQL/pgvector. A Pydantic AI agent orchestrates hybrid retrieval and knowledge graph tools to synthesise answers from ingested documents.

---

## Table of Contents

- [Stack](#stack)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Query Flow (Agent)](#query-flow-agent)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | Pydantic AI |
| LLM / Embeddings | Ollama (local) or any OpenAI-compatible API |
| Vector store | PostgreSQL + pgvector |
| Knowledge graph | Apache AGE (PostgreSQL extension, port 5433) |
| Ingestion | Docling (multi-format → structured chunks; VlmPipeline optional) |
| User memory | Mem0 (backed by same PostgreSQL) |
| Observability | Langfuse (optional) |
| UI | Streamlit — `rag/app/streamlit/streamlit_app.py` |
| REST API | FastAPI — `rag/app/rest_api/api.py` |

---

## Ingestion Pipeline

```
  Documents (rag/documents/)
  PDF · DOCX · PPTX · MD · TXT · Audio
          │
          ▼
  ┌───────────────────────────────────┐
  │   _compute_file_hash()  (MD5)     │
  │   Incremental: skip if unchanged  │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │   _read_document()                │
  │                                   │
  │   .pdf → _get_pdf_converter()     │
  │     VLM_ENABLED=false             │
  │       → StandardPdfPipeline       │
  │         (layout + OCR)            │
  │     VLM_ENABLED=true              │
  │       → VlmPipeline               │
  │         page image → Ollama       │
  │         Qwen2.5-VL → markdown     │
  │         with [Figure:...] tags    │
  │   .docx/.pptx/.xlsx/.html/.md     │
  │       → _get_standard_converter() │
  │         (text layer — no VLM,     │
  │          no OCR)                  │
  │   Audio → Docling ASR + Whisper   │
  │   .txt  → direct read             │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │   chunker.chunk_document()        │
  │                                   │
  │   Docling HybridChunker           │
  │   · token-aware splitting         │
  │   · preserves heading context     │
  │   Fallback: sliding-window        │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │   embedder.embed_chunks()         │
  │                                   │
  │   Batched POST → /v1/embeddings   │
  │   nomic-embed-text → 768-dim      │
  │   Async LRU cache (1000 entries)  │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │   PostgresHybridStore             │
  │                                   │
  │   INSERT INTO documents           │
  │   executemany → chunks            │
  │   (single batch per document)     │
  └───────────────────────────────────┘

  Key file: rag/ingestion/pipeline.py
```

---

## Retrieval Pipeline

```
  User query
      │
      ▼
  ┌───────────────────────────────────┐
  │   ResultCache (LRU, TTL 5 min)    │
  │   Key: sha256(query:type:n)[:24]  │
  │   Hit → return immediately        │
  └───────────────┬───────────────────┘
                  │ miss
                  ▼
  ┌───────────────────────────────────┐
  │   Query embedding                 │
  │                                   │
  │   embed raw query                 │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────────────────────┐
  │   Search  (asyncio.gather for hybrid)             │
  │                                                   │
  │   semantic ──▶ ORDER BY embedding <=> $1 LIMIT N  │
  │   text     ──▶ content_tsv @@ plainto_tsquery()   │
  │   fuzzy    ──▶ word_similarity($1, content) > 0.2 │
  │                         │                         │
  │                         ▼                         │
  │              RRF merge  (k = 60)  — all 3 legs    │
  │              score = Σ 1 / (k + rank_i)           │
  └───────────────────────────────────┬───────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────┐
  │   Rerank  (optional, RERANKER_ENABLED=true)       │
  │                                                   │
  │   Over-fetch:  n × RERANKER_OVERFETCH_FACTOR      │
  │                                                   │
  │   llm          → parallel LLM scoring             │
  │                  asyncio.gather → sort → trim      │
  │   cross_encoder→ sentence-transformers (local)    │
  │                  → sort → trim                    │
  └───────────────────────────────────┬───────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────┐
  │   Score filter                    │
  │   Drop chunks < MIN_RELEVANCE     │
  │   (default 0.4)                   │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │   Cache write                     │
  │   Return list[SearchResult]       │
  └───────────────────────────────────┘

  Key file: rag/retrieval/retriever.py
```

---

## Query Flow (Agent)

```
  User question
        │
        ▼
  ┌─────────────────────────────────────────────────┐
  │  Pydantic AI Agent   rag/agent/rag_agent.py     │
  │                                                 │
  │  ┌─ search_knowledge_base(query) ─────────────┐ │
  │  │  Retriever.retrieve_as_context()           │ │
  │  │  + Mem0Store.get_context_string()          │ │
  │  └────────────────────────────────────────────┘ │
  │                                                 │
  │  ┌─ search_knowledge_graph(query) ────────────┐ │
  │  │  AgeGraphStore.search_entities()           │ │
  │  │  + get_related_entities()                  │ │
  │  └────────────────────────────────────────────┘ │
  └──────────────────────────┬──────────────────────┘
                             │
                             ▼
                       LLM synthesis
                             │
                             ▼
               Response  (JSON · SSE · Streamlit)
```

---

## Database Schema

```
  documents
  ├── id           UUID PK
  ├── title        TEXT
  ├── source       TEXT UNIQUE      ← dedup key
  ├── content      TEXT
  ├── metadata     JSONB            ← file hash, format
  └── created_at   TIMESTAMPTZ

  chunks
  ├── id           UUID PK
  ├── document_id  UUID FK → documents CASCADE
  ├── content      TEXT
  ├── embedding    vector(768)      ← HNSW index (cosine)
  ├── chunk_index  INTEGER
  ├── metadata     JSONB
  ├── token_count  INTEGER
  └── content_tsv  tsvector GENERATED  ← GIN index (full-text)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | DB + embedding API + LLM connectivity |
| POST | `/v1/chat` | Full agent run — tool calls + synthesis |
| POST | `/v1/chat/stream` | SSE-streamed agent response |
| POST | `/v1/retrieve` | Raw retrieval, no LLM synthesis |
| POST | `/v1/ingest` | Trigger document ingestion |

Run: `uvicorn rag.app.rest_api.api:app --port 8000 --reload`
