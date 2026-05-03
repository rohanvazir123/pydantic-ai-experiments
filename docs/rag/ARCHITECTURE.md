# RAG + PDF Question Generator — Architecture

## Overview

Agentic Retrieval-Augmented Generation over ingested documents (CUAD legal contracts).
A Pydantic AI agent orchestrates retrieval tools (vector search, knowledge graph) and synthesises answers with an LLM.
A separate PDF question generator pipeline produces MCQs from ingested documents.

## Stack

| Layer | Technology |
|---|---|
| LLM / Embeddings | Ollama (local) or any OpenAI-compatible API |
| Vector store | PostgreSQL + pgvector (Neon or local Docker) |
| Knowledge graph | Apache AGE (PostgreSQL extension, port 5433) |
| Agent framework | Pydantic AI |
| Ingestion | Docling (PDF → structured chunks) |
| Observability | Langfuse |
| User memory | Mem0 (backed by same PostgreSQL) |
| UI | Streamlit (`apps/rag/streamlit_app.py`) |
| REST API | FastAPI (`apps/rag/api.py`) |

## Components

```
apps/rag/
├── streamlit_app.py   — Chat UI with streaming tool call display
└── api.py             — FastAPI: /health, /v1/chat, /v1/chat/stream,
                         /v1/retrieve, /v1/ingest

rag/
├── config/settings.py         — All config (pydantic-settings, .env)
├── agent/
│   ├── rag_agent.py           — Pydantic AI agent, RAGState, tools
│   └── agent_main.py          — stream_agent_interaction helper
├── ingestion/
│   └── pipeline.py            — DocumentIngestionPipeline (Docling → chunks → embed)
├── retrieval/
│   └── retriever.py           — HyDE + hybrid search + LLM reranker
├── storage/
│   └── vector_store/
│       └── postgres.py        — PostgresHybridStore (asyncpg, pgvector)
├── knowledge_graph/
│   ├── age_graph_store.py     — AgeGraphStore (Apache AGE via asyncpg)
│   └── pipeline.py            — KG extraction from ingested chunks
├── memory/
│   └── mem0_store.py          — Mem0Store (per-user episodic memory)
└── observability/
    └── langfuse_tracer.py     — Langfuse span helpers
```

## Data Flow — Chat Query

```
User question
    │
    ▼
RAGState (lazy-init retriever + AGE store)
    │
    ▼
Pydantic AI agent
    ├── search_knowledge_base(query)
    │       │
    │       ▼
    │   Retriever.retrieve(query, search_type="hybrid")
    │       ├── HyDE: LLM generates hypothetical answer → embed
    │       ├── Semantic search (pgvector cosine)
    │       ├── Full-text search (tsvector BM25)
    │       ├── RRF fusion
    │       └── LLM reranker (parallel scoring)
    │
    ├── search_knowledge_graph(query)
    │       └── AgeGraphStore.search_as_context(query)
    │
    └── run_graph_query(cypher)
            └── AgeGraphStore.run_cypher_query(cypher)
    │
    ▼
LLM synthesis
    │
    ▼
Response (streamed via SSE or Streamlit)
```

## Data Flow — Ingestion

```
PDF files (rag/documents/)
    │
    ▼
DocumentIngestionPipeline.ingest_documents()
    ├── Docling: PDF → structured markdown
    ├── Chunking (RecursiveCharacterSplitter)
    ├── Embedding (OpenAI-compatible API)
    └── PostgresHybridStore.upsert_chunks() (batch executemany)
    │
    ▼
KG pipeline (if KG_BACKEND=age)
    ├── LLM entity extraction per chunk
    ├── AgeGraphStore.upsert_entity() per entity
    └── AgeGraphStore.add_relationship() per relation
```

## Key Configuration (`.env`)

```
DATABASE_URL=postgresql://...          # pgvector store
AGE_DATABASE_URL=postgresql://...      # Apache AGE (port 5433)
KG_BACKEND=age                         # age | postgres
LLM_PROVIDER=ollama                    # ollama | openai | anthropic
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
LANGFUSE_PUBLIC_KEY=...
MEM0_ENABLE=true
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB, embedding API, LLM connectivity |
| POST | `/v1/chat` | Full agent run (tool calls + synthesis) |
| POST | `/v1/chat/stream` | SSE-streamed agent response |
| POST | `/v1/retrieve` | Raw retrieval (no LLM synthesis) |
| POST | `/v1/ingest` | Trigger document ingestion |

## Running

```bash
# UI
streamlit run apps/rag/streamlit_app.py

# API
uvicorn apps.rag.api:app --port 8000 --reload

# Ingest documents
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_folder": "rag/documents"}'
```
