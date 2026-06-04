# RAG System — Architecture Summary

Agentic RAG over PostgreSQL/pgvector. Read this first; see linked docs for depth.

---

## Table of Contents

- [What It Does](#what-it-does)
- [System Overview](#system-overview)
- [AI Service Calls](#ai-service-calls)
- [Key Files](#key-files)
- [Further Reading](#further-reading)

---

## What It Does

1. **Ingests** documents (PDF, DOCX, audio, Markdown) → chunks → embeddings → PostgreSQL
2. **Retrieves** relevant chunks via hybrid search (vector + full-text, fused with RRF)
3. **Answers** questions through a Pydantic AI agent with retrieval and knowledge graph tools
4. **Remembers** users across sessions via Mem0 (pgvector-backed)
5. **Observes** itself with optional Langfuse tracing

---

## System Overview

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Documents                          User Query                      │
  │  PDF · DOCX · Audio · MD            CLI · Streamlit · API · MCP    │
  └──────────────┬──────────────────────────────────┬───────────────────┘
                 │                                  │
    INGESTION    │                                  │   RETRIEVAL
                 ▼                                  ▼
  ┌──────────────────────────┐      ┌───────────────────────────────────┐
  │  Docling                 │      │  Pydantic AI Agent                │
  │  · PDF / DOCX / audio    │      │  rag/agent/rag_agent.py           │
  │  · HybridChunker         │      │                                   │
  │  · title extraction      │      │  search_knowledge_base()  ──────┐ │
  └──────────────┬───────────┘      │  search_knowledge_graph() ────┐ │ │
                 │                  └───────────────────────────────│─│─┘
                 ▼                                                  │ │
  ┌──────────────────────────┐                                      │ │
  │  Embedder                │      ┌───────────────────────────────│─┘
  │  nomic-embed-text 768-dim│      │  Retriever                    │
  │  batched API calls       │      │  rag/retrieval/retriever.py   │
  │  async LRU cache         │      │                               │
  └──────────────┬───────────┘      │  1. Cache check (LRU 5 min)   │
                 │                  │  2. Embed query               │
                 ▼                  │  3. Semantic search (cosine)  │
  ┌──────────────────────────┐      │  4. Full-text search (BM25)   │
  │  PostgresHybridStore     │◀─────│  5. RRF merge  (k=60)         │
  │                          │      │  6. Rerank  (LLM / CrossEnc.) │
  │  documents  ← metadata   │─────▶│  7. Score filter              │
  │  chunks     ← vector(768)│      └───────────────────────────────┘
  │             + tsvector   │
  │  mem0_memories           │      ┌───────────────────────────────┐
  └──────────────────────────┘      │  AgeGraphStore                │
                                    │  kg/age_graph_store.py        │
                                    │                               │
                                    │  search_entities()            │
                                    │  get_related_entities()       │
                                    │  run_cypher_query()           │
                                    └───────────────────────────────┘
```

---

## AI Service Calls

| Caller | Service | When |
|--------|---------|------|
| Agent | LLM | Every query |
| Retriever — embed query | Embeddings | Always |
| Retriever — rerank | LLM or CrossEncoder | `RERANKER_ENABLED=true` |
| Mem0Store | LLM + Embeddings | Every `add()` call |
| Ingestion pipeline | Embeddings | Every chunk |
| Ingestion pipeline — VLM | VLM (Qwen2.5-VL via Ollama) | `VLM_ENABLED=true`, every PDF page |

---

## Key Files

| Purpose | File |
|---------|------|
| Settings | `rag/config/settings.py` |
| Ingestion pipeline | `rag/ingestion/pipeline.py` |
| Docling chunker | `rag/ingestion/chunkers/docling.py` |
| Embedder | `rag/ingestion/embedder.py` |
| Data models | `rag/ingestion/models.py` |
| PostgreSQL store | `rag/storage/vector_store/postgres.py` |
| Retriever | `rag/retrieval/retriever.py` |
| Rerankers | `rag/retrieval/rerankers.py` |
| RAG agent + RAGState | `rag/agent/rag_agent.py` |
| Agent prompts | `rag/agent/prompts.py` |
| REST API | `rag/app/rest_api/api.py` |
| MCP server | `rag/mcp/server.py` |
| CLI entry point | `rag/main.py` |
| Mem0 memory layer | `rag/memory/mem0_store.py` |
| Knowledge graph store | `kg/age_graph_store.py` |

---

## Further Reading

| Doc | What's in it |
|-----|-------------|
| `rag/docs/ARCHITECTURE.md` | Full ingestion + retrieval pipelines with query flow |
| `rag/docs/INGESTION_PIPELINE.md` | Ingestion pipeline step-by-step |
| `rag/docs/RETRIEVAL_PIPELINE.md` | Retrieval pipeline step-by-step |
| `rag/docs/DATASTORE_GUIDE.md` | PostgreSQL schema, indexes, SQL examples |
| `rag/docs/CALL_GRAPH.md` | Method-level call graphs |
| `rag/docs/RAG.md` | Deep dive on chunking, reranking, caching strategies |
| `kg/docs/KG_FAQ.md` | Knowledge graph design decisions and eval results |
