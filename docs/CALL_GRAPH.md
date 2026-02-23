# Call Graphs

Call graphs for the main workflows in this project.

## Table of Contents

- [1. Document Ingestion](#1-document-ingestion)
- [2. Query & Retrieval](#2-query--retrieval)
- [3. RAG Agent (CLI)](#3-rag-agent-cli)
- [4. Mem0 Memory](#4-mem0-memory)
- [5. PDF Question Generator](#5-pdf-question-generator)
- [6. Streamlit App](#6-streamlit-app)
- [7. Architecture Overview](#7-architecture-overview)

---

## 1. Document Ingestion

> See [RAG.md](RAG.md) and [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: `python -m rag.main --ingest`

```
rag/main.py: main()
  ├── validate_config()
  │     └── load_settings()
  └── run_ingestion_pipeline()
        └── DocumentIngestionPipeline
              ├── __init__(config, documents_folder, clean_before_ingest)
              │     ├── load_settings()
              │     ├── create_chunker()           # DoclingHybridChunker
              │     ├── create_embedder()          # EmbeddingGenerator
              │     └── PostgresHybridStore()
              ├── initialize()
              │     └── store.initialize()
              │           ├── asyncpg.create_pool(DATABASE_URL)
              │           ├── CREATE EXTENSION IF NOT EXISTS vector
              │           ├── CREATE TABLE documents (...)
              │           ├── CREATE TABLE chunks (..., embedding vector(768))
              │           └── CREATE INDEX (IVFFlat, GIN, B-tree)
              ├── ingest_documents(progress_callback)
              │     ├── store.clean_collections()  # if --clean
              │     ├── _find_document_files()
              │     └── [for each file]:
              │           _ingest_single_document(file_path)
              │             ├── _read_document(file_path)
              │             │     ├── [PDF/DOCX/PPTX/HTML]  Docling DocumentConverter
              │             │     ├── [MP3/WAV/M4A/FLAC]    Whisper ASR via Docling
              │             │     └── [MD/TXT]              Direct file read
              │             ├── _extract_title()
              │             ├── _extract_document_metadata()
              │             │     └── _compute_file_hash()
              │             ├── chunker.chunk_document()
              │             │     └── DoclingHybridChunker
              │             │           ├── Docling HybridChunker (semantic splitting)
              │             │           └── _simple_fallback_chunk()  # sliding window
              │             ├── embedder.embed_chunks(chunks)
              │             │     └── generate_embeddings_batch(texts)
              │             │           ├── _cached_embed(text, model)  # async_lru cache
              │             │           └── openai.AsyncOpenAI.embeddings.create()
              │             ├── store.save_document(title, source, content, metadata)
              │             │     └── INSERT INTO documents
              │             └── store.add(chunks, document_id)
              │                   └── INSERT INTO chunks (with pgvector embedding)
              └── close()
                    └── store.close()  →  pool.close()
```

**Key files**:

| File | Class/Function | Role |
|------|---------------|------|
| `rag/ingestion/pipeline.py` | `DocumentIngestionPipeline` | Orchestrates full ingestion |
| `rag/ingestion/chunkers/docling.py` | `DoclingHybridChunker` | Smart document chunking |
| `rag/ingestion/embedder.py` | `EmbeddingGenerator` | Embedding generation with caching |
| `rag/storage/vector_store/postgres.py` | `PostgresHybridStore` | PostgreSQL/pgvector storage |
| `rag/ingestion/models.py` | `ChunkData`, `IngestionResult` | Data models |

---

## 2. Query & Retrieval

> See [RAG.md](RAG.md) and [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: `Retriever.retrieve(query)`

```
Retriever.retrieve(query, match_count, search_type="hybrid", use_cache=True)
  ├── _result_cache.get(query, search_type, match_count)  # LRU + 5min TTL
  │     └── cache hit? → return list[SearchResult]
  ├── embedder.embed_query(query, use_cache=True)
  │     ├── _cached_embed(query, model)  # async_lru cache (1000 entries)
  │     └── openai.AsyncOpenAI.embeddings.create()  →  list[float]
  ├── [search_type == "semantic"]
  │     └── store.semantic_search(query_embedding, match_count)
  │           └── SQL: SELECT ... ORDER BY embedding <=> $1::vector LIMIT $2
  ├── [search_type == "text"]
  │     └── store.text_search(query, match_count)
  │           └── SQL: SELECT ... WHERE content_tsv @@ plainto_tsquery('english', $1)
  ├── [search_type == "hybrid"]  (default)
  │     ├── asyncio.gather(
  │     │     semantic_search(embedding, match_count * 2),
  │     │     text_search(query, match_count * 2)
  │     │   )
  │     └── _reciprocal_rank_fusion([semantic_results, text_results])
  │           ├── RRF score = Σ 1/(k=60 + rank)  per result list
  │           ├── Deduplicate by chunk_id
  │           └── Sort descending by combined score  →  top match_count
  ├── _result_cache.set(query, search_type, match_count, results)
  └── return list[SearchResult]

Retriever.retrieve_as_context(query, match_count, search_type)
  └── retrieve(...)
        └── "\n\n---\n\n".join(f"[{r.document_title}]\n{r.content}" for r in results)
```

**Caching layers**:

| Cache | Location | Key | TTL | Max Size |
|-------|----------|-----|-----|----------|
| Embedding cache | `embedder.py` (`async_lru`) | `(text, model)` | None | 1000 |
| Result cache | `retriever.py` (`ResultCache`) | `(query, search_type, count)` | 5 min | 100 |

---

## 3. RAG Agent (CLI)

> See [RAG.md](RAG.md) for details.

**Entry point**: `python -m rag.agent.agent_main`

```
rag/agent/agent_main.py: agent_main()
  ├── RAGState()              # lazy-initialized container
  ├── StateDeps(state)
  └── [input loop]
        └── stream_agent_interaction(user_input, message_history, deps)
              └── _stream_agent(user_input, deps, message_history)
                    └── agent.iter(query, deps=deps, message_history=...)
                          └── [async for node in run]
                                ├── UserPromptNode
                                │     └── _debug_print()
                                ├── ModelRequestNode    # LLM decision
                                │   └── _handle_model_request_node(node, ctx)
                                │         └── node.stream(ctx)
                                │               ├── PartStartEvent
                                │               ├── PartDeltaEvent (TextPartDelta)
                                │               ├── FinalResultEvent
                                │               └── PartEndEvent
                                ├── CallToolsNode       # tool execution
                                │   └── _handle_tool_call_node(node, ctx)
                                │         └── node.stream(ctx)
                                │               ├── FunctionToolCallEvent
                                │               │     → search_knowledge_base(ctx, query, ...)
                                │               │         ├── RAGState.get_retriever()
                                │               │         │     ├── PostgresHybridStore.initialize()
                                │               │         │     └── Retriever(store, embedder)
                                │               │         ├── retriever.retrieve_as_context()
                                │               │         │     └── [see Query & Retrieval]
                                │               │         ├── mem0_store.get_context_string(query, user_id)
                                │               │         │     └── mem0.search(query, user_id)
                                │               │         └── return combined_context
                                │               └── FunctionToolResultEvent
                                ├── ModelRequestNode    # LLM final answer
                                └── EndNode
```

**RAGState lazy initialization** (first call to `get_retriever()`):

```
RAGState.get_retriever()
  ├── (first call) PostgresHybridStore().initialize()
  ├── EmbeddingGenerator()
  ├── Retriever(store, embedder)
  └── Mem0Store()  (if mem0_enabled=true)
```

**Key files**:

| File | Class/Function | Role |
|------|---------------|------|
| `rag/agent/rag_agent.py` | `agent` (PydanticAI `Agent`) | Agent instance with tools |
| `rag/agent/rag_agent.py` | `RAGState` | Lazy-init shared state (store + retriever + mem0) |
| `rag/agent/rag_agent.py` | `search_knowledge_base()` | Agent tool: retrieval + memory |
| `rag/agent/agent_main.py` | `stream_agent_interaction()` | Streams node events to console |
| `rag/agent/prompts.py` | `MAIN_SYSTEM_PROMPT` | LLM system prompt |

---

## 4. Mem0 Memory

> See [RAG.md §16](RAG.md#16-mem0-memory-layer) for details.

**Entry point**: `Mem0Store` methods (used inside `search_knowledge_base` tool)

```
Add Memory:
  Mem0Store.add(text, user_id, metadata, infer=True)
    └── mem0ai.Memory.add(text, user_id, metadata)
          ├── LLM extracts structured facts (if infer=True)
          ├── EmbeddingGenerator embeds facts
          └── INSERT into PostgreSQL mem0 table (pgvector)

Search Memory:
  Mem0Store.search(query, user_id, limit)
    └── mem0ai.Memory.search(query, user_id, limit)
          ├── Embed query
          └── pgvector similarity search

Get Context (used by agent tool):
  Mem0Store.get_context_string(query, user_id, limit=3)
    ├── search(query, user_id, limit)
    └── format memories as "## User Context\n- fact1\n- fact2"

Delete All:
  Mem0Store.delete_all(user_id)
    └── mem0ai.Memory.delete_all(user_id)
```

**Configuration**: Uses `DATABASE_URL` (same PostgreSQL instance as RAG) and existing Ollama models.

---

## 5. PDF Question Generator

> See [PDF_QUESTION_GENERATOR.md](PDF_QUESTION_GENERATOR.md) and [RAG_anything.md](RAG_anything.md) for details.

**Entry point**: `python -m rag.ingestion.processors.pdf_question_generator <pdf_path>`

```
main(pdf_path)
  └── process_pdf(pdf_path)
        ├── MinerU parser (via raganything)
        │     └── Extract: text blocks, tables, images, equations
        ├── Modal processors:
        │     ├── TableModalProcessor.process_multimodal_content()
        │     │     └── LLM call: describe table structure and content
        │     ├── ImageModalProcessor.process_multimodal_content()
        │     │     └── LLM call (vision): describe image content
        │     └── EquationModalProcessor.process_multimodal_content()
        │           └── LLM call: render and explain LaTeX equations
        ├── Combine all extracted content as context string
        ├── LLM call: generate Q&A pairs
        │     ├── get_ollama_llm_funcs() / get_openai_llm_funcs()
        │     │     └── lightrag_utils.py
        │     ├── QUESTION_GENERATION_SYSTEM_PROMPT
        │     └── Returns JSON: {questions: [...], entities: [...], summary: "..."}
        ├── Parse JSON response → ProcessingResult
        └── PDFQuestionStore.save_result(result)
              ├── INSERT INTO pdf_documents
              ├── INSERT INTO pdf_questions (with embeddings)
              └── INSERT INTO pdf_chunks   (with embeddings)
```

**Key files**:

| File | Class/Function | Role |
|------|---------------|------|
| `rag/ingestion/processors/pdf_question_generator.py` | `process_pdf()` | Core orchestration |
| `rag/ingestion/processors/lightrag_utils.py` | `LightRAGConfig` | LLM provider configuration |
| `rag/ingestion/processors/` | Modal processor classes | Per-modality extraction |

---

## 6. Streamlit App

**Entry point**: `streamlit run streamlit_mem0_app.py`

```
streamlit_mem0_app.py  (module level, cached with st.cache_resource)
  ├── get_settings()          # load_dotenv + os.getenv
  ├── get_mem0()
  │     ├── _parse_database_url(DATABASE_URL)
  │     └── mem0ai.Memory.from_config(pgvector_config)
  └── get_agent()
        ├── OpenAIProvider(base_url, api_key)
        ├── OpenAIChatModel(model_name, provider)
        └── Agent(model, system_prompt=...)

Page rerun (each user action)
  ├── Display st.session_state.messages  # full chat history
  ├── Sidebar
  │     ├── st.text_input("User ID")
  │     ├── [Clear Chat]     → st.session_state.messages = []
  │     ├── [Clear Memories] → delete_all_memories(user_id)
  │     │                         └── psycopg2: DELETE FROM mem0 table
  │     └── [Show Memories]  → get_mem0().get_all(user_id)
  └── st.chat_input() → user_message
        ├── get_user_context(user_id)
        │     └── psycopg2: SELECT memories → format as context string
        ├── enhanced_prompt = context + "\n\n" + user_message
        ├── asyncio.run(run_agent(agent, enhanced_prompt))
        │     └── agent.run(enhanced_prompt)   # Pydantic AI (no RAG tool)
        ├── save_to_memory(mem0, user_id, conversation)
        │     └── mem0.add(conversation_text, user_id, infer=True)
        └── st.session_state.messages.append(response)
```

**Session state**:

| Key | Type | Scope | Contents |
|-----|------|-------|----------|
| `messages` | `list[dict]` | Browser session | `[{role, content}, ...]` |
| Cached `agent` | `Agent` | Server lifetime | Pydantic AI Agent |
| Cached `mem0` | `Memory` | Server lifetime | Mem0 Memory instance |
| PostgreSQL `mem0_memories` | table | Persistent | User facts across sessions |

---

## 7. Architecture Overview

```
Streamlit UI (streamlit_mem0_app.py)
    │
    ▼
PydanticAI Agent (rag/agent/rag_agent.py)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
search_knowledge_base tool        Mem0Store
(rag/agent/rag_agent.py)         (rag/memory/mem0_store.py)
    │                                  │
    ▼                                  │
Retriever                              │
(rag/retrieval/retriever.py)           │
    │                                  │
    ├── EmbeddingGenerator             │
    │   (rag/ingestion/embedder.py)     │
    │                                  │
    ▼                                  ▼
PostgresHybridStore ←─────────────────┘
(rag/storage/vector_store/postgres.py)
    │
    ▼
PostgreSQL / pgvector (Neon or local)
  ├── documents
  ├── chunks          ← RAG embeddings
  └── mem0_memories   ← user memory embeddings

Ingestion (rag/ingestion/pipeline.py)
    └──► PostgresHybridStore  (writes documents + chunks)
```
