# Call Graphs

Call graphs for the main workflows in this project.
Links jump directly to the relevant line in source code.

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
rag/main.py:main()                                                L100
  ├── validate_config()                                           L46
  │     └── load_settings()
  └── run_ingestion_pipeline()                                    L633
        └── DocumentIngestionPipeline                             L135
              ├── __init__(config, documents_folder, clean)       L138
              │     ├── load_settings()
              │     ├── create_chunker()                          L302
              │     ├── create_embedder()                         L308
              │     └── PostgresHybridStore()                     L119
              ├── initialize()                                     L173
              │     └── store.initialize()                        L125
              │           ├── asyncpg.create_pool(DATABASE_URL)
              │           ├── CREATE EXTENSION IF NOT EXISTS vector
              │           ├── CREATE TABLE documents
              │           ├── CREATE TABLE chunks (embedding vector(768))
              │           └── CREATE INDEX (IVFFlat, GIN, B-tree)
              ├── ingest_documents(progress_callback)              L470
              │     ├── store.clean_collections()                  L502
              │     ├── _find_document_files()                     L189
              │     └── [for each file]:
              │           _ingest_single_document(file_path)       L404
              │             ├── _read_document(file_path)          L230
              │             │     ├── [PDF/DOCX] Docling DocumentConverter
              │             │     ├── [Audio]    _transcribe_audio()  L299
              │             │     └── [MD/TXT]   direct file read
              │             ├── _extract_title()                   L347
              │             ├── _extract_document_metadata()       L373
              │             │     └── _compute_file_hash()         L357
              │             ├── chunker.chunk_document()           L142
              │             │     ├── Docling HybridChunker
              │             │     └── _simple_fallback_chunk()     L228
              │             ├── embedder.embed_chunks(chunks)      L204
              │             │     └── generate_embeddings_batch()  L181
              │             │           └── _cached_embed()        L117
              │             ├── store.save_document(...)           L467
              │             └── store.add(chunks, document_id)     L214
              └── close()                                          L182
                    └── store.close()                              L206
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/main.py`](../rag/main.py#L100) | `main()` | L100 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L135) | `DocumentIngestionPipeline` | L135 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L470) | `ingest_documents()` | L470 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L404) | `_ingest_single_document()` | L404 |
| [`rag/ingestion/chunkers/docling.py`](../rag/ingestion/chunkers/docling.py#L108) | `DoclingHybridChunker` | L108 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L132) | `EmbeddingGenerator` | L132 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L116) | `PostgresHybridStore` | L116 |

---

## 2. Query & Retrieval

> See [RAG.md](RAG.md) and [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: [`Retriever.retrieve()`](../rag/retrieval/retriever.py#L234)

```
Retriever.retrieve(query, match_count, search_type, use_cache)    L234
  ├── 1. ResultCache.get(query, search_type, match_count)          L266
  │        └── cache hit? → return list[SearchResult]
  │
  ├── 2. Query embedding  (HyDE if hyde_enabled=True)
  │     ├── [hyde_enabled=True]:
  │     │     ├── HyDEProcessor.generate_hypothetical(query)       ← LLM call
  │     │     └── embedder.generate_embedding(hypothetical_doc)
  │     └── [hyde_enabled=False]:
  │           └── EmbeddingGenerator.embed_query(query)            L288
  │                 ├── _cached_embed(text, model)  async_lru(1000)
  │                 └── openai.AsyncOpenAI.embeddings.create()
  │
  ├── 3. fetch_count = match_count × reranker_overfetch_factor     (if reranker_enabled)
  │
  ├── 4. Search
  │     ├── [search_type == "semantic"]
  │     │     └── store.semantic_search(query_embedding, fetch_count)
  │     │           └── SQL: ORDER BY embedding <=> $1::vector LIMIT $2
  │     ├── [search_type == "text"]
  │     │     └── store.text_search(query, fetch_count)
  │     │           └── SQL: WHERE content_tsv @@ plainto_tsquery(...)
  │     └── [search_type == "hybrid"]  (default)
  │           ├── asyncio.gather(semantic_search, text_search)
  │           └── _reciprocal_rank_fusion(results_list)
  │                 └── RRF score = Σ 1/(k=60 + rank), deduplicate, sort
  │
  ├── 5. Rerank  (if reranker_enabled=True)
  │     ├── [reranker_type == "llm"]:
  │     │     └── LLMReranker.rerank(query, results, top_k)
  │     │           └── asyncio.gather(*[_score_document(...) for each result])
  │     └── [reranker_type == "cross_encoder"]:
  │           └── CrossEncoderReranker.rerank(query, results, top_k)
  │
  ├── 6. ResultCache.set(...)
  └── return list[SearchResult]

Retriever.retrieve_as_context(query, match_count, search_type)    L334
  └── retrieve(...)
        └── join chunks as formatted context string
```

**Caching layers**:

| Cache | File | Line | Key | TTL | Size |
|-------|------|------|-----|-----|------|
| Embedding cache (`async_lru`) | [`embedder.py`](../rag/ingestion/embedder.py#L117) | L117 | `(text, model)` | None | 1000 |
| Result cache (`ResultCache`) | [`retriever.py`](../rag/retrieval/retriever.py#L126) | L126 | `(query, type, count)` | 5 min | 100 |

---

## 3. RAG Agent (CLI)

> See [RAG.md](RAG.md) for details.

**Entry point**: `python -m rag.agent.agent_main`  →  [`agent_main()`](../rag/agent/agent_main.py#L75)

```
agent_main.py:stream_agent_interaction()                          L75
  └── _stream_agent(user_input, deps, message_history)
        └── agent.iter(query, deps=deps, message_history=...)
              └── [async for node in run]
                    ├── UserPromptNode
                    │     └── _debug_print()                       L69
                    ├── ModelRequestNode  (LLM decides)
                    │   └── _handle_model_request_node(node, ctx)
                    │         └── node.stream() yields:
                    │               PartStartEvent / PartDeltaEvent / FinalResultEvent
                    ├── CallToolsNode  (tool execution)
                    │   └── _handle_tool_call_node(node, ctx)
                    │         └── node.stream() yields:
                    │               FunctionToolCallEvent
                    │                 → search_knowledge_base(ctx, query, ...)  L225
                    │                     ├── RAGState.get_retriever()           L194
                    │                     │     └── PostgresHybridStore.initialize()  L125
                    │                     ├── retriever.retrieve_as_context()    L299
                    │                     │     └── [see Query & Retrieval]
                    │                     ├── mem0_store.get_context_string()    L207
                    │                     └── return combined_context
                    │               FunctionToolResultEvent
                    ├── ModelRequestNode  (LLM final answer)
                    └── EndNode

RAGState lazy init (first get_retriever() call):                   L194
  ├── PostgresHybridStore().initialize()                           L125
  ├── EmbeddingGenerator()                                         L135
  ├── Retriever(store, embedder)                                   L214
  └── Mem0Store()  (if mem0_enabled)                               L101
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L225) | `search_knowledge_base()` tool | L225 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L171) | `RAGState` | L171 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L194) | `RAGState.get_retriever()` | L194 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L357) | `traced_agent_run()` | L357 |
| [`rag/agent/agent_main.py`](../rag/agent/agent_main.py#L75) | `stream_agent_interaction()` | L75 |
| [`rag/agent/agent_main.py`](../rag/agent/agent_main.py#L58) | `set_verbose_debug()` | L58 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L116) | `PostgresHybridStore` | L116 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L126) | `PostgresHybridStore.initialize()` | L126 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L132) | `EmbeddingGenerator` | L132 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L181) | `Retriever` | L181 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L334) | `Retriever.retrieve_as_context()` | L334 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L93) | `Mem0Store` | L93 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L196) | `Mem0Store.get_context_string()` | L196 |

---

## 4. Mem0 Memory

> See [RAG.md §16](RAG.md#16-mem0-memory-layer) for details.

**Entry point**: [`Mem0Store`](../rag/memory/mem0_store.py#L93) methods (called from `search_knowledge_base`)

```
Mem0Store.__init__()                                              L101
  └── _parse_database_url(DATABASE_URL)                          L112

Add Memory:
  Mem0Store.add(text, user_id, metadata, infer=True)
    └── mem0ai.Memory.add(text, user_id, metadata)
          ├── LLM extracts structured facts  (if infer=True)
          ├── EmbeddingGenerator embeds facts
          └── INSERT into PostgreSQL mem0 table

Search Memory:
  Mem0Store.search(query, user_id, limit)
    └── mem0ai.Memory.search(query, user_id, limit)
          ├── embed query
          └── pgvector similarity search

Get Context (called by agent tool at L225):
  Mem0Store.get_context_string(query, user_id, limit=3)
    ├── search(query, user_id, limit)
    └── format as "## User Context\n- fact1\n- fact2"
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L93) | `Mem0Store` | L93 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L101) | `__init__()` | L101 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L112) | `_parse_database_url()` | L112 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L146) | `_get_memory()` | L146 |

---

## 5. PDF Question Generator

> See [PDF_QUESTION_GENERATOR.md](PDF_QUESTION_GENERATOR.md) and [RAG_anything.md](RAG_anything.md) for details.

**Entry point**: `python -m rag.ingestion.processors.pdf_question_generator <pdf_path>`
→ [`main()`](../rag/ingestion/processors/pdf_question_generator.py#L899) →  [`main_async()`](../rag/ingestion/processors/pdf_question_generator.py#L778)

```
main()                                                            L899
  └── main_async()                                               L778
        └── process_pdf_with_raganything(pdf_path, config)       L269
              │
              ├── 1. Parse PDF with MinerU / RAGAnything
              │        RAGAnything.process_document(pdf_path)
              │        └── Extract: text blocks, tables, images, equations
              │
              ├── 2. Modal processors (per content type):
              │     ├── TableProcessor.process()                  L355
              │     │     ├── _normalize_table()                  L190
              │     │     ├── _build_user_prompt()                L281
              │     │     ├── _call_llm_api()                     L310
              │     │     └── _parse_json_response() → TableAnalysis  L292
              │     │
              │     ├── ImageProcessor.process()                  L281
              │     │     ├── _encode_image()                     L166
              │     │     ├── _build_user_prompt()                L192
              │     │     ├── _call_vision_api()                  L221
              │     │     └── _parse_json_response() → ImageAnalysis  L201
              │     │
              │     └── EquationProcessor.process()               L351
              │           ├── _detect_format()                    L200
              │           ├── _normalize_equation()               L251
              │           ├── _build_user_prompt()                L275
              │           ├── _call_llm_api()                     L306
              │           └── _parse_json_response() → EquationAnalysis  L288
              │
              ├── 3. extract_chunks_from_lightrag()               L167
              │     └── walk LightRAG chunk store → list[ChunkContext]  L116
              │
              ├── 4. format_chunks_as_context(chunks)             L126
              │     └── format as context string for LLM
              │
              ├── 5. LLM call: generate Q&A pairs
              │     ├── get_ollama_llm_funcs() / get_openai_llm_funcs()  L42 / L162
              │     │     └── lightrag_utils.py
              │     ├── QUESTION_GENERATION_SYSTEM_PROMPT
              │     └── Returns JSON: {questions, entities, summary}
              │
              ├── 6. Parse JSON → ProcessingResult                L62
              │
              └── 7. PDFQuestionStore.save_pdf_result(result)     L179
                    ├── INSERT INTO pdf_documents
                    ├── INSERT INTO pdf_questions  (+ embed each question)
                    └── INSERT INTO pdf_chunks     (+ embed each chunk)

Fallback (if RAGAnything unavailable):
  process_pdf_simple(pdf_path, config)                           L432
    ├── Docling DocumentConverter  (text-only extraction)
    ├── format_chunks_as_context()                               L126
    └── LLM call → ProcessingResult                             L62
```

**Key files — PDF Question Generator**:

| File | Symbol | Line |
|------|--------|------|
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L899) | `main()` | L899 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L778) | `main_async()` | L778 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L269) | `process_pdf_with_raganything()` | L269 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L432) | `process_pdf_simple()` | L432 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L167) | `extract_chunks_from_lightrag()` | L167 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L126) | `format_chunks_as_context()` | L126 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L62) | `ProcessingResult` | L62 |
| [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L116) | `ChunkContext` | L116 |
| [`pdf_question_store.py`](../rag/ingestion/processors/pdf_question_store.py#L49) | `PDFQuestionStore` | L49 |
| [`pdf_question_store.py`](../rag/ingestion/processors/pdf_question_store.py#L179) | `save_pdf_result()` | L179 |
| [`pdf_question_store.py`](../rag/ingestion/processors/pdf_question_store.py#L313) | `search_questions()` | L313 |
| [`pdf_question_store.py`](../rag/ingestion/processors/pdf_question_store.py#L445) | `search_chunks()` | L445 |
| [`pdf_question_store.py`](../rag/ingestion/processors/pdf_question_store.py#L582) | `_rrf_merge()` | L582 |
| [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L42) | `get_ollama_llm_funcs()` | L42 |
| [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L162) | `get_openai_llm_funcs()` | L162 |
| [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L335) | `LightRAGConfig` | L335 |
| [`table.py`](../rag/ingestion/processors/table.py#L163) | `TableProcessor` | L163 |
| [`table.py`](../rag/ingestion/processors/table.py#L355) | `TableProcessor.process()` | L355 |
| [`image.py`](../rag/ingestion/processors/image.py#L139) | `ImageProcessor` | L139 |
| [`image.py`](../rag/ingestion/processors/image.py#L281) | `ImageProcessor.process()` | L281 |
| [`equation.py`](../rag/ingestion/processors/equation.py#L173) | `EquationProcessor` | L173 |
| [`equation.py`](../rag/ingestion/processors/equation.py#L351) | `EquationProcessor.process()` | L351 |
| [`base.py`](../rag/ingestion/processors/base.py#L129) | `BaseProcessor` | L129 |

---

## 6. Streamlit App

**Entry point**: `streamlit run streamlit_mem0_app.py`

```
streamlit_mem0_app.py  (cached with @st.cache_resource)
  ├── get_mem0_store()        create_mem0_store() → Mem0Store (PostgreSQL/pgvector)
  └── get_agent()             Agent(OpenAIChatModel, system_prompt)

Page rerun lifecycle:
  ├── Display st.session_state.messages
  ├── Sidebar
  │     ├── [Clear Chat]     → session_state.messages = []
  │     ├── [Clear Memories] → get_mem0_store().delete_all(user_id)
  │     └── [Show Memories]  → get_mem0_store().get_all(user_id)
  └── st.chat_input() → user_message
        ├── mem0_store.get_context_string(query=prompt, user_id=user_id)
        │     └── pgvector similarity search → formatted context string
        ├── enhanced_prompt = context + "\n\n" + user_message
        ├── asyncio.run(agent.run(enhanced_prompt))
        ├── mem0_store.add(conversation, user_id=user_id, infer=True)
        └── session_state.messages.append(response)
```

**Session state**:

| Key | Scope | Contents |
|-----|-------|----------|
| `messages` | Browser session | `[{role, content}, ...]` |
| cached `agent` | Server lifetime | Pydantic AI Agent |
| cached `mem0_store` | Server lifetime | Mem0Store (PostgreSQL-backed) |
| PostgreSQL `mem0_memories` | Persistent | User facts across sessions |

---

## 7. Architecture Overview

```
Streamlit UI (streamlit_mem0_app.py)
    │
    ▼
PydanticAI Agent  rag/agent/rag_agent.py:L171
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
search_knowledge_base()  L225     Mem0Store  L93
    │                                  │
    ▼                                  │
Retriever  L211                        │
    │                                  │
    ├── EmbeddingGenerator  L132       │
    │                                  │
    ▼                                  ▼
PostgresHybridStore  L116 ←───────────┘
    │
    ▼
PostgreSQL / pgvector (Neon or local)
  ├── documents
  ├── chunks          ← RAG embeddings
  └── mem0_memories   ← user memory embeddings

PDF Question Generator  (separate workflow)
  pdf_question_generator.py:L269
    ├── TableProcessor  L163  ─┐
    ├── ImageProcessor  L139  ─┤── modal processors
    ├── EquationProcessor L173 ┘
    └── PDFQuestionStore  L49
          └── PostgreSQL (pdf_documents, pdf_questions, pdf_chunks)

Ingestion CLI  rag/main.py:L100
  └── DocumentIngestionPipeline  L135
        └── PostgresHybridStore  L116
```
