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
- [8. Knowledge Graph — Build (CuadKgBuilder)](#8-knowledge-graph--build-cuadkgbuilder)
- [9. Knowledge Graph — Query (AgeGraphStore / PgGraphStore)](#9-knowledge-graph--query-agegraphstore--pggraphstore)
- [10. Agent KG Tool (search_knowledge_graph)](#10-agent-kg-tool-search_knowledge_graph)
- [11. NL-to-SQL v1 — Sync MVP](#11-nl-to-sql-v1--sync-mvp)
- [12. NL-to-SQL v2 — Async + Retry + Guardrails](#12-nl-to-sql-v2--async--retry--guardrails)
- [13. SQL Discovery Agent (sql_discovery)](#13-sql-discovery-agent-sql_discovery)

---

## 1. Document Ingestion

> See [RAG.md](RAG.md) and [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: `python -m rag.main --ingest`

```
rag/main.py:main()                                                L100
  ├── validate_config()                                           L46
  │     └── load_settings()
  └── run_ingestion_pipeline()                                    L633
        └── DocumentIngestionPipeline                             L136
              ├── __init__(config, documents_folder, clean)       L136
              │     ├── load_settings()
              │     ├── create_chunker()
              │     ├── create_embedder()
              │     └── PostgresHybridStore()
              ├── initialize()
              │     └── store.initialize()
              │           ├── asyncpg.create_pool(DATABASE_URL)
              │           ├── CREATE EXTENSION IF NOT EXISTS vector
              │           ├── CREATE TABLE documents
              │           ├── CREATE TABLE chunks (embedding vector(768))
              │           └── CREATE INDEX (IVFFlat, GIN, B-tree)
              ├── ingest_documents(progress_callback)              L479
              │     ├── store.clean_collections()
              │     ├── _find_document_files()
              │     └── [for each file]:
              │           _ingest_single_document(file_path)       L413
              │             ├── _read_document(file_path)
              │             │     ├── [PDF/DOCX] Docling DocumentConverter
              │             │     ├── [Audio]    _transcribe_audio()
              │             │     └── [MD/TXT]   direct file read
              │             ├── _extract_title()
              │             ├── _extract_document_metadata()
              │             │     └── _compute_file_hash()
              │             ├── chunker.chunk_document()
              │             │     ├── Docling HybridChunker
              │             │     └── _simple_fallback_chunk()
              │             ├── embedder.embed_chunks(chunks)              L207
              │             │     ├── generate_embeddings_batch(texts)   L184
              │             │     │     └── openai.AsyncOpenAI.embeddings.create()
              │             │     │           └── returns list[list[float]]
              │             │     └── chunk.embedding = embedding        L250
              │             │           └── ChunkData.embedding          models.py:L142
              │             ├── store.save_document(...)
              │             └── store.add(chunks, document_id)           L257
              │                   └── conn.executemany()                 L268
              │                         INSERT INTO chunks
              │                           (document_id, content, embedding,
              │                            chunk_index, metadata, token_count)
              │                           VALUES ($1, $2, $3, $4, $5, $6)
              │                         -- $3 = chunk.embedding → vector(N)
              │                         -- N = settings.embedding_dimension
              │                         --     (default 768, fixed at CREATE TABLE)
              └── close()
                    └── store.close()
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/main.py`](../rag/main.py#L100) | `main()` | L100 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L136) | `DocumentIngestionPipeline` | L136 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L479) | `ingest_documents()` | L479 |
| [`rag/ingestion/pipeline.py`](../rag/ingestion/pipeline.py#L413) | `_ingest_single_document()` | L413 |
| [`rag/ingestion/chunkers/docling.py`](../rag/ingestion/chunkers/docling.py) | `DoclingHybridChunker` | |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L135) | `EmbeddingGenerator` | L135 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L207) | `embed_chunks()` — sets `chunk.embedding` | L207 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L184) | `generate_embeddings_batch()` — API call | L184 |
| [`rag/ingestion/models.py`](../rag/ingestion/models.py#L142) | `ChunkData.embedding: list[float] \| None` | L142 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L116) | `PostgresHybridStore` | L116 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L257) | `add()` — INSERT chunks with embedding `$3` | L257 |
| [`rag/config/settings.py`](../rag/config/settings.py#L135) | `embedding_dimension` (default 768) | L135 |

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
| Embedding cache (`async_lru`) | [`embedder.py`](../rag/ingestion/embedder.py) | | `(text, model)` | None | 1000 |
| Result cache (`ResultCache`) | [`retriever.py`](../rag/retrieval/retriever.py#L96) | L96 | `(query, type, count)` | 5 min | 100 |

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
                    ├── ModelRequestNode  (LLM decides)
                    │   └── node.stream() yields:
                    │         PartStartEvent / PartDeltaEvent / FinalResultEvent
                    ├── CallToolsNode  (tool execution)
                    │   └── node.stream() yields:
                    │         FunctionToolCallEvent
                    │           → search_knowledge_base(ctx, query, ...)  L244
                    │               ├── RAGState.get_retriever()           L201
                    │               │     └── PostgresHybridStore.initialize()
                    │               ├── retriever.retrieve_as_context()    L334
                    │               │     └── [see Query & Retrieval]
                    │               ├── mem0_store.get_context_string()
                    │               └── return combined_context
                    │         FunctionToolResultEvent
                    ├── ModelRequestNode  (LLM final answer)
                    └── EndNode

RAGState lazy init (first get_retriever() call):                   L201
  ├── PostgresHybridStore().initialize()
  ├── EmbeddingGenerator()                                         L135
  ├── Retriever(store, embedder)                                   L181
  └── Mem0Store()  (if mem0_enabled)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L244) | `search_knowledge_base()` tool | L244 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L176) | `RAGState` | L176 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L201) | `RAGState.get_retriever()` | L201 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L437) | `traced_agent_run()` | L437 |
| [`rag/agent/agent_main.py`](../rag/agent/agent_main.py#L75) | `stream_agent_interaction()` | L75 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L116) | `PostgresHybridStore` | L116 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L135) | `EmbeddingGenerator` | L135 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L181) | `Retriever` | L181 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L234) | `Retriever.retrieve()` | L234 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L334) | `Retriever.retrieve_as_context()` | L334 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L96) | `ResultCache` | L96 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L94) | `Mem0Store` | L94 |

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
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L94) | `Mem0Store` | L94 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py) | `__init__()` | |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py) | `_parse_database_url()` | |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py) | `_get_memory()` | |

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

## 6. Streamlit Apps

Three apps, each independently runnable.

### App 1 — Legal Contract Assistant

**Entry point**: `streamlit run rag/agent/streamlit_app.py`

```
streamlit_app.py
  ├── init_session_state()
  │     └── RAGState()  (lazy — no network calls yet)
  │           └── StateDeps[RAGState](state=state)  → st.session_state.deps
  └── st.chat_input() → prompt
        └── asyncio.run(stream_agent_response(prompt, deps, history, …))
              └── agent.iter(prompt, deps=deps, message_history=…)
                    ├── ModelRequestNode   → stream text tokens to UI
                    ├── CallToolsNode
                    │     ├── search_knowledge_base  → hybrid RAG chunks
                    │     ├── search_knowledge_graph → entity/relationship lookup
                    │     └── run_graph_query        → custom Cypher results
                    └── EndNode
```

### App 2 — Memory Chat Demo

**Entry point**: `streamlit run streamlit_mem0_app.py`

```
streamlit_mem0_app.py  (@st.cache_resource)
  ├── get_mem0_store()   create_mem0_store() → Mem0Store (PostgreSQL/pgvector)
  └── get_agent()        Agent(OpenAIChatModel, system_prompt)  [no RAG/KG tools]

Page rerun:
  └── st.chat_input() → prompt
        ├── mem0_store.get_context_string(prompt, user_id)
        │     └── pgvector similarity search → formatted context
        ├── asyncio.run(agent.run(context + prompt))
        └── mem0_store.add(conversation, user_id, infer=True)
```

### App 3 — NL-to-SQL Explorer

**Entry point**: `streamlit run streamlit_nlsql_app.py`

```
streamlit_nlsql_app.py  (@st.cache_resource)
  └── _build_manager() → (ConversationManager, schema_text)
        ├── duckdb.connect(":memory:")
        ├── INSTALL/LOAD postgres_scanner
        ├── ATTACH DATABASE_URL AS rag_db (READ_ONLY)
        ├── generate schema from rag_db.information_schema
        ├── OpenAIModel(settings.llm_model, …)
        ├── Agent(model, result_type=str, system_prompt=SQL_PROMPT)
        └── ConversationManager(conn, agent, schema_text, …)  L303

Page rerun:
  └── st.chat_input() → prompt
        └── asyncio.run(manager.run_query(prompt)) → QueryResult
              ├── NL cache hit?
              ├── agent.run(schema + history + question) → SQL
              ├── _check_readonly(sql)    [guardrail]
              ├── _apply_row_cap(sql)     [guardrail]
              ├── _execute_with_timeout() [guardrail]
              └── render SQL code block + markdown results table
```

**Session state summary**:

| Key | App | Scope | Contents |
|-----|-----|-------|----------|
| `messages` | App 1 | Browser | `[{role, content}]` chat display |
| `message_history` | App 1 | Browser | Pydantic AI `ModelMessage` list |
| `deps` (RAGState) | App 1 | Browser | lazy store/retriever/kg |
| `messages` | App 2 | Browser | `[{role, content}]` chat display |
| `nl_messages` | App 3 | Browser | `[{role, content}]` chat display |
| cached `agent` | App 2 | Server | plain Pydantic AI Agent |
| cached `mem0_store` | App 2 | Server | Mem0Store |
| cached `manager` | App 3 | Server | ConversationManager (holds DuckDB conn + caches) |

---

## 7. Architecture Overview

```
── App 1: Legal Contract Assistant  rag/agent/streamlit_app.py ──────────
    │
    ▼
PydanticAI Agent  rag/agent/rag_agent.py:L176
    │
    ├──────────────────────┬──────────────────────┬────────────────────┐
    ▼                      ▼                      ▼                    ▼
search_knowledge_base()  search_knowledge_graph() run_graph_query()  Mem0Store
L245                     L347                    L409                L94
    │                      │                      │                    │
    ▼                      ▼                      ▼                    │
Retriever  L181      create_kg_store()      create_kg_store()         │
    │                 ├── AgeGraphStore L98   └── AgeGraphStore L98   │
    ├── Embedder      │     search_entities        run_cypher_query    │
    │                 │     get_related_entities    L525               │
    ▼                 └── PgGraphStore L72 (legacy)                   ▼
PostgresHybridStore L116 ←─────────────────────────────────────────────┘
    │
    ▼
PostgreSQL / pgvector (Neon or local)
  ├── documents  ├── chunks  └── mem0_memories

Apache AGE (docker-compose port 5433)
  └── legal_graph  ← Entity vertices + directed edges
        ├── 13,262 entities  (Party, Jurisdiction, Date, *Clause)
        └── 13,603 relationships (PARTY_TO, GOVERNED_BY_LAW, HAS_LICENSE, …)

── App 2: Memory Chat  streamlit_mem0_app.py ─────────────────────────
    └── Plain Agent + Mem0Store (no RAG/KG tools)

── App 3: NL-to-SQL Explorer  streamlit_nlsql_app.py ────────────────
    └── ConversationManager (nlp_sql_postgres_v2.py:L303)
          ├── DuckDB in-memory + postgres scanner
          ├── ATTACH DATABASE_URL AS rag_db (READ_ONLY)
          ├── Pydantic AI Agent (same LLM settings as App 1)
          └── 3-attempt retry + guardrails (_check_readonly, _apply_row_cap)

PDF Question Generator  (separate workflow)
  pdf_question_generator.py:L269
    ├── TableProcessor L163 ─┐
    ├── ImageProcessor L139  ─┤── modal processors
    ├── EquationProcessor L173┘
    └── PDFQuestionStore L49
          └── PostgreSQL (pdf_documents, pdf_questions, pdf_chunks)

NL-to-SQL (nlp2sql/)   (library, used by App 3)
  ├── nlp_sql_postgres_v1.py  ConversationManager L136 (sync MVP)
  ├── nlp_sql_postgres_v2.py  ConversationManager L303 (async + retry)
  │     └── HistoryStore L189  (asyncpg persistence)
  └── sql_discovery.py        sql_agent (multi-DB discovery)

Ingestion CLI  rag/main.py:L100
  └── DocumentIngestionPipeline L136  └── PostgresHybridStore L116

KG Build CLI  rag/knowledge_graph/cuad_kg_ingest.py:L115
  └── build_cuad_kg()  └── AgeGraphStore (AGE-only)
```

---

## 8. Knowledge Graph — Build (build_cuad_kg)

> See [HYBRID_KG_QUESTIONS.md](HYBRID_KG_QUESTIONS.md) for evaluation queries over this graph.

**Entry point**: `python -m rag.knowledge_graph.cuad_kg_ingest [--eval-path ...] [--limit N]`

```
main()                                                              cuad_kg_ingest.py
  ├── load_settings()
  ├── AgeGraphStore()  ← graph backend (AGE only)
  ├── asyncpg.create_pool(settings.database_url)  ← doc lookups
  └── build_cuad_kg(store, doc_pool, eval_path, limit)

build_cuad_kg(store, doc_pool, eval_path, limit)
  ├── load cuad_eval.json  → list[{question_type, answers, contract_title}]
  ├── [for each QA pair]:
  │     ├── _get_document_id(doc_pool, contract_title, cache)
  │     │     └── doc_pool → conn.fetchrow(SELECT id WHERE title …)
  │     │           (result cached in local dict)
  │     ├── entity_type_for(question_type)      ← constants.py
  │     │     └── ENTITY_TYPE_MAP.get(question_type, "Clause")
  │     ├── relationship_type_for(entity_type)  ← constants.py
  │     │     └── RELATIONSHIP_MAP.get(entity_type, "HAS_CLAUSE")
  │     ├── store.upsert_entity("Contract", …)   ← contract node
  │     └── [for each answer_text]:
  │           ├── store.upsert_entity(entity_type, answer_text, …)
  │           └── store.add_relationship(entity_id, contract_id, rel_type, …)
  └── return {"entities": N, "relationships": N, "skipped": N}
```

**Entity type map (35+ CUAD question types → 9 entity types)**:

| CUAD question type | Entity type | Relationship |
|---|---|---|
| `Parties` | `Party` | `PARTY_TO` |
| `Governing Law` | `Jurisdiction` | `GOVERNED_BY_LAW` |
| `Effective Date`, `Expiration Date`, … | `Date` | `HAS_DATE` |
| `License Grant`, `Non-Transferable License`, … | `LicenseClause` | `HAS_LICENSE` |
| `Termination For Convenience`, … | `TerminationClause` | `HAS_TERMINATION` |
| `Non-Compete`, `Exclusivity`, … | `RestrictionClause` | `HAS_RESTRICTION` |
| `IP Ownership Assignment`, `Work For Hire`, … | `IPClause` | `HAS_IP_CLAUSE` |
| `Liability Cap`, `Uncapped Liability`, … | `LiabilityClause` | `HAS_LIABILITY` |
| *(everything else)* | `Clause` | `HAS_CLAUSE` |

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/knowledge_graph/cuad_kg_ingest.py`](../rag/knowledge_graph/cuad_kg_ingest.py) | `build_cuad_kg()` | L62 |
| [`rag/knowledge_graph/cuad_kg_ingest.py`](../rag/knowledge_graph/cuad_kg_ingest.py) | `_get_document_id()` | L38 |
| [`rag/knowledge_graph/constants.py`](../rag/knowledge_graph/constants.py) | `entity_type_for()` | L141 |
| [`rag/knowledge_graph/constants.py`](../rag/knowledge_graph/constants.py) | `relationship_type_for()` | L146 |
| [`rag/knowledge_graph/constants.py`](../rag/knowledge_graph/constants.py) | `ENTITY_TYPE_MAP` | L83 |
| [`rag/knowledge_graph/constants.py`](../rag/knowledge_graph/constants.py) | `RELATIONSHIP_MAP` | L127 |
| [`rag/knowledge_graph/cuad_kg_ingest.py`](../rag/knowledge_graph/cuad_kg_ingest.py) | `main()` | L115 |
| [`rag/knowledge_graph/__init__.py`](../rag/knowledge_graph/__init__.py) | `create_kg_store()` | |

---

## 9. Knowledge Graph — Query (AgeGraphStore / PgGraphStore)

Both stores share the same public interface; swap via `KG_BACKEND` env var.

```
── AgeGraphStore (default, Apache AGE)  L98 ──────────────────────────
── PgGraphStore  (legacy, SQL tables)   L72 ──────────────────────────

initialize()                                       AGE:L124  PG:L85
  ├── [AGE] asyncpg.create_pool(init=_age_init)    L92
  │     └── _age_init(conn)   per-connection setup
  │           ├── LOAD '$libdir/plugins/age'
  │           └── SET search_path = ag_catalog, "$user", public
  ├── [AGE] ag_catalog.create_graph(graph_name)
  ├── [AGE] CREATE UNIQUE constraint on (normalized_name, entity_type, document_id)
  └── [PG]  CREATE TABLE kg_entities, kg_relationships + indexes

upsert_entity(name, entity_type, document_id, metadata) → UUID
  ├── _normalize(name)                             L67  (from pg_graph_store)
  ├── [AGE] _conn() → _cypher("MERGE (e:Entity {…})")
  │     └── _unquote_agtype(result)                L82
  └── [PG]  INSERT INTO kg_entities ON CONFLICT DO UPDATE (merge metadata)

add_relationship(src_id, tgt_id, rel_type, document_id, props) → UUID
  ├── [AGE] _cypher("MATCH (s),(t) MERGE (s)-[r:REL_TYPE {…}]->(t)")
  └── [PG]  INSERT INTO kg_relationships ON CONFLICT DO NOTHING

search_entities(query, entity_type, limit) → list[dict]
  ├── [AGE] MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($query)
  │         [optional] AND e.entity_type = $entity_type
  └── [PG]  WHERE name_tsv @@ plainto_tsquery($query)
            [optional] AND entity_type = $entity_type
            ORDER BY ts_rank(…)

get_related_entities(entity_id, rel_type, limit) → list[dict]
  ├── [AGE] MATCH (e)-[r]-(other)  WHERE id(e) = $id
  │         [optional] AND type(r) = $rel_type
  └── [PG]  UNION ALL of outgoing (source_id=$id) + incoming (target_id=$id)

find_contracts_by_entity(entity_name, entity_type, limit) → list[dict]
  ├── _normalize(entity_name)
  ├── [AGE] MATCH (e:Entity {normalized_name:$n})-[]->(c:Entity {entity_type:"Contract"})
  └── [PG]  JOIN kg_entities with documents WHERE normalized_name=$n

search_as_context(query, limit) → str             AGE:L413  PG:L403
  ├── search_entities(query, limit=limit)
  ├── [for each entity]: get_related_entities(entity_id, limit=5)
  └── format as "## Knowledge Graph — Facts\n- [TYPE] name\n  └─ REL → target"
        fallback: bullet list of entities if no relationships found

get_graph_stats() → dict                          AGE:L452  PG:L454
  ├── [AGE] MATCH (e:Entity) RETURN e.entity_type, count(*)
  │         MATCH ()-[r]->() RETURN type(r), count(*)
  └── [PG]  SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY 1
            SELECT relationship_type, COUNT(*) FROM kg_relationships GROUP BY 1

run_cypher_query(cypher) → str                    AGE:L525  PG:L473
  ├── [AGE] guard: block CREATE/MERGE/SET/DELETE/REMOVE/DROP/DETACH
  │         _parse_return_aliases(cypher)          L74  → list of display names
  │           ├── regex-find RETURN clause
  │           ├── paren-depth comma split
  │           └── extract AS alias or last identifier token
  │         build AS (c0 agtype, c1 agtype, …) from alias count
  │         _conn() → conn.fetch(_cypher(cypher) + AS clause)
  │         format as pipe-separated table: "col1 | col2\n---\nv1 | v2\n(N rows)"
  └── [PG]  returns "Cypher requires AGE backend" message (stub)
```

**Internal helpers (AgeGraphStore)**:

| Helper | Line | Purpose |
|--------|------|---------|
| `_normalize(name)` | pg L67 | `lower(re.sub(r"\s+", " ", name.strip()))` — imported from pg_graph_store |
| `_unquote_agtype(value)` | L82 | strips surrounding `"` from AGE agtype strings |
| `_age_init(conn)` | L92 | asyncpg pool `init=` callback; loads AGE extension + sets search_path |
| `_conn()` | L181 | async context manager; acquires connection + re-runs AGE setup |
| `_cypher(body)` | L194 | wraps body in `SELECT * FROM ag_catalog.cypher('graph', $$…$$, NULL) AS (v agtype)` |
| `_parse_return_aliases(cypher)` | L74 | parses RETURN clause → display name list for the AS column declaration |

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L98) | `AgeGraphStore` | L98 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L74) | `_parse_return_aliases()` | L74 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L82) | `_unquote_agtype()` | L82 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L92) | `_age_init()` | L92 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L181) | `_conn()` | L181 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L194) | `_cypher()` | L194 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L525) | `run_cypher_query()` | L525 |
| [`rag/knowledge_graph/pg_graph_store.py`](../rag/knowledge_graph/pg_graph_store.py#L72) | `PgGraphStore` | L72 |
| [`rag/knowledge_graph/pg_graph_store.py`](../rag/knowledge_graph/pg_graph_store.py#L67) | `_normalize()` | L67 |
| [`rag/knowledge_graph/pg_graph_store.py`](../rag/knowledge_graph/pg_graph_store.py#L473) | `run_cypher_query()` stub | L473 |

---

## 10. Agent KG Tools (search_knowledge_graph + run_graph_query)

**Entry point**: Two Pydantic AI tools registered on the RAG agent for KG access.

```
── Tool: search_knowledge_graph ─────────────────────────────────── L347
search_knowledge_graph(ctx, query, entity_type, limit)
  ├── RAGState.get_kg_store()  (lazy init)                          L217
  ├── [entity_type provided]:
  │     └── kg.search_entities(query, entity_type, limit)
  │           → "## Knowledge Graph — {entity_type} entities\n- [TYPE] name …"
  └── [no entity_type]:
        └── kg.search_as_context(query, limit)
              ├── search_entities(query)
              ├── get_related_entities(entity_id) per entity
              └── → pipe-formatted context string

── Tool: run_graph_query ─────────────────────────────────────────── L409
run_graph_query(ctx, cypher)
  ├── RAGState.get_kg_store()  (lazy init, same cached instance)    L217
  └── kg.run_cypher_query(cypher)                                   AGE:L525
        ├── guard: block mutating keywords
        ├── _parse_return_aliases(cypher)  → display name list      L74
        ├── build AS (c0 agtype, …) clause
        ├── conn.fetch(_cypher(cypher) + AS clause)
        └── → pipe-separated table string

── Shared lazy init ──────────────────────────────────────────────── L217
RAGState.get_kg_store()
  ├── [first call] create_kg_store()  reads KG_BACKEND
  │     ├── [default "age"]      → AgeGraphStore().initialize()
  │     └── [KG_BACKEND=postgres] → PgGraphStore().initialize()
  └── cache result in self._kg_store  (reused for all KG tool calls)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L347) | `search_knowledge_graph()` tool | L347 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L409) | `run_graph_query()` tool | L409 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L217) | `RAGState.get_kg_store()` | L217 |
| [`rag/knowledge_graph/age_graph_store.py`](../rag/knowledge_graph/age_graph_store.py#L525) | `run_cypher_query()` | L525 |
| [`rag/knowledge_graph/__init__.py`](../rag/knowledge_graph/__init__.py) | `create_kg_store()` | |

---

## 11. NL-to-SQL v1 — Sync MVP

> `nlp2sql/nlp_sql_postgres_v1.py`

**Entry point**: `python nlp2sql/nlp_sql_postgres_v1.py`

```
main()  [sync]
  ├── duckdb.connect()            in-memory DuckDB
  ├── UnifiedDataSource(conn, gcs_bucket, gcs_prefix, postgres_dbs)  L222
  │     ├── load_gcs_tables()                                          L236
  │     │     ├── conn.execute("INSTALL httpfs / LOAD httpfs")
  │     │     ├── CREATE SECRET (GCS HMAC credentials)
  │     │     ├── list GCS prefixes  → table names
  │     │     └── CREATE VIEW tbl AS parquet_scan('gs://…')
  │     ├── attach_postgres_dbs()                                      L270
  │     │     ├── conn.execute("INSTALL/LOAD postgres_scanner")
  │     │     └── ATTACH 'dsn' AS alias (TYPE postgres, READ_ONLY)
  │     ├── generate_schema() → str                                    L281
  │     │     ├── information_schema.tables for each source
  │     │     ├── information_schema.columns per table
  │     │     └── format: "Table: alias.main.tbl\n  col type\n  …"
  │     ├── init_agent(model="gpt-4o") → Agent                        L324
  │     │     └── Pydantic AI Agent(OpenAIChatModel, system_prompt)
  │     └── conversation_manager(cache_size) → ConversationManager     L337
  │
  └── ConversationManager.run_query(nl_query)  [× 5 test queries]    L170

ConversationManager.run_query(nl_query)                               L170
  ├── 1. NL cache hit?  (exact key match, OrderedDict LRU)
  ├── 2. _build_prompt(nl_query)                                       L161
  │     └── schema_text + _history_context(last 3 success) + question
  ├── 3. agent.run_sync(prompt) → raw SQL string
  ├── 4. strip_sql_fences(sql)                                         L109
  ├── 5. SQL hash cache hit?  (_hash = MD5)                            L151
  ├── 6. conn.execute(sql).fetchall() → rows
  ├── 7. history.append((nl_query, sql, rows))
  └── return rows (or None on error)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L136) | `ConversationManager` | L136 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L170) | `run_query()` | L170 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L222) | `UnifiedDataSource` | L222 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L236) | `load_gcs_tables()` | L236 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L281) | `generate_schema()` | L281 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L324) | `init_agent()` | L324 |
| [`nlp2sql/nlp_sql_postgres_v1.py`](../nlp2sql/nlp_sql_postgres_v1.py#L109) | `strip_sql_fences()` | L109 |

---

## 12. NL-to-SQL v2 — Async + Retry + Guardrails

> `nlp2sql/nlp_sql_postgres_v2.py`

**Entry point**: `python nlp2sql/nlp_sql_postgres_v2.py`

```
main()  [async]                                                       L634
  ├── UnifiedDataSource setup  (same as v1, but async)
  ├── optional: HistoryStore.create(dsn)                              L189
  └── ConversationManager via conversation_manager(…, session_id)    L589

── HistoryStore (asyncpg-backed persistence) ─────────────────────── L189
HistoryStore.create(dsn, **pool_kwargs)  [classmethod, async]
  ├── asyncpg.create_pool(dsn)
  └── _init_schema()
        └── CREATE TABLE conversation_history
              (session_id, nl_query, sql, rows JSONB, created_at)
              INDEX on session_id

save(session_id, nl_query, sql, qr: QueryResult)  [async]
  └── INSERT INTO conversation_history

load(session_id) → list[(nl_query, sql, QueryResult)]  [async]
  └── SELECT … ORDER BY created_at  → reconstruct QueryResult objects

sessions() → list[str]  [async]
  └── SELECT DISTINCT session_id ORDER BY MIN(created_at)

── ConversationManager (async + retry) ──────────────────────────── L303
__init__(conn, agent, schema_text, cache_size, max_retries,
         max_result_rows, query_timeout, history_store, session_id,
         _initial_history)
  └── warm caches from _initial_history (resumed sessions)

run_query(nl_query) → QueryResult  [async]                          L376
  ├── _normalize_nl(nl_query)                                        L340
  ├── 1. NL cache hit? (_nl_cache, OrderedDict LRU)
  ├── 2. for attempt in range(1, max_retries + 1):
  │     ├── [attempt == 1] _build_prompt(nl_query)                  L356
  │     │     └── schema + _history_context(last 3 successes) + question
  │     ├── [attempt > 1]  _build_correction_prompt(nl, bad_sql, error)  L365
  │     │     └── original prompt + "Previous SQL failed: …\nError: …"
  │     ├── await agent.run(prompt) → raw SQL
  │     ├── strip_sql_fences(sql)                                    L112
  │     ├── ── Guardrails ──────────────────────────────────────────
  │     ├── _check_readonly(sql)                                     L65
  │     │     └── regex for DROP/INSERT/UPDATE/DELETE/ALTER/CREATE/TRUNCATE
  │     │           → return error string or None
  │     ├── _apply_row_cap(sql, max_result_rows)                     L73
  │     │     └── append LIMIT if none present (regex)
  │     ├── _hash(sql) → MD5                                         L336
  │     ├── 3. SQL cache hit? (_sql_cache)
  │     ├── 4. _execute_with_timeout(conn, sql, query_timeout)       L81
  │     │     ├── threading.Timer(timeout, conn.interrupt)
  │     │     ├── conn.execute(sql) → cursor
  │     │     └── (columns from cursor.description, rows as list[tuple])
  │     ├── ── Success path ─────────────────────────────────────────
  │     ├── create QueryResult(nl, sql, columns, rows, attempts=attempt)
  │     ├── _cache_put(_nl_cache, key, qr)                          L343
  │     ├── _cache_put(_sql_cache, hash, qr)
  │     ├── history.append((nl, sql, qr))
  │     ├── await history_store.save(session_id, …)  [if provided]
  │     └── return qr
  │     ├── ── Error path ───────────────────────────────────────────
  │     └── last_error = str(exc); continue to next attempt
  └── return QueryResult(error=last_error, attempts=max_retries)

QueryResult                                                          L140
  ├── nl_query, sql, columns, rows, error, cached, attempts
  ├── .success → bool  (error is None)
  ├── .pretty_print(max_rows=20)  → tabulate to stdout
  └── .to_dataframe()             → pandas DataFrame
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L140) | `QueryResult` | L140 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L189) | `HistoryStore` | L189 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L303) | `ConversationManager` | L303 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L376) | `run_query()` | L376 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L65) | `_check_readonly()` | L65 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L73) | `_apply_row_cap()` | L73 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L81) | `_execute_with_timeout()` | L81 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L494) | `UnifiedDataSource` | L494 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L575) | `init_agent()` | L575 |
| [`nlp2sql/nlp_sql_postgres_v2.py`](../nlp2sql/nlp_sql_postgres_v2.py#L589) | `conversation_manager()` | L589 |

**v1 → v2 feature delta**:

| Feature | v1 | v2 |
|---------|----|----|
| Async | No (sync) | Yes (`await`) |
| Retry on SQL error | No | Yes (up to `max_retries`) |
| Correction prompt | No | Yes (feeds bad SQL + error back) |
| SELECT-only guardrail | No | Yes (`_check_readonly`) |
| Row cap | No | Yes (`_apply_row_cap`) |
| Query timeout | No | Yes (threading timer + `conn.interrupt()`) |
| Session persistence | No | Yes (`HistoryStore`, asyncpg) |
| Structured result | Raw rows | `QueryResult` dataclass |
| Provider support | OpenAI only | OpenAI + Anthropic |

---

## 13. SQL Discovery Agent (sql_discovery)

> `nl2sql/sql_discovery.py` — lightweight Pydantic AI agent that discovers schema on-the-fly via tools rather than a pre-built schema string. Integrated into `UnifiedDataSource` via `discovery_query()` in `nl2sql/nlp_sql_postgres_v2.py`.

```
── Entry point ────────────────────────────────────────────────────────────────

UnifiedDataSource.discovery_query(prompt, pg_alias=None)             L616
  nlp_sql_postgres_v2.py
  ├── resolve pg_dsn from self.postgres_dbs by alias (default: first)
  ├── asyncpg.create_pool(pg_dsn)
  ├── MultiDBDeps(pg_pool, duck_conn=self.conn)
  └── sql_agent.run(prompt, deps=MultiDBDeps)  ──────────────────────┐
                                                                      │
── Agent ──────────────────────────────────────────────────────────── │ ──────

sql_agent = Agent(                                           L43  <───┘
  model=_make_model()     # reads LLM_MODEL/LLM_BASE_URL from .env
  deps_type=MultiDBDeps,
  result_type=SQLResponse,
  system_prompt="…Discover the schema first."
)

  The agent receives the prompt and no schema string — it discovers
  schema by calling the tools below before generating SQL.

── Tools (registered via @sql_agent.tool) ─────────────────────────────────────

list_tables(ctx, db_type) → list[str]                                L53
  ├── [postgres]  SELECT table_name FROM information_schema.tables
  │               WHERE table_schema = 'public'  (via asyncpg pg_pool)
  └── [duckdb]    SHOW TABLES  (via duck_conn)

describe_table(ctx, db_type, table_name) → str                       L67
  ├── [postgres]  SELECT column_name, data_type FROM information_schema.columns
  └── [duckdb]    DESCRIBE {table_name}

── Agent inference loop ────────────────────────────────────────────────────────

  turn 1:  LLM calls list_tables("postgres") + list_tables("duckdb")
  turn 2:  LLM calls describe_table(db_type, tbl) for relevant tables only
  turn N:  LLM returns SQLResponse(database_type, sql, explanation)

── Execution & return ──────────────────────────────────────────────────────────

  result.output → SQLResponse
  ├── [postgres]  asyncpg pg_conn.fetch(sql)  → columns + rows as dicts
  └── [duckdb]    self.conn.execute(sql)       → columns + rows as tuples
  return (SQLResponse, columns, rows)

── Supporting types ────────────────────────────────────────────────────────────

MultiDBDeps                                                           L37
  ├── pg_pool: asyncpg.Pool
  └── duck_conn: duckdb.DuckDBPyConnection

SQLResponse                                                           L29
  ├── database_type: str   ("postgres" | "duckdb")
  ├── sql: str
  └── explanation: str
```

**Contrast with v1/v2**: v1/v2 call `generate_schema()` once up-front and stuff the full schema string into every prompt; `sql_discovery` lets the LLM call tools to explore schema dynamically — fewer prompt tokens for narrow queries, more LLM round-trips for broad ones.

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L43) | `sql_agent` | L43 |
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L29) | `SQLResponse` | L29 |
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L37) | `MultiDBDeps` | L37 |
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L53) | `list_tables()` tool | L53 |
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L67) | `describe_table()` tool | L67 |
| [`nl2sql/sql_discovery.py`](../../nl2sql/sql_discovery.py#L88) | `run_query()` | L88 |
| [`nl2sql/nlp_sql_postgres_v2.py`](../../nl2sql/nlp_sql_postgres_v2.py#L616) | `UnifiedDataSource.discovery_query()` | L616 |
