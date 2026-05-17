# Call Graphs

Call graphs for the main workflows in this project.
Links jump directly to the relevant line in source code.

## Table of Contents

- [1. Document Ingestion](#1-document-ingestion)
- [2. Query & Retrieval](#2-query--retrieval)
- [3. RAG Agent (CLI)](#3-rag-agent-cli)
- [4. Mem0 Memory](#4-mem0-memory)
- [5. Streamlit Apps](#5-streamlit-apps)
- [6. Architecture Overview](#6-architecture-overview)
- [7. Knowledge Graph — Build (CuadKgBuilder)](#7-knowledge-graph--build-cuadkgbuilder)
- [8. Knowledge Graph — Query (AgeGraphStore / PgGraphStore)](#8-knowledge-graph--query-agegraphstore--pggraphstore)
- [9. Agent KG Tools (search_knowledge_graph)](#9-agent-kg-tools-search_knowledge_graph)

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

## 5. Streamlit Apps

Three apps, each independently runnable.

### App 1 — Legal Contract Assistant

**Entry point**: `streamlit run rag/app/streamlit/streamlit_app.py`

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

**Entry point**: `streamlit run rag/app/streamlit/streamlit_mem0_app.py`

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

**Entry point**: `streamlit run nl2sql/app/streamlit/streamlit_app.py`

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

## 6. Architecture Overview

```
── App 1: Legal Contract Assistant  rag/app/streamlit/streamlit_app.py ──
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
PostgreSQL / pgvector (local)
  ├── documents  ├── chunks  └── mem0_memories

Apache AGE (docker-compose port 5433)
  └── legal_graph  ← Entity vertices + directed edges
        ├── 13,262 entities  (Party, Jurisdiction, Date, *Clause)
        └── 13,603 relationships (PARTY_TO, GOVERNED_BY_LAW, HAS_LICENSE, …)

── App 2: Memory Chat  rag/app/streamlit/streamlit_mem0_app.py ─────────
    └── Plain Agent + Mem0Store (no RAG/KG tools)

── App 3: NL-to-SQL Explorer  nl2sql/app/streamlit/streamlit_app.py ────
    └── ConversationManager (nlp_sql_postgres_v2.py:L303)
          ├── DuckDB in-memory + postgres scanner
          ├── ATTACH DATABASE_URL AS rag_db (READ_ONLY)
          ├── Pydantic AI Agent (same LLM settings as App 1)
          └── 3-attempt retry + guardrails (_check_readonly, _apply_row_cap)

Ingestion CLI  rag/main.py:L100
  └── DocumentIngestionPipeline L136  └── PostgresHybridStore L116

KG Build CLI  kg/legal/ingestion/cuad_kg_ingest.py
  └── build_cuad_kg()  └── AgeGraphStore (AGE-only)
```

---

## 7. Knowledge Graph — Build (build_cuad_kg)

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

## 8. Knowledge Graph — Query (AgeGraphStore / PgGraphStore)

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

## 9. Agent KG Tools (search_knowledge_graph + run_graph_query)

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
