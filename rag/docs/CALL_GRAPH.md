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
- [8. Knowledge Graph — Query (AgeGraphStore)](#8-knowledge-graph--query-agegraphstore)
- [9. Agent KG Tools (search_knowledge_graph)](#9-agent-kg-tools-search_knowledge_graph)

---

## 1. Document Ingestion

> See [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: `python -m rag.main --ingest`

```
rag/main.py:main()                                                L100
  ├── validate_config()                                           L46
  │     └── load_settings()
  └── run_ingestion_pipeline()                                    L646
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
| [`rag/main.py`](../rag/main.py#L646) | `run_ingestion_pipeline()` | L646 |
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

> See [DATASTORE_GUIDE.md](DATASTORE_GUIDE.md) for details.

**Entry point**: [`Retriever.retrieve()`](../rag/retrieval/retriever.py#L234)

```
Retriever.retrieve(query, match_count, search_type, use_cache)    L234
  ├── 1. ResultCache.get(query, search_type, match_count)          L266
  │        └── cache hit? → return list[SearchResult]
  │
  ├── 2. Query embedding
  │     └── EmbeddingGenerator.embed_query(query)                  L263
  │           ├── _cached_embed(text, model)  async_lru(1000)
  │           └── openai.AsyncOpenAI.embeddings.create()
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
  ├── 5. Rerank  (if reranker_enabled=True; off by default)
  │     ├── [reranker_type == "llm"]  ← default; no extra deps
  │     │     └── LLMReranker.rerank(query, results, top_k)
  │     │           └── asyncio.gather(*[_score_document(...) for each result])
  │     └── [reranker_type == "cross_encoder"]  requires sentence-transformers
  │           └── CrossEncoderReranker.rerank(query, results, top_k)
  │
  ├── 6. ResultCache.set(...)
  └── return list[SearchResult]

Retriever.retrieve_as_context(query, match_count, search_type)    L353
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
                    │           → search_knowledge_base(ctx, query, ...)  L247
                    │               ├── RAGState.get_retriever()           L202
                    │               │     └── PostgresHybridStore.initialize()
                    │               ├── retriever.retrieve_as_context()    L353
                    │               │     └── [see Query & Retrieval]
                    │               ├── mem0_store.get_context_string()
                    │               └── return combined_context
                    │         FunctionToolResultEvent
                    ├── ModelRequestNode  (LLM final answer)
                    └── EndNode

RAGState lazy init (first get_retriever() call):                   L202
  ├── PostgresHybridStore().initialize()
  ├── EmbeddingGenerator()                                         L135
  ├── Retriever(store, embedder)                                   L181
  └── Mem0Store()  (if mem0_enabled)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L247) | `search_knowledge_base()` tool | L247 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L177) | `RAGState` | L177 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L202) | `RAGState.get_retriever()` | L202 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L663) | `traced_agent_run()` | L663 |
| [`rag/agent/agent_main.py`](../rag/agent/agent_main.py#L75) | `stream_agent_interaction()` | L75 |
| [`rag/storage/vector_store/postgres.py`](../rag/storage/vector_store/postgres.py#L116) | `PostgresHybridStore` | L116 |
| [`rag/ingestion/embedder.py`](../rag/ingestion/embedder.py#L135) | `EmbeddingGenerator` | L135 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L181) | `Retriever` | L181 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L234) | `Retriever.retrieve()` | L234 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L353) | `Retriever.retrieve_as_context()` | L353 |
| [`rag/retrieval/retriever.py`](../rag/retrieval/retriever.py#L96) | `ResultCache` | L96 |
| [`rag/memory/mem0_store.py`](../rag/memory/mem0_store.py#L94) | `Mem0Store` | L94 |

---

## 4. Mem0 Memory

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
PydanticAI Agent  rag/agent/rag_agent.py:L177
    │
    ├──────────────────────┬──────────────────────┬────────────────────┐
    ▼                      ▼                      ▼                    ▼
search_knowledge_base()  search_knowledge_graph() run_graph_query()  Mem0Store
L247                     L349                    L478                L94
    │                      │                      │                    │
    ▼                      ▼                      ▼                    │
Retriever  L181      create_kg_store()      create_kg_store()         │
    │                 └── AgeGraphStore L139  └── AgeGraphStore L139  │
    ├── Embedder            search_entities        run_cypher_query    │
    │                       get_related_entities   L574                │
    ▼                                                                  ▼
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
| [`kg/legal/ingestion/cuad_kg_ingest.py`](../../kg/legal/ingestion/cuad_kg_ingest.py) | `build_cuad_kg()` | L61 |
| [`kg/legal/ingestion/cuad_kg_ingest.py`](../../kg/legal/ingestion/cuad_kg_ingest.py) | `_get_document_id()` | L41 |
| [`kg/legal/common/cuad_ontology.py`](../../kg/legal/common/cuad_ontology.py) | `entity_type_for()` | L141 |
| [`kg/legal/common/cuad_ontology.py`](../../kg/legal/common/cuad_ontology.py) | `relationship_type_for()` | L146 |
| [`kg/legal/common/cuad_ontology.py`](../../kg/legal/common/cuad_ontology.py) | `ENTITY_TYPE_MAP` | L83 |
| [`kg/legal/common/cuad_ontology.py`](../../kg/legal/common/cuad_ontology.py) | `RELATIONSHIP_MAP` | L127 |
| [`kg/legal/ingestion/cuad_kg_ingest.py`](../../kg/legal/ingestion/cuad_kg_ingest.py) | `main()` | L162 |
| [`kg/__init__.py`](../../kg/__init__.py) | `create_kg_store()` | L79 |

---

## 8. Knowledge Graph — Query (AgeGraphStore)

Both stores share the same public interface; swap via `KG_BACKEND` env var.

```
── AgeGraphStore  kg/age_graph_store.py:L139 ───────────────────────

initialize()                                                L161
  ├── asyncpg.create_pool(init=_age_init)           L133 (_age_init)
  │     └── _age_init(conn)   per-connection setup
  │           ├── LOAD '$libdir/plugins/age'
  │           └── SET search_path = ag_catalog, "$user", public
  └── ag_catalog.create_graph(graph_name)

upsert_entity(name, entity_type, document_id, metadata) → UUID       L252
  ├── _normalize(name)                             L57
  └── _conn() → _cypher("MERGE (e:Entity {…})")    L234
        └── _unquote_agtype(result)                L123

add_relationship(src_id, tgt_id, rel_type, document_id, props) → UUID  L305
  └── _cypher("MATCH (s),(t) MERGE (s)-[r:REL_TYPE {…}]->(t)")

search_entities(query, entity_type, limit) → list[dict]               L361
  └── MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($query)
        [optional] AND e.entity_type = $entity_type

get_related_entities(entity_id, rel_type, limit) → list[dict]          L422
  └── MATCH (e)-[r]-(other)  WHERE id(e) = $id
        [optional] AND type(r) = $rel_type

find_contracts_by_entity(entity_name, entity_type, limit) → list[dict]  L463
  ├── _normalize(entity_name)
  └── MATCH (e:Entity {normalized_name:$n})-[]->(c:Entity {entity_type:"Contract"})

search_as_context(query, limit) → str                                  L503
  ├── search_entities(query, limit=limit)
  ├── [for each entity]: get_related_entities(entity_id, limit=5)
  └── format as "## Knowledge Graph — Facts\n- [TYPE] name\n  └─ REL → target"
        fallback: bullet list of entities if no relationships found

get_graph_stats() → dict                                               L542
  ├── MATCH (e:Entity) RETURN e.entity_type, count(*)
  └── MATCH ()-[r]->() RETURN type(r), count(*)

run_cypher_query(cypher) → str                                         L574
  ├── guard: block CREATE/MERGE/SET/DELETE/REMOVE/DROP/DETACH
  ├── _parse_return_aliases(cypher)                L74  → list of display names
  │     ├── regex-find RETURN clause
  │     ├── paren-depth comma split
  │     └── extract AS alias or last identifier token
  ├── build AS (c0 agtype, c1 agtype, …) from alias count
  ├── _conn() → conn.fetch(_cypher(cypher) + AS clause)
  └── format as pipe-separated table: "col1 | col2\n---\nv1 | v2\n(N rows)"
```

**Internal helpers (AgeGraphStore)**:

| Helper | Line | Purpose |
|--------|------|---------|
| `_normalize(name)` | L57 | `lower(re.sub(r"\s+", " ", name.strip()))` |
| `_unquote_agtype(value)` | L123 | strips surrounding `"` from AGE agtype strings |
| `_age_init(conn)` | L133 | asyncpg pool `init=` callback; loads AGE extension + sets search_path |
| `_conn()` | L234 | async context manager; acquires connection + re-runs AGE setup |
| `_cypher(body)` | L247 | wraps body in `SELECT * FROM ag_catalog.cypher('graph', $$…$$, NULL) AS (v agtype)` |
| `_parse_return_aliases(cypher)` | L74 | parses RETURN clause → display name list for the AS column declaration |

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L139) | `AgeGraphStore` | L139 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L74) | `_parse_return_aliases()` | L74 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L123) | `_unquote_agtype()` | L123 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L133) | `_age_init()` | L133 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L234) | `_conn()` | L234 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L247) | `_cypher()` | L247 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L574) | `run_cypher_query()` | L574 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L57) | `_normalize()` | L57 |

---

## 9. Agent KG Tools (search_knowledge_graph + run_graph_query)

**Entry point**: Two Pydantic AI tools registered on the RAG agent for KG access.

```
── Tool: search_knowledge_graph ─────────────────────────────────── L347
search_knowledge_graph(ctx, query, entity_type, limit)
  ├── RAGState.get_kg_store()  (lazy init)                          L216
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
  ├── RAGState.get_kg_store()  (lazy init, same cached instance)    L216
  └── kg.run_cypher_query(cypher)                                   L574
        ├── guard: block mutating keywords
        ├── _parse_return_aliases(cypher)  → display name list      L74
        ├── build AS (c0 agtype, …) clause
        ├── conn.fetch(_cypher(cypher) + AS clause)
        └── → pipe-separated table string

── Shared lazy init ──────────────────────────────────────────────── L216
RAGState.get_kg_store()
  ├── [first call] create_kg_store()  reads KG_BACKEND
  │     ├── [default "age"]      → AgeGraphStore().initialize()

  └── cache result in self._kg_store  (reused for all KG tool calls)
```

**Key files**:

| File | Symbol | Line |
|------|--------|------|
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L347) | `search_knowledge_graph()` tool | L347 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L409) | `run_graph_query()` tool | L409 |
| [`rag/agent/rag_agent.py`](../rag/agent/rag_agent.py#L216) | `RAGState.get_kg_store()` | L216 |
| [`kg/age_graph_store.py`](../../kg/age_graph_store.py#L574) | `run_cypher_query()` | L574 |
| [`kg/__init__.py`](../../kg/__init__.py) | `create_kg_store()` | L79 |
