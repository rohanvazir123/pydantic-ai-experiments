# RAG + PDF Question Generator — Call Graph

## Chat Query (full path)

```
apps/rag/api.py:chat()  OR  apps/rag/streamlit_app.py:stream_agent_response()
    └── rag/agent/rag_agent.py:traced_agent_run()               [Langfuse trace start]
            └── agent.run() / agent.run_stream()                [Pydantic AI]
                    │
                    ├── [tool call] search_knowledge_base()     rag_agent.py:~290
                    │       ├── RAGState.get_retriever()        rag_agent.py:~150
                    │       │       └── Retriever.__init__()    retriever.py:~50
                    │       │               ├── PostgresHybridStore.initialize()
                    │       │               └── asyncpg.create_pool(..., init=register_vector)
                    │       │
                    │       ├── Retriever.retrieve()            retriever.py:~120
                    │       │       ├── _hyde_query()           retriever.py:~200
                    │       │       │       └── LLM.generate(hypothetical answer)
                    │       │       ├── _semantic_search()      retriever.py:~240
                    │       │       │       └── PostgresHybridStore.search_semantic()
                    │       │       │               └── asyncpg: SELECT ... ORDER BY embedding <=> $1
                    │       │       ├── _text_search()          retriever.py:~270
                    │       │       │       └── PostgresHybridStore.search_text()
                    │       │       │               └── asyncpg: SELECT ... ts_rank(tsv, query)
                    │       │       ├── _rrf_fusion()           retriever.py:~300
                    │       │       └── _rerank()               retriever.py:~310
                    │       │               └── asyncio.gather(LLM.score() × N)
                    │       │
                    │       └── Mem0Store.get_context()         mem0_store.py (if MEM0_ENABLE)
                    │
                    ├── [tool call] search_knowledge_graph()    rag_agent.py:~350
                    │       └── AgeGraphStore.search_as_context()  age_graph_store.py:~480
                    │               ├── _conn() context manager (LOAD 'age' + search_path)
                    │               └── asyncpg: SELECT * FROM ag_catalog.cypher(...)
                    │
                    └── [tool call] run_graph_query()           rag_agent.py:~400
                            └── AgeGraphStore.run_cypher_query()  age_graph_store.py:~551
                                    ├── _parse_return_aliases()
                                    ├── read-only guardrail check
                                    └── asyncpg: SELECT * FROM ag_catalog.cypher(...)
```

## Direct Retrieval (`/v1/retrieve`)

```
apps/rag/api.py:retrieve()
    └── RAGState.get_retriever()
            └── Retriever.retrieve(query, search_type, match_count)
                    ├── [hybrid]  _hyde_query → _semantic_search + _text_search → _rrf_fusion → _rerank
                    ├── [semantic] embed(query) → pgvector cosine search
                    └── [text]    tsvector plainto_tsquery search
```

## Document Ingestion

```
apps/rag/api.py:ingest()  OR  notebook/ingestion_pipeline.ipynb
    └── create_pipeline()                                    pipeline.py:~50
            └── DocumentIngestionPipeline.ingest_documents()  pipeline.py:~120
                    ├── _get_converter()                     [cached Docling DocumentConverter]
                    │       └── DocumentConverter.convert(pdf_path)
                    │               └── → DocumentConversionResult → markdown
                    ├── _chunk_document()                    pipeline.py:~200
                    │       └── RecursiveCharacterSplitter.split()
                    ├── _embed_chunks()                      pipeline.py:~250
                    │       └── httpx.post(embedding_base_url/embeddings)  [batch]
                    └── PostgresHybridStore.upsert_chunks()  postgres.py:~150
                            └── asyncpg.executemany(INSERT INTO chunks ...)
```

## KG Extraction (during ingestion)

```
KGExtractionPipeline.extract_and_store()      kg/pipeline.py
    ├── LLM.extract_entities(chunk_text)       → list[Entity]
    ├── AgeGraphStore.upsert_entity()          age_graph_store.py:~255
    │       └── asyncpg: cypher MERGE (e:Label {...}) SET ...
    └── AgeGraphStore.add_relationship()       age_graph_store.py:~300
            └── asyncpg: cypher MATCH (s) MATCH (t) CREATE (s)-[r:TYPE {}]->(t)
```

## Streamlit Streaming

```
apps/rag/streamlit_app.py:render_chat()
    └── asyncio.run(stream_agent_response())
            └── agent.iter(user_input, deps=deps, message_history=...)
                    ├── UserPromptNode      → (no UI update)
                    ├── ModelRequestNode   → node.stream(ctx)
                    │       └── PartDeltaEvent / TextPartDelta → response_placeholder.markdown()
                    ├── CallToolsNode      → node.stream(ctx)
                    │       ├── FunctionToolCallEvent  → status_placeholder.info()
                    │       └── FunctionToolResultEvent → status_placeholder.success()
                    └── EndNode            → (no UI update)
```

## Key File Locations

| Symbol | File | Line |
|---|---|---|
| `agent` (Pydantic AI agent) | `rag/agent/rag_agent.py` | ~230 |
| `RAGState` | `rag/agent/rag_agent.py` | ~100 |
| `search_knowledge_base` tool | `rag/agent/rag_agent.py` | ~290 |
| `search_knowledge_graph` tool | `rag/agent/rag_agent.py` | ~350 |
| `run_graph_query` tool | `rag/agent/rag_agent.py` | ~400 |
| `traced_agent_run` | `rag/agent/rag_agent.py` | ~450 |
| `Retriever.retrieve` | `rag/retrieval/retriever.py` | ~120 |
| `PostgresHybridStore` | `rag/storage/vector_store/postgres.py` | ~50 |
| `AgeGraphStore` | `rag/knowledge_graph/age_graph_store.py` | ~154 |
| `DocumentIngestionPipeline` | `rag/ingestion/pipeline.py` | ~50 |
| `Mem0Store` | `rag/memory/mem0_store.py` | ~1 |
