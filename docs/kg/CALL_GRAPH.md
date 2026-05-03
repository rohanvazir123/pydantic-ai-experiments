# Knowledge Graph — Call Graph

## KG Population (during ingestion)

```
rag/ingestion/pipeline.py:DocumentIngestionPipeline.ingest_documents()
    └── rag/knowledge_graph/pipeline.py:KGExtractionPipeline.extract_and_store(chunk)
            │
            ├── _extract_entities(chunk_text)
            │       └── agent.run(extraction_prompt)       [Pydantic AI → LLM]
            │               → list[{"name": str, "type": str}]
            │
            ├── AgeGraphStore.upsert_entity(name, entity_type, document_id)
            │       └── age_graph_store.py:~255
            │               ├── _safe_label(entity_type)   [whitelist validation]
            │               ├── _normalize(name)           [lowercase + strip]
            │               └── _conn() context manager
            │                       ├── LOAD 'age'
            │                       ├── SET search_path = ag_catalog, ...
            │                       └── asyncpg.fetch(
            │                               "SELECT * FROM ag_catalog.cypher('graph', $$"
            │                               "  MERGE (e:{label} {normalized_name, document_id})"
            │                               "  SET e.uuid = COALESCE(e.uuid, {new_uuid}), ..."
            │                               "  RETURN e.uuid"
            │                               "$$) AS (uuid agtype)"
            │                           )
            │
            └── AgeGraphStore.add_relationship(src_uuid, tgt_uuid, rel_type)
                    └── age_graph_store.py:~300
                            ├── _safe_rel_type(rel_type)   [whitelist validation]
                            └── asyncpg.fetch(
                                    "MATCH (s {uuid: src}), (t {uuid: tgt})"
                                    "CREATE (s)-[r:{rel_type} {uuid, document_id}]->(t)"
                                    "RETURN r.uuid"
                                )
```

## Entity Search (API + Streamlit)

```
apps/kg/api.py:search()   OR   apps/kg/streamlit_app.py:_show_search()
    └── AgeGraphStore.search_entities(query, entity_type, limit)
            └── age_graph_store.py:~354
                    ├── _safe_label(entity_type)          [label filter if provided]
                    └── _conn()
                            └── asyncpg.fetch(
                                    "MATCH (e:{label})"        [or MATCH (e) if no type]
                                    "WHERE toLower(e.name) CONTAINS {query_esc}"
                                    "RETURN e.uuid, e.name, e.label, e.document_id"
                                    "LIMIT {limit}"
                                )
```

## Context Retrieval (RAG agent tool + API)

```
rag/agent/rag_agent.py:search_knowledge_graph()   OR   apps/kg/api.py:context()
    └── AgeGraphStore.search_as_context(query, limit)
            └── age_graph_store.py:~480
                    └── _conn()
                            └── asyncpg.fetch(
                                    "MATCH (e)-[r]->(t)"
                                    "WHERE toLower(e.name) CONTAINS {query_esc}"
                                    "   OR toLower(t.name) CONTAINS {query_esc}"
                                    "RETURN e.name, e.label, type(r), t.name, t.label"
                                    "LIMIT {limit}"
                                )
                    └── [fallback if no rows] search_entities(query)
                    └── → "## Knowledge Graph — Facts\n- [Party] Amazon --PARTY_TO--> ..."
```

## Custom Cypher (Streamlit REPL + API)

```
apps/kg/streamlit_app.py:_show_cypher()   OR   apps/kg/api.py:cypher()
    └── AgeGraphStore.run_cypher_query(cypher)
            └── age_graph_store.py:~551
                    ├── re.search(CREATE|MERGE|SET|DELETE|...)  [read-only guardrail]
                    ├── _parse_return_aliases(cypher)
                    │       └── regex: RETURN clause → alias list
                    │               ["e.name", "type(r)", "c.name"] → ["name", "r", "name"]
                    └── _conn()
                            └── asyncpg.fetch(
                                    "SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$)"
                                    "AS ({alias} agtype, ...)"
                                )
                    └── → "name | r | name\n--------\nAmazon | PARTY_TO | Contract A\n(1 row)"
```

## Graph Statistics

```
apps/kg/api.py:stats()   OR   apps/kg/streamlit_app.py:_show_stats()
    └── AgeGraphStore.get_graph_stats()
            └── age_graph_store.py:~519
                    ├── asyncpg.fetch("MATCH (e) RETURN e.label, count(*)")
                    └── asyncpg.fetch("MATCH ()-[r]->() RETURN type(r), count(*)")
                    └── → {total_entities, total_relationships, entities_by_type, relationships_by_type}
```

## Pool Initialization

```
AgeGraphStore._do_initialize()
    └── asyncpg.create_pool(
            age_database_url,
            init=_age_init,              ← called for every new connection in the pool
        )
            └── _age_init(conn)
                    ├── conn.execute("LOAD 'age'")
                    └── conn.execute("SET search_path = ag_catalog, \"$user\", public")

AgeGraphStore._conn()                    ← context manager used by all methods
    └── pool.acquire() as conn
            ├── conn.execute("LOAD 'age'")     ← re-applied on every acquire
            └── conn.execute("SET search_path = ...")   ← asyncpg resets state on return
```

The double setup (pool init + per-acquire) is intentional: asyncpg's `RESET ALL`
on connection return clears both the loaded extension and the search_path.

## Key File Locations

| Symbol | File | Line |
|---|---|---|
| `AgeGraphStore` | `rag/knowledge_graph/age_graph_store.py` | ~154 |
| `AgeGraphStore.initialize` | `rag/knowledge_graph/age_graph_store.py` | ~180 |
| `AgeGraphStore.upsert_entity` | `rag/knowledge_graph/age_graph_store.py` | ~255 |
| `AgeGraphStore.add_relationship` | `rag/knowledge_graph/age_graph_store.py` | ~300 |
| `AgeGraphStore.search_entities` | `rag/knowledge_graph/age_graph_store.py` | ~354 |
| `AgeGraphStore.search_as_context` | `rag/knowledge_graph/age_graph_store.py` | ~480 |
| `AgeGraphStore.get_graph_stats` | `rag/knowledge_graph/age_graph_store.py` | ~519 |
| `AgeGraphStore.run_cypher_query` | `rag/knowledge_graph/age_graph_store.py` | ~551 |
| `_age_init` | `rag/knowledge_graph/age_graph_store.py` | ~148 |
| `_parse_return_aliases` | `rag/knowledge_graph/age_graph_store.py` | ~89 |
| `create_kg_store` factory | `rag/knowledge_graph/__init__.py` | ~1 |
| KG extraction pipeline | `rag/knowledge_graph/pipeline.py` | ~1 |
| FastAPI app | `apps/kg/api.py` | ~1 |
| Streamlit app | `apps/kg/streamlit_app.py` | ~1 |
