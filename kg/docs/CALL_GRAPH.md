# Knowledge Graph — Call Graph

## KG Population (`python -m kg.extraction_pipeline`)

```
ExtractionPipeline.process_contract(contract_id, title, text)
    │
    ├── [Bronze] _process_chunk(contract_id, i, chunk_text) × N chunks
    │       ├── _pass_entities(chunk)
    │       │       └── _entity_agent.run(prompt)    → list[ExtractedEntity]
    │       ├── _pass_relationships(chunk, entities)
    │       │       └── _rel_agent.run(prompt)        → list[ExtractedRelationship]
    │       ├── _pass_hierarchy(chunk)
    │       │       └── _hierarchy_agent.run(prompt)  → (list[HierarchyNode], list[HierarchyEdge])
    │       ├── _pass_cross_refs(chunk, contract_id)
    │       │       └── _cross_ref_agent.run(prompt)  → list[CrossContractRef]
    │       └── _pass_validate(chunk, entities, rels)
    │               └── _validation_agent.run(prompt) → filtered list[ExtractedRelationship]
    │       BronzeArtifact → BronzeStore.save()       → kg_raw_extractions (JSONB)
    │
    ├── _save_json(contract_id, title, artifacts)
    │       → entity_relationships/jsons/<title>_<id[:8]>.json
    │
    ├── [Silver] SilverNormalizer.normalize(contract_id, artifacts)
    │       ├── INSERT kg_staging_entities / kg_staging_relationships (raw, per-chunk)
    │       └── DISTINCT ON (label, normalized_name) → INSERT kg_canonical_entities
    │           JOIN staging → deduplicate → INSERT kg_canonical_relationships
    │
    ├── [Gold] GoldProjector.project(contract_id)
    │       ├── SELECT kg_canonical_entities → AgeGraphStore.upsert_entity()
    │       │       └── MERGE (e:{label} {normalized_name, document_id}) SET ...
    │       └── SELECT kg_canonical_relationships → AgeGraphStore.add_relationship()
    │               └── MATCH (s {uuid}), (t {uuid}) CREATE (s)-[r:{type} {...}]->(t)
    │
    └── RiskGraphBuilder.build(contract_id)
            └── Rule-based: LiabilityClause / IndemnityClause → Risk nodes
                            + INCREASES_RISK_FOR / CAUSES edges → AGE
```

### Replay (Silver + Gold only, no LLM)

```
ExtractionPipeline.project_contract(contract_id)
    ├── BronzeStore.load_for_contract(contract_id)  → list[BronzeArtifact]
    ├── SilverNormalizer.normalize(...)
    ├── GoldProjector.project(...)
    └── RiskGraphBuilder.build(...)
```

## NL → Cypher Query

```
NL2CypherConverter.convert(question)
    ├── IntentParser.parse(question)           → IntentMatch(intent, params)
    └── QUERY_CAPABILITIES[intent](params)     → Cypher string

AgeGraphStore.run_cypher_query(cypher)
    ├── re.search(CREATE|MERGE|SET|DELETE|…)   [read-only guardrail]
    ├── _parse_return_aliases(cypher)          → AS clause aliases
    └── asyncpg: SELECT * FROM ag_catalog.cypher(graph, $$ cypher $$) AS (...)
```

## Entity Search (API + Streamlit)

```
AgeGraphStore.search_entities(query, entity_type, limit)
    └── MATCH (e:{label}) WHERE toLower(e.name) CONTAINS {query}
        RETURN e.uuid, e.name, e.label, e.document_id LIMIT {limit}
```

## Context Retrieval (RAG agent tool + API)

```
AgeGraphStore.search_as_context(query, limit)
    └── MATCH (e)-[r]->(t)
        WHERE toLower(e.name) CONTAINS {query} OR toLower(t.name) CONTAINS {query}
        RETURN e.name, e.label, type(r), t.name, t.label LIMIT {limit}
        → "## Knowledge Graph — Facts\n- [Party] Acme Corp --SIGNED_BY--> ..."
```

## AGE Pool Initialization

```
AgeGraphStore._do_initialize()
    └── asyncpg.create_pool(age_database_url, init=_age_init)
            └── _age_init(conn)          ← called for every new connection
                    ├── LOAD 'age'
                    └── SET search_path = ag_catalog, "$user", public

AgeGraphStore._conn()                    ← context manager on every acquire
    └── pool.acquire()
            ├── LOAD 'age'               ← re-applied (asyncpg RESET ALL on return)
            └── SET search_path = ...
```

## Key Files

| Symbol | File |
|---|---|
| `ExtractionPipeline` | `kg/extraction_pipeline.py` |
| `BronzeStore` | `kg/extraction_pipeline.py` |
| `SilverNormalizer` | `kg/extraction_pipeline.py` |
| `GoldProjector` | `kg/extraction_pipeline.py` |
| `RiskGraphBuilder` | `kg/risk_graph_builder.py` |
| `AgeGraphStore` | `kg/age_graph_store.py` |
| `NL2CypherConverter` | `kg/nl2cypher.py` |
| `IntentParser` | `kg/intent_parser.py` |
| `QUERY_CAPABILITIES` | `kg/query_builder.py` |
| `VALID_LABELS`, `VALID_REL_TYPES` | `kg/constants.py` |
| `create_kg_store()` | `kg/__init__.py` |
| FastAPI app | `apps/kg/api.py` |
| Streamlit app | `apps/kg/streamlit_app.py` |
