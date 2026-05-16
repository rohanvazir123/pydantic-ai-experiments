# Knowledge Graph — Call Graph

Method-level call graphs for all KG workflows.  Line numbers reference the source files as of 2026-05-16.

---

## KG Population (`python -m kg.legal.ingestion.extraction_pipeline`)

```
ExtractionPipeline.process_contract(contract_id, title, text)    L885
    │
    ├── [Bronze] _process_chunk(contract_id, i, chunk_text) × N  L837
    │       ├── _pass_entities(chunk)                             L751
    │       │       └── _entity_agent.run(prompt)    → list[ExtractedEntity]
    │       ├── _pass_relationships(chunk, entities)              L763
    │       │       └── _rel_agent.run(prompt)        → list[ExtractedRelationship]
    │       ├── _pass_hierarchy(chunk)                            L779
    │       │       └── _hierarchy_agent.run(prompt)  → (list[HierarchyNode], list[HierarchyEdge])
    │       ├── _pass_cross_refs(chunk, contract_id)              L796
    │       │       └── _cross_ref_agent.run(prompt)  → list[CrossContractRef]
    │       └── _pass_validate(chunk, entities, rels)             L811
    │               └── _validation_agent.run(prompt) → filtered list[ExtractedRelationship]
    │       BronzeArtifact → BronzeStore.save()                   L341
    │               → INSERT kg_raw_extractions (JSONB)
    │
    ├── _save_json(contract_id, title, artifacts)                 L862
    │       → kg/evals/jsons/<title>_<id[:8]>.json
    │
    ├── [Silver] SilverNormalizer.normalize(contract_id, artifacts)  L468
    │       ├── INSERT kg_staging_entities / kg_staging_relationships (raw, per-chunk)
    │       └── DISTINCT ON (label, normalized_name) → INSERT kg_canonical_entities
    │           JOIN staging → deduplicate → INSERT kg_canonical_relationships
    │
    ├── [Gold] GoldProjector.project(contract_id)                 L658
    │       ├── SELECT kg_canonical_entities → AgeGraphStore.upsert_entity()
    │       │       └── MERGE (e:{label} {normalized_name, document_id}) SET ...
    │       └── SELECT kg_canonical_relationships → AgeGraphStore.add_relationship()
    │               └── MATCH (s {uuid}), (t {uuid}) CREATE (s)-[r:{type} {...}]->(t)
    │
    └── RiskGraphBuilder.build(contract_id)                       L46  (risk_graph_builder.py)
            └── Rule-based: LiabilityClause / IndemnityClause → Risk nodes
                            + INCREASES_RISK_FOR / CAUSES edges → AGE
```

### Silver + Gold replay (no LLM, fast)

```
ExtractionPipeline.project_contract(contract_id)                 L933
    ├── BronzeStore.load_for_contract(contract_id)                L360  → list[BronzeArtifact]
    ├── SilverNormalizer.normalize(...)                           L468
    ├── GoldProjector.project(...)                                L658
    └── RiskGraphBuilder.build(...)                               L46
```

### Contract loading

```
_fetch_contracts(pool, contract_ids, limit)                       L948
    └── SELECT document_id, title, content FROM chunks
        WHERE document_id IN (...) ORDER BY chunk_index
        → list[(contract_id, title, list[str chunk_text])]
```

---

## CUAD Fast Ingest (`python -m kg.legal.ingestion.cuad_kg_ingest`)

```
build_cuad_kg(store, cuad_eval_path)                             L61  (cuad_kg_ingest.py)
    ├── json.load(cuad_eval.json)       → annotations dict
    ├── AgeGraphStore.upsert_entity()   → vertex per annotation
    └── AgeGraphStore.add_relationship() → edge per annotation pair
    (No LLM.  No Bronze/Silver tables.  Direct JSON → AGE.)
```

---

## NL → Cypher Query

```
NL2CypherConverter.convert(question, schema="")                  L26  (nl2cypher.py)
    ├── IntentParser.parse(question)                              L113 (intent_parser.py)
    │       → IntentMatch(intent, params)
    └── QUERY_CAPABILITIES[intent](params)                        (query_builder.py)
            → Cypher string

AgeGraphStore.run_cypher_query(cypher)                           L574 (age_graph_store.py)
    ├── re.search(CREATE|MERGE|SET|DELETE|…)  [read-only guardrail]
    ├── _parse_return_aliases(cypher)                             L74  (age_graph_store.py)
    │       → AS clause aliases
    └── asyncpg: SELECT * FROM ag_catalog.cypher(graph, $$ cypher $$) AS (...)
```

---

## Entity Search (API + Streamlit)

```
AgeGraphStore.search_entities(query, entity_type, limit)         L361 (age_graph_store.py)
    └── MATCH (e:{label}) WHERE toLower(e.name) CONTAINS {query}
        RETURN e.uuid, e.name, e.label, e.document_id LIMIT {limit}
```

---

## Context Retrieval (RAG agent tool + API)

```
AgeGraphStore.search_as_context(query, limit)                    L503 (age_graph_store.py)
    └── MATCH (e)-[r]->(t)
        WHERE toLower(e.name) CONTAINS {query} OR toLower(t.name) CONTAINS {query}
        RETURN e.name, e.label, type(r), t.name, t.label LIMIT {limit}
        → "## Knowledge Graph — Facts\n- [Party] Acme Corp --SIGNED_BY--> ..."
```

---

## AGE Pool Initialization

```
AgeGraphStore._do_initialize()                                   L168 (age_graph_store.py)
    └── asyncpg.create_pool(age_database_url, init=_age_init)
            └── _age_init(conn)                                   L133 (age_graph_store.py)
                    ├── LOAD 'age'
                    └── SET search_path = ag_catalog, "$user", public

AgeGraphStore._conn()                                            L234 (age_graph_store.py)
    └── pool.acquire()
            ├── LOAD 'age'               ← re-applied (asyncpg RESET ALL on return)
            └── SET search_path = ...
```

---

## Key Files

| Symbol | File | Line |
|---|---|---|
| `ExtractionPipeline` | `kg/legal/ingestion/extraction_pipeline.py` | L719 |
| `BronzeStore` | `kg/legal/ingestion/extraction_pipeline.py` | L315 |
| `SilverNormalizer` | `kg/legal/ingestion/extraction_pipeline.py` | L387 |
| `GoldProjector` | `kg/legal/ingestion/extraction_pipeline.py` | L651 |
| `process_contract()` | `kg/legal/ingestion/extraction_pipeline.py` | L885 |
| `project_contract()` | `kg/legal/ingestion/extraction_pipeline.py` | L933 |
| `_fetch_contracts()` | `kg/legal/ingestion/extraction_pipeline.py` | L948 |
| `_save_json()` | `kg/legal/ingestion/extraction_pipeline.py` | L862 |
| `RiskGraphBuilder` | `kg/legal/ingestion/risk_graph_builder.py` | L46 |
| `build_cuad_kg()` | `kg/legal/ingestion/cuad_kg_ingest.py` | L61 |
| `AgeGraphStore` | `kg/age_graph_store.py` | L168 |
| `run_cypher_query()` | `kg/age_graph_store.py` | L574 |
| `search_entities()` | `kg/age_graph_store.py` | L361 |
| `search_as_context()` | `kg/age_graph_store.py` | L503 |
| `NL2CypherConverter` | `kg/legal/retrieval/nl2cypher.py` | L26 |
| `IntentParser` | `kg/legal/retrieval/intent_parser.py` | L113 |
| `QUERY_CAPABILITIES` | `kg/legal/retrieval/query_builder.py` | — |
| `VALID_LABELS`, `VALID_REL_TYPES` | `kg/legal/common/cuad_ontology.py` | — |
| `create_kg_store()` | `kg/__init__.py` | — |
| Ingestion eval | `kg/legal/ingestion/eval_pipeline.py` | — |
| Retrieval eval | `kg/legal/retrieval/eval_pipeline.py` | — |
| Retrieval CLI | `kg/legal/retrieval/cli.py` | — |
| FastAPI app | `apps/kg/api.py` | — |
| Streamlit app | `apps/kg/streamlit_app.py` | — |
