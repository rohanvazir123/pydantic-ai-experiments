# Knowledge Graph — Architecture

## Overview

A property graph built over ingested legal contracts using Apache AGE
(a PostgreSQL extension that adds native openCypher query support).
KG population is a separate CLI step from document ingestion — run
`python -m kg.extraction_pipeline` after ingestion is complete.

## Stack

| Layer | Technology |
|---|---|
| Graph database | Apache AGE (PostgreSQL 15 + AGE extension, port 5433) |
| Query language | openCypher (via `ag_catalog.cypher()` SQL wrapper) |
| Client | asyncpg |
| LLM (extraction) | Ollama or any OpenAI-compatible API |
| UI | Streamlit (`apps/kg/streamlit_app.py`) |
| REST API | FastAPI (`apps/kg/api.py`) |

## Module Layout

```
kg/
├── extraction_pipeline.py  — ExtractionPipeline: Bronze / Silver / Gold
├── age_graph_store.py      — AgeGraphStore: all AGE Cypher operations
├── risk_graph_builder.py   — RiskGraphBuilder: rule-based risk inference
├── constants.py            — VALID_LABELS, VALID_REL_TYPES (single source of truth)
├── nl2cypher.py            — NL2CypherConverter: question → Cypher (no LLM)
├── intent_parser.py        — IntentParser: regex intent matching
├── query_builder.py        — QUERY_CAPABILITIES: intent → Cypher builder
├── graph_router.py         — GraphRouter: question → list[GraphType]
├── schemas.py              — GraphType enum, get_schema()
├── cuad_kg_ingest.py       — build_cuad_kg(): CUAD annotations → AGE (no LLM)
└── __init__.py             — create_kg_store() factory
```

## Bronze / Silver / Gold Pipeline

The `ExtractionPipeline` follows a three-tier medallion architecture.
Each tier is independent — Bronze is always written first; Silver and Gold
can be replayed from Bronze without re-running the LLM.

### Bronze — raw per-chunk extraction (LLM)

For each 1 500-character chunk of a contract the pipeline runs five
sequential LLM passes using Pydantic AI agents:

| Pass | Agent | Output |
|---|---|---|
| 1 — Entities | `_entity_agent` | `list[ExtractedEntity]` |
| 2 — Relationships | `_rel_agent` | `list[ExtractedRelationship]` |
| 3 — Hierarchy | `_hierarchy_agent` | `list[HierarchyNode]`, `list[HierarchyEdge]` |
| 4 — Cross-contract refs | `_cross_ref_agent` | `list[CrossContractRef]` |
| 5 — Validation | `_validation_agent` | filtered `list[ExtractedRelationship]` |

Each chunk produces one `BronzeArtifact` which is written to two places:

- **PostgreSQL** — `kg_raw_extractions` (JSONB, deduplicated on `contract_id + chunk_index + model_version`)
- **JSON file** — `entity_relationships/jsons/<title>_<id[:8]>.json`

### Silver — canonical deduplication (PostgreSQL)

`SilverNormalizer.normalize()` reads all Bronze artifacts for a contract and:

1. Stages raw rows into `kg_staging_entities` / `kg_staging_relationships`
2. Deduplicates by `(label, normalized_name)`, keeping highest confidence
3. Writes deduplicated rows to `kg_canonical_entities` / `kg_canonical_relationships`

### Gold — AGE projection (Cypher)

`GoldProjector.project()` reads the canonical Silver tables and upserts
everything into Apache AGE using `MERGE` Cypher queries.

After Gold, `RiskGraphBuilder.build()` runs a rule-based pass that infers
`Risk` nodes and `INCREASES_RISK_FOR` / `CAUSES` edges — no LLM needed.

### Replay from Bronze

```bash
# Re-run Silver + Gold without paying LLM cost again
python -m kg.extraction_pipeline --project --all
python -m kg.extraction_pipeline --project --contract-id <uuid>
```

## Ontology (from `kg/constants.py`)

**Entity labels** (vertex types in AGE):

`Contract`, `Section`, `Clause`, `Party`, `Jurisdiction`,
`EffectiveDate`, `ExpirationDate`, `RenewalTerm`, `LiabilityClause`,
`IndemnityClause`, `PaymentTerm`, `ConfidentialityClause`,
`TerminationClause`, `GoverningLawClause`, `Obligation`,
`Risk`, `Amendment`, `ReferenceDocument`

**Relationship types** (edge labels in AGE):

`SIGNED_BY`, `GOVERNED_BY`, `INDEMNIFIES`, `HAS_TERMINATION`, `HAS_RENEWAL`,
`HAS_PAYMENT_TERM`, `REFERENCES`, `AMENDS`, `SUPERCEDES`, `REPLACES`,
`OBLIGATES`, `LIMITS_LIABILITY`, `DISCLOSES_TO`, `HAS_CLAUSE`,
`HAS_SECTION`, `HAS_CHUNK`, `ATTACHES`, `INCORPORATES_BY_REFERENCE`,
`INCREASES_RISK_FOR`, `CAUSES`

## How AGE Works with asyncpg

AGE exposes Cypher through a SQL wrapper:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (e:Party)-[:SIGNED_BY]->(c:Contract)
    RETURN e.name, c.name
$$) AS (party agtype, contract agtype)
```

`agtype` columns are returned as strings — `_unquote_agtype()` strips quotes.

Every connection must load the AGE extension before issuing Cypher:

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
```

This is applied in both the pool `init` callback and on every `pool.acquire()`
(asyncpg resets session state on connection return).

## NL → Cypher (no LLM)

```
NL2CypherConverter.convert(question)
    ├── IntentParser.parse(question)         → IntentMatch(intent, params)
    └── QUERY_CAPABILITIES[intent](params)   → Cypher string
```

`IntentParser` uses regex patterns — deterministic, no prompt injection surface.
`QUERY_CAPABILITIES` maps each recognised intent to a Cypher builder function.

## Configuration (`.env`)

```
KG_BACKEND=age
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
AGE_GRAPH_NAME=legal_graph
KG_LLM_MODEL=llama3.1:8b          # falls back to LLM_MODEL if unset
KG_LLM_BASE_URL=http://localhost:11434/v1
KG_EXTRACTION_CHUNK_SIZE=1500
KG_CONFIDENCE_THRESHOLD=0.7
```

## Docker

```bash
docker-compose up age     # start AGE on port 5433
docker-compose down age   # stop
```

## CLI

```bash
# Full pipeline (Bronze + Silver + Gold) for all contracts
python -m kg.extraction_pipeline --all [--limit N] [--verbose]

# Full pipeline for one contract
python -m kg.extraction_pipeline --contract-id <uuid>

# Replay Silver + Gold from existing Bronze (no LLM)
python -m kg.extraction_pipeline --project --all
python -m kg.extraction_pipeline --project --contract-id <uuid>
```

## API Endpoints (`apps/kg/api.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | AGE connectivity |
| GET | `/v1/stats` | Entity + relationship counts by type |
| POST | `/v1/search` | Entity name substring search |
| POST | `/v1/context` | LLM-ready context string for a query |
| POST | `/v1/related` | Entities connected to a UUID |
| POST | `/v1/contracts` | Contracts mentioning a named entity |
| POST | `/v1/cypher` | Execute a read-only Cypher MATCH query |
