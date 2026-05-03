# Knowledge Graph — Architecture

## Overview

A property graph built over ingested legal contracts using Apache AGE
(a PostgreSQL extension that adds native openCypher query support).
The KG is populated during document ingestion via LLM-based entity extraction.
It runs in a separate PostgreSQL instance (port 5433) from the vector store.

> **Note:** `rag/agent/kg_agent.py` is legacy code that uses Neo4j and is NOT
> the production KG. The production backend is `rag/knowledge_graph/age_graph_store.py`
> (Apache AGE). The Neo4j file should be treated as dead code.

## Stack

| Layer | Technology |
|---|---|
| Graph database | Apache AGE (PostgreSQL 15 + AGE extension) |
| Query language | openCypher (via `ag_catalog.cypher()` SQL wrapper) |
| Client | asyncpg (standard PostgreSQL protocol) |
| LLM (entity extraction) | Ollama or any OpenAI-compatible API |
| UI | Streamlit (`apps/kg/streamlit_app.py`) |
| REST API | FastAPI (`apps/kg/api.py`) |

## Components

```
apps/kg/
├── streamlit_app.py   — KG Explorer: stats, entity search, Cypher REPL
└── api.py             — FastAPI: /health, /v1/stats, /v1/search,
                         /v1/context, /v1/related, /v1/contracts, /v1/cypher

rag/knowledge_graph/
├── age_graph_store.py     — AgeGraphStore: all AGE operations
├── pg_graph_store.py      — PgGraphStore: fallback (SQL tables, no AGE)
├── pipeline.py            — KGExtractionPipeline: LLM extraction → store
├── constants.py           — VALID_LABELS, VALID_REL_TYPES
└── __init__.py            — create_kg_store() factory (selects by KG_BACKEND)
```

## Graph Model

**Vertex labels** (entity types — label IS the type, one vertex table per label):

| Label | Meaning |
|---|---|
| `Party` | Contract parties (companies, individuals) |
| `Contract` | Contract documents |
| `Jurisdiction` | Governing law jurisdictions |
| `Clause` | Generic contract clauses |
| `LicenseClause` | License grant clauses |
| `TerminationClause` | Termination clauses |
| `IndemnificationClause` | Indemnification clauses |

All vertices carry: `uuid`, `name`, `normalized_name`, `label`, `document_id`, `metadata`.

**Edge types** (relationship types):

| Type | Meaning |
|---|---|
| `PARTY_TO` | Party → Contract |
| `GOVERNED_BY_LAW` | Contract → Jurisdiction |
| `HAS_CLAUSE` | Contract → Clause |
| `RELATED_TO` | Generic relationship |
| `LICENSED_TO` | Licensor → Licensee |

All edges carry: `uuid`, `document_id`, `properties`.

## How AGE Works with asyncpg

AGE exposes Cypher through a SQL function:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (e:Party)-[:PARTY_TO]->(c:Contract)
    RETURN e.name, c.name
$$) AS (party agtype, contract agtype)
```

The `agtype` columns are returned as strings by asyncpg — they look like
JSON scalars (`"Acme Corp"`), so `_unquote_agtype()` strips the surrounding quotes.

**Every connection must run setup statements** before issuing Cypher:
```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
```

This is registered as an `init` callback on the asyncpg pool, and also re-applied
in the `_conn()` context manager on every acquire (asyncpg resets state on return).

## Data Flow — KG Population (during ingestion)

```
DocumentIngestionPipeline.ingest_documents()
    │
    └── KGExtractionPipeline.extract_and_store(chunk)
            ├── LLM.extract_entities(chunk_text)
            │       → [{"name": "Amazon", "type": "Party"}, ...]
            ├── AgeGraphStore.upsert_entity(name, entity_type, document_id)
            │       └── asyncpg: MERGE (e:Party {normalized_name, document_id})
            │               SET e.uuid = COALESCE(e.uuid, new_uuid), ...
            └── AgeGraphStore.add_relationship(src_uuid, tgt_uuid, rel_type)
                    └── asyncpg: MATCH (s {uuid}), (t {uuid})
                            CREATE (s)-[r:PARTY_TO {...}]->(t)
```

## Data Flow — KG Query

```
User query: "Which contracts have Amazon as a party?"
    │
    ├── AgeGraphStore.search_as_context("Amazon")          [via RAG agent tool]
    │       └── asyncpg: MATCH (e)-[r]->(t)
    │                   WHERE toLower(e.name) CONTAINS "amazon"
    │                      OR toLower(t.name) CONTAINS "amazon"
    │                   RETURN e.name, e.label, type(r), t.name, t.label
    │
    ├── AgeGraphStore.search_entities("Amazon", "Party")   [via API]
    │       └── asyncpg: MATCH (e:Party)
    │                   WHERE toLower(e.name) CONTAINS "amazon"
    │                   RETURN e.uuid, e.name, e.label, e.document_id
    │
    └── AgeGraphStore.run_cypher_query(custom_cypher)      [via Cypher REPL]
            ├── read-only guardrail (blocks CREATE/MERGE/SET/DELETE/…)
            ├── _parse_return_aliases() → AS clause
            └── asyncpg: SELECT * FROM ag_catalog.cypher(...)
```

## Key Configuration (`.env`)

```
KG_BACKEND=age                              # age | postgres
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
AGE_GRAPH_NAME=legal_graph                  # graph name within AGE
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
```

## Docker

The AGE instance runs in a separate container:

```bash
docker-compose up age       # start AGE on port 5433
docker-compose down age     # stop
```

Container name: `rag_age`. Uses the `apache/age:v1.5.0-pg15` image.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | AGE container connectivity |
| GET | `/v1/stats` | Entity + relationship counts by type |
| POST | `/v1/search` | Entity name substring search |
| POST | `/v1/context` | LLM-ready context for a query |
| POST | `/v1/related` | Entities connected to a UUID |
| POST | `/v1/contracts` | Contracts that mention a named entity |
| POST | `/v1/cypher` | Execute a read-only Cypher MATCH query |

## Running

```bash
# Start AGE
docker-compose up age

# UI
streamlit run apps/kg/streamlit_app.py

# API
uvicorn apps.kg.api:app --port 8002 --reload

# Example queries
curl http://localhost:8002/v1/stats
curl -X POST http://localhost:8002/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Amazon", "entity_type": "Party"}'
curl -X POST http://localhost:8002/v1/cypher \
  -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (p:Party)-[:PARTY_TO]->(c:Contract) RETURN p.name, c.name LIMIT 10"}'
```
