# Legal Knowledge Graph — Ingestion Pipeline

Two independent pipelines share the same AGE graph (`legal_graph`).
Run ingestion first; retrieval works on whatever is already in the graph.

---

## Table of Contents

1. [Pipeline 1 — Ingestion (populate the graph)](#pipeline-1--ingestion-populate-the-graph)
   - [LLM Passes — Prompts and Outputs](#llm-passes--prompts-and-outputs)
   - [JSON Output File](#json-output-file)
   - [Bronze → Silver → Gold PostgreSQL Schema](#bronze--silver--gold-postgresql-schema)
   - [Ontology](#ontology-from-kglegalcuad_ontologypy)
   - [CLI](#cli)
   - [Configuration](#configuration)
   - [Docker](#docker)
2. [Pipeline 2 — Retrieval (answer queries from the graph)](#pipeline-2--retrieval-answer-queries-from-the-graph)
3. [Module Layout](#module-layout)
4. [API Endpoints](#api-endpoints-appskg/apipy)

---

## Pipeline 1 — Ingestion (populate the graph)

```
CUAD contracts (already in chunks table via Docling)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESSING  (kg/legal/ingestion/extraction_pipeline.py)   │
│                                                     │
│  Read semantic chunks from chunks table.            │
│  Fallback: character-stride split at                │
│  KG_EXTRACTION_CHUNK_SIZE (1 500 chars) if no       │
│  Docling chunks exist for a document.               │
└───────────────────────┬─────────────────────────────┘
                        │  list[chunk_text]
                        ▼
┌─────────────────────────────────────────────────────┐
│  BRONZE — 5 sequential LLM passes per chunk         │
│  Model: qwen2.5:14b via Ollama (KG_LLM_MODEL)       │
│  Context window: 128K (num_ctx=131072)               │
│  Confidence threshold: 0.7                          │
│                                                     │
│  Pass 1  Entity extraction                          │
│  Pass 2  Relationship extraction                    │
│  Pass 3  Document hierarchy extraction              │
│  Pass 4  Cross-contract reference extraction        │
│  Pass 5  Validation (second LLM opinion)            │
│                                                     │
│  Output: BronzeArtifact → kg_raw_extractions (JSONB)│
│          + kg/evals/jsons/<title>.json  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  SILVER — deduplication (pure SQL, no LLM)          │
│                                                     │
│  Stages raw rows → deduplicates by                  │
│  (normalize(name), label) → canonical tables        │
│                                                     │
│  kg_staging_entities / kg_staging_relationships     │
│           ↓                                         │
│  kg_canonical_entities / kg_canonical_relationships │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  GOLD — AGE projection (Cypher MERGE, no LLM)       │
│                                                     │
│  (:Party)  (:Contract)  (:Jurisdiction)  (:Clause)  │
│  [:SIGNED_BY]  [:GOVERNED_BY]  [:INDEMNIFIES]  …   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  RISK GRAPH — rule-based inference (no LLM)         │
│                                                     │
│  RiskGraphBuilder scans canonical entities for      │
│  missing clauses and writes Risk nodes +            │
│  INCREASES_RISK_FOR / CAUSES edges to AGE.          │
└─────────────────────────────────────────────────────┘
```

### LLM Passes — Prompts and Outputs

All passes: `temperature=0`, JSON-only output, no markdown.

---

#### Pass 1 — Entity Extraction

**System prompt (abbreviated):**
```
You are a legal contract entity extraction system specialized in the CUAD dataset.
Extract ONLY legally meaningful entities. Do NOT extract generic nouns.
Use ONLY these labels: Contract, Section, Clause, Party, Jurisdiction,
EffectiveDate, ExpirationDate, RenewalTerm, LiabilityClause, IndemnityClause,
PaymentTerm, ConfidentialityClause, TerminationClause, GoverningLawClause,
Obligation, Risk, Amendment, ReferenceDocument.
If unsure, use "Clause". confidence must be 0–1.
Return ONLY valid JSON. No explanations.
```

**User message:** `Contract text:\n\n{chunk}`

**LLM output:**
```json
{
  "entities": [
    {
      "entity_id": "party:acme_corp",
      "label": "Party",
      "canonical_name": "Acme Corp",
      "text_span": "Acme Corporation",
      "confidence": 0.98
    },
    {
      "entity_id": "jurisdiction:delaware",
      "label": "Jurisdiction",
      "canonical_name": "Delaware",
      "text_span": "the State of Delaware",
      "confidence": 0.95
    }
  ]
}
```

---

#### Pass 2 — Relationship Extraction

**System prompt (abbreviated):**
```
You are a legal contract relationship extraction system.
Use ONLY: SIGNED_BY, GOVERNED_BY, INDEMNIFIES, HAS_TERMINATION, HAS_RENEWAL,
HAS_PAYMENT_TERM, REFERENCES, AMENDS, SUPERCEDES, OBLIGATES, LIMITS_LIABILITY,
DISCLOSES_TO, HAS_CLAUSE.
Only create relationships explicitly supported by the text. Do NOT infer.
source/target entity_ids must reference entity_ids from the entity list.
evidence_text must be exact supporting text. confidence 0–1.
Return ONLY valid JSON.
```

**User message:** `Contract text:\n\n{chunk}\n\nExtracted entities:\n{entity_list_json}`

**LLM output:**
```json
{
  "relationships": [
    {
      "relationship_id": "rel_001",
      "source_entity_id": "party:acme_corp",
      "target_entity_id": "party:beta_inc",
      "relationship_type": "INDEMNIFIES",
      "evidence_text": "Acme Corp shall indemnify and hold harmless Beta Inc",
      "confidence": 0.95
    }
  ]
}
```

---

#### Pass 3 — Document Hierarchy Extraction

**System prompt (abbreviated):**
```
You are a legal document structure extraction system.
Extract: Contract → Section → Clause → Chunk hierarchy.
Relationships: HAS_SECTION, HAS_CLAUSE, HAS_CHUNK.
Every node: node_id, node_type (Contract|Section|Clause|Chunk), title, sequence_number.
Every edge: source_id, target_id, relationship_type.
Preserve document ordering. Return ONLY valid JSON.
```

**User message:** `Contract text:\n\n{chunk}`

**LLM output:**
```json
{
  "nodes": [
    {"node_id": "section:12", "node_type": "Section", "title": "Indemnification", "sequence_number": 12},
    {"node_id": "clause:12.1", "node_type": "Clause", "title": "Mutual Indemnification", "sequence_number": 1}
  ],
  "edges": [
    {"source_id": "section:12", "target_id": "clause:12.1", "relationship_type": "HAS_CLAUSE"}
  ]
}
```

---

#### Pass 4 — Cross-Contract Reference Extraction

**System prompt (abbreviated):**
```
You are a legal contract lineage extraction system.
Identify explicit references to other contracts or legal documents.
Relationship types: REFERENCES, AMENDS, SUPERCEDES, REPLACES, ATTACHES,
  INCORPORATES_BY_REFERENCE.
Extract ONLY explicit references. Do not infer.
Return ONLY valid JSON.
```

**User message:** `source_contract_id: {contract_id}\n\nContract text:\n\n{chunk}`

**LLM output:**
```json
{
  "references": [
    {
      "source_contract_id": "uuid-...",
      "target_document_name": "Master Services Agreement",
      "relationship_type": "AMENDS",
      "evidence_text": "This Amendment amends the Master Services Agreement dated...",
      "confidence": 0.93
    }
  ]
}
```

---

#### Pass 5 — Validation

**System prompt (abbreviated):**
```
You are a legal knowledge graph validation system.
Validate extracted relationships against the source text.
Remove: relationships not supported by evidence, hallucinated entities,
ontologically inconsistent source/target pairs.
Return ONLY valid JSON.
```

**User message:** `Contract text:\n\n{chunk}\n\nExtracted entities:\n{entities_json}\n\nExtracted relationships:\n{relationships_json}`

**LLM output:**
```json
{
  "valid_relationships": [
    {"relationship_id": "rel_001", ...}
  ],
  "invalid_relationships": [
    {"relationship_id": "rel_002", "reason": "no supporting evidence in text"}
  ]
}
```

> **Note:** If the validation pass returns an empty `valid_relationships` list
> (e.g. the model refused or returned garbage), the pipeline keeps all relationships
> from Pass 2 rather than discarding everything. See `_pass_validate()` in
> `kg/legal/extraction_pipeline.py`.

---

### JSON Output File

After Bronze, results are written to disk at:
```
kg/evals/jsons/<contract_title>_<contract_id[:8]>.json
```

Written by `ExtractionPipeline._save_json()` at
`kg/legal/ingestion/extraction_pipeline.py:862`.
Called from `ExtractionPipeline.process_contract()` at line `885` after all
five passes complete for the contract.  The `kg/evals/jsons/`
folder is created automatically if it does not exist.

Shape:
```json
{
  "contract_id": "uuid-...",
  "title": "Acme Distribution Agreement",
  "model_version": "qwen2.5:14b",
  "chunks": [
    {
      "chunk_index": 0,
      "entities": [ { "entity_id": "...", "label": "Party", ... } ],
      "relationships": [ { "relationship_id": "...", "relationship_type": "SIGNED_BY", ... } ],
      "hierarchy_nodes": [ ... ],
      "cross_refs": [ ... ]
    }
  ]
}
```

This is your **graph triplets** file — entities + typed relationships + evidence text per chunk.

---

### Bronze → Silver → Gold PostgreSQL Schema

```sql
-- Bronze (immutable, one row per chunk × model)
kg_raw_extractions (
    id              UUID PK,
    contract_id     UUID → documents(id),
    chunk_index     INT,
    model_version   TEXT,            -- "qwen2.5:14b"
    ontology_version TEXT,           -- "1.0"
    raw_json        JSONB,           -- full BronzeArtifact
    UNIQUE (contract_id, chunk_index, model_version)
)

-- Silver staging (cleared and rebuilt on each run)
kg_staging_entities (id, contract_id, chunk_index, entity_id_raw, label, canonical_name, text_span, confidence)
kg_staging_relationships (id, contract_id, chunk_index, source_entity_id_raw, target_entity_id_raw, relationship_type, evidence_text, confidence, validated)

-- Silver canonical (deduplicated)
kg_canonical_entities (id UUID PK, contract_id, label, canonical_name, confidence, evidence_count, source_chunk_indices JSONB,
    UNIQUE (contract_id, label, canonical_name))
kg_canonical_relationships (id UUID PK, contract_id, source_entity_id UUID, target_entity_id UUID,
    relationship_type, confidence, evidence_texts JSONB,
    UNIQUE (contract_id, source_entity_id, target_entity_id, relationship_type))
```

---

### Ontology (from `kg/legal/cuad_ontology.py`)

**18 vertex labels:**
`Contract`, `Section`, `Clause`, `Party`, `Jurisdiction`,
`EffectiveDate`, `ExpirationDate`, `RenewalTerm`, `LiabilityClause`,
`IndemnityClause`, `PaymentTerm`, `ConfidentialityClause`, `TerminationClause`,
`GoverningLawClause`, `Obligation`, `Risk`, `Amendment`, `ReferenceDocument`

**35 relationship types** — grouped by graph:

| Graph | Relationship types |
|---|---|
| Legal entity | `PARTY_TO`, `SIGNED_BY`, `GOVERNED_BY`, `GOVERNED_BY_LAW`, `INDEMNIFIES`, `HAS_TERMINATION`, `HAS_RENEWAL`, `HAS_PAYMENT_TERM`, `HAS_LICENSE`, `HAS_RESTRICTION`, `HAS_IP_CLAUSE`, `HAS_LIABILITY`, `HAS_PAYMENT`, `HAS_OBLIGATION`, `HAS_CLAUSE`, `HAS_DATE`, `EFFECTIVE_ON`, `EXPIRES_ON`, `OBLIGATES`, `LIMITS_LIABILITY`, `DISCLOSES_TO`, `GRANTS_LICENSE_TO`, `OWES_OBLIGATION_TO`, `ASSIGNS_IP_TO`, `CAN_TERMINATE` |
| Document hierarchy | `HAS_SECTION`, `HAS_CHUNK` |
| Cross-contract lineage | `REFERENCES`, `AMENDS`, `SUPERCEDES`, `REPLACES`, `ATTACHES`, `INCORPORATES_BY_REFERENCE` |
| Risk dependency | `INCREASES_RISK_FOR`, `CAUSES` |

---

### CLI

```bash
# Full pipeline (Bronze + Silver + Gold) for all contracts
python -m kg.legal.ingestion.extraction_pipeline --all [--limit N] [--verbose]

# Full pipeline for one contract
python -m kg.legal.ingestion.extraction_pipeline --contract-id <uuid>

# Replay Silver + Gold from existing Bronze (no LLM cost)
python -m kg.legal.ingestion.extraction_pipeline --project --all
python -m kg.legal.ingestion.extraction_pipeline --project --contract-id <uuid>

# Evaluate ingestion quality (entities/chunk, dedup rate, confidence)
python -m kg.legal.ingestion.eval_pipeline                        # all contracts
python -m kg.legal.ingestion.eval_pipeline --contract-id <uuid>  # single contract
python -m kg.legal.ingestion.eval_pipeline --output report.json  # save JSON

# Interactive CLI + retrieval eval
python -m kg.legal.retrieval.cli                            # REPL
python -m kg.legal.retrieval.cli --question "Who are the parties?"
python -m kg.legal.retrieval.eval_pipeline --dry-run        # intent + Cypher only
python -m kg.legal.retrieval.eval_pipeline                  # full eval with live AGE
```

### Configuration

```
KG_LLM_MODEL=qwen2.5:14b           # dedicated extraction model
KG_LLM_BASE_URL=http://localhost:11434/v1
KG_LLM_API_KEY=ollama
LLM_NUM_CTX=131072                  # 128K context window
KG_EXTRACTION_CHUNK_SIZE=1500       # fallback char split (not used if Docling chunks exist)
KG_CONFIDENCE_THRESHOLD=0.7
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
AGE_GRAPH_NAME=legal_graph
```

### Docker

```bash
docker-compose up -d      # start AGE on port 5433
docker-compose down       # stop
```

---

## Pipeline 2 — Retrieval (answer queries from the graph)

See **[KG_RETRIEVAL_PIPELINE.md](./KG_RETRIEVAL_PIPELINE.md)** for the full
routing diagram with all three paths.

High-level:

```
USER QUERY
    │
    ▼
HybridIntentClassifier  (rag/retrieval/intent_classifier.py)
    │
    ├── HYBRID (default) ──────┬────────────────────────┐
    │                          │                        │
    │                    Vector+BM25              KG path
    │                    (pgvector+tsvector       (IntentParser →
    │                     RRF k=60)                Cypher → AGE)
    │                          │                        │
    │                          └──────── fuse ──────────┘
    │                                      │
    ├── STRUCTURED ──── KG path only ──────┤
    │                                      │
    └── SEMANTIC ─── Vector+BM25 only ─────┘
                                           │
                                           ▼
                                   LLM answer synthesis
                                   (rag/agent/rag_agent.py)
```

---

## Module Layout

```
kg/
├── __init__.py                  # create_kg_store() factory; re-exports all public symbols
├── age_graph_store.py           # AgeGraphStore — all AGE Cypher operations
├── entity_index.py              # EntityIndex — shadow table (tsvector GIN + pgvector IVFFlat)
└── legal/
    ├── common/
    │   └── cuad_ontology.py         # VALID_LABELS, VALID_REL_TYPES, ENTITY_TYPE_MAP (single source of truth)
    ├── ingestion/
    │   ├── extraction_pipeline.py   # ExtractionPipeline: Bronze / Silver / Gold
    │   ├── cuad_kg_ingest.py        # build_cuad_kg(): CUAD annotation path (no LLM)
    │   ├── risk_graph_builder.py    # RiskGraphBuilder: rule-based risk inference
    │   └── eval_pipeline.py         # Ingestion quality metrics (entities/chunk, dedup rate, etc.)
    └── retrieval/
        ├── schemas.py               # GraphType enum, get_schema() for NL→Cypher routing
        ├── graph_router.py          # GraphRouter: question → list[GraphType]
        ├── intent_parser.py         # IntentParser: regex → IntentMatch(intent, params)
        ├── query_builder.py         # QUERY_CAPABILITIES: intent → Cypher builder
        ├── nl2cypher.py             # NL2CypherConverter: orchestrates intent → Cypher
        ├── eval_pipeline.py         # Retrieval quality metrics (intent match, hit rate, latency)
        └── cli.py                   # Interactive CLI: ask legal questions in natural language
```

---

## API Endpoints (`apps/kg/api.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | AGE connectivity check |
| GET | `/v1/stats` | Entity + relationship counts by type |
| POST | `/v1/search` | Entity name substring search |
| POST | `/v1/context` | LLM-ready context string for a query |
| POST | `/v1/related` | Entities connected to a UUID |
| POST | `/v1/contracts` | Contracts mentioning a named entity |
| POST | `/v1/cypher` | Execute a read-only Cypher MATCH query |
