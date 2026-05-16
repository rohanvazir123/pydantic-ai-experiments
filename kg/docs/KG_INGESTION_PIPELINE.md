# Legal Knowledge Graph Pipeline

Bronze / Silver / Gold extraction pipeline for CUAD legal contracts using
Apache AGE and Ollama (`llama3.1:8b`).

---

## Architecture Overview

```
CUAD contracts
    │  (already ingested by Docling into the chunks table)
    ▼
┌─────────────────────────────────────────────────┐
│  PREPROCESSING                                  │
│  Read semantic chunks from the chunks table     │
│  (Docling HybridChunker — respects sentence     │
│  and paragraph boundaries).                     │
│  Fallback: character-stride split at            │
│  KG_EXTRACTION_CHUNK_SIZE (1 500 chars) if a    │
│  document has no pre-computed chunks.           │
└──────────────────────┬──────────────────────────┘
                       │  chunk_text × N
                       ▼
┌─────────────────────────────────────────────────┐
│  EXTRACTION  (5 sequential LLM passes,          │
│  temperature=0, llama3.1:8b via Ollama)         │
│                                                 │
│  Pass 1  entity extraction                      │
│  Pass 2  relationship extraction                │
│           (receives entity list from pass 1)    │
│  Pass 3  document hierarchy extraction          │
│  Pass 4  cross-contract reference extraction    │
│  Pass 5  validation pass                        │
│           (second LLM opinion — filters         │
│            hallucinated / unsupported edges)    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  BRONZE  (kg_raw_extractions — PostgreSQL JSONB)│
│  Immutable.  One row per (contract, chunk).     │
│  Stores validated relationships only (post      │
│  pass 5).  Replayable: Silver/Gold can be       │
│  rebuilt at any time without re-running LLM.   │
│                                                 │
│  Also written to disk:                          │
│  entity_relationships/jsons/<title>_<id>.json  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  SILVER  (PostgreSQL staging + canonical tables)│
│                                                 │
│  kg_staging_entities                            │
│  kg_staging_relationships                       │
│       │                                         │
│       │  dedup by (normalize(name), label)      │
│       │  confidence threshold ≥ 0.7             │
│       │  resolve cross-chunk entity refs        │
│       ▼                                         │
│  kg_canonical_entities                          │
│  kg_canonical_relationships                     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  GOLD  (Apache AGE — distinct vertex labels)    │
│                                                 │
│  (:Party)  (:Contract)  (:Clause)               │
│  (:Jurisdiction)  (:Obligation)  …              │
│                                                 │
│  [:SIGNED_BY]  [:GOVERNED_BY]  [:INDEMNIFIES]  │
│  [:AMENDS]  [:SUPERCEDES]  [:HAS_CLAUSE]  …    │
└─────────────────────────────────────────────────┘
```

---

## Vertex Labels (entity ontology)

| Label | Meaning |
|---|---|
| `Contract` | The agreement itself |
| `Section` | Named section of a contract |
| `Clause` | A specific contractual provision |
| `Party` | Contracting party, organisation, individual |
| `Jurisdiction` | Governing law, state or country |
| `EffectiveDate` | When the contract takes effect |
| `ExpirationDate` | When the contract expires |
| `RenewalTerm` | Automatic renewal period |
| `LiabilityClause` | Limitation of liability, uncapped liability |
| `IndemnityClause` | Indemnification obligations |
| `PaymentTerm` | Fees, royalties, revenue sharing |
| `ConfidentialityClause` | NDA, confidentiality obligations |
| `TerminationClause` | Termination for cause / convenience |
| `GoverningLawClause` | Choice of law provision |
| `Obligation` | A duty a party must perform |
| `Risk` | A risk factor or compliance gap |
| `Amendment` | Modification to an existing contract |
| `ReferenceDocument` | External document referenced by the contract |

---

## Relationship Types

### Semantic (legal entity graph)

| Type | Valid source → target |
|---|---|
| `SIGNED_BY` | Contract → Party |
| `GOVERNED_BY` | Contract → Jurisdiction |
| `INDEMNIFIES` | Party → Party |
| `HAS_TERMINATION` | Contract → TerminationClause |
| `HAS_RENEWAL` | Contract → RenewalTerm |
| `HAS_PAYMENT_TERM` | Contract → PaymentTerm |
| `OBLIGATES` | Contract → Obligation |
| `LIMITS_LIABILITY` | Contract → LiabilityClause |
| `DISCLOSES_TO` | Party → Party |
| `HAS_CLAUSE` | Contract → Clause (fallback) |

### Document hierarchy graph

| Type | Source → target |
|---|---|
| `HAS_SECTION` | Contract → Section |
| `HAS_CLAUSE` | Section → Clause |
| `HAS_CHUNK` | Clause → Chunk |

### Cross-contract lineage graph

| Type | Source → target |
|---|---|
| `REFERENCES` | Contract → ReferenceDocument |
| `AMENDS` | Contract → Contract |
| `SUPERCEDES` | Contract → Contract |
| `REPLACES` | Contract → Contract |
| `ATTACHES` | Contract → ReferenceDocument |
| `INCORPORATES_BY_REFERENCE` | Contract → ReferenceDocument |

### Risk dependency graph (built last — requires inference)

| Type | Source → target |
|---|---|
| `INCREASES_RISK_FOR` | Risk → Party |
| `CAUSES` | Risk → Risk |

---

## Four Graph Types

All four live in the same AGE graph (`legal_graph`).  They are distinguished
by the vertex labels and relationship types used, not by separate graphs.

1. **Document Hierarchy Graph** — physical structure for neighbouring-chunk
   retrieval during RAG.  Enables "give me the surrounding clauses" queries.

2. **Legal Entity Graph** — semantic relationships extracted by the LLM.
   Enables "find all contracts where Party X indemnifies Party Y" queries that
   pure vector search cannot answer.

3. **Cross-Contract Lineage Graph** — amendment / supersession chains.
   Ensures the agent does not retrieve clauses from expired or amended
   contracts.  Requires a name-resolution step: `target_document_name` is
   fuzzy-matched to a known `contract_id` at Silver time.

4. **Risk Dependency Graph** — built last and separately.  Requires rule-based
   inference on top of the entity graph (e.g., "missing IndemnityClause" is a
   gap, not something the LLM extracted directly).

---

## Extraction Prompts

All prompts use `temperature=0`.  No markdown, no prose, return only valid JSON.

### Prompt 1 — Entity Extraction

```
You are a legal contract entity extraction system specialized in the CUAD dataset.
Extract ONLY entities from the provided contract text.

RULES
1. Extract ONLY legally meaningful entities.
2. Do NOT extract generic nouns.
3. Use ONLY these labels: Contract, Section, Clause, Party, Jurisdiction,
   EffectiveDate, ExpirationDate, RenewalTerm, LiabilityClause, IndemnityClause,
   PaymentTerm, ConfidentialityClause, TerminationClause, GoverningLawClause,
   Obligation, Risk, Amendment, ReferenceDocument
4. If unsure, use "Clause".
5. Every entity must have: entity_id, label, canonical_name, text_span, confidence.
6. canonical_name must be normalised and concise.
7. confidence must be between 0 and 1.

Return ONLY valid JSON. No explanations. No hallucinations.

OUTPUT FORMAT
{"entities": [{"entity_id": "party:acme_corp", "label": "Party",
  "canonical_name": "Acme Corp", "text_span": "Acme Corporation",
  "confidence": 0.98}]}
```

### Prompt 2 — Relationship Extraction

```
You are a legal contract relationship extraction system.
Extract semantic relationships between the provided entities.

INPUT: contract text + previously extracted entities (JSON)

RULES
1. Use ONLY: SIGNED_BY, GOVERNED_BY, INDEMNIFIES, HAS_TERMINATION, HAS_RENEWAL,
   HAS_PAYMENT_TERM, REFERENCES, AMENDS, SUPERCEDES, OBLIGATES, LIMITS_LIABILITY,
   DISCLOSES_TO, HAS_CLAUSE
2. Only create relationships explicitly supported by the text.
3. Do NOT infer speculative relationships.
4. Every relationship: relationship_id, source_entity_id, target_entity_id,
   relationship_type, evidence_text, confidence.
5. evidence_text must contain the exact supporting text.

Return ONLY valid JSON.

OUTPUT FORMAT
{"relationships": [{"relationship_id": "rel_001", "source_entity_id": "party:acme_corp",
  "target_entity_id": "party:beta_inc", "relationship_type": "INDEMNIFIES",
  "evidence_text": "Acme Corp shall indemnify Beta Inc", "confidence": 0.95}]}
```

### Prompt 3 — Document Hierarchy Extraction

```
You are a legal document structure extraction system.
Extract the hierarchical structure: Contract → Section → Clause → Chunk.

RELATIONSHIPS: HAS_SECTION, HAS_CLAUSE, HAS_CHUNK
Every node: node_id, node_type, title, sequence_number.
Every edge: source_id, target_id, relationship_type.
Preserve document ordering. Return ONLY valid JSON.
```

### Prompt 4 — Cross-Contract Reference Extraction

```
You are a legal contract lineage extraction system.
Identify references between contracts and external legal documents.

RELATIONSHIP TYPES: REFERENCES, AMENDS, SUPERCEDES, REPLACES, ATTACHES,
  INCORPORATES_BY_REFERENCE

RULES
1. Extract ONLY explicit references.
2. Include referenced document names exactly as written.
3. Include supporting evidence text.
4. Do not infer references.

OUTPUT FORMAT
{"references": [{"source_contract_id": "...", "target_document_name": "Master Services
  Agreement", "relationship_type": "AMENDS", "evidence_text": "...",
  "confidence": 0.93}]}
```

### Prompt 5 — Validation (second LLM pass)

```
You are a legal knowledge graph validation system.
Validate extracted relationships against the source text.

INPUT: contract text + extracted entities + extracted relationships

VALIDATION RULES
1. Remove unsupported relationships (no evidence in text).
2. Remove hallucinated entities.
3. Ensure ontology consistency (source/target types logically valid).
4. Verify confidence scores.

OUTPUT FORMAT
{"valid_relationships": [...], "invalid_relationships":
  [{"relationship_id": "...", "reason": "unsupported by text"}]}
```

---

## PostgreSQL Schema

### Bronze

```sql
kg_raw_extractions (
    id              UUID PK,
    contract_id     UUID → documents(id),
    chunk_index     INT,
    model_version   TEXT,
    ontology_version TEXT DEFAULT '1.0',
    raw_json        JSONB,
    created_at      TIMESTAMPTZ
)
UNIQUE (contract_id, chunk_index, model_version)
```

### Silver

```sql
kg_staging_entities (
    id, contract_id, chunk_index, artifact_id,
    entity_id_raw TEXT,   -- from LLM ("party:acme_corp")
    label TEXT,
    canonical_name TEXT,
    text_span TEXT,
    confidence FLOAT
)

kg_staging_relationships (
    id, contract_id, chunk_index, artifact_id,
    source_entity_id_raw TEXT,
    target_entity_id_raw TEXT,
    relationship_type TEXT,
    evidence_text TEXT,
    confidence FLOAT,
    validated BOOLEAN
)

kg_canonical_entities (
    id UUID PK,
    contract_id UUID,
    label TEXT,
    canonical_name TEXT,
    confidence FLOAT,
    evidence_count INT,
    UNIQUE (contract_id, label, canonical_name)
)

kg_canonical_relationships (
    id UUID PK,
    contract_id UUID,
    source_entity_id UUID → kg_canonical_entities(id),
    target_entity_id UUID → kg_canonical_entities(id),
    relationship_type TEXT,
    confidence FLOAT,
    evidence_texts JSONB,
    UNIQUE (contract_id, source_entity_id, target_entity_id, relationship_type)
)
```

---

## AGE Vertex Model: distinct labels

Previous model (flat):
```cypher
(:Entity {entity_type: "Party", name: "Acme Corp", ...})
```

Current model (distinct labels):
```cypher
(:Party   {uuid: "...", name: "Acme Corp",         label: "Party",        ...})
(:Contract{uuid: "...", name: "Distribution Agmt", label: "Contract",     ...})
(:Clause  {uuid: "...", name: "Section 12.1",      label: "Clause",       ...})
```

Benefits: cleaner Cypher traversals, label-specific indexes, idiomatic graph model.

---

## CLI

```bash
# Full pipeline for one contract (Bronze + Silver + Gold)
python -m kg.extraction_pipeline --contract-id <uuid>

# Full pipeline for all contracts
python -m kg.extraction_pipeline --all [--limit N]

# Silver + Gold only (replay from Bronze, no LLM calls)
python -m kg.extraction_pipeline --project --contract-id <uuid>
python -m kg.extraction_pipeline --project --all
```

---

## Example Cypher Queries

```cypher
-- All parties to contracts governed by Delaware law
MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS 'delaware'
MATCH (p:Party)-[:SIGNED_BY]-(c)
RETURN p.name, c.name

-- Multi-hop: find indemnifying parties across all contracts
MATCH (a:Party)-[:INDEMNIFIES]->(b:Party)
RETURN a.name AS indemnifier, b.name AS indemnified

-- Contract lineage: what does contract X supersede?
MATCH path = (c:Contract {name: $name})-[:SUPERCEDES|AMENDS*1..5]->(old)
RETURN path

-- Missing indemnity (gap detection — feeds risk graph)
MATCH (c:Contract) WHERE NOT (c)-[:HAS_CLAUSE]->(:IndemnityClause)
RETURN c.name
```

---

## Hybrid Retrieval Architecture

The system answers questions by routing through two parallel paths and fusing
the results.

```
USER QUERY
     │
     ▼
Intent Classification  (rag/retrieval/intent_classifier.py)
     │
+----+----+
│         │
▼         ▼
Semantic  Structured
Retrieval Reasoning
(tsvector (KG / AGE /
+ pgvector  SQL)
+ RRF)
│         │
+----+----+
     │
     ▼
Context Fusion Layer
     │
     ▼
Final LLM Reasoning
```

### Intent Classification

Three intents drive which paths activate:

| Intent | When | Paths active |
|---|---|---|
| `HYBRID` | Default — clause text questions | Both paths in parallel |
| `STRUCTURED` | Count/aggregate/distribution queries | KG/SQL only |
| `SEMANTIC` | Pure text retrieval | Vector + BM25 only |

Signals for `STRUCTURED`: "how many", "distribution of", "average number",
"which year", "ratio of", "most common", "top N". If the query also contains
text-retrieval words ("exact language", "what does the clause say") it stays
`HYBRID` even if count patterns are present.

### Semantic Retrieval Path

Existing `Retriever` class — unchanged. Runs pgvector cosine similarity +
PostgreSQL tsvector BM25, merges with RRF (k=60), optionally reranks.

### Structured Reasoning Path

`HybridKGRetriever._structured_retrieve()` in `rag/retrieval/hybrid_kg_retriever.py`:

1. `kg_store.search_entities(query, limit=10)` — RRF hybrid search via `EntityIndex`
   (tsvector GIN + pgvector IVFFlat in main DB). Falls back to O(n) AGE CONTAINS scan
   if the shadow index is not available.
2. For each matched entity (cap 5): `kg_store.get_related_entities(id, limit=5)`.
3. Returns `list[dict]` with entity and relationship facts.

### Source Chunk Index Tracking

Each canonical entity in `kg_canonical_entities` carries a
`source_chunk_indices JSONB` column — the chunk indices from which that entity
was extracted. Foundation for KG→chunk boosting in a future RRF step:

```
KG entity hit → source_chunk_indices → chunk UUIDs
    → add to RRF list alongside vector + BM25 hits
```

### Context Fusion

`_fuse()` produces one context block with KG facts above text passages.
The agent tool `search_hybrid_kg` returns this block directly to the LLM.

### Agent Tool

```python
@agent.tool
async def search_hybrid_kg(ctx, query, match_count=5) -> str
```

Prefer this over calling `search_knowledge_base` + `search_knowledge_graph`
separately for questions requiring both clause text and graph facts.

### Test Suite

```bash
# Unit tests (mocked, no external deps):
pytest rag/tests/test_hybrid_kg_retrieval.py -v

# Integration tests + record all 100 answers for review:
pytest rag/tests/test_hybrid_kg_retrieval.py -m integration --record-answers -v
# → docs/qa_results/hybrid_kg_results.json
```

---

## Implementation Notes

- **Do not insert LLM output directly into AGE.** Always go Bronze → Silver → Gold.
- Cross-contract lineage requires a name-resolution step in Silver: fuzzy-match
  `target_document_name` to a known `contract_id` in the `documents` table.
- The Risk Dependency Graph is built last, separately, using rule-based inference
  on the entity graph — not direct LLM extraction.
- All LLM calls use `llama3.1:8b` via Ollama (`localhost:11434`). No external API.
- Confidence threshold for Silver → Gold promotion: 0.7 (configurable via
  `KG_CONFIDENCE_THRESHOLD` in `.env`).
- **Entity search** uses `kg/entity_index.py` (`EntityIndex`) — a shadow table
  `kg_entity_index` in the main PostgreSQL DB with tsvector GIN + pgvector IVFFlat.
  Each `upsert_entity()` call in AGE also mirrors the row into `EntityIndex`.
  `AgeGraphStore.search_entities()` uses RRF (k=60) over both indexes; falls back
  to the O(n) AGE CONTAINS scan if the index is unreachable.
- **Chunk source**: the Bronze loop reads from the `chunks` table (Docling semantic
  chunks) rather than re-splitting `documents.content` at a fixed character offset.
  This ensures KG chunk boundaries match RAG chunk boundaries exactly, enabling
  future KG→chunk boosting in the RRF pipeline.
