# KG Retrieval Architecture

This document covers how user queries are answered from the knowledge graph.
It is distinct from **extraction** (how the graph is populated — see `KG_PIPELINE.md`).

---

## Separation of concerns

| Concern | Uses LLM? | Key files |
|---------|-----------|-----------|
| **Extraction** — populate the graph from documents | Yes (5-pass Bronze→Silver→Gold) | `kg/extraction_pipeline.py`, `kg/legal_extractor.py` |
| **Retrieval** — answer user queries from the graph | No — fully deterministic | `kg/intent_parser.py`, `kg/query_builder.py`, `kg/nl2cypher.py` |

LLMs are used only at the **answer synthesis** step (the final RAG agent response), never for generating Cypher.

Why no LLM for Cypher generation in a legal system:
- Hallucinated edge names produce silent wrong answers
- Schema drift goes undetected until runtime
- Injection risk through prompt manipulation
- Non-determinism is unacceptable for legal queries

---

## Retrieval pipeline

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  INTENT CLASSIFICATION  (kg/intent_parser.py)           │
│                                                         │
│  Regex patterns → IntentMatch(intent, params)           │
│  ~25 intents, matched in specificity order              │
│  Extracts entity name param from quoted strings or      │
│  title-case phrases.  No LLM.  ~0 ms.                   │
└─────────────────┬───────────────────────────────────────┘
                  │  IntentMatch(intent="find_indemnification",
                  │              params={"name": "Acme Corp"})
                  ▼
┌─────────────────────────────────────────────────────────┐
│  CAPABILITY SELECTION  (kg/query_builder.py)            │
│                                                         │
│  QUERY_CAPABILITIES[intent] → builder function          │
│                                                         │
│  Intents map to graph capabilities:                     │
│    find_parties, find_indemnification,                  │
│    find_jurisdictions, find_termination_clauses,        │
│    find_confidentiality_clauses, find_payment_terms,    │
│    find_obligations, find_liability_clauses,            │
│    find_effective_dates, find_expiration_dates,         │
│    find_renewal_terms, find_disclosures,                │
│      → semantic_graph (Legal Entity Graph)              │
│                                                         │
│    find_sections                                        │
│      → hierarchy_graph (Document Hierarchy Graph)       │
│                                                         │
│    find_superseded_contracts, find_amendments,          │
│    find_references, find_incorporated_documents,        │
│    find_attachments, find_replacements                  │
│      → lineage_graph (Cross-Contract Lineage Graph)     │
│                                                         │
│    find_all_risks, find_risk_chains,                    │
│    find_missing_indemnity, find_missing_termination     │
│      → risk_graph (Risk Dependency Graph)               │
│                                                         │
│    list_contracts  → fallback (any graph)               │
└─────────────────┬───────────────────────────────────────┘
                  │  builder function + params
                  ▼
┌─────────────────────────────────────────────────────────┐
│  DETERMINISTIC QUERY BUILDER  (kg/query_builder.py)     │
│                                                         │
│  builder(params) → Cypher string                        │
│                                                         │
│  Each builder:                                          │
│    - encodes schema knowledge directly (no prompt)      │
│    - calls _esc() on every user-supplied value          │
│    - applies optional name filter if param present,     │
│      otherwise returns all results (LIMIT 50)           │
│    - returns raw Cypher MATCH..RETURN..LIMIT            │
│                                                         │
│  Example output:                                        │
│    MATCH (p1:Party)-[:INDEMNIFIES]->(p2:Party)          │
│    WHERE p1.name CONTAINS 'Acme Corp'                   │
│    RETURN p1.name AS indemnifier,                       │
│           p2.name AS indemnified                        │
│    LIMIT 20                                             │
└─────────────────┬───────────────────────────────────────┘
                  │  Cypher string
                  ▼
┌─────────────────────────────────────────────────────────┐
│  CYPHER EXECUTION  (kg/age_graph_store.py)              │
│                                                         │
│  AgeGraphStore.run_cypher_query(cypher)                 │
│    1. Read-only guardrail (blocks CREATE/MERGE/etc.)    │
│    2. Wrap: SELECT * FROM ag_catalog.cypher(            │
│              'legal_graph', $$<cypher>$$) AS (...)      │
│    3. Execute via asyncpg                               │
│    4. Format results as a table string                  │
└─────────────────┬───────────────────────────────────────┘
                  │  table string (e.g. "indemnifier | indemnified\n...")
                  ▼
┌─────────────────────────────────────────────────────────┐
│  LLM ANSWER SYNTHESIS  (rag/agent/rag_agent.py)         │
│                                                         │
│  Graph results are fused with RAG chunk context and     │
│  passed to the Pydantic AI agent for final answer.      │
│  This is the only LLM call in the retrieval path.       │
└─────────────────────────────────────────────────────────┘
```

---

## Intent → graph capability mapping

| Intent | Graph | Edge types queried |
|--------|-------|--------------------|
| `find_parties` | semantic | `SIGNED_BY` |
| `find_indemnification` | semantic | `INDEMNIFIES` |
| `find_jurisdictions` | semantic | `GOVERNED_BY` |
| `find_termination_clauses` | semantic | `HAS_TERMINATION` |
| `find_confidentiality_clauses` | semantic | `HAS_CLAUSE → ConfidentialityClause` |
| `find_payment_terms` | semantic | `HAS_PAYMENT_TERM` |
| `find_obligations` | semantic | `OBLIGATES` |
| `find_liability_clauses` | semantic | `LIMITS_LIABILITY` |
| `find_effective_dates` | semantic | `HAS_CLAUSE → EffectiveDate` |
| `find_expiration_dates` | semantic | `HAS_CLAUSE → ExpirationDate` |
| `find_renewal_terms` | semantic | `HAS_RENEWAL` |
| `find_disclosures` | semantic | `DISCLOSES_TO` |
| `find_sections` | hierarchy | `HAS_SECTION`, `HAS_CLAUSE` |
| `find_superseded_contracts` | lineage | `SUPERCEDES` |
| `find_amendments` | lineage | `AMENDS` |
| `find_references` | lineage | `REFERENCES` |
| `find_incorporated_documents` | lineage | `INCORPORATES_BY_REFERENCE` |
| `find_attachments` | lineage | `ATTACHES` |
| `find_replacements` | lineage | `REPLACES` |
| `find_all_risks` | risk | `INCREASES_RISK_FOR` |
| `find_risk_chains` | risk | `CAUSES` |
| `find_missing_indemnity` | risk | `NOT HAS_CLAUSE → IndemnityClause` |
| `find_missing_termination` | risk | `NOT HAS_TERMINATION → TerminationClause` |
| `list_contracts` | any | `Contract` vertices |

---

## Key files

| File | Role |
|------|------|
| `kg/intent_parser.py` | `IntentParser.parse(query) → IntentMatch` |
| `kg/query_builder.py` | `build_*_query(params) → Cypher`; `QUERY_CAPABILITIES` dict |
| `kg/nl2cypher.py` | `NL2CypherConverter.convert(question) → Cypher` (thin orchestrator) |
| `kg/graph_router.py` | `GraphRouter.route(query) → list[GraphType]` (schema selection) |
| `kg/schemas.py` | `GraphType` enum + compact schema strings per graph type |
| `kg/age_graph_store.py` | `run_cypher_query(cypher) → str` (executes, formats results) |

---

## Adding a new intent

1. Add a regex pattern + intent name to `_PATTERNS` in `kg/intent_parser.py`.
2. Write a `build_<intent>_query(params: dict) -> str` function in `kg/query_builder.py`.
3. Register it in `QUERY_CAPABILITIES`.
4. Done — no prompt changes, no model dependency.

---

## Hybrid retrieval (KG + RAG)

The KG retrieval path runs in parallel with the RAG vector/BM25 path inside
`HybridKGRetriever` (see `KG_PIPELINE.md` — Hybrid Retrieval Architecture section).
The intent classifier (`rag/retrieval/intent_classifier.py`) decides which paths
are active:

| Intent | Paths |
|--------|-------|
| `HYBRID` (default) | Both: vector+BM25 AND KG |
| `STRUCTURED` | KG only (count/aggregate queries) |
| `SEMANTIC` | Vector+BM25 only |

This is a separate classifier from `IntentParser` — it routes at the tool level
(which retrieval modalities to use), while `IntentParser` routes within the KG
path (which graph and which Cypher template).
