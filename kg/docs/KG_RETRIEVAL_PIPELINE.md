# Legal Knowledge Graph — Retrieval Pipeline

Retrieval never calls an LLM to generate Cypher. The only LLM call is the
final answer synthesis in `rag/agent/rag_agent.py`.

---

## Table of Contents

1. [Top-level routing — HybridIntentClassifier](#top-level-routing--hybridintentclassifier)
2. [Path A — HYBRID (default)](#path-a--hybrid-default)
3. [Path B — STRUCTURED (KG only)](#path-b--structured-kg-only)
4. [Path C — SEMANTIC (vector+BM25 only)](#path-c--semantic-vectorbm25-only)
5. [Query examples for each path](#query-examples-for-each-path)
6. [Intent → graph mapping](#intent--graph-mapping)
7. [How schemas feed into query building](#how-schemas-feed-into-query-building)
8. [KG entity lookup (within Path A and B)](#kg-entity-lookup-within-path-a-and-b)
9. [Two-level intent routing](#two-level-intent-routing)
10. [Key files](#key-files)
11. [Adding a new KG intent](#adding-a-new-kg-intent)

---

## Top-level routing — HybridIntentClassifier

`rag/retrieval/intent_classifier.py` classifies every query into one of three intents
and activates retrieval paths accordingly.

```
USER QUERY
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  HybridIntentClassifier                                   │
│  (rag/retrieval/intent_classifier.py)                     │
│                                                           │
│  Signals for STRUCTURED:                                  │
│    "how many", "distribution of", "average number",       │
│    "which year", "ratio of", "most common", "top N"       │
│                                                           │
│  Overrides to HYBRID if STRUCTURED signals co-occur with  │
│  text-retrieval words: "exact language", "clause says",   │
│    "what does ... say", "text of"                         │
│                                                           │
│  Default: HYBRID                                          │
└──────┬─────────────────────┬──────────────────────────────┘
       │                     │                     │
  HYBRID (default)      STRUCTURED              SEMANTIC
       │                     │                     │
       ▼                     ▼                     ▼
  [Path A]             [Path B]              [Path C]
```

---

## Path A — HYBRID (default)

Both retrieval modalities run in parallel and their results are fused.

```
QUERY
  │
  ├────────────────────────────────┐
  │                                │
  ▼                                ▼
┌────────────────────┐   ┌─────────────────────────────────┐
│  SEMANTIC PATH     │   │  STRUCTURED PATH (KG)           │
│                    │   │                                 │
│  pgvector cosine   │   │  IntentParser (regex)           │
│  + tsvector BM25   │   │    → IntentMatch(intent,params) │
│  merged via RRF    │   │  → Cypher builder               │
│  (k=60)            │   │    → Cypher string              │
│                    │   │  → AGE execution                │
│  list[SearchResult]│   │    → KG facts table string      │
└────────┬───────────┘   └────────────────┬────────────────┘
         │                                │
         └──────────────┬─────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │  CONTEXT FUSION      │
             │  _fuse()             │
             │                      │
             │  KG facts block      │
             │  ── separator ──     │
             │  RAG text passages   │
             └──────────┬───────────┘
                        │
                        ▼
             LLM answer synthesis
             (rag/agent/rag_agent.py)
```

---

## Path B — STRUCTURED (KG only)

Vector search is skipped entirely. Only the KG path runs.

```
QUERY
  │
  ▼
IntentParser (kg/legal/retrieval/intent_parser.py)
  │  regex, ~25 patterns, no LLM, ~0 ms
  │
  ▼  IntentMatch(intent="find_indemnification", params={"name": "Acme Corp"})
  │
  ▼
QUERY_CAPABILITIES[intent]  (kg/legal/retrieval/query_builder.py)
  │  maps intent → Cypher builder function
  │
  ▼  builder(params) → Cypher string
  │
  │  Example:
  │    MATCH (p1:Party)-[:INDEMNIFIES]->(p2:Party)
  │    WHERE p1.name CONTAINS 'Acme Corp'
  │    RETURN p1.name AS indemnifier, p2.name AS indemnified
  │    LIMIT 20
  │
  ▼
AgeGraphStore.run_cypher_query(cypher)  (kg/age_graph_store.py)
  │  1. Blocks mutating keywords (CREATE/MERGE/DELETE/SET/REMOVE)
  │  2. Wraps: SELECT * FROM ag_catalog.cypher('legal_graph', $$...$$)
  │  3. Executes via asyncpg
  │  4. Formats as pipe-separated table string
  │
  ▼  "indemnifier | indemnified\nAcme Corp | Beta Inc\n..."
  │
  ▼
LLM answer synthesis
```

---

## Path C — SEMANTIC (vector+BM25 only)

KG is skipped entirely. Standard RAG retrieval runs.

```
QUERY
  │
  ▼
Retriever.retrieve_as_context()  (rag/retrieval/retriever.py)
  │
  ├── pgvector cosine similarity search (embedding)
  ├── tsvector BM25 full-text search
  └── Reciprocal Rank Fusion (k=60)
      │
      ▼  list[SearchResult] (chunks + metadata)
      │
      ▼
LLM answer synthesis
```

---

## Query examples for each path

### Path A — HYBRID examples

These questions trigger both vector search AND a KG traversal in parallel.

**Example 1 — named-contract parties + text context**
```
Query:   "What are the termination clauses and who are the parties in the
          Lightbridge agreement?"

Classifier → HYBRID (has "what does" text-retrieval signal alongside structured terms)

Semantic path:  pgvector + BM25 on "termination clauses Lightbridge"
                → top-5 text chunks about termination
KG path:        intent=find_termination_clauses  params={"name": "Lightbridge"}
                Cypher:
                  MATCH (c:Contract)-[:HAS_TERMINATION]->(t:TerminationClause)
                  WHERE c.name CONTAINS 'Lightbridge'
                  RETURN c.name AS contract, t.name AS clause
                  LIMIT 20
Fused context:
  [KG Facts]
  contract | clause
  LIGHTBRIDGECORP_11_23_2015 | Termination for Convenience
  ...
  ──────────────────────────
  [Text passages]
  "Either party may terminate this Agreement upon 30 days written notice..."
```

**Example 2 — indemnification + evidence text**
```
Query:   "What does the indemnification clause say in the AdaptImmune contract?"

Classifier → HYBRID  (contains "what does ... say")

KG path:    intent=find_indemnification  params={"name": "AdaptImmune"}
            → MATCH (p1:Party)-[:INDEMNIFIES]->(p2:Party)
              WHERE p1.name CONTAINS 'AdaptImmune' ...
Semantic:   top-5 chunks matching "indemnification AdaptImmune"
```

**Example 3 — risk + clause text**
```
Query:   "What compliance risks exist and what does the liability clause say?"

Classifier → HYBRID  (structured "compliance risks" + text "what does ... say")

KG path:    intent=find_all_risks  → Risk nodes + INCREASES_RISK_FOR edges
Semantic:   top-5 chunks matching "liability clause"
```

---

### Path B — STRUCTURED examples

These are count/distribution queries where the KG alone is sufficient.

**Example 1 — pure count**
```
Query:   "How many contracts are governed by Delaware law?"

Classifier → STRUCTURED  ("how many" signal)
KG only:    intent=find_jurisdictions
            Cypher:
              MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
              WHERE j.name CONTAINS 'Delaware'
              RETURN c.name AS contract, j.name AS jurisdiction
              LIMIT 50
Result:  pipe-separated table of matching contracts
```

**Example 2 — most common**
```
Query:   "Which parties appear most commonly across all contracts?"

Classifier → STRUCTURED  ("most common" signal)
KG only:    intent=find_parties
            Cypher:
              MATCH (c:Contract)-[:SIGNED_BY]->(p:Party)
              RETURN c.name AS contract, p.name AS party
              LIMIT 50
```

**Example 3 — gap analysis**
```
Query:   "Which contracts are missing an indemnity clause?"

Classifier → STRUCTURED  (analytical gap query)
KG only:    intent=find_missing_indemnity
            Cypher:
              MATCH (c:Contract)
              OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(ic:IndemnityClause)
              WHERE ic IS NULL
              RETURN c.name AS contract_missing_indemnity
              LIMIT 50
```

---

### Path C — SEMANTIC examples

These are open-ended questions where clause text matters more than graph structure.

**Example 1 — exact clause language**
```
Query:   "What does the exact language of the non-compete clause say?"

Classifier → SEMANTIC  ("exact language" override signal)
Vector+BM25 on "non-compete clause language"
→ top-5 text chunks containing the non-compete language verbatim
```

**Example 2 — general explanation**
```
Query:   "Explain the payment structure in this contract."

Classifier → SEMANTIC  (no structured signals; default HYBRID, but no KG
                         intent matched strongly → falls back to semantic)
→ top-5 chunks about payment terms, fees, royalties
```

**Example 3 — clause text retrieval**
```
Query:   "What does the confidentiality clause say?"

Classifier → SEMANTIC  ("what does ... say" overrides to HYBRID, but if
                         STRUCTURED signals absent → pure SEMANTIC)
→ top-5 chunks about confidentiality / NDA obligations
```

---

## Intent → graph mapping

| Intent group | Intents | Graph queried |
|---|---|---|
| Legal entity | `find_parties`, `find_indemnification`, `find_jurisdictions`, `find_termination_clauses`, `find_confidentiality_clauses`, `find_payment_terms`, `find_obligations`, `find_liability_clauses`, `find_effective_dates`, `find_expiration_dates`, `find_renewal_terms`, `find_disclosures` | Legal Entity Graph |
| Document structure | `find_sections` | Document Hierarchy Graph |
| Lineage | `find_superseded_contracts`, `find_amendments`, `find_references`, `find_incorporated_documents`, `find_attachments`, `find_replacements` | Cross-Contract Lineage Graph |
| Risk / gaps | `find_all_risks`, `find_risk_chains`, `find_missing_indemnity`, `find_missing_termination` | Risk Dependency Graph |
| Fallback | `list_contracts` | Any graph |

---

## How schemas feed into query building

Four logical graph schemas are defined in `kg/legal/retrieval/schemas.py` as compact
text strings (one per `GraphType` enum value). They describe the vertex labels,
edge types, and property names for each subgraph.

```
GraphType.ENTITY    → ENTITY_SCHEMA    (Party, Contract, Jurisdiction, clause types…)
GraphType.HIERARCHY → HIERARCHY_SCHEMA (Contract, Section, Clause, HAS_SECTION…)
GraphType.LINEAGE   → LINEAGE_SCHEMA   (Contract, ReferenceDocument, AMENDS…)
GraphType.RISK      → RISK_SCHEMA      (Risk, Party, INCREASES_RISK_FOR, CAUSES)
```

All four schemas live in the **same physical AGE graph** (`legal_graph`). The schema
strings are used at two points:

**1. GraphRouter** (`kg/legal/retrieval/graph_router.py`)
Regex keyword matching maps a question to a `list[GraphType]`.
`get_schema(types)` concatenates the relevant compact strings and returns them
for callers that need the schema text (e.g., hypothetical LLM-driven Cypher
generation — currently not used in the main pipeline).

**2. NL2CypherConverter** (`kg/legal/retrieval/nl2cypher.py`)
```python
async def convert(self, question: str, schema: str = "") -> str:
    match = self._parser.parse(question)   # IntentParser — regex only
    builder = QUERY_CAPABILITIES[match.intent]  # query_builder.py
    return builder(match.params)
```
The `schema` argument is **accepted but unused**. Schema knowledge is
**encoded directly in the builder functions** in `query_builder.py` — each
builder hardcodes the correct vertex labels and edge types for its intent.
This means the pipeline is completely deterministic: no LLM, no prompt,
no schema drift possible.

The schema strings are available for future use (e.g., a fallback LLM-driven
path for queries with no matched intent) without changing the public API.

---

## KG entity lookup (within Path A and B)

When the KG path needs to find entities before traversing relationships:

```
search_entities(query, limit=10)
  │
  ├── EntityIndex (kg/entity_index.py)
  │   Shadow table kg_entity_index in main PostgreSQL DB
  │   ├── tsvector GIN full-text search
  │   └── pgvector IVFFlat cosine similarity
  │   Merged via RRF (k=60)
  │
  └── Fallback: O(n) AGE CONTAINS scan if EntityIndex unavailable
```

---

## Two-level intent routing

There are **two separate classifiers** — do not confuse them:

| Classifier | File | Question answered | Output |
|---|---|---|---|
| `HybridIntentClassifier` | `rag/retrieval/intent_classifier.py` | Which retrieval modalities to activate? | `HYBRID` / `STRUCTURED` / `SEMANTIC` |
| `IntentParser` | `kg/legal/retrieval/intent_parser.py` | Which graph and Cypher template to use? | `IntentMatch(intent, params)` |

The first routes at the tool level. The second routes within the KG path.

---

## Key files

| File | Role |
|---|---|
| `rag/retrieval/intent_classifier.py` | Top-level HYBRID/STRUCTURED/SEMANTIC routing |
| `rag/retrieval/hybrid_kg_retriever.py` | Orchestrates parallel paths + fusion |
| `kg/legal/retrieval/intent_parser.py` | KG-level regex intent detection |
| `kg/legal/retrieval/query_builder.py` | Intent → Cypher builder; `QUERY_CAPABILITIES` dict |
| `kg/legal/retrieval/nl2cypher.py` | Thin orchestrator: `convert(question) → Cypher` |
| `kg/legal/retrieval/graph_router.py` | `route(query) → list[GraphType]` (schema selection) |
| `kg/legal/retrieval/schemas.py` | `GraphType` enum + compact schema strings (unused by NL2Cypher) |
| `kg/legal/retrieval/cli.py` | Interactive CLI: `python -m kg.legal.retrieval.cli` |
| `kg/legal/retrieval/eval_pipeline.py` | Retrieval quality eval (intent match, hit rate, latency) |
| `kg/legal/ingestion/eval_pipeline.py` | Ingestion quality eval (entities/chunk, dedup, confidence) |
| `kg/age_graph_store.py` | `run_cypher_query(cypher) → str` |
| `kg/entity_index.py` | Entity full-text + vector shadow table |

---

## Adding a new KG intent

1. Add a regex pattern + intent name to `_PATTERNS` in `kg/legal/retrieval/intent_parser.py`.
2. Write `build_<intent>_query(params: dict) -> str` in `kg/legal/retrieval/query_builder.py`.
3. Register it in `QUERY_CAPABILITIES`.
4. No prompt changes, no model dependency.
