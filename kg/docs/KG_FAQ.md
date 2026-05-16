# Knowledge Graph Pipeline — FAQ & Findings

Living document. Updated as we run experiments and make architectural changes.
See `kg/docs/KG_INGESTION_PIPELINE.md` for the ingestion design reference.

---

## Table of Contents

**Practical / operational (start here)**

1. [How are contracts chunked for KG extraction?](#how-are-contracts-chunked-for-kg-extraction)
2. [How many AGE graphs are created? Can I view them with Apache AGE Viewer?](#how-many-age-graphs-are-created-can-i-view-them-with-apache-age-viewer)
3. [Hybrid retrieval strategy](#hybrid-retrieval-strategy-rrf-across-modalities-vs-route--execute--synthesize)
4. [How does intent classification work, and why is it rule-based?](#how-does-intent-classification-work-and-why-is-it-rule-based-instead-of-llm-based)
5. [Ingestion pipeline CLI](#ingestion-pipeline-cli)
6. [Retrieval pipeline CLI](#retrieval-pipeline-cli)

**Evaluation**

7. [Evaluation overview](#evaluation-overview)
8. [What are the Bronze JSON audit files written by `_save_json()`?](#what-are-the-bronze-json-audit-files-written-by-_save_json)
9. [Ingestion eval — how to run and what it measures](#ingestion-eval--how-to-run-and-what-it-measures)
10. [Ingestion eval — latest results (2026-05-16, 14 contracts)](#ingestion-eval--latest-results-2026-05-16-14-contracts)
11. [Retrieval eval — how to run and what it measures](#retrieval-eval--how-to-run-and-what-it-measures)
12. [Retrieval eval — latest results (2026-05-16, 34 questions)](#retrieval-eval--latest-results-2026-05-16-34-questions)
13. [Known issues found by evals](#known-issues-found-by-evals)

**Issues, performance & improvements**

14. [JSON parsing failures from llama3.1:8b](#json-parsing-failures-from-llama318b)
15. [Performance & timing](#performance--timing)
16. [Planned improvements](#planned-improvements)
17. [Guardrails](#guardrails)
18. [Observed issues & fixes](#observed-issues--fixes)
19. [Run log](#run-log)

**Design rationale (read if you want the "why")**

20. [What is the CUAD fast ingest and why is it so much faster than LLM extraction?](#what-is-the-cuad-fast-ingest-and-why-is-it-so-much-faster-than-llm-extraction)
21. [Why Bronze / Silver / Gold?](#why-bronze--silver--gold-instead-of-writing-directly-to-age)
22. [Why 5 separate LLM passes?](#why-5-separate-llm-passes-instead-of-one-combined-prompt)
23. [Why distinct vertex labels?](#why-distinct-vertex-labels-party-contract-instead-of-flat-entity)
24. [Where does Ollama do the heavy lifting — and where can it ruin the graph?](#where-does-ollama-do-the-heavy-lifting--and-where-can-it-ruin-the-graph)
25. [Why Apache AGE?](#why-apache-age-for-the-knowledge-graph)

---

## Architecture decisions

### How are contracts chunked for KG extraction?

**Short answer: Docling semantic chunks from the `chunks` table, with a character-stride fallback.**

The KG extraction pipeline reads pre-built semantic chunks from the PostgreSQL `chunks` table — the same chunks produced by the Docling `HybridChunker` during ingestion — rather than re-splitting the raw document text.

#### How it works

`_fetch_contracts()` (`kg/legal/ingestion/extraction_pipeline.py:948`) runs:

```sql
SELECT content FROM chunks WHERE document_id = $1::uuid ORDER BY chunk_index
```

If a document has no rows in `chunks` (e.g., ingested by a path that doesn't go through Docling), it falls back to a fixed character-stride split at `KG_EXTRACTION_CHUNK_SIZE` (default 1500 chars).

#### What Docling chunks give us

| Property | Docling HybridChunker |
|---|---|
| Boundary logic | Document structure — headings, paragraphs, tables |
| Token limit | `max_tokens` (default 512 tokens) — stays within embedding model window |
| Context prefix | Each chunk carries heading ancestry — self-contained for the LLM |
| Overlap | Configurable (default 100 chars) with sentence-boundary snapping |

This means KG chunk boundaries now match RAG chunk boundaries exactly. A future improvement can exploit this: KG entity hit → `chunk_index` → retrieve the exact matching chunk for context boosting in the RRF pipeline.

#### Known limitation: cross-chunk relationships

Each chunk is sent through 5 LLM passes independently.  The LLM sees only the text within that chunk; it cannot see entities from adjacent chunks.  A relationship whose source and target entity fall in different chunks will be missed.

- Chunk N ends: "…Party A shall indemnify"
- Chunk N+1 starts: "Party B from all claims…"
- The relationship `Party A –INDEMNIFIES→ Party B` is never extracted.

Silver deduplication partially mitigates this for entities (matching `(label, normalized_name)` across all chunks of a contract), but relationships that span a chunk boundary are lost.  The `source_chunk_indices` column in `kg_canonical_entities` records which chunks each entity appeared in — foundation for a future cross-chunk relationship re-extraction pass.

---

### How many AGE graphs are created? Can I view them with Apache AGE Viewer?

**One graph, named `legal_graph`. All pipelines write into it.**

#### How many graphs

Every code path that writes to the graph goes through `AgeGraphStore`, which reads a single setting:

```python
# rag/config/settings.py
age_graph_name: str = Field(default="legal_graph", ...)
```

`AgeGraphStore.initialize()` calls `SELECT create_graph('legal_graph')` once (idempotent — no-ops if it already exists). There is no code that creates a second graph or switches the name per pipeline.

The three pipelines that all write to the **same** `legal_graph`:

| Pipeline | File | What it writes |
|---|---|---|
| CUAD annotation ingest | `kg/legal/ingestion/cuad_kg_ingest.py` | Entities + relationships from `cuad_eval.json` CUAD annotations |
| LLM extraction (Bronze→Silver→Gold) | `kg/legal/ingestion/extraction_pipeline.py` (`GoldProjector`) | Entities + relationships from LLM extraction of contract text |
| RAG agent KG tools | `kg/age_graph_store.py` (called by `rag_agent.py`) | Runtime entity/relationship lookups (read-only at query time) |

The PostgreSQL *database* in `docker-compose.yml` is also named `legal_graph` (`POSTGRES_DB: legal_graph`) — this is the PostgreSQL database that AGE stores its internal graph metadata in. The AGE *graph object* inside it is also called `legal_graph`. They share a name but are distinct things (the database is the container; the graph is an object inside it).

To create a second graph (e.g. for a test environment), set `AGE_GRAPH_NAME=test_graph` in `.env` before running.

#### Viewing with Apache AGE Viewer

AGE Viewer is bundled in `docker-compose.yml` as the `age-viewer` service.

```powershell
docker compose up -d age age-viewer
```

Then open **http://localhost:3001** and connect with:

| Field      | Value         |
|------------|---------------|
| Host       | `age`         |
| Port       | `5432`        |
| Database   | `legal_graph` |
| User       | `age_user`    |
| Password   | `age_pass`    |
| Graph Path | `legal_graph` |

#### Cypher queries for all four graphs

Paste these directly into the AGE Viewer (Chrome) query box:

**1. Legal Semantic Graph** — parties, jurisdictions, clause types, indemnification
```cypher
MATCH (c:Contract)-[r]->(n)
RETURN c, r, n
LIMIT 60
```

**2. Document Hierarchy Graph** — contract → section → clause structure
```cypher
MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, s, cl
LIMIT 80
```

**3. Cross-Contract Lineage Graph** — amendments, supersessions, references
```cypher
MATCH (c1:Contract)-[r:AMENDS|SUPERCEDES|REPLACES|REFERENCES|INCORPORATES_BY_REFERENCE|ATTACHES]->(c2)
RETURN c1, r, c2
LIMIT 60
```

**4. Risk Dependency Graph** — compliance gaps and risk cascades
```cypher
MATCH (r:Risk)-[rel]->(n)
RETURN r, rel, n
LIMIT 60
```

For extended queries and filter examples see **[GRAPH_VIEWER.md](GRAPH_VIEWER.md)**.

---

### Hybrid retrieval strategy: RRF across modalities vs Route → Execute → Synthesize

**Question asked:** "Can we use full-text + vector + RRF to combine KG results with text chunks?"

**Decision: parallel execution with Context Fusion, not cross-modality RRF.**

RRF merges ranked lists of the *same result unit* (chunk IDs). KG hits are
entities and relationships — not chunk IDs — so they cannot directly enter the
RRF pool.

The chosen architecture runs both paths in parallel and fuses results in a
single context block:

```
USER QUERY
     │
     ▼
Intent Classification
     │
+----+----+
│         │
▼         ▼
Semantic  Structured
Retrieval Reasoning
(tsvector (KG / AGE)
+ pgvector
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

**RRF is still used** — but only *inside* the Semantic Retrieval box (BM25 +
vector over the `chunks` table). This is the existing `hybrid_search()` in
`postgres.py`.

**Future cross-path RRF is possible** by projecting KG entity hits to their
`source_chunk_indices` (stored in `kg_canonical_entities` since 2026-05-02).
A KG entity match → chunk indices → chunk UUIDs that can enter the RRF list
alongside vector and BM25 hits. This is not yet wired up; the column is there
as foundation.

**Analytical questions (Q89-100)** — count/aggregate/distribution queries —
route to `STRUCTURED` only. The LLM synthesizes SQL/Cypher results without
needing text chunks.

**Implementation:** `rag/retrieval/intent_classifier.py`,
`rag/retrieval/hybrid_kg_retriever.py`, `search_hybrid_kg` agent tool.
Tests + answer recording: `rag/tests/test_hybrid_kg_retrieval.py`.

---

### How does intent classification work, and why is it rule-based instead of LLM-based?

**There are two completely separate layers — don't conflate them.**

#### Layer 1: Which tool to call — the LLM decides

The Pydantic AI agent exposes four tools. Each tool has a docstring that tells the LLM when to use it:

| Tool | When to use (from docstring) |
|---|---|
| `search_knowledge_base` | Semantic / text retrieval over document chunks |
| `search_knowledge_graph` | Named entity lookup — parties, jurisdictions, clause types |
| `search_hybrid_kg` | Questions needing both clause text and graph facts |
| `run_graph_query` | Multi-hop Cypher, aggregations, analytics |

The LLM reads those descriptions at inference time and picks which tool(s) to call — and in what order. There is no code intercepting the question. The agent may call multiple tools in a single turn.

#### Layer 2: Within `search_hybrid_kg` — rule-based routing

When the LLM calls `search_hybrid_kg`, it enters `HybridKGRetriever.retrieve()`, which runs `IntentClassifier.classify()` **before** any database call. This classifies into three buckets:

```
STRUCTURED  →  KG path only   (skip the vector store entirely)
SEMANTIC    →  chunks only    (skip the KG entirely)
HYBRID      →  both paths in parallel  (the safe default)
```

The classifier is two compiled regex patterns (`rag/retrieval/intent_classifier.py`):

```python
_ANALYTICAL = re.compile(
    r"\b(how many|count|total number|on average|distribution of|
        most common|top \d+|aggregate|median|...)\b"
)

_GRAPH_TRAVERSAL = re.compile(
    r"\b(traverse|two hops?|shortest path|connected to|path between|...)\b"
)
```

Decision logic:

```
_ANALYTICAL matches AND query doesn't also ask for clause text  →  STRUCTURED
otherwise                                                        →  HYBRID
```

`SEMANTIC` is never returned by `classify()` — it is reserved for explicit caller overrides. HYBRID is the safe default: both paths run with `asyncio.gather` and their results are fused.

#### Why rule-based here, not LLM-based

**1. It controls parallelism, not meaning.**
`STRUCTURED` skips the vector store to save latency. `HYBRID` launches both database calls with `asyncio.gather`. This is a performance routing decision. Getting it wrong just wastes a database call — it cannot produce a wrong answer, because both paths merge into the same fused context block regardless.

**2. The patterns are unambiguous.**
"How many contracts" is always an aggregation. "Shortest path between" always needs graph traversal. There is no fuzzy case where a regex gives a wrong answer that an LLM would get right.

**3. Speed and cost.**
This runs on every `search_hybrid_kg` call, inside what may already be a streaming response. A 100 ms LLM classification call would be noticeable overhead for zero additional accuracy gain.

The same principle applies to the read-only guardrail in `run_graph_query`: `age_graph_store.run_cypher_query` blocks `CREATE/MERGE/SET/DELETE` with a regex because safety must be guaranteed, not estimated.

---

### Ingestion pipeline CLI

The ingestion module has two entry points — CUAD fast ingest and LLM extraction — plus an eval pipeline that reads the tables they populate.

#### CUAD fast ingest (no LLM)

```powershell
# Populate graph from CUAD human annotations — 510 contracts, minutes, no LLM
python -m kg.legal.ingestion.cuad_kg_ingest
```

Calls `build_cuad_kg()` (`kg/legal/ingestion/cuad_kg_ingest.py:61`) and writes directly to AGE.  No Bronze or Silver tables, no LLM calls.  Run this first to get a high-quality baseline graph before LLM enrichment.

#### LLM extraction pipeline (Bronze → Silver → Gold → Risk)

```powershell
# All contracts in the database (~5 days local on CPU)
python -m kg.legal.ingestion.extraction_pipeline --all

# Small batch for testing
python -m kg.legal.ingestion.extraction_pipeline --limit 20

# Single contract by UUID
python -m kg.legal.ingestion.extraction_pipeline --contract-id <uuid>

# Replay Silver + Gold from existing Bronze — no LLM, fast
python -m kg.legal.ingestion.extraction_pipeline --project --all
```

Key entry points in `kg/legal/ingestion/extraction_pipeline.py`:

| Method | Line | Purpose |
|---|---|---|
| `process_contract()` | L885 | Full pipeline per contract |
| `project_contract()` | L933 | Silver + Gold replay, no LLM |
| `_fetch_contracts()` | L948 | Loads contracts from the `chunks` table |
| `_save_json()` | L862 | Writes Bronze JSON audit files to `kg/evals/jsons/` |

#### Ingestion eval

```powershell
# Print summary table to stdout
python -m kg.legal.ingestion.eval_pipeline

# Also write JSON to file
python -m kg.legal.ingestion.eval_pipeline --output kg/evals/ingest_eval_latest.json
```

---

### Retrieval pipeline CLI

The retrieval CLI (`kg/legal/retrieval/cli.py`) queries the live AGE graph.  Requires the AGE container: `docker compose up -d age`.

#### Interactive REPL

```powershell
python -m kg.legal.retrieval.cli
```

Type a question at the `>` prompt.  Results render as a Rich table.  `exit` or Ctrl-C to quit.

#### Single question

```powershell
python -m kg.legal.retrieval.cli --question "Which parties indemnify each other?"

# Show the generated Cypher alongside results
python -m kg.legal.retrieval.cli --question "What is the governing law?" --show-cypher
```

#### Stdin mode (for piping / batch)

```powershell
echo "Who are the parties?" | python -m kg.legal.retrieval.cli --stdin

# Batch from file
Get-Content questions.txt | python -m kg.legal.retrieval.cli --stdin
```

#### Retrieval eval

```powershell
# Dry-run — tests IntentParser only, no AGE connection needed
python -m kg.legal.retrieval.eval_pipeline --dry-run

# Full eval against live AGE
python -m kg.legal.retrieval.eval_pipeline --output kg/evals/retrieval_eval_latest.json
```

---

## Evaluation

### Evaluation overview

The pipeline produces three categories of evaluation artifacts, all written to `kg/evals/` (gitignored):

| Artifact | Where | What it contains | Requires live AGE? |
|---|---|---|---|
| Bronze JSON audit files | `kg/evals/jsons/<title>_<id[:8]>.json` | Per-contract extraction output: entities, validated relationships, hierarchy nodes, cross-refs — one file per `process_contract()` call | No — written during extraction |
| Ingestion eval report | `kg/evals/ingest_eval_latest.json` | Aggregated Bronze/Silver/Gold metrics across all processed contracts | No — reads PostgreSQL tables |
| Retrieval eval report | `kg/evals/retrieval_eval_latest.json` | Intent match rate, Cypher validity, result hit rate, latency for 34 predefined questions | Yes — executes Cypher against AGE |

#### Eval pipeline reports

Both eval pipelines summarise the tables, not re-run extraction:

```powershell
# Ingestion metrics (no AGE needed)
python -m kg.legal.ingestion.eval_pipeline --output kg/evals/ingest_eval_latest.json

# Retrieval quality (needs live AGE)
python -m kg.legal.retrieval.eval_pipeline --output kg/evals/retrieval_eval_latest.json
```

#### End-to-end integration test

`kg/tests/test_extraction_pipeline_e2e.py` runs the full Bronze → Silver → Gold → Risk → NL2Cypher stack with mocked LLM calls against real PostgreSQL + AGE:

```powershell
pytest kg/tests/test_extraction_pipeline_e2e.py -m integration -v
```

Skips automatically if PostgreSQL or AGE is unreachable.  Does not call Ollama.

---

### What are the Bronze JSON audit files written by `_save_json()`?

**Short answer: per-contract snapshots of raw LLM extraction output, written automatically after every `process_contract()` call — the primary audit trail for debugging Silver/Gold data.**

Written by `_save_json()` (`kg/legal/ingestion/extraction_pipeline.py:862`) at the end of every `process_contract()` call.  File name: `<title_slug>_<contract_id[:8]>.json`.  Stored in `kg/evals/jsons/` (tracked by git).

**Contents per chunk** (example from `kg/evals/jsons/E2E_Test___Acme___Beta_LLC_e2e00000.json`):

```json
{
  "contract_id": "e2e00000-0000-0000-0000-000000000001",
  "title": "E2E Test — Acme + Beta LLC",
  "model_version": "qwen2.5:14b",
  "chunks": [
    {
      "chunk_index": 0,
      "entities": [
        {"entity_id": "ent-e2e-0001", "label": "Party",        "canonical_name": "Acme Corp", "text_span": "Acme Corp", "confidence": 0.95},
        {"entity_id": "ent-e2e-0002", "label": "Party",        "canonical_name": "Beta LLC",  "text_span": "Beta LLC",  "confidence": 0.95},
        {"entity_id": "ent-e2e-0003", "label": "Jurisdiction", "canonical_name": "Delaware",  "text_span": "Delaware",  "confidence": 0.95}
      ],
      "relationships": [
        {"relationship_id": "rel-e2e-0001", "source_entity_id": "ent-e2e-0001", "target_entity_id": "ent-e2e-0003", "relationship_type": "GOVERNED_BY",  "evidence_text": "governed by the laws of Delaware",    "confidence": 0.9},
        {"relationship_id": "rel-e2e-0002", "source_entity_id": "ent-e2e-0001", "target_entity_id": "ent-e2e-0002", "relationship_type": "INDEMNIFIES",  "evidence_text": "Acme Corp shall indemnify Beta LLC", "confidence": 0.9}
      ],
      "hierarchy_nodes": [],
      "cross_refs": []
    }
  ]
}
```

Fields:
- `entities` — Pass 1 output, filtered to confidence ≥ threshold
- `relationships` — Pass 5 output (post-validation); **Pass 2 raw relationships are not saved**
- `hierarchy_nodes` — Pass 3 output (Section/Clause structure)
- `cross_refs` — Pass 4 output (AMENDS / REFERENCES / INCORPORATES_BY_REFERENCE)

**What is NOT saved:** the raw relationships from Pass 2 before Pass 5 (validation) filters them.  If you need to audit what validation dropped, add a `raw_relationships` field to `BronzeArtifact` (tracked in Planned improvements).

These files are the primary audit trail for LLM extraction.  If Silver or Gold data looks wrong, read the JSON to see exactly what the LLM returned per chunk.

---

### Ingestion eval — how to run and what it measures

```powershell
python -m kg.legal.ingestion.eval_pipeline
python -m kg.legal.ingestion.eval_pipeline --output kg/evals/ingest_eval_latest.json
```

Reads `kg_raw_extractions` (Bronze JSONB), `kg_canonical_entities`, and `kg_canonical_relationships` (Silver).

**Metrics:**

| Metric | What it tells you |
|---|---|
| `contracts_evaluated` | How many contracts have Bronze data |
| `total_chunks` | Total Bronze rows processed |
| `total_entities` / `total_relationships` | Raw Bronze counts before confidence filter |
| `mean_entities_per_chunk` | LLM extraction density |
| `mean_confidence` | Average confidence across all extracted entities |
| `label_distribution` | Which entity labels the LLM produced |
| `rel_type_distribution` | Which relationship types the LLM produced |
| `silver_gold.raw_entities` | Entities written to Silver staging (after confidence ≥ 0.7) |
| `silver_gold.canonical_entities` | After Silver deduplication |
| `silver_gold.canonical_relationships` | After Silver deduplication |

**What to watch:** off-ontology labels (anything not in `VALID_LABELS`) signal prompt leakage; high dedup rates (> 50%) mean the LLM is duplicating entities across chunks; low confidence means uncertainty.

---

### Ingestion eval — latest results (2026-05-16, 14 contracts)

Source: `kg/evals/ingest_eval_latest.json`.

**Summary:**

| Metric | Value |
|---|---|
| Contracts evaluated | 14 |
| Chunks processed | 199 |
| Raw entities (Bronze) | 777 |
| Staged entities (Silver pre-dedup) | 587 |
| Canonical entities (Silver) | 436 (25.7% dedup) |
| Raw relationships (Bronze) | 423 |
| Staged relationships (Silver pre-dedup) | 320 |
| Canonical relationships (Silver) | 248 (22.5% dedup) |
| Mean entities per chunk | 3.9 |
| Mean relationships per chunk | 2.1 |
| Mean confidence | 96.4% |

**Label distribution (top 10):**

| Label | Count | In ontology? |
|---|---|---|
| Party | 324 | Yes |
| Clause | 136 | Yes |
| Contract | 61 | Yes |
| Obligation | 39 | Yes |
| Jurisdiction | 38 | Yes |
| EffectiveDate | 35 | Yes |
| Section | 29 | Yes |
| RenewalTerm | 18 | Yes |
| LiabilityClause | 18 | Yes |
| Risk | 12 | Yes |

**Off-ontology labels detected:** `Role (Reviewer)`, `Person (Reviewer)`, `Person (Author)`, `Person`, `Company`, `Location`, `Government Authority`, `Law`, `Product`, `Product/Service`, `Address`, `Term`, `Renewal Term`, `EffectiveDate/ExpirationDate`. Blocked by `safe_label()` before Silver write; inflate Bronze counts but never reach AGE.

**Relationship type distribution (top 10):**

| Rel type | Count |
|---|---|
| GOVERNED_BY | 96 |
| REFERENCES | 92 |
| OBLIGATES | 64 |
| DISCLOSES_TO | 34 |
| HAS_TERMINATION | 28 |
| HAS_CLAUSE | 27 |
| SUPERCEDES | 22 |
| LIMITS_LIABILITY | 22 |
| HAS_PAYMENT_TERM | 12 |
| INDEMNIFIES | 11 |

---

### Retrieval eval — how to run and what it measures

```powershell
# Dry-run: tests IntentParser + Cypher generation only, no AGE connection
python -m kg.legal.retrieval.eval_pipeline --dry-run

# Full eval against live AGE
python -m kg.legal.retrieval.eval_pipeline --output kg/evals/retrieval_eval_latest.json
```

34 predefined questions covering all 17 intent categories (2 per intent: one generic, one filtered by name param).

**Metrics:**

| Metric | What it tells you |
|---|---|
| `intent_match_rate` | Fraction where `IntentParser.parse()` returned the expected intent |
| `cypher_valid_rate` | Fraction of queries that executed without AGE syntax error |
| `result_hit_rate` | Fraction of queries that returned ≥ 1 row |
| `mean_latency_ms` | Average end-to-end time (IntentParser + Cypher execute) |
| `p95_latency_ms` | 95th-percentile latency |

`intent_match_rate` and `cypher_valid_rate` test the NL→Cypher pipeline itself — should be 100%.  `result_hit_rate` depends on graph population.

---

### Retrieval eval — latest results (2026-05-16, 34 questions)

Source: `kg/evals/retrieval_eval_latest.json`.  Graph state: 52 contracts (CUAD fast ingest) + 14 contracts (LLM extraction).

**Summary:**

| Metric | Value |
|---|---|
| Total questions | 34 |
| Intent match rate | 100% (34/34) |
| Cypher valid rate | 100% (34/34) |
| Result hit rate | 26.5% (9/34) |
| Mean latency | 220.6 ms |
| p95 latency | 228.9 ms |

**Intents with results:**

| Intent | Example question | Rows returned |
|---|---|---|
| `find_indemnification` | Which parties indemnify each other? | 5 |
| `find_jurisdictions` | What is the governing law? | 3 |
| `find_disclosures` | What disclosures are made between parties? | 9 |
| `find_references` | What documents are referenced? | 3 |
| `find_all_risks` | What are the compliance risks? | 52 |
| `find_risk_chains` | What risk factors cause other risks? | 12 |
| `find_missing_indemnity` | Which contracts lack an indemnity clause? | 52 |
| `find_missing_termination` | Which contracts are missing a termination clause? | 52 |
| `list_contracts` | List all contracts. | 52 |

**Intents with 0 results:** `find_parties`, `find_termination_clauses`, `find_confidentiality_clauses`, `find_payment_terms`, `find_obligations`, `find_liability_clauses`, `find_effective_dates`, `find_expiration_dates`, `find_renewal_terms`, `find_sections`, `find_superseded_contracts`, `find_amendments`, `find_incorporated_documents`, `find_attachments`, `find_replacements`.

The 26.5% hit rate reflects graph sparsity, not NL→Cypher accuracy.  Risk queries and absence queries (`find_missing_*`) hit because `RiskGraphBuilder` populates Risk nodes for all 52 CUAD contracts rule-based; most other edge types require LLM extraction.

---

### Known issues found by evals

**1. "Which" extracted as name param**

Questions like "Which contracts supersede others?" cause IntentParser to extract "Which" as the name filter, producing Cypher with `WHERE c.name CONTAINS 'Which'` — always 0 rows.  Affected intents: `find_superseded_contracts`, `find_amendments`, `find_replacements`, `find_missing_indemnity`, `find_missing_termination`.

Fix: add a guard in `IntentParser.parse()` (`kg/legal/retrieval/intent_parser.py:113`) to drop name params that match interrogative words: `{"Which", "What", "Who", "How", "When", "Where"}`.

**2. HAS_TERMINATION / HAS_PAYMENT_TERM discrepancy**

Ingestion eval shows 28 HAS_TERMINATION and 12 HAS_PAYMENT_TERM relationships in Bronze.  Retrieval eval returns 0 rows for `find_termination_clauses` and `find_payment_terms`.  Silver dedup should not reduce 28 to 0.

Likely cause: Gold projection writes the relationship but the target node uses the generic `Clause` label (LLM fallback) rather than `TerminationClause`.  The Cypher `MATCH (c)-[:HAS_TERMINATION]->(t:TerminationClause)` then matches nothing.

Fix: add a Gold projection validation step — count HAS_TERMINATION / HAS_PAYMENT_TERM edges in AGE and compare to Silver canonical counts.

**3. Off-ontology label inflation**

14 off-ontology label types detected (latest run).  They never reach AGE but inflate Bronze counts and skew the `label_distribution` metrics.  Improve entity extraction prompt examples for `Person`, `Company`, and date-hybrid types.

**4. 52-contract baseline from CUAD**

`list_contracts` returns 52 rows, not 14.  This is correct — the CUAD fast ingest populated 52 Contract nodes before LLM extraction ran.  Retrieval eval results for relationship-level queries reflect only LLM extraction yield (14 contracts), while risk and absence queries reflect the full 52-contract CUAD baseline.

---

## Design rationale

### What is the CUAD fast ingest and why is it so much faster than LLM extraction?

**Short answer: no LLM involved — it reads pre-built human annotations from a JSON file and reformats them into AGE.**

#### What CUAD is

[CUAD](https://huggingface.co/datasets/theatticusproject/cuad) (Contract Understanding Atticus Dataset) is a public legal NLP benchmark released by the Atticus Project — a group of lawyers and ML researchers.  They manually annotated 510 commercial contracts from SEC EDGAR filings for 41 specific clause types: parties, governing law, termination provisions, renewal terms, liability caps, indemnification, etc.  The annotations ship as `cuad_eval.json`.

#### What `cuad_kg_ingest.py` does

`build_cuad_kg()` in `kg/cuad_kg_ingest.py` reads `cuad_eval.json` and calls `AgeGraphStore.upsert_entity()` / `add_relationship()` directly — no chunking, no LLM calls, no Bronze/Silver pipeline.  It is a pure format conversion: CUAD annotation → AGE vertex/edge.

#### Why it finishes in minutes vs days

| Step | LLM extraction | CUAD ingest |
|---|---|---|
| Read contract text | Yes (1500-char chunks) | No |
| LLM call per chunk | 5 passes × N chunks × 522 contracts | None |
| Parse + validate JSON | Yes (`_parse_json` repair loop) | No |
| Bronze → Silver dedup | Yes | No |
| Write to AGE | Yes (Gold projection) | Yes (directly) |
| **Bottleneck** | Local Ollama inference (~20 s/call) | Disk I/O + asyncpg inserts |

LLM extraction at ~15 min/contract = ~5 days for 522 contracts.  CUAD ingest = minutes.

#### Trade-offs

| | CUAD ingest | LLM extraction |
|---|---|---|
| Speed | Minutes | Days (local) / Hours (API) |
| Coverage | 41 fixed clause types only | Any entity the LLM can name |
| Quality | Human-annotated (high precision) | Model-dependent (variable) |
| Cross-contract lineage | Not in CUAD annotations | Extracted by cross-ref pass |
| Risk graph | Not in CUAD (built separately) | Built by `RiskGraphBuilder` post-Gold |

#### Recommended workflow

1. **Always run CUAD ingest first** — gives a high-quality, complete graph in minutes.
2. **Run LLM extraction selectively** — use `--limit N` or `--contract-id` to enrich specific high-value contracts with lineage and relationship data not covered by CUAD annotations.
3. **Run `RiskGraphBuilder`** — rule-based risk inference runs on top of whatever Silver data exists (CUAD or LLM), no LLM needed.

```bash
# Step 1: Fast graph from CUAD annotations (minutes)
python -m kg.legal.ingestion.cuad_kg_ingest

# Step 2: Optional — LLM enrichment for a handful of contracts
# JSON artifacts written to kg/evals/jsons/ automatically
python -m kg.legal.ingestion.extraction_pipeline --limit 20

# Step 3: Risk graph — called automatically by ExtractionPipeline,
# or replay Silver+Gold from existing Bronze without re-running LLM:
python -m kg.legal.ingestion.extraction_pipeline --project --all
```

---

### Where does Ollama do the heavy lifting — and where can it ruin the graph?

Ollama (`llama3.1:8b`) runs five sequential passes per chunk during Bronze extraction.  Each pass is an independent LLM call.  Here is exactly where it is responsible, and where things can go wrong.

#### Where Ollama is doing the work

| Pass | What it produces | Why it can't be rule-based |
|---|---|---|
| 1 — Entity extraction | `list[ExtractedEntity]` — canonical names, labels, confidence | The model must read legal prose and decide "Acme Corporation" is a Party and "Section 12.1" is a Clause. No regex can do this reliably across 510 contracts with varied formatting. |
| 2 — Relationship extraction | `list[ExtractedRelationship]` — typed edges with evidence text | Deciding that "Acme shall indemnify Beta" is an `INDEMNIFIES` edge requires semantic understanding, not keyword matching. |
| 3 — Document hierarchy | `list[HierarchyNode]`, `list[HierarchyEdge]` | Reconstructing Section → Clause → Chunk structure from varying document layouts. |
| 4 — Cross-contract references | `list[CrossContractRef]` — AMENDS / SUPERCEDES / REFERENCES | Spotting "this Agreement amends the Master Services Agreement dated …" and extracting the referenced document name exactly. |
| 5 — Validation | Filtered `list[ExtractedRelationship]` | Second-opinion pass: the model re-reads the text and removes relationships it deems unsupported. |

#### Where hallucinations actually damage the graph

**Pass 2 is the highest risk.** The model receives the entity list from Pass 1 and must only create relationships between those entity IDs. In practice:

- It invents entity IDs that don't exist in the Pass 1 output — relationships point to ghost nodes.
- It creates edges that sound plausible but have no evidence in the text ("Party A shall not disclose" does not imply `DISCLOSES_TO`).
- It uses the wrong relationship type — `REFERENCES` when the text says "in accordance with", not "pursuant to the Agreement".

**Pass 1 also hallucinates entities**, but less dangerously:

- Extracts generic nouns as entities ("the Agreement", "the parties") that the ontology doesn't model.
- Splits one entity into two slightly different canonical names ("Acme Corp" + "Acme Corporation").

**Pass 3 and 4 are lower risk** because the outputs are more structural and the model has less room to invent:

- Hierarchy extraction can invent Section numbering that doesn't match the actual document.
- Cross-contract references can invent document names that don't exist — these ghost references land in Bronze and Silver but fail Silver's name-resolution step when the `documents` table has no matching row.

**Pass 5 (validation) is the last guardrail** — but it can fail to catch what Pass 2 invented, particularly for relationships with plausible-sounding evidence text.

#### How we contain the damage

1. **Ontology whitelist** — `VALID_LABELS` and `VALID_REL_TYPES` block anything outside the known schema before it reaches AGE.
2. **Confidence threshold** — entities and relationships below 0.7 are dropped at Silver staging.
3. **Silver deduplication** — duplicate entities with slightly different names are merged on `(label, normalized_name)`, collapsing "Acme Corp" / "Acme Corporation" into one canonical node.
4. **Pass 5 validation** — second LLM opinion specifically tasked with removing unsupported edges.
5. **Bronze as the safety net** — if Silver produces garbage, drop the Silver + Gold tables and replay from Bronze. Bronze is immutable and never re-processed.

#### What is NOT protected against

- A hallucinated relationship that passes validation and has a confidence of 0.85 will make it into the graph. There is no automated ground-truth check.
- A relationship with two valid entity IDs but wrong direction (`INDEMNIFIES` reversed) passes all filters.
- Pass 5 can itself hallucinate — it may "validate" a relationship by fabricating evidence text that wasn't in the original chunk.

**Bottom line:** treat everything in the `kg_canonical_relationships` table as "model opinion, confidence ≥ 0.7, second-opinion validated" — not as ground truth.  The CUAD fast ingest path (human annotations) is the high-precision baseline; LLM extraction enriches it with lineage and risk data that the annotations don't cover.

---

### Why Bronze / Silver / Gold instead of writing directly to AGE?

Direct LLM → AGE inserts are dangerous: you get hallucinated edges, duplicate
nodes with slightly different names, and no way to replay or fix without
re-running expensive LLM calls.

The medallion split gives you:
- **Bronze** — immutable record of exactly what the LLM said.  Re-run Silver/Gold
  any time for free (no LLM calls).
- **Silver** — deduplicated, confidence-filtered canonical tables in PostgreSQL.
  The dedup logic runs in SQL, is fast, and is easy to tune.
- **Gold** — AGE graph built from trusted Silver data only.  Projection is
  idempotent: re-project as many times as needed.

### Why distinct vertex labels (`:Party`, `:Contract`) instead of flat `:Entity`?

Flat model:
```cypher
(:Entity {entity_type: "Party", name: "Acme Corp"})
```
Distinct labels:
```cypher
(:Party {name: "Acme Corp", label: "Party"})
```

Benefits:
- Cleaner Cypher — `MATCH (p:Party)-[:INDEMNIFIES]->(q:Party)` vs filtering on property
- Label-specific AGE indexes (better traversal performance at scale)
- Semantically correct — labels ARE the type in property graph model

Trade-off: `MATCH (n)` (no label) scans all vertex tables.  Mitigated by always
using the `e.label` property for cross-type queries.

### Why 5 separate LLM passes instead of one combined prompt?

One combined prompt asking for entities + relationships + hierarchy + cross-refs
+ validation in a single call regularly produces:
- mixed-up entity IDs in relationships (references IDs not in the entity list)
- skipped fields when output is long
- hallucinated relationships when the model tries to fill all slots

Separate focused passes with `temperature=0` and small chunks (1–3 clauses)
produce cleaner, more consistent JSON per call.  The extra LLM calls are the
cost — see timing section below.

---

### Why Apache AGE for the knowledge graph?

**Short answer: multi-hop traversal, LLM-driven ad-hoc Cypher, and native graph semantics at CUAD scale.**

#### What AGE gives us

**Multi-hop traversal in a single query.**  Legal analysis regularly requires following relationship chains: "find all contracts that reference a document that was amended by a contract governed by Delaware law."  In Cypher this is one `MATCH` pattern.  SQL cannot express variable-depth graph traversal without recursive CTEs that become unreadable past two hops.

**The agent can ask ad-hoc questions.**  `run_cypher_query` (in `AgeGraphStore`) accepts whatever Cypher the LLM generates and executes it at query time.  There is no equivalent in a relational model — you cannot "run whatever JOIN chain the LLM just wrote" over plain tables.

**Entity type as a schema concept, not a string value.**  `:Party` is a vertex label with its own storage in AGE.  `MATCH (e:Party)` touches only Party nodes, not the full entity set.  At CUAD scale (510 contracts, thousands of entities) that distinction matters for query performance.

| Capability | AgeGraphStore |
|---|---|
| 1-hop neighbour | `MATCH (s)-[r]-(t)` |
| 2-hop path | `MATCH (a)-[*1..2]->(b)` |
| Variable-depth | `MATCH (a)-[*1..N]->(b)` |
| LLM ad-hoc Cypher | `run_cypher_query` executes Cypher |
| Type-filtered scan | `MATCH (e:Party)` — label-pruned |
| Pattern matching | `MATCH (p:Party)-[:INDEMNIFIES]->(q:Party)-[:SIGNED_BY]->(c:Contract)` |

#### The one trade-off: no tsvector inside AGE

AGE does not expose PostgreSQL's `tsvector` / GIN index.  Entity name search inside Cypher would be an O(n) CONTAINS scan.

**Resolved by `EntityIndex` (`kg/entity_index.py`)** — a shadow table `kg_entity_index` in the main PostgreSQL DB that mirrors every entity written to AGE.  It carries a `tsvector GENERATED` column (GIN index) and a `vector(N)` column (pgvector IVFFlat).  `AgeGraphStore.search_entities()` uses RRF (k=60) over both; the CONTAINS fallback only fires if the index is unreachable.

#### Infrastructure cost

AGE requires a patched PostgreSQL 15/16 image (`apache/age:latest`, port 5433), separate from the main pgvector DB.  Every connection must run `LOAD 'age'` and `SET search_path = ag_catalog` — handled automatically by the `_conn()` context manager and pool `init=_age_init` callback.

---

## Observed issues & fixes

### JSON parsing failures from llama3.1:8b

**Observed failure rate:** 21.5% (14/65 calls) on first real run (2026-05-02).

**Four error types seen in production (2026-05-02 first run):**

| Error | Cause | Fix |
|---|---|---|
| `Extra data: line N column M` | Model outputs two JSON objects in a row | `JSONDecoder.raw_decode()` — stops at end of first valid object |
| `Expecting ',' delimiter` | Model omits commas between object keys | `_clean_json()` best-effort repair; chunk skipped if unrepairable |
| `Invalid control character` | Unescaped `\n` or `\t` inside string values | Strip `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` before retry |
| `Illegal trailing comma` | JSON5-style `[...,]` / `{...,}` | `re.sub(r',\s*([}\]])', r'\1', s)` |

**`_parse_json` repair pipeline (in order):**
1. Strip markdown fences
2. `raw_decode` from first `{` — handles extra-data
3. `_clean_json` (control chars + trailing commas) + `raw_decode` — handles the rest
4. Return `{}` if both fail — chunk contributes 0 entities, graceful skip

**Architectural implication:** Chunk failures are silent (entity/relationship count
is just lower).  Consider logging a per-chunk extraction quality metric to Bronze
so we can see which chunks consistently fail and investigate prompt adjustments.

**TODO:** Add `parse_failed: bool` field to `BronzeArtifact` so we can track
repair rate over time.

### Missing commas in JSON — root cause

llama3.1:8b uses its context window for entity text and sometimes loses track of
JSON syntax in longer outputs.  Mitigation options (in priority order):

1. **Smaller chunks** — reduce from 1500 to 800–1000 chars, limiting output size
2. **Ollama JSON mode** — pass `format: "json"` to force structured output
   (currently not set — see improvement below)
3. **Structured output via pydantic-ai** — `result_type=EntityExtractionResult`
   lets pydantic-ai handle JSON enforcement

### AGE_DATABASE_URL missing from .env on first run

**Symptom:** `ValueError: AGE_DATABASE_URL is not set`
**Fix:** Added to `.env`:
```
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
AGE_GRAPH_NAME=legal_graph
```
Both values match the `rag_age` Docker container credentials.

---

## Performance & timing

### First real run (2026-05-02)

**Contract:** LIGHTBRIDGE CORP — Strategic Alliance Agreement
**Size:** 19,429 chars → 13 chunks (1500 chars/chunk)
**Total LLM calls:** 65 (13 × 5 passes)
**Average call duration:** ~21 seconds (llama3.1:8b local, MacBook/Windows CPU)
**Total extraction time:** ~23 minutes for one contract

**Extrapolation:**
| Contracts | Est. time |
|---|---|
| 1 | ~23 min |
| 10 | ~4 hrs |
| 100 | ~38 hrs |
| 522 (full CUAD) | ~8 days |

Full CUAD at this rate is not feasible on a single machine.  See scaling options.

### Scaling options (in order of effort)

1. **Reduce passes** — skip hierarchy and cross-ref passes for bulk runs; run
   them separately on high-value contracts only
2. **Enable Ollama JSON mode** — reduces parse failures, may speed up inference
3. **Reduce chunk size** — 800 chars gives the model less to output, faster calls
4. **Process chunks in parallel** — currently sequential.  Ollama handles one
   request at a time anyway, but with a GPU or remote inference this would help
5. **Batch smaller contracts** — many CUAD contracts are under 5k chars; group
   into one LLM call to amortise prompt overhead
6. **Switch to a faster model for bulk** — llama3.2:3b for pass 1 (entity
   extraction), llama3.1:8b for pass 5 (validation) only

---

## Planned improvements

### Near-term

- [ ] **Ollama JSON mode** — add `extra_body={"format": "json"}` to agent calls
  via pydantic-ai model settings.  Eliminates most JSON parse failures.
- [ ] **`parse_failed` field on BronzeArtifact** — track repair rate per chunk
- [ ] **Chunk size tuning** — experiment with 800 vs 1500 vs 2000 chars; measure
  entity yield per chunk
- [ ] **Skip hierarchy + cross-ref passes by default** — make them opt-in flags
  (`--passes entity,relationship,validate` CLI argument)

### Medium-term

- [ ] **Cross-contract name resolution** — Silver layer should fuzzy-match
  `target_document_name` from cross-ref extraction to known `contract_id` values
  in the `documents` table.  Currently stored but not resolved.
- [x] **Risk Dependency Graph** — implemented in `kg/risk_graph_builder.py` (2026-05-03).
  Rule-based gap detection on Silver canonical entities → Risk vertices + `INCREASES_RISK_FOR` /
  `CAUSES` edges in AGE.  Called automatically by `ExtractionPipeline` after Gold projection.
- [ ] **Incremental Bronze** — skip chunks already in `kg_raw_extractions` for
  the same `(contract_id, chunk_index, model_version)`.  Allows resuming
  interrupted runs without re-processing.
- [ ] **QA reporter** — read Silver tables, emit `qa_reports/` JSON files:
  duplicate entities, orphan nodes, invalid edges

### Architectural

- [ ] **Protocol / ABC for graph store** — replace `AgeGraphStore` direct type references with a `KGStore` protocol for easier testing
- [ ] **Chunk overlap** — entity references at chunk boundaries get missed; add
  ~200 char overlap between consecutive chunks
- [ ] **Graph-aware RAG retrieval** — after vector retrieval, expand context
  using AGE: `MATCH (matched_entity)-[*1..2]->(neighbour)` to pull in related
  clauses that vector search missed

---

## Guardrails

### Extraction path — what exists

| Guardrail | Where | What it does |
|---|---|---|
| Label whitelist | `ExtractedEntity.safe_label()` | Validates label against `VALID_LABELS`; falls back to `"Clause"` |
| Relationship whitelist | `ExtractedRelationship.safe_rel_type()` | Returns `None` for unknown types; filtered out before Bronze write |
| Confidence threshold | Silver staging + `_pass_*` methods | Drops entities / relationships below 0.7 (configurable via `KG_CONFIDENCE_THRESHOLD`) |
| JSON repair | `_parse_json` / `_clean_json` | Handles 4 llama3 failure modes; returns `{}` on total failure — chunk skipped gracefully |
| Pass 5 validation | `_pass_validate()` | Second LLM opinion — filters hallucinated / unsupported relationships |
| Bronze dedup | `UNIQUE (contract_id, chunk_index, model_version)` | Idempotent re-runs; re-processing a contract doesn't duplicate Bronze rows |
| Silver dedup | `DISTINCT ON (label, normalized_name)` | Merges duplicate entities across chunks; keeps highest confidence |
| Pydantic validation | Try/except in all `_pass_*` methods | Silently drops malformed LLM output rows |

### Extraction path — gaps

| Gap | Risk | Fix |
|---|---|---|
| **Context overflow** | Ollama defaults `num_ctx=2048`; dense passes (validation) can hit it; output silently garbled | Set `num_ctx=4096` via `KG_LLM_NUM_CTX` + pre-call `_truncate_to_budget()` guard |
| **Duplicate ontology** | `VALID_LABELS` / `VALID_REL_TYPES` defined in both `extraction_pipeline.py` and `kg/constants.py` — drift risk if one is updated without the other | Import from `kg.constants` only; remove the local copies |
| **No output size cap** | Abnormally large LLM response could cause memory pressure on long contracts | Add `num_predict=1024` to `ChatOllama` to cap output tokens |
| **Prompt injection** | Contract text goes into LLM prompts unsanitized | Low practical risk (source is legal text, not user input), but worth noting |

### Retrieval path — what exists

| Guardrail | Where | What it does |
|---|---|---|
| Label whitelist | `AgeGraphStore._safe_label()` | Validates against `VALID_LABELS` before Cypher string interpolation; falls back to `"Clause"` |
| Relationship whitelist | `AgeGraphStore._safe_rel_type()` | Returns `None` for unknown types; write is skipped |
| Read-only Cypher guard | `AgeGraphStore.run_cypher_query()` | Regex blocks `CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH` before query is executed |

### Retrieval path — gaps

| Gap | Risk | Fix |
|---|---|---|
| **Entity name in Cypher** | `normalized_name` / `name` are interpolated into Cypher via f-string — a malicious party name like `}) DETACH DELETE (` would break the query | Escape or parameterize name values in `upsert_entity` and `add_relationship` |
| **No query result size cap** | `run_cypher_query` has no `LIMIT` enforcement; an unfiltered `MATCH` returns the entire graph | Inject `LIMIT 500` if the query contains no `LIMIT` clause |
| **Substring search escaping** | `search_entities` / `search_as_context` interpolate the search term into `CONTAINS` — verify apostrophes and backslashes are escaped | Confirm `_escape_for_cypher()` covers all special characters |

---

## Run log

| Date | Contract | Chars | Chunks | LLM calls | Entities (raw) | Entities (canonical) | Rels (raw) | Rels (canonical) | Parse failures | Duration | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-02 | Recording1 | 576 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | ~2 min | Not a legal contract |
| 2026-05-02 | LIGHTBRIDGE CORP — Strategic Alliance Agreement | 19,429 | 13 | 65 | 48 | 42 | 41 | 25 | 14 (21.5%) | 23 min 40 sec | First real run; `_parse_json` fix not yet applied; Silver dedup removed 6 entities (12.5%), 16 rels (39%) |
