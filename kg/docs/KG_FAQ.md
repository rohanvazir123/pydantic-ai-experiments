# Knowledge Graph Pipeline — FAQ & Findings

Living document. Updated as we run experiments and make architectural changes.
See `kg/docs/KG_PIPELINE.md` for the full design reference.

---

## Architecture decisions

### Why Apache AGE over the original kg_entities / kg_relationships tables?

**Short answer: multi-hop traversal, LLM-driven ad-hoc Cypher queries, and native graph semantics.**

#### What the table approach gave us

`PgGraphStore` stored the graph in two plain PostgreSQL tables: `kg_entities` and `kg_relationships`.  Writes were simple `INSERT … ON CONFLICT DO UPDATE`.  Reads used SQL JOINs and `to_tsvector` full-text search.  No extra infrastructure needed.

#### What broke down

**Multi-hop traversal requires recursive CTEs.**  A single-hop "give me entities connected to X" is one JOIN.  Two hops is three JOINs.  Variable-depth paths (`*1..3`) need a `WITH RECURSIVE` block that is hard to write, hard to read, and not what the LLM produces via nl2cypher.

**`run_cypher_query` cannot be implemented.**  The LLM-to-graph pipeline (`nl2cypher` → `run_graph_query` tool → AGE) lets the agent write arbitrary read-only Cypher at query time.  `PgGraphStore.run_cypher_query` returns a hardcoded error — there is no equivalent without a graph engine.

**Type filtering is a column value, not schema.**  In PgGraphStore, `entity_type = 'Party'` is a `WHERE` filter on a shared 
table.  In AGE, `:Party` is an indexed vertex label — the planner prunes non-Party vertex tables before touching properties.

#### Concrete wins in AgeGraphStore

| Capability | PgGraphStore | AgeGraphStore |
|---|---|---|
| 1-hop neighbour query | 1 JOIN | `MATCH (s)-[r]-(t)` |
| 2-hop path | 3 JOINs | `MATCH (a)-[*1..2]->(b)` |
| Variable-depth path | Recursive CTE | `MATCH (a)-[*1..N]->(b)` |
| LLM ad-hoc queries | Not possible (`run_cypher_query` returns error) | `run_cypher_query` executes Cypher |
| Type-filtered scan | `WHERE entity_type = ?` on full table | `MATCH (e:Party)` — label-pruned |
| Graph pattern matching | Multi-table JOIN chain | `MATCH (p:Party)-[:PARTY_TO]->(c:Contract)-[:GOVERNED_BY_LAW]->(j:Jurisdiction)` |

#### What we gave up

- **Full-text search** — `to_tsvector` / GIN index is unavailable in AGE.  We fell back to `toLower() CONTAINS` substring scan (`search_entities`, `search_as_context`).
- **Simpler connection management** — every connection needs `LOAD 'age'` and `SET search_path = ag_catalog` (handled by the `_conn()` context manager and pool `init=_age_init`).
- **A separate Docker container** — AGE requires a patched PostgreSQL 15/16 image (`apache/age:latest`, port 5433), separate from the main pgvector DB.

#### Why PgGraphStore still exists

`PgGraphStore` is kept as the `doc_store` in `CuadKgBuilder` because it needs to JOIN against the `documents` table which lives in the main PostgreSQL DB (port 5434).  AGE runs on a separate Docker instance (port 5433) and cannot cross-database JOIN.  All graph traversal and LLM query work goes through `AgeGraphStore`.

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

The four pipelines that all write to the **same** `legal_graph`:

| Pipeline | File | What it writes |
|---|---|---|
| CUAD annotation ingest | `kg/cuad_kg_ingest.py` | Entities + relationships from `cuad_eval.json` CUAD annotations |
| LLM extraction (Bronze→Silver→Gold) | `kg/extraction_pipeline.py` (`GoldProjector`) | Entities + relationships from LLM extraction of contract text |
| Legal entity extractor | `kg/legal_extractor.py` | Named entities from legal text via LLM |
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

### Is KG chunking semantic/clause-aware or fixed-size?

**Short answer: fixed-size, no overlap. Completely separate from the RAG chunker.**

There are two chunking pipelines in this codebase. They are independent and should not be confused:

#### RAG chunker — structure-aware (used for the vector store)

`rag/ingestion/chunkers/docling.py` wraps Docling's `HybridChunker`:

- **Structure-aware**: respects heading hierarchy, paragraph boundaries, tables, code blocks — it follows the document's logical sections, not character counts.
- **Token-bounded**: controlled by `max_tokens` (default 512) using a HuggingFace tokenizer (`sentence-transformers/all-MiniLM-L6-v2`). A chunk will never exceed the embedding model's context window.
- **Contextualized**: each chunk is prefixed with its heading ancestry so a retrieved chunk is self-contained ("Section 3 > Non-Compete Restrictions > …").
- **Fallback**: when no `DoclingDocument` is available (plain text input), it uses a sliding window with sentence-boundary snapping and configurable overlap.

These chunks land in the `chunks` PostgreSQL table and are indexed for vector + BM25 search.

#### KG extraction chunker — dumb fixed-size (used for entity/relationship extraction)

`kg/extraction_pipeline.py`, method `_chunk()`, line ~747:

```python
def _chunk(self, text: str) -> list[str]:
    chunks = []
    for i in range(0, len(text), self._chunk_size):
        chunks.append(text[i : i + self._chunk_size])
    return chunks
```

- **Fixed 1500-character slices.** Pure Python string slicing — `text[0:1500]`, `text[1500:3000]`, etc.
- **No overlap.** A clause that straddles a boundary is split between two chunks.
- **No sentence/paragraph awareness.** The cut can land mid-sentence.
- Default `chunk_size=1500` is set in `ExtractionPipeline.__init__`.

Each slice is sent to a separate LLM extraction call (entity pass → relationship pass → hierarchy pass → validation pass). The LLM sees only the text within one slice; it cannot see entities from adjacent slices.

#### Why two different chunkers

| | RAG chunker | KG extraction chunker |
|---|---|---|
| Purpose | Embedding for semantic similarity search | LLM context window for entity extraction |
| Boundary logic | Document structure (headings, paragraphs) | Hard character count |
| Overlap | Configurable (default 100 chars) | None |
| Output destination | `chunks` table → pgvector index | Bronze artifacts → Silver dedup → AGE graph |
| Tokenizer | HuggingFace (token-accurate) | None (character estimate) |

The KG chunker is simpler because LLM extraction is forgiving about cut boundaries — the entity extraction prompt tells the model to extract "only entities from the provided text", and missed entities at boundaries are partially recovered by Silver deduplication (which merges matching `(label, normalized_name)` pairs across all chunks of a contract). A relationship whose source and target entity fall in different slices will be missed entirely, however.

#### Known consequence: split-boundary relationship loss

If an indemnification clause spans characters 1490–1550 and the slice boundary falls at 1500:

- Chunk N sees "…Party A shall indemnify" — no target entity yet
- Chunk N+1 sees "Party B from all claims…" — no source entity in this chunk
- The relationship `Party A –INDEMNIFIES→ Party B` is never extracted

The `source_chunk_indices` column added to `kg_canonical_entities` (Silver) records which chunks each entity appeared in, enabling future cross-chunk relationship re-extraction. This is not yet wired up.

**Practical impact on the CUAD corpus:** Most party/jurisdiction/date entities appear multiple times across a contract and are extracted in at least one chunk. Clause-level relationships (e.g. `Contract –HAS_CLAUSE→ TerminationClause`) are less affected because the clause label and its governing entity typically co-occur well within 1500 chars.

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

```powershell
# Step 1: Fast graph from CUAD annotations (minutes)
python -m kg.cuad_kg_ingest

# Step 2: Optional — LLM enrichment for a handful of contracts
python -m kg.extraction_pipeline --limit 20

# Step 3: Risk graph over whatever Silver data exists
# (automatically called by ExtractionPipeline — or call standalone)
```

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

### Why keep PgGraphStore if AGE is the active backend?

`PgGraphStore` is still used in `CuadKgBuilder` as a `doc_store` — it looks up
the `documents` table which lives in the main PostgreSQL DB (port 5434), not in
the AGE Docker instance (port 5433).  AGE can't cross-database JOIN.  Everything
else goes through `AgeGraphStore`.

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

- [ ] **Protocol / ABC for graph stores** — `AgeGraphStore | PgGraphStore` union
  type appears in many signatures; replace with a `KGStore` protocol
- [ ] **Chunk overlap** — entity references at chunk boundaries get missed; add
  ~200 char overlap between consecutive chunks
- [ ] **Graph-aware RAG retrieval** — after vector retrieval, expand context
  using AGE: `MATCH (matched_entity)-[*1..2]->(neighbour)` to pull in related
  clauses that vector search missed

---

## Run log

| Date | Contract | Chars | Chunks | LLM calls | Entities (raw) | Entities (canonical) | Rels (raw) | Rels (canonical) | Parse failures | Duration | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-02 | Recording1 | 576 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | ~2 min | Not a legal contract |
| 2026-05-02 | LIGHTBRIDGE CORP — Strategic Alliance Agreement | 19,429 | 13 | 65 | 48 | 42 | 41 | 25 | 14 (21.5%) | 23 min 40 sec | First real run; `_parse_json` fix not yet applied; Silver dedup removed 6 entities (12.5%), 16 rels (39%) |
