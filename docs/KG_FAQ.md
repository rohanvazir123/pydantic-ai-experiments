# Knowledge Graph Pipeline — FAQ & Findings

Living document. Updated as we run experiments and make architectural changes.
See `docs/KG_PIPELINE.md` for the full design reference.

---

## Architecture decisions

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
- [ ] **Risk Dependency Graph** — rule-based inference on entity graph:
  "no IndemnityClause → ComplianceRisk node"; separate from LLM extraction
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
