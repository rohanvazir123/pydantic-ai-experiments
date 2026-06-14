# Production Risk Analysis — Failure Modes & Limits

This document maps where the system can fail in production across all four components:
**Hybrid RAG · Knowledge Graph (KG) · NL→SQL · NL→Cypher**

---

> **Hard constraint throughout this document: all LLM inference runs locally via Ollama.**
> No SDK API calls to OpenAI, Anthropic, or any cloud LLM provider. This constraint is not optional and affects almost every failure mode below. Local models are smaller, slower, less instruction-following, and more prone to hallucination than frontier cloud models. Every mitigation must account for this.

---

## Table of Contents

1. [Local LLM Failures — Shared Across All Components](#1-local-llm-failures--shared-across-all-components)
2. [Document Ingestion & Chunking](#2-document-ingestion--chunking)
3. [Embedding Quality](#3-embedding-quality)
4. [Hybrid RAG: Retrieval Failures](#4-hybrid-rag-retrieval-failures)
5. [Hybrid RAG: Answer Synthesis Failures](#5-hybrid-rag-answer-synthesis-failures)
6. [Knowledge Graph: Construction Failures](#6-knowledge-graph-construction-failures)
7. [Knowledge Graph: NL→Cypher Failures](#7-knowledge-graph-nlcypher-failures)
8. [NL→SQL Failures](#8-nlsql-failures)
9. [Inter-Component Failures — Where the Systems Collide](#9-inter-component-failures--where-the-systems-collide)
10. [Infrastructure & Connection Failures](#10-infrastructure--connection-failures)
11. [Token & Context Window Limits](#11-token--context-window-limits)
12. [Data Consistency & Staleness](#12-data-consistency--staleness)
13. [Silent Failures — Wrong Answers That Look Right](#13-silent-failures--wrong-answers-that-look-right)
14. [Risk Matrix](#14-risk-matrix)

---

## 1. Local LLM Failures — Shared Across All Components

Every component that calls an LLM (RAG answer synthesis, KG entity extraction, NL→SQL generation, NL→Cypher generation) is exposed to these failures. They compound every other failure below.

---

### 1.1 Instruction following degrades sharply with model size

**What goes wrong:** Smaller local models (7B–14B parameters) frequently ignore explicit output format instructions. The system prompt says "Return ONLY plain SQL. No Markdown fences, no explanation, no comments." A 7B model will often return:

````
Sure! Here is the SQL query you requested:

```sql
SELECT COUNT(*) FROM orders WHERE ...
```

This query counts the number of orders where...
````

`strip_sql_fences()` handles the Markdown code block, but the surrounding prose is not stripped. The SQL parser then fails because the string starts with "Sure!".

Similarly for NL→Cypher: "respond with exactly two blocks: `<cypher>` … `</cypher>` `<columns>` … `</columns>`" is ignored by 7B models roughly 30–40% of the time.

**How likely in production:** Certain with 7B models. Rare with 13B+ models. Uncommon but non-zero with Qwen 2.5 Coder 7B on SQL tasks.

**Current mitigation:** `strip_sql_fences()` handles the common Markdown case. Retry loop re-prompts on parse failure.

**What's missing:** Structured output enforcement via Ollama's JSON mode or grammar-constrained generation. A pre-execution parse check that catches "this isn't SQL" before trying DuckDB.

---

### 1.2 Quantisation degrades output quality

**What goes wrong:** Local models are typically run in quantised form (Q4_K_M, Q5_K_M, Q8_0) to fit in VRAM. Quantisation reduces model weights to lower precision. The effect on SQL and Cypher generation:

- Q4 (4-bit) models make more reasoning errors on multi-step queries (complex JOINs, date arithmetic, multi-hop Cypher paths). They are more prone to hallucinating table names and column names.
- Q8 (8-bit) is close to full precision but requires 2× the VRAM.
- The gap between Q4 and Q8 quality is substantial for code generation tasks — worse than the gap between a 7B-Q8 and a 13B-Q4.

**What the user sees:** More frequent SQL errors, more retries, more hallucinated column names, more Cypher format failures — all relative to what the same model at higher precision would produce.

**How likely in production:** Certain — almost all Ollama deployments use Q4 or Q5 models to fit in 8–16GB VRAM.

**Current mitigation:** None. The system doesn't know or track which quantisation level is in use.

**What's missing:** Document which quantisation tier is acceptable per task. For NL→SQL: Q5_K_M minimum for Qwen 2.5 Coder. For KG entity extraction (more reasoning-heavy): Q8 or a larger Q4 model.

---

### 1.3 Ollama inference is slow — timeout cascades

**What goes wrong:** Local LLMs are significantly slower than cloud APIs. A Qwen 2.5 Coder 7B Q4 on a consumer GPU generates ~25–50 tokens/second. A 512-token SQL generation + explanation response takes 10–20 seconds. The NL→SQL system has a `query_timeout=30s` for DuckDB execution, but no timeout on the Ollama inference call itself.

If Ollama is under load (another request is running), inference may queue for 30–60 seconds before even starting. The total wall-clock time for a user query can exceed 90 seconds.

With the retry loop (up to 3 attempts), a worst-case path is: 20s (attempt 1) + 20s (attempt 2) + 20s (attempt 3) = 60s of LLM time alone, before DuckDB execution.

**What the user sees:** Long waits with no progress indicator. If there's a global timeout (e.g. API gateway, browser), the request may be killed before an answer is returned.

**Current mitigation:** None. No timeout on Ollama calls. No streaming response to the user.

**What's missing:** An `asyncio.wait_for()` wrapper around every `agent.run()` call with a configurable LLM timeout (separate from the DuckDB timeout). Streaming output to the user so they see partial results. A queue depth check before accepting new requests if Ollama is already saturated.

---

### 1.4 Ollama VRAM exhaustion — model eviction mid-request

**What goes wrong:** Ollama manages model loading into VRAM automatically. If VRAM is shared between multiple models (e.g. embedding model + LLM), Ollama may evict the LLM from VRAM to load the embedding model mid-session. The next LLM request triggers a cold load (10–30s for a 7B model, 60–120s for a 13B model) before inference begins.

**What the user sees:** Suddenly very slow responses after a previously fast interaction. No error — just latency.

**How likely in production:** Any deployment running both Ollama LLM and Ollama embeddings on the same GPU. This system does exactly that (`nomic-embed-text` + `llama3.1:8b`).

**Current mitigation:** None.

**What's missing:** Pin the LLM model in VRAM with `OLLAMA_KEEP_ALIVE=-1` (never unload). Use a separate Ollama instance or separate GPU for embeddings vs generation.

---

### 1.5 Local models have worse hallucination floors

**What goes wrong:** Frontier models (GPT-4o, Claude 3.5 Sonnet) follow "say I don't know" instructions reliably. A local 7B or 13B model does not. When the retrieved context is insufficient, a local model will construct a plausible answer from its training data rather than admit ignorance. This is not a bug in the model — it's the expected behaviour of smaller instruction-tuned models.

**Concrete example:** User asks "What is the governing law for the Acme contract?" The relevant chunk wasn't retrieved. A local 7B model returns: "Based on the contract, the governing law is likely Delaware, as this is common for technology agreements." — entirely fabricated, stated as fact.

**How likely in production:** Certain, on some fraction of queries, with any local model below ~70B parameters.

**Current mitigation:** System prompt instructs the model to say "I don't have that information." Actual compliance: low for 7B models, moderate for 13B models.

**What's missing:** Citation enforcement (structured output requiring the model to quote the chunk supporting each claim). Refusal detection: if the model's answer does not contain any text from the retrieved chunks, flag it as potentially confabulated.

---

### 1.6 Model output is non-deterministic across runs

**What goes wrong:** Local models use sampling (temperature > 0 by default in many Ollama configurations). The same question asked twice may produce different SQL, different Cypher, or different answers. This makes debugging hard and caching unreliable for borderline inputs.

**What the user sees:** Different answers to the same question on repeat queries. The NL cache (keyed on normalized NL string) hides this once a result is cached — but the first cached result may be the bad one.

**Current mitigation:** NL cache means the same question gets the same answer once cached. But the cached answer may have been generated from a bad sample.

**What's missing:** Temperature=0 for SQL and Cypher generation tasks (determinism over diversity). Temperature > 0 is only useful for N-candidate generation (Gap 4 in `SYSTEM_DESIGN.md`).

---

## 2. Document Ingestion & Chunking

### 2.1 Bad chunk boundaries from Docling

**What goes wrong:** Docling's `HybridChunker` splits on detected document structure. When structure detection fails — scanned PDFs, multi-column layouts, heavily formatted DOCX — chunks break mid-sentence or mid-clause. A legal obligation "…shall indemnify and hold harmless [split] …against any claims arising from…" becomes two meaningless half-clauses.

**What the user sees:** Partial quotes in RAG answers. KG entity extraction on a half-clause may extract wrong or incomplete relationships.

**How likely in production:** High — 10–20% of real-world document corpora have structurally ambiguous files.

**Current mitigation:** `chunk_overlap` partially covers sentence-level splits. `_simple_fallback_chunk()` fires on total Docling failure but produces semantically meaningless sliding-window chunks.

**What's missing:** Chunk quality scoring. Post-chunking check: minimum token count, sentence completeness (ends with `.`/`?`/`)`). A `chunker: str` field in `ChunkData.metadata` recording `"docling"` or `"fallback"` so retrieval metrics can be broken down by chunker type.

---

### 2.2 OCR errors in scanned PDFs

**What goes wrong:** Docling OCR introduces errors on legal terminology: `"lndemnification"` (capital I misread as lowercase l), `"$1,00O"` (zero misread as O), garbled entity names. These propagate into both the vector store and the KG extraction pipeline.

**What the user sees:** Retrieval misses — BM25 won't match `"lndemnification"` against a query for `"indemnification"`. KG extraction may create a spurious entity `"lndemnification"` instead of a `Clause` node.

**Current mitigation:** None. Docling OCR output is trusted as-is.

**What's missing:** OCR confidence scores from `DoclingDocument` (available but not used). A spell-check pass against a legal domain dictionary. Chunks below an OCR confidence threshold flagged in metadata.

---

### 2.3 Table extraction failures

**What goes wrong:** Merged cells, multi-row headers, borderless tables — Docling produces garbled output: columns merged, rows repeated, or the entire table collapsed to a single line. Payment schedules, milestone tables, SLA tables are high-risk.

**What the user sees:** "What is the payment schedule?" retrieves `"Month 1 2 3 4 5 6 Amount $10,000 $10,000…"` — the table flattened. The LLM synthesises around it; the answer may be wrong.

**Current mitigation:** Tables are chunked as text. No table-specific handling.

**What's missing:** Table-aware chunking that serialises `TableItem` objects as Markdown (`| col | col |`) before embedding. Markdown table text embeds and retrieves more reliably than collapsed text.

---

### 2.4 Audio transcription (Whisper) — domain errors

**What goes wrong:** Whisper ASR errors on domain-specific vocabulary: "NDA" → "en dee ay", "indemnification" → "in den if occasion", company names phonetically approximated. These errors are not systematic — they vary per recording quality.

**What the user sees:** Meeting notes or depositions that are ingested but retrieved incorrectly because key terms are wrong.

**Current mitigation:** Total failure → `[Error: Could not transcribe audio file …]` placeholder. Partial transcription with domain errors: no mitigation.

**What's missing:** Whisper post-processing with a legal/domain vocabulary hint. Per-segment confidence scoring. Speaker diarisation metadata is discarded entirely.

---

### 2.5 Silent fallback to simple chunking

**What goes wrong:** When `DoclingDocument` is unavailable (plain text files, conversion failures), `_simple_fallback_chunk()` runs — a sliding window with no sentence or paragraph awareness. Chunks start mid-sentence. No indication in metadata.

**What the user sees:** Nothing immediately. Retrieval quality for that document is silently degraded.

**What's missing:** `chunker: str` metadata field. Alerting when the fallback rate exceeds a threshold (e.g. >5% of documents use fallback → investigate Docling configuration).

---

## 3. Embedding Quality

### 3.1 Domain mismatch — `nomic-embed-text` on legal/financial text

**What goes wrong:** `nomic-embed-text` (768-dim) is trained on general web text. Legal contracts use stylised boilerplate ("the Company", "the Licensor", defined terms with specific intra-document meanings) that the model has seen only rarely. Synonymous legal terms may be far apart in vector space.

**Concrete failure:** "Governing Law" and "Choice of Law" are synonyms. An embedding model trained on web text may place them far apart if the legal usage is rare in training data. A query for "governing law" may miss a chunk containing "choice of law clause".

**Current mitigation:** Hybrid search (BM25 fallback) partially compensates on exact keyword matches.

**What's missing:** Domain-adapted embeddings. Fine-tuning `nomic-embed-text` on a legal corpus, or switching to `law-ai/legal-bert-base-uncased` for the vector path. Evaluation of retrieval quality broken down by clause type to quantify the gap.

---

### 3.2 Embedding model swap invalidates the entire vector store

**What goes wrong:** If `EMBEDDING_MODEL` is changed, all existing embeddings are in the old vector space. New documents embed in the new space. Cosine similarity between old and new embeddings is meaningless — they're in different spaces. The system doesn't know this and returns garbage retrieval results with no error.

**How likely in production:** Certain to happen eventually. Any model upgrade, provider change, or dimension change triggers this.

**Current mitigation:** None. No per-chunk model tracking.

**What's missing:** `embedding_model: str` column in `chunks`. A startup check that detects model mismatch. A re-embedding job for stale chunks. Until implemented, treat embedding model as immutable infrastructure — change it requires a full re-ingest.

---

### 3.3 Degenerate chunks embed poorly — headers, page numbers, dates

**What goes wrong:** Chunks with very few tokens (section headers, "Page 12 of 47", standalone dates) produce embeddings that cluster near meaningless centroids. They are retrieved by queries that have nothing to do with their content, displacing useful chunks from the top-K.

**Current mitigation:** `min_chunk_tokens` / `max_tokens` parameters, but no post-ingestion filtering.

**What's missing:** A minimum token count filter before embedding (`< 20 tokens → skip or merge`). A `chunk_quality_score` in metadata. Degenerate chunks excluded from retrieval at query time via a metadata filter.

---

## 4. Hybrid RAG: Retrieval Failures

### 4.1 Top-K misses the relevant chunk — the fundamental problem

**What goes wrong:** The relevant chunk exists but falls outside the top-K results. Causes: (a) query-chunk vocabulary mismatch (embedding gap), (b) chunk is a fragment of a longer relevant section, (c) IVFFlat index returns approximate nearest neighbours — not guaranteed exact.

**Current measured hit rate: 26.5% on 34 legal evaluation questions.** For production corpora with higher diversity, expect lower.

**What the user sees:** "I don't have information about that" or a confabulated answer. There is no way to distinguish "the document doesn't contain this" from "retrieval failed."

**Current mitigation:** Hybrid search (semantic + BM25 via RRF). Configurable `match_count`.

**What's missing:** Reranker (disabled). Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would significantly improve precision. HyDE (also disabled) would help for queries phrased very differently from document language. Both must run locally.

---

### 4.2 RRF fusion promotes irrelevant chunks

**What goes wrong:** RRF gives equal weight to vector rank and BM25 rank. A chunk that ranks highly on BM25 due to keyword overlap but is semantically irrelevant gets promoted. Example: a chunk mentioning "termination" in a schedule header promotes over the actual termination clause.

**Current mitigation:** None. Weights are fixed 1:1.

**What's missing:** Tunable RRF weights. A local reranker as a final stage to re-score the fused list.

---

### 4.3 Score threshold silently drops all results

**What goes wrong:** The score filter in `Retriever` drops results below a configurable threshold. If the threshold is too high, all results are dropped and the retriever returns empty — the LLM then answers from training data alone (pure confabulation).

**Current mitigation:** Configurable `score_threshold`, but no adaptive fallback.

**What's missing:** If fewer than K results pass the threshold, relax the threshold incrementally before returning empty. Log when empty retrieval occurs — this is the highest-risk state in the whole pipeline.

---

### 4.4 No reranker, no HyDE — both disabled

The two highest-impact retrieval improvements are disabled:
- **Reranker** — would re-score top-K results using a cross-encoder. Must run locally (Ollama or a small ONNX model).
- **HyDE** — generates a hypothetical document that would answer the query, then embeds that. Helps bridge the vocabulary gap between query and document. Adds one LLM call per query; with local inference this means +10–20s latency.

Both are disabled because of latency cost on local LLMs. The trade-off is measurable: hit rate 26.5% without them.

---

## 5. Hybrid RAG: Answer Synthesis Failures

### 5.1 LLM confabulation when retrieved context is incomplete

**What goes wrong:** When the relevant information is absent (retrieval miss), a local 7B–13B model will construct a plausible answer from training data rather than say "I don't know." This is the highest-risk failure in the system.

**Why it's worse with local models:** Frontier models (GPT-4o, Claude 3.5) follow "say I don't know" instructions reliably. Local models below ~70B do not — the instruction-following capability simply isn't there. A local Llama 3.1 8B will hallucinate confidently.

**What the user sees:** A confident, well-written answer that is factually wrong. For legal or financial questions this can be materially harmful.

**Current mitigation:** System prompt instructs the model to say "I don't have that information." Actual compliance: low for 7B models.

**What's missing:** Citation enforcement — structured output requiring the model to cite the specific chunk that supports each claim. Any claim not traceable to a retrieved chunk is rejected. Refusal detection: if the answer does not overlap with any retrieved chunk text, flag it.

---

### 5.2 Context window overflow — retrieved chunks + history + question

**What goes wrong:** 5 chunks × 512 tokens + system prompt (~300 tokens) + conversation history + question = potentially 3,500–4,000 tokens of input. For a local 7B model with an effective context window of 4K–8K tokens, this leaves little room for careful reasoning. Chunks at the end of the context are often ignored by smaller models (the "lost in the middle" problem — middle-of-context chunks are attended to least).

**Current mitigation:** None. `match_count` and `max_tokens` are configurable but not coordinated with the model's actual context window.

**What's missing:** A context budget calculation: `available_tokens = model_context_window - system_prompt_tokens - history_tokens - question_tokens`. Fetch only as many chunks as fit within the budget. Prioritise chunks by relevance score when truncating.

---

### 5.3 The "lost in the middle" problem with local models

**What goes wrong:** Research on attention in transformer models shows that relevant information in the middle of a long context is attended to less than information at the beginning or end. With 5 retrieved chunks, the most relevant chunk placed 3rd in context may contribute less to the answer than the 1st or 5th chunk, even if it scored highest.

**What the user sees:** Answers that rely on the first or last retrieved chunk rather than the highest-relevance one.

**Current mitigation:** None. Chunks are passed in rank order.

**What's missing:** Re-ordering retrieved chunks to place the highest-scoring chunk first and last (bookending), with lower-scoring chunks in the middle.

---

## 6. Knowledge Graph: Construction Failures

### 6.1 Local LLM entity extraction is inconsistent and slow

**What goes wrong:** The Bronze/Silver/Gold extraction pipeline uses a local LLM to extract entities and relationships from contract text. Local models (7B–14B) are significantly worse at structured information extraction than frontier models:

- They produce inconsistent entity types for the same clause across documents
- They frequently output malformed JSON that the pipeline must repair
- A 14-contract corpus takes hours to process — a 1,000-contract corpus would take days

**Concrete failure:** "Acme Corp shall indemnify Beta Inc against any IP claims" produces:
- Attempt 1: `(Acme Corp)-[:INDEMNIFIES]->(Beta Inc)` — correct
- Attempt 2 (different document, same clause structure): `(Acme Corp)-[:HAS_OBLIGATION]->(Beta Inc)` — wrong relationship type
- Attempt 3: `{"entity": "Acme Corp", "type": "Party", "relationship": "indemnification"}` — flat JSON, not a graph triple

**Current mitigation:** Silver→Gold confidence filtering. `ENTITY_TYPE_MAP` for canonical type normalisation. Bronze JSON audit files.

**What's missing:** Per-extraction structured output enforcement (Ollama JSON mode). A consistency metric: for the same clause type across documents, how often does the LLM produce the same relationship type? Below a threshold, flag for human review.

---

### 6.2 Entity deduplication fails across documents

**What goes wrong:** "Acme Corporation", "Acme Corp.", "ACME CORP", "Acme Corp (the 'Company')" become separate nodes. Graph traversal for "Acme Corp" misses the others. In a corpus of 100 contracts, the same Fortune 500 company may appear under 5–10 surface forms.

**Current mitigation:** Canonical entity names via Silver deduplication. Some string normalisation.

**What's missing:** Fuzzy string matching or embedding-based entity resolution across documents before writing to Gold. This is a hard problem with local models — a dedicated entity resolution model or rule-based approach (Levenshtein distance + known alias dictionaries) is more reliable than prompting a small LLM.

---

### 6.3 Bronze→Silver→Gold pipeline fails silently mid-contract

**What goes wrong:** An Ollama timeout, OOM, or connection drop during Silver processing for contract N leaves it in Bronze with no Silver/Gold representation. The pipeline continues to N+1. No user-visible error.

**Current mitigation:** Bronze JSON audit files show what reached Bronze. No cross-stage tracking.

**What's missing:** A `pipeline_status` table: `(contract_id, stage, status, error, ts)`. Re-run capability for failed contracts. Alerting when the pipeline completion rate drops below 100%.

---

### 6.4 KG construction with local LLMs is fundamentally slow

**What goes wrong:** Entity and relationship extraction requires one LLM call per contract (or per chunk, for chunk-level extraction). At 10–20 seconds per call on a local 7B model, processing 510 CUAD contracts requires 85–170 minutes of pure LLM time, serial. Any error requires a re-run of the affected contract.

**What the user sees:** Long ingestion times. A document that "should be in the KG" isn't there yet.

**Current mitigation:** The rule-based fast ingest (`cuad_kg_ingest.py`) bypasses the LLM entirely for CUAD annotations. But for new contracts without annotation, LLM extraction is the only path.

**What's missing:** Parallelised extraction with `asyncio.gather` across contracts (not chunks within a contract — that would overload Ollama). A batch processing mode with progress tracking. The option to skip LLM extraction for contracts that already have structured annotations (e.g. CUAD).

---

## 7. Knowledge Graph: NL→Cypher Failures

### 7.1 IntentParser catch-all swallows complex queries — silently

**What goes wrong:** The `list_contracts` intent is `re.compile(r".*")` — matches everything. Any question not covered by the 23 specific patterns gets `MATCH (c:Contract) RETURN c.name LIMIT 50`. This looks like a real answer.

**Queries that silently fail:**
```
"Which contracts governed by California law also have an indemnification clause?"
→ Needs: find_jurisdictions AND find_indemnification
→ Gets: MATCH (c:Contract) RETURN c.name LIMIT 50

"Show me all contracts that expire in 2025 where the liability cap exceeds $1M"
→ Date filter + financial filter: not modelled
→ Gets: MATCH (c:Contract) RETURN c.name LIMIT 50

"What is the governing law for contracts signed after 2020?"
→ find_jurisdictions fires correctly, BUT "after 2020" is silently ignored
→ Gets: ALL jurisdictions, not filtered by date
```

**Current mitigation:** None.

**What's missing:** Log every `list_contracts` hit. Expose intent to the user ("I couldn't find a specific pattern for this — showing all contracts"). Implement the LLM fallback for the catch-all case.

---

### 7.2 LLM-based Cypher generation (planned) will fail on local models

**What goes wrong (when Gap 7 is implemented):** The AGE-specific Cypher format is almost entirely absent from local model training data. The specific failures on local 7B–14B models are systematic (see `nl2sql/docs/FAQ.md`):

| Failure | Frequency on 7B models |
|---------|----------------------|
| Missing `LIMIT` clause | ~60% of queries |
| `LIKE` instead of `CONTAINS` | ~50% of queries |
| `<columns>` count mismatch vs RETURN | ~30–40% of queries |
| Hallucinated relationship type | ~20% of free-form queries |
| Attempt to use `$1` params (unsupported in AGE) | ~10% of queries |
| Write keyword in query (CREATE/MERGE/SET) | Rare but catastrophic if unguarded |

With 3-attempt retry and validation, the effective success rate on free-form queries is probably 40–60% for a 7B model. For a 13B+ model (Qwen 2.5 Coder 14B, Llama 3.1 70B), expect 70–80%.

**Current mitigation:** LLM is not called for Cypher today (rule-based pipeline). The risk is future.

**What's missing:** Before implementing Gap 7, build the complete validation pipeline first: write-keyword guard + RETURN↔columns parity check + label/relationship allowlist. Only then add the LLM fallback. Start with the largest practical local model (Qwen 2.5 Coder 14B or better).

---

### 7.3 `_extract_name()` extracts wrong entity

**What goes wrong:** Multi-word title-case matching can grab question phrases as entity names. Fixed for interrogatives (`Which`, `What`, etc.) but not for common nouns in question templates:

```
"Which contracts have termination clauses?"
→ _TITLED matches "Which contracts" → strips "Which" → name = "contracts"
→ Cypher: WHERE c.name CONTAINS 'contracts'  (wrong)
→ Returns 0 results silently
```

**Current mitigation:** `_INTERROGATIVES` frozenset. Partially effective.

**What's missing:** More aggressive filtering of common English nouns appearing in question templates. Unit tests covering phrasing patterns that produce title-case matches without a real entity.

---

### 7.4 No date or numeric filtering in any Cypher builder

**What goes wrong:** All 23 Cypher builders filter only on entity name (string `CONTAINS` match). Date ranges, numeric thresholds, and status filters are not modelled. Any question involving time ("contracts expiring in Q4") or quantity ("liability cap > $1M") is either mis-routed to the catch-all or routes to the right intent but ignores the filter entirely.

**What the user sees:** Too many results (the filter was silently ignored) or `list_contracts` output.

**Current mitigation:** None.

**What's missing:** Extend `IntentMatch.params` to include `date_range: tuple | None` and `numeric_filter: dict | None`. Extend the relevant builders to interpolate these into the Cypher `WHERE` clause. This is buildable without an LLM.

---

## 8. NL→SQL Failures

### 8.1 Schema text overflows the local model's context window

**What goes wrong:** `UnifiedDataSource.generate_schema()` serialises every table and column from every source into a single string. For a schema with 50 tables × 10 columns, this is 4,000–6,000 tokens. A local 7B model with an effective context window of 4K–8K tokens has almost no room left for reasoning after the schema, history, and question.

**Local model specific:** Cloud models (GPT-4o: 128K context, Claude: 200K) can absorb large schemas trivially. Local 7B models cannot. The schema text is the primary prompt engineering challenge for NL→SQL on local hardware.

**What the user sees:** The model truncates or ignores parts of the schema, hallucinating column names that don't exist.

**Current mitigation:** None. Full schema is always sent.

**What's missing:** Semantic schema retrieval: embed the NL query, retrieve only the top-K relevant table/column chunks, inject only those. The full schema design is in `SYSTEM_DESIGN.md §5`. Critical for any real-world schema.

---

### 8.2 Local models generate wrong DuckDB dialect

**What goes wrong:** Local models are trained predominantly on PostgreSQL/MySQL/standard SQL. DuckDB dialect differences are underrepresented in training data:
- `QUALIFY` window filter (not in PostgreSQL)
- `COLUMNS(*)` expression
- `LIST_AGG` / `ARRAY_AGG` syntax differences
- DuckDB's table naming convention (`alias.main.table`) is entirely custom — no model has seen it in training
- `STRUCT`, `MAP`, `LIST` types

**What the user sees:** SQL errors on first attempt. DuckDB error messages are descriptive so the self-correcting retry often fixes dialect errors. But if the model consistently uses wrong syntax (e.g. always uses `ILIKE` which DuckDB doesn't support), all 3 retries fail.

**What's missing:** A few-shot system prompt section showing DuckDB-specific syntax examples. SQLGlot pre-execution validation that converts the generated SQL to DuckDB dialect before execution.

---

### 8.3 Self-correction retry eats LLM quota

**What goes wrong:** With local models, each LLM call takes 10–20 seconds. A 3-attempt retry means 30–60 seconds of LLM time per failed query. Under concurrent load (multiple users), the Ollama queue grows, increasing latency further. The retry loop designed for fast cloud APIs becomes a throughput bottleneck on local hardware.

**Current mitigation:** Max 3 retries.

**What's missing:** Reduce max retries to 2 for local models (saves 10–20s per failure). Cache known-bad SQL patterns: if the model generates the same wrong SQL on retry as on attempt 1, bail out immediately instead of waiting for DuckDB to fail again.

---

### 8.4 Correct SQL, semantically wrong answer — no validation layer

**What goes wrong:** SQL executes, returns rows, but answers the wrong question. Examples:
- "Revenue last quarter" → rolling 90 days, not the calendar quarter
- "Top customers" → by order count, not by revenue
- "Active contracts" → no filter (model doesn't know what "active" means in this schema)

**What the user sees:** A result that looks plausible, presented with confidence. Cached and returned for future similar questions.

**How likely in production:** Very high. The most common NL→SQL failure mode in practice.

**Current mitigation:** None.

**What's missing:** `<thinking>` reasoning extraction (Gap 2) exposes the model's interpretation. Query result sanity checks: if the question uses "most/least/top/bottom", verify `ORDER BY` is in the SQL. If the question has a date constraint, verify a date column is in the `WHERE` clause.

---

## 9. Inter-Component Failures — Where the Systems Collide

These failures arise specifically from the interaction between components. They are harder to detect because each component individually may look healthy.

---

### 9.1 RAG vector store and KG are out of sync

**What goes wrong:** A document is ingested into RAG (`documents` + `chunks` tables) but KG extraction fails or is never run. The RAG system retrieves text from 14 contracts; the KG answers for only 13. The discrepancy is invisible unless both paths are queried for the same question.

**Hybrid path specific:** In Path A (HYBRID), the KG result and the RAG text are fused by `_fuse()`. If the KG is missing a contract, the fused context has a gap. The LLM sees partial facts from the KG and fuller text from RAG — the inconsistency may produce a confident but partially wrong answer.

**Current mitigation:** None. The two pipelines are independent.

**What's missing:** A document registry tracking pipeline completion per document: `(doc_id, rag_ingested, kg_extracted, kg_stage)`. A consistency check query: `documents` rows with no `kg_entities` rows are KG-incomplete. Alerting on KG-incomplete documents older than X hours.

---

### 9.2 NL→SQL and KG query the same facts — can return conflicting answers

**What goes wrong:** A user asks "How many contracts are governed by Delaware law?" via NL→SQL, getting an answer from the relational `chunks` table via DuckDB. Another user (or the same user via a different path) asks the same question via the KG, getting an answer from AGE graph traversal. These can return different numbers because:
- RAG stores full document text; some documents may not have been KG-extracted
- KG entity extraction may have missed some governing law clauses
- NL→SQL may generate SQL that counts documents rather than clauses

**What the user sees:** Two different answers to the same question depending on which path was used. Both look authoritative.

**Current mitigation:** None. There is no unified answer layer.

**What's missing:** A query router that picks the authoritative source per question type. A reconciliation layer that compares KG and RAG answers and flags discrepancies.

---

### 9.3 Embedding model used at query time differs from ingestion time

**What goes wrong:** Ingestion runs with `nomic-embed-text` (768-dim). Someone runs a query after changing `EMBEDDING_MODEL` in `.env` without re-ingesting. The query embedding is in a different vector space from the stored chunk embeddings. Cosine similarities are garbage; retrieval returns random chunks.

**What the user sees:** Completely wrong retrieval results with no error. Every query returns a confident but wrong answer.

**Current mitigation:** None.

**What's missing:** `embedding_model` stored in a `system_config` table checked at startup. If the configured model doesn't match the ingested model, block queries and require re-ingest.

---

### 9.4 Local LLM bottleneck shared across all components

**What goes wrong:** All four components make LLM calls to the same Ollama instance: RAG answer synthesis, KG entity extraction during ingestion, NL→SQL generation, NL→Cypher generation (when implemented). Under concurrent use:
- KG ingestion (slow, high-token calls for entity extraction) starves NL→SQL (fast, low-token calls)
- Multiple simultaneous user queries queue behind each other
- Ollama serves requests serially by default (no parallel batching for most local models)

**What the user sees:** Queries that normally take 15 seconds take 60–90 seconds under load. No indication that Ollama is saturated.

**Current mitigation:** None. Single Ollama instance, no queue management.

**What's missing:** Separate Ollama instances for ingestion vs query-time tasks. A queue with priority: query-time requests preempt background ingestion. Or: run KG ingestion only during off-hours.

---

### 9.5 NL→Cypher catch-all + LLM confabulation = fabricated KG facts

**What goes wrong:** The KG `list_contracts` catch-all returns all contracts. The final LLM answer synthesis in the RAG agent receives this generic result and constructs an answer. For a question like "Which contracts have missing indemnity clauses?", the LLM receives a list of all contracts and may fabricate: "Contracts A, B, C are missing indemnity clauses" — selected arbitrarily from the full list.

**What the user sees:** A confident legal finding that is completely fabricated from the wrong KG query output.

**Current mitigation:** None. The LLM receives whatever the KG path returns.

**What's missing:** Confidence signal passed from the KG path to the answer synthesis: "intent=list_contracts (catch-all, low confidence) — do not fabricate findings from this result." The LLM should be instructed to say "I couldn't determine a specific graph query for this question" rather than hallucinating from the catch-all output.

---

## 10. Infrastructure & Connection Failures

### 10.1 AGE not loaded on new asyncpg connections

**What goes wrong:** AGE requires `LOAD 'age'` and `SET search_path` on every connection. The `init=` callback handles this for new pool connections, but if a connection is recycled without going through `init=` (e.g. pool configuration error, connection reuse bug), Cypher queries fail with `function cypher(...) does not exist`.

**What the user sees:** All KG queries fail intermittently. RAG queries continue. The failure appears and disappears as the pool cycles connections.

**Current mitigation:** `_init_age_conn` registered on `create_pool`. No health check.

**What's missing:** A `SELECT 1 FROM ag_catalog.ag_graph LIMIT 1` probe on every acquired connection before use. Automatic re-execution of `LOAD 'age'` on probe failure.

---

### 10.2 Ollama unavailable mid-request

**What goes wrong:** Ollama crashes, restarts, or becomes unavailable mid-inference. The `httpx` call raises `ConnectError` or `ReadTimeout`. This propagates as an unhandled exception through the Pydantic AI agent. The NL→SQL retry loop doesn't distinguish between SQL errors (retriable by re-prompting) and Ollama errors (not retriable by re-prompting).

**Current mitigation:** None. Ollama errors propagate to the caller.

**What's missing:** Exception type differentiation: `OllamaUnavailableError` vs `SQLExecutionError`. Exponential backoff retry specifically for transient Ollama errors. A readiness probe endpoint that checks Ollama before accepting requests.

---

### 10.3 asyncpg pool exhaustion under concurrent load

**What goes wrong:** The asyncpg pool has a fixed max size. Under concurrent load, all connections are held. New requests block waiting. If held connections are running slow vector similarity scans, blocking cascades.

**Current mitigation:** Configurable pool sizing.

**What's missing:** Pool acquisition timeout (reject instead of block indefinitely). Per-query timeout at the pool level. Monitoring of pool utilisation and wait time.

---

### 10.4 DuckDB memory exhaustion on large result sets

**What goes wrong:** DuckDB runs in `:memory:` mode. Queries with large intermediate result sets — full table scan of a large GCS Parquet before a WHERE filter, large JOIN before aggregation — can exhaust RAM. DuckDB spills to disk only if `temp_directory` is configured (default: none).

**Current mitigation:** Row cap (`LIMIT 10000`) on final results, not on intermediate stages.

**What's missing:** `SET temp_directory='/tmp/duckdb_spill'`. Per-query memory limit (`SET memory_limit='2GB'`). Query complexity check before execution (reject queries without WHERE clauses on large tables).

---

## 11. Token & Context Window Limits

| Limit | Value | What happens when exceeded |
|-------|-------|---------------------------|
| nomic-embed-text max tokens | 8192 | Silent truncation — embedding covers only first 8192 tokens |
| Local 7B model context | 4K–8K (effective) | Schema/history truncated; model ignores tail of context |
| Local 13B model context | 8K–16K | Better but still limited vs cloud (128K+) |
| NL→SQL schema text (50 tables) | ~4,000–6,000 tokens | Leaves <2K for reasoning in 7B models |
| RAG context (5 chunks × 512 tokens) | ~2,560 tokens | Compound with schema/history → overflow |
| KG extraction per contract | 1,000–3,000 tokens | One LLM call per contract — slow on local hardware |
| NL→Cypher schema context (all labels) | ~500–1,000 tokens | Manageable, but scales with graph size |

**The core local-model constraint:** A local 7B model has an effective context window of 4K–8K tokens. The NL→SQL schema alone can consume most of this. Adding conversation history and retrieved RAG chunks can push the total past the model's attention capacity, causing silent quality degradation — not an error, just wrong answers.

**What's missing across all components:** A unified token budget manager that knows the configured model's context window and allocates tokens explicitly across (system prompt) + (schema/graph context) + (history) + (retrieved chunks) + (question). Each component currently allocates its context portion independently with no coordination.

---

## 12. Data Consistency & Staleness

### 12.1 NL→SQL schema is static — no live reload

**What goes wrong:** Schema is captured at startup. New tables, dropped columns, renamed columns → running system is wrong. Generated SQL references columns that no longer exist.

**Current mitigation:** Restart required.

**What's missing:** Schema version hash stored at startup. Background re-generation on change detection. Or: schema re-generated per session (adds startup latency but guarantees freshness).

---

### 12.2 Both caches serve stale results after data updates

**What goes wrong:** NL cache and SQL hash cache serve cached `QueryResult` indefinitely (LRU eviction only, no TTL). After new documents are ingested or data is updated, cached results are stale. The system serves the old answer until the cache is naturally evicted.

**What the user sees:** "How many documents are in the system?" returns 42 after 10 more documents were ingested.

**What's missing:** TTL-based cache invalidation. An explicit cache invalidation hook triggered by `POST /v1/ingest`.

---

### 12.3 KG and RAG not re-synced after document updates

**What goes wrong:** A document is updated (re-ingested with changed content). The RAG vector store updates correctly. The KG extraction is not re-run — the graph retains entities and relationships from the old version of the document. Now RAG and KG are out of sync for that document.

**Current mitigation:** None.

**What's missing:** A document version hash in both the RAG store and the KG. On re-ingest, detect hash change, delete old KG entities for that document, re-run extraction.

---

## 13. Silent Failures — Wrong Answers That Look Right

The most dangerous failure category: the system produces output with no error signal. Detection requires ground-truth evaluation or user feedback loops — neither is implemented.

| Scenario | What the system returns | What's actually happening |
|----------|------------------------|--------------------------|
| Retrieval miss + local LLM confabulation | Confident paragraph citing the contract | No retrieved chunk supports it; 7B model invented it |
| KG catch-all fires | List of 50 contracts | User asked about a specific clause type; intent parsing failed |
| `_extract_name()` extracts question noun | Cypher: `WHERE c.name CONTAINS 'contracts'` | Entity name is a question word, not a real entity |
| SQL semantic error (correct syntax, wrong logic) | A plausible number or table | Date range wrong, aggregation wrong, column semantics wrong |
| Stale cached result | Old QueryResult | Data was updated; cache wasn't invalidated |
| Embedding model mismatch | Random top-K chunks | Old + new embeddings in different vector spaces |
| OCR error in chunk | Chunk retrieved, answer plausible | LLM synthesises around garbled text; answer is partially wrong |
| KG entity dedup failure | Partial count | Same party under 3 surface forms → 3 separate nodes |
| KG out of sync with RAG | Inconsistent counts from the two paths | One document processed by RAG but not KG |
| Local model ignores "say I don't know" | Fabricated answer | Small local model doesn't follow refusal instructions |

**The detection gap:** Every one of these failures produces output, not an error. Without a ground-truth evaluation set running continuously in production, these failures are invisible until a user notices and reports.

---

## 14. Risk Matrix

**Likelihood:** H High · M Medium · L Low  
**Impact:** H High (wrong answer, data loss, outage) · M Medium (degraded quality) · L Low  
**Mitigation:** ✅ Exists · 🔶 Partial · ❌ None

| # | Failure | Likelihood | Impact | Mitigation | Notes |
|---|---------|-----------|--------|------------|-------|
| 1.1 | Local model ignores output format instructions | H | M | 🔶 | Worse on 7B; use 13B+ for format-critical tasks |
| 1.2 | Quantisation degrades SQL/Cypher quality | H | M | ❌ | Use Q5_K_M minimum for code tasks |
| 1.3 | Ollama inference timeout — no LLM timeout set | H | H | ❌ | No `asyncio.wait_for()` on LLM calls |
| 1.4 | VRAM exhaustion — model eviction mid-session | M | M | ❌ | Pin LLM with `OLLAMA_KEEP_ALIVE=-1` |
| 1.5 | Local model confabulates instead of refusing | H | H | 🔶 | Use citation enforcement |
| 1.6 | Non-deterministic output caches a bad result | M | M | 🔶 | Use temperature=0 for SQL/Cypher generation |
| 2.1 | Bad Docling chunk boundaries | H | M | 🔶 | Chunk quality scoring missing |
| 2.2 | OCR errors in scanned PDFs | M | M | ❌ | No confidence scoring |
| 2.3 | Table extraction failures | H | M | ❌ | Table-aware chunking missing |
| 3.2 | Embedding model swap → stale vectors | M | H | ❌ | **Critical** — no per-chunk model tracking |
| 4.1 | Top-K misses relevant chunk (hit rate 26.5%) | H | H | 🔶 | Reranker + HyDE disabled |
| 4.3 | Score threshold drops all results | M | H | 🔶 | No adaptive fallback |
| 5.1 | LLM confabulation on retrieval miss | H | H | 🔶 | **Critical** — worse with local 7B models |
| 5.2 | Context window overflow | M | M | ❌ | No token budget management |
| 6.1 | LLM entity extraction inconsistency | M | M | 🔶 | Structured output enforcement missing |
| 6.3 | Bronze→Silver→Gold fails silently | M | M | ❌ | No pipeline status tracking |
| 6.4 | KG construction slow with local LLMs | H | M | 🔶 | Async parallelism partial |
| 7.1 | IntentParser catch-all fires silently | H | H | ❌ | **Critical** — no logging, no user signal |
| 7.2 | LLM Cypher generation fails on local models | H | H | ❌ | Not yet implemented; validation needed first |
| 7.4 | No date/numeric filtering in Cypher builders | H | M | ❌ | Queries silently ignore temporal constraints |
| 8.1 | Schema text overflows local model context | H | H | ❌ | **Critical** — semantic schema retrieval needed |
| 8.4 | Correct SQL, semantically wrong answer | H | H | ❌ | **Critical** — no semantic validation |
| 9.1 | RAG store and KG out of sync | H | M | ❌ | No document registry |
| 9.2 | NL→SQL and KG return conflicting answers | M | H | ❌ | No unified answer layer |
| 9.4 | Ollama bottleneck shared across all components | H | M | ❌ | Single Ollama instance, no queue management |
| 9.5 | KG catch-all + LLM fabricates findings | H | H | ❌ | **Critical** — confidence signal missing |
| 10.2 | Ollama unavailable mid-request | M | H | ❌ | No retry/backoff on provider errors |
| 11.* | Token budget unmanaged across components | H | M | ❌ | Each component allocates independently |
| 13.* | Silent failures — no ground truth evaluation | H | H | ❌ | **Critical** — no detection mechanism |

### The eight Critical failures

These are the items most likely to cause material harm in production with no current mitigation:

1. **Local model confabulation** — 7B models don't follow "say I don't know." Users receive fabricated legal findings.
2. **Schema text overflows local model context** — On any real schema, the LLM can't reason correctly. NL→SQL quality collapses.
3. **Correct SQL, semantically wrong answer** — No semantic validation. Wrong answers look correct and get cached.
4. **IntentParser catch-all fires silently** — Complex KG queries return a list of all contracts with no indication of failure.
5. **KG catch-all + LLM fabricates findings** — Catch-all output fed to LLM produces fabricated legal facts stated as findings.
6. **Embedding model swap → stale vectors** — Entire vector store becomes garbage with no error signal.
7. **Top-K retrieval miss at 26.5% hit rate** — Almost 3 in 4 queries may fail to retrieve the relevant chunk.
8. **No ground-truth evaluation running in production** — All the above failures are invisible without continuous evaluation.
