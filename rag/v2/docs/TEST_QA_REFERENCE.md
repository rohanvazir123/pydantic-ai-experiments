# RAG v2 — Test & QA Reference

> Single source of truth for every metric, formula, threshold, and acceptance criterion used across RAG v2 testing. Read this before writing or running any test.

---

## Table of Contents

- [1. Retrieval Metrics](#1-retrieval-metrics)
  - [1.1 Metric Definitions and Formulas](#11-metric-definitions-and-formulas)
  - [1.2 Relevance Judgement Rules](#12-relevance-judgement-rules)
  - [1.3 Thresholds by Corpus](#13-thresholds-by-corpus)
  - [1.4 Interpreting Results](#14-interpreting-results)
  - [1.5 K Values and When to Use Each](#15-k-values-and-when-to-use-each)
- [2. Ingestion Metrics](#2-ingestion-metrics)
  - [2.1 Throughput and Latency SLAs](#21-throughput-and-latency-slas)
  - [2.2 Chunk Quality Metrics](#22-chunk-quality-metrics)
  - [2.3 Incremental Ingestion Correctness](#23-incremental-ingestion-correctness)
- [3. Generation and Answer Quality Metrics](#3-generation-and-answer-quality-metrics)
  - [3.1 Faithfulness](#31-faithfulness)
  - [3.2 Answer Relevance](#32-answer-relevance)
  - [3.3 Answer Correctness (Ground Truth Required)](#33-answer-correctness-ground-truth-required)
  - [3.4 Thresholds and Alert Levels](#34-thresholds-and-alert-levels)
- [4. Confidence Scoring](#4-confidence-scoring)
  - [4.1 How Confidence Is Computed](#41-how-confidence-is-computed)
  - [4.2 Pipeline Gate Thresholds](#42-pipeline-gate-thresholds)
  - [4.3 Abstention Metrics](#43-abstention-metrics)
  - [4.4 Calibration Workflow](#44-calibration-workflow)
- [5. Latency and Cost Metrics](#5-latency-and-cost-metrics)
  - [5.1 Per-Stage Latency Budgets](#51-per-stage-latency-budgets)
  - [5.2 Token and Cost Accounting](#52-token-and-cost-accounting)
- [6. Scale Test Plan](#6-scale-test-plan)
  - [6.1 Load Model](#61-load-model)
  - [6.2 Baseline Load Scenarios](#62-baseline-load-scenarios)
  - [6.3 Stress and Exhaustion Tests](#63-stress-and-exhaustion-tests)
- [7. Chaos and Resilience Test Plan](#7-chaos-and-resilience-test-plan)
  - [7.1 Failure Injection Matrix](#71-failure-injection-matrix)
  - [7.2 Recovery Acceptance Criteria](#72-recovery-acceptance-criteria)
- [8. Gold Dataset Format](#8-gold-dataset-format)
  - [8.1 JSONL Schema](#81-jsonl-schema)
  - [8.2 Relevance Rules](#82-relevance-rules)
  - [8.3 Adding New Samples](#83-adding-new-samples)
- [9. Regression Thresholds](#9-regression-thresholds)
  - [9.1 Quality Regression Tolerances](#91-quality-regression-tolerances)
  - [9.2 Performance Regression Tolerances](#92-performance-regression-tolerances)
  - [9.3 How to Run a Regression Comparison](#93-how-to-run-a-regression-comparison)

---

## 1. Retrieval Metrics

These are the five standard Information Retrieval (IR) metrics used to measure whether the retriever surfaces the right documents. All are computed over a gold dataset of (query, relevant\_sources) pairs.

### 1.1 Metric Definitions and Formulas

All metrics are **mean over all queries** in the evaluation set.

#### Hit Rate@K

> "Does the system find *anything* useful in the top K?"

Binary per query: 1.0 if at least one relevant document appears in the top-K results, 0.0 otherwise.

```
hit_rate@K = mean(1 if any relevant in top-K else 0  for each query)
```

**Use when:** you want to know whether the system is usable at all for a given query set. A low Hit Rate means the corpus doesn't contain what users are asking, or the retriever is completely missing relevant documents.

---

#### MRR@K — Mean Reciprocal Rank

> "How *early* does the first relevant result appear?"

```
rr(query) = 1 / rank_of_first_relevant_result   (0.0 if none in top-K)
mrr@K     = mean(rr(q) for each query q)
```

**Example:** if the first relevant result is rank 1 → RR = 1.0; rank 2 → RR = 0.5; rank 3 → RR = 0.33; not found in top-K → RR = 0.0.

**Use when:** you care about how much the user has to scroll. An MRR of 0.5 means the first relevant result is typically at rank 2.

---

#### NDCG@K — Normalised Discounted Cumulative Gain

> "Are relevant results ranked *above* irrelevant ones?"

```
DCG@K  = Σ_{i=0}^{K-1}  rel_i / log2(i + 2)        # i is 0-indexed
IDCG@K = Σ_{i=0}^{min(n_relevant, K)-1}  1 / log2(i + 2)   # ideal ranking
ndcg@K = DCG@K / IDCG@K     (0.0 if no relevant docs exist)
```

Uses binary relevance (0 or 1). IDCG is computed by assuming all relevant documents are ranked first.

**Use when:** you want a ranking quality signal that rewards putting relevant docs at the top more than at the bottom. NDCG is the most informative single retrieval metric for ranked lists.

---

#### Precision@K

> "What fraction of returned results are actually relevant?"

```
precision@K = mean(count(relevant in top-K) / K  for each query)
```

**Use when:** result quality matters — you want every returned document to be useful, not just the first one. Low Precision@K means the results are noisy.

---

#### Recall@K

> "What fraction of *all* known relevant documents are found?"

```
recall@K = mean(count(relevant in top-K) / total_relevant  for each query)
```

`total_relevant` is the number of gold relevant sources listed for that query.

**Use when:** completeness matters — you want to know if the system is missing relevant documents. Important for corpora where multiple documents are needed to fully answer a question.

---

### 1.2 Relevance Judgement Rules

Relevance is determined by checking whether any gold source **stem** appears as a substring (case-insensitive) in the retrieved document's `document_source` path.

```python
def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    src_lower = document_source.lower()
    return any(stem.lower() in src_lower for stem in relevant_sources)
```

**Example:**
- Gold: `relevant_sources = ["team-handbook"]`
- Retrieved source: `/rag/documents/team-handbook.md` → **relevant** ✓
- Retrieved source: `/rag/documents/company-overview.md` → **not relevant** ✗

**Rule for multi-document queries:** if a query has `relevant_sources = ["company-overview", "team-handbook"]`, any result matching *either* stem is relevant. `total_relevant` for Recall is set to the number of gold stems, not the number of matching documents in the corpus.

---

### 1.3 Thresholds by Corpus

These are minimum acceptable scores at K=5. Failing any threshold in CI blocks the merge.

#### NeuralFlow AI corpus (10 gold queries)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Hit Rate@5 | ≥ 0.60 | 6 of 10 queries must find a relevant doc |
| MRR@5 | ≥ 0.40 | First relevant result typically in top 3 |
| Precision@5 | ≥ 0.15 | ~1 relevant result per 5 returned |
| Recall@5 | ≥ 0.40 | 40% of known relevant docs surfaced |
| NDCG@5 | ≥ 0.40 | Relevant results ranked above irrelevant |
| P95 latency | ≤ 10,000 ms | End-to-end wall clock per query |

These thresholds are *baselines*, not targets. The target for a well-tuned system is:

| Metric | Target |
|--------|--------|
| Hit Rate@5 | ≥ 0.85 |
| MRR@5 | ≥ 0.65 |
| NDCG@5 | ≥ 0.70 |
| P95 latency | ≤ 600 ms |

#### Individual search path sanity checks

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Semantic Hit Rate@5 | ≥ 0.40 | Embedding search is working |
| Text Hit Rate@5 | ≥ 0.40 | Full-text search is working |
| Hybrid ≥ Semantic − 10pp | Hit Rate@5 | RRF fusion is not degrading results |

#### Legal / CUAD corpus (10 gold queries)

| Metric | Threshold |
|--------|-----------|
| Hit Rate@5 | ≥ 0.50 |
| MRR@5 | ≥ 0.35 |
| NDCG@5 | ≥ 0.35 |

#### Corpus isolation checks (both corpora ingested)

| Query type | Expected behaviour |
|------------|-------------------|
| Legal clause query | Top-5 must contain ≥ 1 legal doc; must contain 0 NeuralFlow docs |
| Company query | Top-5 must contain ≥ 1 NeuralFlow doc; must contain 0 legal docs |

A corpus isolation failure is a **critical bug** regardless of the Hit Rate score — it means RBAC or `corpus_id` filtering is broken.

---

### 1.4 Interpreting Results

Use this table to diagnose what a failing metric means:

| Symptom | Likely cause | What to investigate |
|---------|-------------|---------------------|
| Hit Rate@5 = 0.0 | Retriever not connected, or corpus not ingested | Check DB connection, chunk count, `corpus_id` filter |
| Hit Rate@5 low but not zero | Wrong embedding model, embedding dimension mismatch, or insufficient data | Check `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, chunk count |
| Hit Rate high, MRR low | Relevant docs found but buried at rank 4–5 | Reranker off, or RRF weights need tuning |
| Hit Rate high, NDCG low | Same as above — relevant docs ranked poorly | Reranker quality issue |
| High Hit Rate, low Precision | Too many irrelevant results returned | Confidence threshold too low; reranker not effective |
| High Hit Rate, low Recall | Corpus coverage gap — not all relevant docs are indexed | Check ingestion completeness, incremental mode |
| Corpus isolation failure | `corpus_id` filter not applied, or wrong RBAC | Check `WHERE corpus_id = $1` in all queries |

---

### 1.5 K Values and When to Use Each

| K | When to use |
|---|------------|
| K=1 | Chatbot with a single "best answer" context — the first result must be right |
| K=3 | Default for tight context windows or fast responses |
| K=5 | Default for quality evaluation — balances precision and recall |
| K=10 | Long-context models or when completeness is critical |

All tests report K ∈ {1, 3, 5}. Gate thresholds use K=5.

---

## 2. Ingestion Metrics

### 2.1 Throughput and Latency SLAs

End-to-end job latency from submission to `status=completed`. Measured from the API `POST /v1/ingest` response until the `IngestCompleteEvent` lands in `knowledge:events`.

| Document type | P50 | P95 | P99 |
|---------------|-----|-----|-----|
| Plain text / Markdown (< 10 KB) | < 5 s | < 15 s | < 30 s |
| PDF, < 20 pages | < 30 s | < 90 s | < 3 min |
| PDF, 20–100 pages | < 2 min | < 6 min | < 12 min |
| DOCX / PPTX | < 20 s | < 60 s | < 2 min |
| Audio, 60 min (Whisper Turbo) | < 5 min | < 12 min | < 20 min |
| Any type + graph extraction | +50–100% on all tiers | | |
| Batch, 100 documents | < 30 min | < 90 min | < 3 h |

**Sub-stage budget** (10-page PDF baseline, P95):

| Stage | Budget |
|-------|--------|
| API → Redis XADD (job accepted) | < 150 ms |
| Worker pickup (XREADGROUP) | < 5 s |
| Docling parse | < 20 s |
| HybridChunker | < 3 s |
| Embedding batch (65 chunks) | < 15 s |
| Vector store upsert | < 5 s |
| Graph extraction (optional) | < 90 s |

---

### 2.2 Chunk Quality Metrics

These verify that chunking is producing usable, well-formed output. Checked in integration tests, not in CI on every run.

| Metric | How to measure | Acceptable range |
|--------|---------------|-----------------|
| Mean chunk token count | `mean(chunk.token_count for chunk in chunks)` | 150–450 tokens |
| Max chunk token count | `max(chunk.token_count)` | ≤ `max_tokens` setting |
| Empty chunk rate | `count(chunks where len(content.strip()) == 0) / total` | 0.0% (zero empty chunks) |
| Contextualized chunk rate | `count(chunks where has_context=True) / total` | ≥ 80% for structured docs (PDF/DOCX) |
| Heading context present | First line of chunk content starts with heading text | Verified in `test_contextualize_output` |
| Fallback rate | `count(chunks where chunk_method="simple_fallback") / total` | ≤ 5% for structured formats |

**Contextualization check:** when `HybridChunker.contextualize(chunk)` is called, the output must differ from `chunk.text` — it prepends the heading hierarchy. Verify:

```python
contextualized = chunker.contextualize(chunk=chunk)
assert contextualized != chunk.text
assert len(contextualized) > len(chunk.text)  # context adds content
```

---

### 2.3 Incremental Ingestion Correctness

Run after ingesting a corpus, modifying one file, then running incremental ingest again.

| Assertion | How to check |
|-----------|-------------|
| Unchanged files are skipped | Job result: `skipped=True`; DB: same `content_hash` before and after |
| Modified file is re-ingested | Old chunks deleted; new chunks inserted; `content_hash` updated |
| Deleted file is removed | After file deletion, run incremental; verify doc and chunks removed from DB |
| No duplicate chunks after repeated runs | `SELECT COUNT(*) FROM chunks WHERE document_id = $1` stays constant on re-run of unchanged file |
| Fingerprint cache populated | After first ingest, Redis `cache:doc_fingerprint:{sha256}` key exists |
| Fingerprint cache hit skips DB | Second run with unchanged file: no DB write; Redis hit counter increments |

---

## 3. Generation and Answer Quality Metrics

These measure whether the LLM produces good answers, not just whether retrieval found good context.

### 3.1 Faithfulness

**What it measures:** Are all claims in the answer supported by the retrieved context? A faithfulness score of 1.0 means every claim is grounded; 0.0 means fully hallucinated.

**How it is computed:**

1. Decompose the answer into atomic claims via the nano model:
   > *"List every factual claim made in this answer as individual sentences."*

2. For each claim, verify against the retrieved context via the nano model:
   > *"Context: {context}\nClaim: {claim}\nIs this claim supported by the context? YES or NO"*

3. `faithfulness = count(supported_claims) / count(total_claims)`

**Important:** faithfulness does not require a ground-truth answer. It measures the answer against the *retrieved context*, not against a known correct answer. A faithful answer to bad context can still be factually wrong — faithfulness and correctness are independent metrics.

---

### 3.2 Answer Relevance

**What it measures:** Does the answer address the question? Independent of whether the answer is factually correct.

**How it is computed:**

1. Generate N=3 reverse questions from the answer via the nano model:
   > *"Generate 3 questions that this answer would be a good response to."*

2. Embed the original query and each reverse question.

3. `answer_relevance = mean(cosine_sim(query_emb, reverse_q_emb) for each reverse question)`

**Range:** 0.0–1.0. High relevance (≥ 0.8) means the answer is on-topic. Low relevance (< 0.5) typically means the answer drifted or refused to answer without saying so.

---

### 3.3 Answer Correctness (Ground Truth Required)

Used only for gold samples that have a `ground_truth_answer` field. Multiple metrics give different views:

| Metric | Library | What it measures | Typical range |
|--------|---------|-----------------|---------------|
| **BLEU-4** | `nltk.translate.bleu_score` | N-gram precision up to 4-grams; penalises short answers | 0–1; ≥ 0.3 is good for RAG |
| **ROUGE-1-F** | `rouge-score` | Unigram overlap (F1 of precision + recall) | 0–1 |
| **ROUGE-2-F** | `rouge-score` | Bigram overlap; phrase-level match | 0–1 |
| **ROUGE-L-F** | `rouge-score` | Longest common subsequence F1; preserves word order | 0–1 |
| **METEOR** | `nltk.translate.meteor_score` | Recall-weighted + synonym matching; better than BLEU for short texts | 0–1 |
| **BERTScore-F** | `bert-score` | Contextual embedding F1 via BERT; best for paraphrase detection | 0.8–1.0 typical |
| **Semantic Similarity** | cosine(embed(answer), embed(ground_truth)) | Fast embedding-level match | 0–1 |

**Which to use:**
- Start with **Semantic Similarity** — fastest, no heavy models needed.
- Add **ROUGE-L-F** — good balance of speed and signal.
- Use **BERTScore** only when you need paraphrase detection (high compute cost).
- Use **BLEU-4** only when comparing against exact phrasing is meaningful (rarely the case in RAG).

---

### 3.4 Thresholds and Alert Levels

| Metric | Minimum acceptable | Alert threshold | Human review threshold |
|--------|-------------------|-----------------|------------------------|
| Faithfulness | ≥ 0.70 | < 0.70 | < 0.50 |
| Answer Relevance | ≥ 0.65 | < 0.65 | < 0.40 |
| ROUGE-L-F | ≥ 0.25 | < 0.25 | — |
| Semantic Similarity | ≥ 0.60 | < 0.60 | — |

An answer with `faithfulness < 0.50` is considered hallucinated and triggers a human-review flag. An answer with `faithfulness = 0.0` (no supported claims) must never reach a user — the judge gate or citation gate should have caught it.

---

## 4. Confidence Scoring

### 4.1 How Confidence Is Computed

Confidence is a calibrated 0–1 score attached to every `SearchResult`. It is not the same as the raw search score.

| Search path | Raw score (`raw_score`) | Confidence (`confidence`) |
|-------------|------------------------|--------------------------|
| Hybrid (default) | RRF score (`Σ 1/(60+rank)`, max ~0.05) | `sigmoid(cross_encoder_logit)` after reranking |
| Semantic only | `1 - pgvector cosine distance` (0–1) | Same as raw cosine (no reranker needed) |
| Text only | `ts_rank` output (unbounded) | `None` — ts_rank is not calibrated |

**Key rule:** `confidence` is only set after CrossEncoder reranking. For the hybrid path (always-on reranker), every result gets a confidence score. For standalone text-only search, confidence remains `None` and the confidence filter does not fire.

**Sigmoid calibration:**
```python
import math
confidence = 1.0 / (1.0 + math.exp(-cross_encoder_logit))
```

CrossEncoder logits are not bounded; sigmoid maps them to (0, 1). A logit of 0 → confidence = 0.5; logit of 3 → confidence ≈ 0.95; logit of −3 → confidence ≈ 0.05.

---

### 4.2 Pipeline Gate Thresholds

Three gates in sequence. If any gate fires, the pipeline returns an abstention response without calling the next stage.

| Gate | Layer | Condition to abstain | Default threshold |
|------|-------|----------------------|-------------------|
| Retrieval confidence gate | 1 | `sum(confidence for top-K results) < retrieval_confidence_threshold` | 1.5 (= avg 0.30 per chunk at K=5) |
| Citation gate | 2 | `len(uncited_claims) > 0` in generation output | Any uncited claim |
| Judge gate | 3 | `verdict == "unsupported"` OR `judge_confidence < judge_confidence_threshold` | 0.60 |

**Layer 1 aggregate confidence explained:**
- K=5 results, threshold=1.5 → requires average per-chunk confidence ≥ 0.30
- This is a deliberately low floor — it blocks only completely empty or garbage retrieval
- Tighten per corpus once confidence distributions are measured (see §4.4)

**When `verdict == "partial"` (Layer 3):** the pipeline proceeds to answer but appends: *"Note: This answer may be incomplete based on the available context."*

---

### 4.3 Abstention Metrics

Track these per corpus to diagnose calibration problems.

| Metric | Formula | Target |
|--------|---------|--------|
| Abstention rate | `abstained / total` | < 15% on gold dataset |
| False abstention rate | `abstained_on_answerable / answerable` | < 5% |
| Layer 1 share | `abstained_layer1 / abstained` | Diagnoses retrieval gaps |
| Layer 2 share | `abstained_layer2 / abstained` | Diagnoses hallucination pressure |
| Layer 3 share | `abstained_layer3 / abstained` | Diagnoses judge threshold calibration |
| Partial answer rate | `partial / answered` | < 20% on gold dataset |

**Diagnosing from abstention layer share:**
- High Layer 1 → corpus coverage problem; add more documents
- High Layer 2 → LLM is hallucinating or prompt is too permissive; tighten citation prompt
- High Layer 3 → judge threshold too high or judge model too conservative; lower `judge_confidence_threshold`
- False abstention rate > 5% → thresholds too aggressive; lower `retrieval_confidence_threshold` or `judge_confidence_threshold`

---

### 4.4 Calibration Workflow

Run this after every significant ingestion batch (new documents shift the confidence distribution).

1. Run eval with `retrieval_confidence_threshold = 0` (disable Layer 1) to get a baseline Hit Rate.
2. Sweep `retrieval_confidence_threshold` from 0.5 → 3.0 in steps of 0.25.
3. Plot: abstention rate (y-axis) vs. false abstention rate (x-axis). Pick the knee point — the threshold where abstention rate starts climbing steeply for minimal false abstention gain.
4. Repeat for `judge_confidence_threshold` from 0.40 → 0.80 in steps of 0.05.
5. Re-run after every significant corpus change (> 20% new documents).

Store calibration sweep results in `eval_runs.report_json` for the run.

---

## 5. Latency and Cost Metrics

### 5.1 Per-Stage Latency Budgets

Every request produces a latency span tree. These are the P95 targets and alert thresholds.

| Stage | P95 target | Alert threshold | Measure from |
|-------|-----------|-----------------|-------------|
| Schema validation (V1–V2) | < 2 ms | > 10 ms | Request receipt |
| Content policy check (V5, nano) | < 50 ms | > 150 ms | After V2 pass |
| Query routing (nano) | < 80 ms | > 250 ms | After V5 pass |
| L2 Redis cache lookup | < 5 ms | > 20 ms | After routing |
| Query embedding | < 80 ms | > 250 ms | Embed call start |
| Hybrid retrieval (vector + text, parallel) | < 120 ms | > 400 ms | After embedding |
| CrossEncoder rerank | < 200 ms | > 600 ms | After retrieval |
| L3 semantic cache lookup | < 40 ms | > 100 ms | After rerank |
| LLM first token (small model, TTFT) | < 600 ms | > 1,500 ms | After prompt assembled |
| LLM full generation (~300 output tokens) | < 1,200 ms | > 3,000 ms | |
| Judge gate (nano) | < 80 ms | > 250 ms | After generation |
| **Total — search-only** | **< 600 ms** | **> 1,200 ms** | End-to-end |
| **Total — chat, small model** | **< 2,000 ms** | **> 4,000 ms** | End-to-end |

**Streaming TTFT** (time to first token in SSE response): < 300 ms P50, < 800 ms P95.

**PagerDuty alerts fire when:**
- `chat_latency_p95 > 3 s` sustained 5 min
- `search_latency_p99 > 1.5 s`
- `streaming_ttft_p95 > 1,000 ms`
- `l3_cache_hit_rate < 15%`

---

### 5.2 Token and Cost Accounting

Per-query token budget (standard config):

| Stage | Model tier | Input tokens | Output tokens |
|-------|-----------|-------------|--------------|
| Content policy (V5) | nano | 200 | 30 |
| Query routing | nano | 150 | 30 |
| Query embedding | embedding | 50 | — |
| Retrieved context (top-5, 200 tok avg each) | — | 1,000 | — |
| LLM generation (system + context + query) | small | 1,350 | 300 |
| Judge gate | nano | 1,700 | 100 |
| **Total — standard config** | | **3,200** | **430** |

**Cloud model cost reference** (for cost estimation in `metrics/performance.py`):

| Model | Input $/MTok | Output $/MTok |
|-------|-------------|--------------|
| `claude-haiku-4-5` | $0.25 | $1.25 |
| `claude-sonnet-4-6` | $3.00 | $15.00 |
| `claude-opus-4-8` | $15.00 | $75.00 |
| Local Ollama models | $0.00 | $0.00 |

`estimate_cost(model_id, prompt_tokens, completion_tokens)` uses per-1K-token rates: divide the above by 1,000 to get the per-1K values stored in `COST_PER_1K_TOKENS`.

---

## 6. Scale Test Plan

### 6.1 Load Model

Derived from 10,000 DAU target:

| Parameter | Value |
|-----------|-------|
| Daily active users | 10,000 |
| Queries per user per day (median) | 5 |
| Total queries per day | 50,000 |
| Active window | 8 h (business hours, UTC) |
| Average RPS | 1.7 req/s |
| Peak RPS (3× burst) | 5 req/s |
| Peak concurrency | ~10 in-flight |
| Documents ingested per day | 100–500 |

**Cache offload assumption** (reduces LLM calls):

| Layer | Hit rate | Queries reaching LLM |
|-------|---------|----------------------|
| L2 Redis (exact match, 5 min TTL) | ~10% | — |
| L3 semantic cache (cosine ≥ 0.95, 60 min TTL) | ~30% of remainder | — |
| **Combined — queries reaching LLM** | | **~60% (30,000/day)** |

---

### 6.2 Baseline Load Scenarios

Run with `locust` from `backend/tests/load/locustfile.py`. All scenarios run against staging with production-equivalent ingested data (≥ 5,000 chunks).

| Scenario | RPS | Duration | Pass criteria |
|----------|-----|----------|--------------|
| Baseline search only | 1 RPS | 5 min | P95 < 600 ms; 0% errors |
| Baseline chat (small model) | 1 RPS | 5 min | P95 < 2,000 ms; 0% errors |
| Ramp — find breaking point | 1 → 20 RPS over 10 min | 10 min | Record RPS where error rate > 1% |
| Sustained peak | 5 RPS | 30 min | P95 < 2,000 ms; error rate < 0.1% |
| Burst | 0 → 15 RPS spike for 60 s | 5 min | System recovers within 2 min; 0 DLQ entries |
| Cache warmup | 1 RPS, 100 unique queries | 5 min | L2 hit rate ≥ 10% by end |
| Cache cold | 5 RPS, 1,000 unique queries | 10 min | P95 < 2,000 ms (no cache benefit) |

**Locust task weights:**
- `search` (weight 5): `POST /api/v1/search` with random gold query
- `chat` (weight 3): `POST /api/v1/chat` with random gold query
- `ingest_small_doc` (weight 1): `POST /api/v1/ingest` with a small test document

---

### 6.3 Stress and Exhaustion Tests

These probe the system's ceilings. Run manually on demand, not in CI.

| Test | Scenario | What to measure |
|------|----------|----------------|
| DB connection pool exhaustion | 20 RPS sustained for 10 min | Pool waiters visible in `/health`; requests queue, not crash; P99 degrades gracefully |
| Redis memory ceiling | Fill semantic cache to `semantic_cache_max_rows` | Pruning job fires; no OOM; cache hit rate stable after pruning |
| Embedding API rate limit | Ingest 500 documents in 10 min | `RateLimitError` triggers backoff; 0 jobs lost to DLQ; total ingest < 3 h |
| LLM context overflow | 50 queries each with 8,000+ token context | Context trimming fires; `context_truncated: true` in response; 0 HTTP 500s |
| DLQ depth | Inject 20 permanently-failing ingest jobs | DLQ depth counter increments; alert email sent per entry; `/health` shows `dlq_depth > 0` |
| Tenant budget exhaustion | Exhaust Pro tier LLM budget mid-load | Chat returns `402`; search continues; alert email sent |

---

## 7. Chaos and Resilience Test Plan

### 7.1 Failure Injection Matrix

Run each scenario with 3 RPS background load. Kill the component, wait 60 s, restart it, verify recovery.

| Component killed | Expected degraded mode | Header |
|-----------------|----------------------|--------|
| **Redis** | `no_cache` — all queries hit DB; rate limiting falls back to DB counter | `X-Degraded-Mode: no_cache` |
| **Ollama / LLM** | `search_only` — retrieval works; generation returns `503 LLM_CIRCUIT_OPEN` | `X-Degraded-Mode: search_only` |
| **PostgreSQL** | `unavailable` — all endpoints return 503 | `X-Degraded-Mode: unavailable` |
| **Apache AGE** | `no_graph` — vector + text retrieval works; graph traversal skipped | `X-Degraded-Mode: no_graph` |
| **Embedding service** | `no_new_queries` — L2/L3 cache hits served; new queries fail | `X-Degraded-Mode: no_new_queries` |
| **CrossEncoder reranker** | `rrf_only` — results returned by RRF score only; confidence gates skip | `X-Degraded-Mode: rrf_only` |
| **All ingest workers** | Queue grows in Redis stream; no data loss | No degraded mode header (retrieval unaffected) |

---

### 7.2 Recovery Acceptance Criteria

For every chaos scenario, **all of the following must be true**:

1. **No HTTP 500s** during the failure period — every error returns a structured error code (`LLM_CIRCUIT_OPEN`, `DB_UNAVAILABLE`, etc.)
2. **`X-Degraded-Mode` header** present on every response during degradation
3. **No data corruption** — tenant isolation holds; no chunks written to wrong corpus
4. **Circuit breaker state shared** — if a circuit opens on one API pod, it is open on all pods immediately (state is in Redis)
5. **Alert email sent** within 60 s of circuit opening to `rohan.vazirani@gmail.com`
6. **Recovery within SLA:**

| Component | Recovery target after restart |
|-----------|------------------------------|
| Ollama | Circuit OPEN → HALF-OPEN → CLOSED within 90 s |
| Redis | All queries served within 5 s of restart |
| PostgreSQL | All queries resume within 10 s; no in-flight job data lost |

7. **DLQ is zero** after all components are restored — no jobs silently lost

**What must never happen in any scenario:**
- Unhandled Python exception returning HTTP 500
- Data written to wrong tenant (RLS bypass)
- DLQ entries accumulating without alert
- Circuit breaker state reset by API pod restart

---

## 8. Gold Dataset Format

### 8.1 JSONL Schema

Each gold sample is one JSON object per line in a `.jsonl` file located at `backend/knowledge/evaluation/data/{corpus_id}.jsonl`.

```jsonl
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "corpus_id": "neuralflow:default",
  "query": "What is the PTO policy?",
  "relevant_doc_sources": ["team-handbook"],
  "ground_truth_answer": "Employees accrue 15 days of PTO per year, with up to 5 days carried over.",
  "difficulty": "easy",
  "tags": ["factual", "hr-policy"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID string | Yes | Stable identifier; never change after creation |
| `corpus_id` | string | Yes | Corpus this sample belongs to (e.g. `"neuralflow:default"`) |
| `query` | string | Yes | Natural language query exactly as a user would type it |
| `relevant_doc_sources` | list[string] | Yes | Filename stems of relevant documents (substring match) |
| `ground_truth_answer` | string or null | No | Required for correctness metrics; omit if unknown |
| `difficulty` | `"easy"`, `"medium"`, `"hard"` | No | Default: `"medium"` |
| `tags` | list[string] | No | E.g. `["factual", "multi-hop", "aggregation", "temporal"]` |

---

### 8.2 Relevance Rules

A retrieved document is **relevant** to a gold sample if any string in `relevant_doc_sources` is a case-insensitive substring of the retrieved document's `document_source` field.

```python
# Exact implementation used in metric computation
def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    src_lower = document_source.lower()
    return any(stem.lower() in src_lower for stem in relevant_sources)
```

**Examples:**

| `relevant_doc_sources` | Retrieved `document_source` | Relevant? |
|------------------------|----------------------------|-----------|
| `["team-handbook"]` | `rag/documents/team-handbook.md` | ✓ Yes |
| `["team-handbook"]` | `rag/documents/company-overview.md` | ✗ No |
| `["Recording4"]` | `rag/documents/Recording4.mp3` | ✓ Yes (case-insensitive) |
| `["company-overview", "mission"]` | `rag/documents/mission-and-goals.md` | ✓ Yes (matches "mission") |

**Rule:** stems must be specific enough to not accidentally match unrelated documents. Avoid single-word stems like `"policy"` that could match many documents.

---

### 8.3 Adding New Samples

1. Assign a new UUID (use `python -c "import uuid; print(uuid.uuid4())"`).
2. Write the query exactly as a real user would ask it.
3. Identify the relevant document(s) by checking `document_source` values in the DB.
4. Choose a stem that is unique enough to match only the intended document(s).
5. Add `ground_truth_answer` if you know the correct answer; otherwise omit the field.
6. Tag with appropriate difficulty and tags.
7. Run the eval harness against the new sample in isolation before committing.
8. Commit the `.jsonl` file — it is the version-controlled source of truth for the gold dataset.

**Minimum gold dataset sizes:**

| Corpus size | Minimum gold samples | Target |
|-------------|---------------------|--------|
| < 20 documents | 10 samples | 20 |
| 20–100 documents | 20 samples | 50 |
| > 100 documents | 50 samples | 100+ |

---

## 9. Regression Thresholds

### 9.1 Quality Regression Tolerances

A regression is declared when the current run's metric drops below the baseline by more than the tolerance. This blocks CI merges.

| Metric | Tolerance | Direction |
|--------|-----------|-----------|
| Hit Rate@K | −0.05 (5 percentage points) | Lower is worse |
| MRR@K | −0.05 | Lower is worse |
| NDCG@K | −0.05 | Lower is worse |
| Precision@K | −0.05 | Lower is worse |
| Recall@K | −0.05 | Lower is worse |
| Faithfulness | −0.05 | Lower is worse |
| Answer Relevance | −0.05 | Lower is worse |
| ROUGE-L-F | −0.03 | Lower is worse |
| Semantic Similarity | −0.05 | Lower is worse |
| Abstention Rate | +0.05 | Higher is worse (more abstentions = regression) |
| False Abstention Rate | +0.02 | Higher is worse |

---

### 9.2 Performance Regression Tolerances

| Metric | Tolerance | Direction |
|--------|-----------|-----------|
| P95 total latency | +200 ms | Higher is worse |
| P95 retrieval latency | +100 ms | Higher is worse |
| P95 LLM first token | +200 ms | Higher is worse |
| Estimated cost per query | +20% | Higher is worse |
| L2 cache hit rate | −5 pp | Lower is worse |
| L3 cache hit rate | −5 pp | Lower is worse |

---

### 9.3 How to Run a Regression Comparison

```bash
# Trigger an eval run against a baseline
python -m knowledge.evaluation.runner \
  --corpus-id neuralflow:default \
  --baseline-run-id <previous_run_id> \
  --fail-on-regression

# View the report
curl http://localhost:8000/api/v1/evaluate/compare?a=<baseline_id>&b=<current_id>
```

The reporter outputs:
- Per-metric delta (`current - baseline`)
- `REGRESSION` flag on any metric exceeding tolerance
- Markdown summary posted as GitHub PR comment in CI

**In CI:** the eval step runs with `--fail-on-regression` which exits non-zero if any metric exceeds its tolerance, blocking the merge.
