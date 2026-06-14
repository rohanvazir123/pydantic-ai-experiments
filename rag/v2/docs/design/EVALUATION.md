# RAG v2 — Evaluation & Testing

## Table of Contents

- [Evaluation System — Offline & Online Metrics](#evaluation-system--offline--online-metrics)
  - [Module Layout](#module-layout)
  - [Data Schemas (`evaluation/schemas.py`)](#data-schemas-evaluationschemasy)
  - [Database Tables](#database-tables)
  - [Offline Metric Definitions](#offline-metric-definitions)
    - [Context Relevance — Retrieval Quality](#context-relevance--retrieval-quality)
    - [Faithfulness (no ground truth needed)](#faithfulness-no-ground-truth-needed)
    - [Answer Relevance (no ground truth needed)](#answer-relevance-no-ground-truth-needed)
    - [Answer Correctness (requires ground truth)](#answer-correctness-requires-ground-truth)
  - [Performance & Cost Metrics](#performance--cost-metrics)
    - [Latency Breakdown](#latency-breakdown)
    - [Token Accounting](#token-accounting)
    - [Cost Estimation](#cost-estimation)
    - [Storage Metrics](#storage-metrics)
  - [Evaluation Pipeline Flow](#evaluation-pipeline-flow)
  - [Regression Detection (`reporter.py`)](#regression-detection-reporterpy)
  - [CI Integration](#ci-integration)
  - [API Endpoints (additions to API Layer)](#api-endpoints-additions-to-api-layer)
  - [Prometheus Metrics (additions)](#prometheus-metrics-additions)
  - [Grafana Dashboard Panels (suggested layout)](#grafana-dashboard-panels-suggested-layout)
- [Load & Chaos Testing Strategy](#load--chaos-testing-strategy)
  - [Philosophy](#philosophy)
  - [Phase 1 — Baseline Load (single component, no failures)](#phase-1--baseline-load-single-component-no-failures)
  - [Phase 2 — Dependency Failure Injection (chaos)](#phase-2--dependency-failure-injection-chaos)
  - [Phase 3 — Sustained Load & Resource Exhaustion](#phase-3--sustained-load--resource-exhaustion)
  - [Phase 4 — Regression Gate (CI)](#phase-4--regression-gate-ci)
  - [Observability During Load Tests](#observability-during-load-tests)
- [Docling-Graph Evaluation Checklist](#docling-graph-evaluation-checklist)
- [Implementation Phases](#implementation-phases)
  - [Phase A — Housekeeping (no new features, before any refactor)](#phase-a--housekeeping-no-new-features-before-any-refactor)
  - [Phase B — Rate Limiting, Timeouts, Retries (in-progress, see section below)](#phase-b--rate-limiting-timeouts-retries-in-progress-see-section-below)
  - [Phase C — Module Skeleton](#phase-c--module-skeleton)
  - [Phase C2 — Validation + Hooks + Model Router Skeletons](#phase-c2--validation--hooks--model-router-skeletons)
  - [Phase D — Ingestion Pipeline Port](#phase-d--ingestion-pipeline-port)
  - [Phase E — Retrieval Port + Caching](#phase-e--retrieval-port--caching)
  - [Phase F — Security Layer](#phase-f--security-layer)
  - [Phase G — API Port](#phase-g--api-port)
  - [Phase H — Docker Compose + Local TLS](#phase-h--docker-compose--local-tls)
  - [Phase I — Cloud IaC Skeleton](#phase-i--cloud-iac-skeleton)
  - [Phase J — Evaluation System](#phase-j--evaluation-system)
  - [Phase K — Confidence-Based Scoring](#phase-k--confidence-based-scoring)
  - [Phase L — Confidence-Aware Pipeline](#phase-l--confidence-aware-pipeline)

---

### Evaluation System — Offline & Online Metrics

The evaluation system is a first-class citizen of the architecture, not an afterthought. Every retrieval or generation change must be measurable before and after.

#### Module Layout

```
knowledge/evaluation/
├── harness.py               # EvaluationHarness: orchestrates full eval runs
├── datasets.py              # GoldDataset: load/save/validate gold samples; supports JSONL + PostgreSQL
├── runner.py                # async runner; publishes EvalJob to knowledge:eval Redis stream
├── reporter.py              # metric aggregation, trend comparison, regression detection, CI report
├── schemas.py               # all Pydantic models (see Data Schemas below)
└── metrics/
    ├── retrieval.py         # HitRate@k, MRR@k, NDCG@k, Precision@k, Recall@k
    ├── faithfulness.py      # claim decomposition + NLI-style LLM verification
    ├── answer_relevance.py  # reverse-question generation + embedding cosine similarity
    ├── correctness.py       # BLEU-4, ROUGE-1/2/L-F1, METEOR, BERTScore-F, semantic-sim
    ├── performance.py       # latency breakdowns, token counts, cost estimates
    └── online.py            # user feedback aggregation + implicit signal processing
```

#### Data Schemas (`evaluation/schemas.py`)

```python
class GoldSample(BaseModel):
    id: UUID
    corpus_id: str
    query: str
    relevant_doc_sources: list[str]   # for retrieval metrics (source stem matching)
    ground_truth_answer: str | None   # for answer correctness; optional
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = []              # "factual", "multi-hop", "aggregation", "temporal"

class EvalRun(BaseModel):
    id: UUID
    corpus_id: str
    git_commit: str                   # reproducibility anchor
    model_tier: str                   # "small" | "large"
    search_type: str                  # "hybrid" | "semantic" | "text"
    k: int = 5                        # top-K for retrieval metrics
    started_at: datetime
    completed_at: datetime | None
    status: Literal["queued", "running", "completed", "failed"]
    sample_count: int = 0
    baseline_run_id: UUID | None      # for regression diff

class EvalResult(BaseModel):
    id: UUID
    run_id: UUID
    sample_id: UUID
    # --- Retrieval (Context Relevance) ---
    hit_rate: float                   # binary: any relevant in top-K
    mrr: float                        # 1/rank_of_first_relevant
    ndcg: float                       # normalised discounted cumulative gain
    precision_at_k: float             # relevant_in_topk / k
    recall_at_k: float                # relevant_in_topk / total_relevant
    # --- Generation ---
    faithfulness: float | None        # supported_claims / total_claims  (0-1)
    answer_relevance: float | None    # mean cosine_sim(question, reverse_questions)  (0-1)
    # --- Answer Correctness (requires ground_truth_answer) ---
    bleu_4: float | None
    rouge_1_f: float | None
    rouge_2_f: float | None
    rouge_l_f: float | None
    meteor: float | None
    bert_score_f: float | None
    semantic_similarity: float | None  # cosine_sim(answer_emb, gt_emb); fast alternative to BERTScore
    # --- Performance ---
    retrieval_ms: int
    llm_first_token_ms: int | None    # time-to-first-token for streamed responses
    generation_ms: int
    total_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None  # None for local models
    cache_tier_hit: str | None        # "l1" | "l2" | "l3" | None

class UserFeedback(BaseModel):
    id: UUID
    request_id: UUID
    user_id: str
    corpus_id: str
    query_hash: str               # SHA-256 of query (never store plaintext)
    session_id: str | None
    rating: int | None            # 1–5 stars
    thumbs: bool | None           # True=up, False=down
    correction: str | None        # user's suggested correction (stored encrypted)
    tags: list[str] = []          # "hallucinated" | "irrelevant" | "incomplete" | "outdated" | "correct"
    submitted_at: datetime

class ImplicitSignal(BaseModel):
    id: UUID
    session_id: str
    user_id: str
    corpus_id: str
    signal_type: Literal[
        "query_reformulation",    # user re-asked similar question → likely unsatisfied
        "follow_up_question",     # proxy for incomplete answer
        "session_abandoned",      # no further interaction after response
        "copy_action",            # user copied response → likely satisfied
        "escalation",             # user escalated to human support
    ]
    request_id: UUID | None
    recorded_at: datetime
```

#### Database Tables

```sql
-- Gold dataset samples (version-controlled via JSONL in git; also mirrored to DB)
CREATE TABLE gold_samples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id   TEXT NOT NULL,
    query       TEXT NOT NULL,
    relevant_doc_sources TEXT[] NOT NULL,
    ground_truth_answer  TEXT,
    difficulty  TEXT NOT NULL DEFAULT 'medium',
    tags        TEXT[] DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Evaluation runs (one row per triggered eval)
CREATE TABLE eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id       TEXT NOT NULL,
    git_commit      TEXT NOT NULL,
    model_tier      TEXT NOT NULL,
    search_type     TEXT NOT NULL,
    k               INT NOT NULL DEFAULT 5,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'queued',
    sample_count    INT DEFAULT 0,
    baseline_run_id UUID REFERENCES eval_runs(id),
    report_json     JSONB          -- regression diff + per-metric deltas written by reporter.py
);
CREATE INDEX ON eval_runs (corpus_id, started_at DESC);

-- Per-sample results (normalised; join with eval_runs for context)
CREATE TABLE eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    sample_id           UUID NOT NULL REFERENCES gold_samples(id),
    hit_rate            FLOAT,
    mrr                 FLOAT,
    ndcg                FLOAT,
    precision_at_k      FLOAT,
    recall_at_k         FLOAT,
    faithfulness        FLOAT,
    answer_relevance    FLOAT,
    bleu_4              FLOAT,
    rouge_1_f           FLOAT,
    rouge_2_f           FLOAT,
    rouge_l_f           FLOAT,
    meteor              FLOAT,
    bert_score_f        FLOAT,
    semantic_similarity FLOAT,
    retrieval_ms        INT,
    llm_first_token_ms  INT,
    generation_ms       INT,
    total_ms            INT,
    prompt_tokens       INT,
    completion_tokens   INT,
    total_tokens        INT,
    estimated_cost_usd  FLOAT,
    cache_tier_hit      TEXT,
    -- Confidence scoring fields (from Confidence-Based Scoring section)
    mean_confidence     FLOAT,      -- mean post-rerank confidence across top-K
    min_confidence      FLOAT,      -- lowest confidence chunk used
    low_confidence_flag BOOLEAN DEFAULT FALSE,
    -- Confidence-aware pipeline fields (from Confidence-Aware Pipeline section)
    pipeline_status             TEXT,   -- answered | abstained_retrieval | abstained_citation | abstained_judge
    abstention_layer            INT,    -- 1, 2, or 3 (NULL if answered)
    retrieval_aggregate_confidence FLOAT,
    citation_trustworthy        BOOLEAN,
    judge_verdict               TEXT,
    judge_confidence            FLOAT,
    false_abstention            BOOLEAN DEFAULT FALSE  -- abstained on a gold query with a known GT answer
);
CREATE INDEX ON eval_results (run_id);

-- Online user feedback (append-only)
CREATE TABLE user_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id  UUID NOT NULL,
    user_id     TEXT NOT NULL,
    corpus_id   TEXT NOT NULL,
    query_hash  TEXT NOT NULL,
    session_id  TEXT,
    rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
    thumbs      BOOLEAN,
    correction  TEXT,       -- stored as JWE if corpus marks data as sensitive
    tags        TEXT[] DEFAULT '{}',
    submitted_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON user_feedback (corpus_id, submitted_at DESC);
CREATE INDEX ON user_feedback (request_id);

-- Implicit behavioural signals
CREATE TABLE implicit_signals (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    corpus_id    TEXT NOT NULL,
    signal_type  TEXT NOT NULL,
    request_id   UUID,
    recorded_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON implicit_signals (corpus_id, signal_type, recorded_at DESC);
```

#### Offline Metric Definitions

##### Context Relevance — Retrieval Quality

Computed against `gold_samples.relevant_doc_sources`. Relevance = case-insensitive substring match of any gold source stem against `SearchResult.document_source`.

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Hit Rate@k** | `mean(any_relevant_in_topk)` | Does the system find *something* useful? |
| **MRR@k** | `mean(1 / rank_first_relevant)` | How *early* does the first relevant result appear? |
| **NDCG@k** | `mean(DCG@k / IDCG@k)` | Are relevant results ranked *above* irrelevant ones? |
| **Precision@k** | `mean(relevant_in_topk / k)` | What fraction of returned results are relevant? |
| **Recall@k** | `mean(relevant_in_topk / total_relevant)` | What fraction of all known-relevant docs are found? |

##### Faithfulness (no ground truth needed)

Measures whether the LLM answer is grounded in the retrieved context. Prevents hallucination reporting.

```
1. Decompose answer into atomic claims via nano-model:
   prompt: "List every factual claim made in this answer as individual sentences."
   → claims: list[str]

2. For each claim, verify against context via nano-model:
   prompt: "Context: {context}\nClaim: {claim}\nIs this claim supported by the context? YES or NO"
   → supported: bool

3. faithfulness = count(supported) / count(claims)
   Range: 0.0 (fully hallucinated) → 1.0 (fully grounded)
```

Alert threshold: `faithfulness < 0.7` → emit alert; `< 0.5` → flag for human review.

##### Answer Relevance (no ground truth needed)

Measures whether the answer addresses the question, not whether it's correct.

```
1. Generate N=3 reverse questions from answer via nano-model:
   prompt: "Generate 3 questions that this answer would be a good response to."
   → reverse_questions: list[str]

2. Embed original query + each reverse question.

3. answer_relevance = mean(cosine_sim(query_emb, rq_emb) for rq in reverse_questions)
   Range: 0.0 → 1.0
```

##### Answer Correctness (requires ground truth)

| Metric | Library | What it captures |
|--------|---------|-----------------|
| **BLEU-4** | `nltk.translate.bleu_score` | N-gram precision up to 4-grams; brevity penalty for short answers |
| **ROUGE-1-F** | `rouge-score` | Unigram recall + precision F1; word-level coverage |
| **ROUGE-2-F** | `rouge-score` | Bigram F1; phrase-level coverage |
| **ROUGE-L-F** | `rouge-score` | Longest common subsequence F1; preserves word order |
| **METEOR** | `nltk.translate.meteor_score` | Recall-weighted + synonym matching + fragmentation penalty; better than BLEU for short texts |
| **BERTScore-F** | `bert-score` | Contextual embedding F1 via BERT; best for paraphrase detection |
| **Semantic Similarity** | cosine_sim(embed(answer), embed(gt)) | Fast embedding-level match; use when BERTScore is too slow |

Use the full correctness suite when ground truth is available. Use faithfulness + answer relevance for corpora without ground truth.

#### Performance & Cost Metrics

Tracked on every request (production + eval), not just during evaluation runs.

##### Latency Breakdown

Every request records a span tree. Stored on `EvalResult` for eval runs; on `audit_events.response_ms` for production.

```
total_ms = validation_ms + routing_ms + cache_lookup_ms
         + embed_ms + retrieval_ms + rerank_ms
         + semantic_cache_ms + llm_first_token_ms + llm_stream_ms
```

| Span | Target (P95) | Alert threshold |
|------|-------------|----------------|
| `validation_ms` | < 20 ms | > 50 ms |
| `routing_ms` | < 100 ms | > 300 ms |
| `cache_lookup_ms` (L2 Redis) | < 5 ms | > 20 ms |
| `embed_ms` | < 100 ms | > 300 ms |
| `retrieval_ms` | < 150 ms | > 500 ms |
| `rerank_ms` | < 300 ms (if enabled) | > 1 000 ms |
| `llm_first_token_ms` | < 800 ms | > 2 000 ms |
| `total_ms` | < 2 000 ms | > 5 000 ms |

Latency spans are emitted as OpenTelemetry spans → Langfuse (LLM spans) + X-Ray/Cloud Trace (infra spans).

##### Token Accounting

Every LLM call records token counts from the provider response. Aggregated per corpus, per model tier, per day.

```python
class TokenUsage(BaseModel):
    request_id: UUID
    corpus_id: str
    model_tier: str          # "nano" | "small" | "large"
    model_id: str            # exact model name
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int       # provider-level prompt cache hits (if supported)
    timestamp: datetime
```

Stored in `token_usage` table. Prometheus counter: `llm_tokens_total{tier, model, corpus, type}`.

##### Cost Estimation

Local Ollama models: cost = 0 (track for sizing only). Cloud models: use per-token pricing table.

```python
# knowledge/evaluation/metrics/performance.py
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-6": {"input": 0.003,   "output": 0.015},
    "claude-opus-4-8":   {"input": 0.015,   "output": 0.075},
    # updated as pricing changes; local models omitted (cost = 0)
}

def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = COST_PER_1K_TOKENS.get(model_id)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000
```

Aggregated cost dashboards: daily spend per corpus, per model tier, per user (for multi-tenant billing awareness).

##### Storage Metrics

| Metric | Source | Tracked in |
|--------|--------|-----------|
| `pg_table_bytes{table}` | PostgreSQL `pg_total_relation_size()` | Prometheus via pg_exporter |
| `vector_index_bytes` | `pg_indexes_size('chunks_embedding_idx')` | Prometheus |
| `chunk_count{corpus}` | `COUNT(*) FROM chunks WHERE corpus_id = $1` | Eval run report |
| `redis_memory_bytes` | `INFO memory` → `used_memory` | Prometheus via redis_exporter |
| `redis_keys_total{prefix}` | `SCAN` + pattern count | Prometheus |

Storage cost estimate (cloud): `pg_table_bytes × Aurora_GB_month_price + redis_memory × ElastiCache_GB_month_price`. Refreshed nightly as a background job.

#### Evaluation Pipeline Flow

```
Offline Eval:
  POST /v1/evaluate/run
      │  body: { corpus_id, k, model_tier, search_type, baseline_run_id? }
      │
      └── Publish EvalJob → knowledge:eval Redis stream
              │
              └── Evaluation Worker (knowledge/evaluation/runner.py)
                      ├── Load GoldSamples for corpus from DB
                      ├── For each sample (concurrently, semaphore-limited):
                      │   ├── retriever.retrieve(query, k)        → SearchResult[]
                      │   ├── metrics.retrieval.*                 → hit_rate, mrr, ndcg, p@k, r@k
                      │   ├── agent.run(query, context)           → answer, token counts, latency
                      │   ├── metrics.faithfulness.*              → faithfulness score
                      │   ├── metrics.answer_relevance.*          → relevance score
                      │   ├── metrics.correctness.*               → BLEU/ROUGE/METEOR/BERTScore (if GT)
                      │   └── metrics.performance.*               → latency spans, token costs
                      ├── INSERT eval_results rows
                      ├── UPDATE eval_runs SET status='completed'
                      ├── reporter.generate_report()              → regression diff vs baseline_run_id
                      └── Publish EvalCompleteEvent → knowledge:events

Online Feedback:
  POST /v1/feedback
      └── INSERT user_feedback (async background task)
      └── Publish FeedbackEvent → knowledge:events stream
              └── Online metrics worker
                      ├── Increment Redis counters:
                      │   cache:feedback:{corpus_id}:thumbs_up   INCR
                      │   cache:feedback:{corpus_id}:thumbs_down INCR
                      │   cache:feedback:{corpus_id}:rating_sum  INCRBY rating
                      └── Flush aggregated rows → user_feedback table every 60 s

Implicit Signals:
  Client SDK / middleware emits:
      POST /v1/signals (internal, service token)
      └── INSERT implicit_signals (fire-and-forget)
```

#### Regression Detection (`reporter.py`)

When `baseline_run_id` is provided, `reporter.generate_report()` computes delta for every metric:

```python
delta = current_metric - baseline_metric
regression = delta < -REGRESSION_TOLERANCE[metric]
```

Default tolerances:

| Metric | Tolerance |
|--------|-----------|
| Hit Rate@k | -0.05 (5 pp) |
| MRR@k | -0.05 |
| NDCG@k | -0.05 |
| Faithfulness | -0.05 |
| Answer Relevance | -0.05 |
| ROUGE-L-F | -0.03 |
| P95 Latency | +200 ms |
| Estimated cost/query | +20% |

Report emitted as:
- JSON to `eval_runs.report_json` column
- Markdown summary posted as GitHub PR comment (CI integration)
- Prometheus gauge: `eval_metric{metric, corpus, run_id}` — enables Grafana trend charts

#### CI Integration

Add to GitHub Actions workflow (after unit tests, before staging deploy):

```yaml
- name: Offline eval
  run: |
    python -m knowledge.evaluation.runner \
      --corpus-id $EVAL_CORPUS_ID \
      --baseline-run-id $BASELINE_RUN_ID \
      --fail-on-regression
  env:
    DATABASE_URL: ${{ secrets.STAGING_DATABASE_URL }}
    # ...
```

`--fail-on-regression` exits non-zero if any metric crosses its tolerance → blocks merge.

#### API Endpoints (additions to API Layer)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/evaluate/run` | `admin` | Trigger offline eval run; returns `run_id` |
| `GET`  | `/v1/evaluate/run/{id}` | `admin` | Poll run status + aggregated results |
| `GET`  | `/v1/evaluate/run/{id}/results` | `admin` | Per-sample results (paginated) |
| `GET`  | `/v1/evaluate/compare?a={id}&b={id}` | `admin` | Regression diff between two runs |
| `POST` | `/v1/feedback` | `reader` | Submit explicit user feedback |
| `POST` | `/v1/signals` | `service` | Submit implicit behavioural signal |
| `GET`  | `/v1/metrics/satisfaction` | `admin` | Rolling satisfaction scores per corpus |
| `GET`  | `/v1/metrics/cost` | `admin` | Token + storage cost breakdown |

#### Prometheus Metrics (additions)

```
# Retrieval quality (updated per eval run)
eval_hit_rate{corpus, run_id}
eval_mrr{corpus, run_id}
eval_ndcg{corpus, run_id}

# Generation quality
eval_faithfulness{corpus, run_id}
eval_answer_relevance{corpus, run_id}

# Latency (production, rolling)
request_latency_seconds{stage, corpus}      # histogram; stages: embed, retrieve, rerank, llm
request_total_ms{corpus, tier, cache_hit}   # histogram

# Token & cost
llm_tokens_total{tier, model, corpus, type} # counter; type=prompt|completion|cached
llm_cost_usd_total{tier, model, corpus}     # counter

# Online feedback
feedback_rating_total{corpus}               # counter (sum of all ratings)
feedback_count_total{corpus, sentiment}     # counter; sentiment=positive|negative|neutral
implicit_signals_total{corpus, signal_type} # counter

# Storage
pg_table_bytes{table}                       # gauge (via pg_exporter)
redis_memory_bytes                          # gauge (via redis_exporter)
```

#### Grafana Dashboard Panels (suggested layout)

**Row 1 — Retrieval Quality Trends**
- Line chart: Hit Rate@5 over last 30 eval runs, per corpus
- Line chart: MRR@5 + NDCG@5 over last 30 runs
- Stat panels: current Hit Rate, MRR, NDCG vs baseline delta (red/green)

**Row 2 — Generation Quality**
- Line chart: faithfulness + answer relevance over eval runs
- Stat: % queries with faithfulness < 0.7 (hallucination risk)

**Row 3 — Answer Correctness**
- Line chart: ROUGE-L-F + semantic similarity over eval runs (when GT available)

**Row 4 — Latency Breakdown**
- Heatmap: `request_latency_seconds` by stage
- Line chart: P50 / P95 / P99 total_ms, by model tier
- Stat: SLA compliance (% requests < 2 s)

**Row 5 — Cost**
- Bar chart: daily token usage by model tier
- Line chart: daily estimated cost by corpus
- Stat: cost per 1 000 queries (rolling 7-day average)

**Row 6 — Online Feedback**
- Time series: rolling 7-day satisfaction score per corpus
- Bar: feedback tag distribution (hallucinated / irrelevant / incomplete / correct)
- Table: top-10 lowest-rated request IDs (link to Langfuse trace)

**Row 7 — Storage**
- Area chart: `pg_table_bytes` by table over time
- Stat: Redis memory used vs limit; eviction count

---

### Load & Chaos Testing Strategy

The happy path working locally is table stakes. What matters before production is knowing *exactly* what breaks first, at what load, and with what degradation profile. This section is the pre-production test plan that validates the SLAs from [System Design Constraints](#system-design-constraints) and surfaces failure modes before real users hit them.

#### Philosophy

- Every SLA number in this document is a hypothesis. Load testing converts it to a measurement.
- Break things deliberately, in isolation, before the system breaks in production at the worst time.
- A test that only validates the happy path is not a test — it is optimism.

#### Phase 1 — Baseline Load (single component, no failures)

Goal: establish real throughput and latency curves before any chaos. Run on staging environment with production-equivalent data (ingested gold dataset, ~5K chunks).

**Tool**: `locust` (Python, async-compatible, Grafana-integrated).

```python
# tests/load/locustfile.py
class RAGUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def search(self):
        self.client.post("/api/v1/search", json={
            "query": random.choice(GOLD_QUERIES),
            "corpus_ids": ["acme-corp:hr-policies"],
        }, headers={"Authorization": f"Bearer {JWT}"})

    @task(3)
    def chat(self):
        self.client.post("/api/v1/chat", json={
            "query": random.choice(GOLD_QUERIES),
            "corpus_ids": ["acme-corp:hr-policies"],
        }, headers={"Authorization": f"Bearer {JWT}"})

    @task(1)
    def ingest_small_doc(self):
        self.client.post("/api/v1/ingest", json={
            "corpus_id": "acme-corp:hr-policies",
            "document_url": TEST_DOC_URL,
        }, headers={"Authorization": f"Bearer {JWT}"})
```

**Test matrix** (run each scenario independently, record P50/P95/P99 and error rate):

| Scenario | RPS | Duration | Pass criteria |
|---|---|---|---|
| Baseline — search only | 1 RPS | 5 min | P95 < 600 ms, 0% errors |
| Baseline — chat (small model) | 1 RPS | 5 min | P95 < 2,000 ms, 0% errors |
| Ramp — find breaking point | 1→20 RPS over 10 min | 10 min | Record RPS where error rate > 1% |
| Sustained peak | 5 RPS (design peak) | 30 min | P95 < 2,000 ms, error rate < 0.1% |
| Burst | 0→15 RPS spike for 60s | 5 min | System recovers within 2 min; no DLQ entries |
| Cache warmup | 1 RPS, 100 unique queries | 5 min | L2 hit rate reaches > 10% by end |
| Cache cold | 5 RPS, 1000 unique queries | 10 min | P95 < 2,000 ms (no cache benefit) |

**Deliverable**: Grafana dashboard screenshot + `locust` HTML report committed to `tests/load/results/baseline-{date}.html`. This becomes the regression baseline.

#### Phase 2 — Dependency Failure Injection (chaos)

Goal: verify graceful degradation matrix from [Error Handling Strategy](#error-handling-strategy) holds under load. Each scenario kills one component while load continues at 3 RPS.

**Tool**: `chaos-mesh` (K8s) in staging, or direct `docker compose stop <service>` for local runs. For local runs, the `Makefile` provides targets:

```makefile
chaos-kill-redis:
    docker compose stop redis
    sleep 60
    docker compose start redis

chaos-kill-ollama:
    docker compose stop ollama
    sleep 120
    docker compose start ollama

chaos-kill-postgres:
    docker compose stop postgres
    sleep 30
    docker compose start postgres
```

**Chaos scenario matrix:**

| Component killed | Expected degraded mode | Acceptance criteria |
|---|---|---|
| **Redis** | `no_cache`; rate limiting falls back to DB counter | No 500s; P95 ≤ 2× baseline; `X-Degraded-Mode: no_cache` header present |
| **Ollama (LLM)** | `search_only`; generation returns 503 | Search responses succeed; chat returns `503 LLM_CIRCUIT_OPEN`; circuit opens within 60s; alert email sent |
| **Ollama (recovery)** | Circuit transitions OPEN → HALF-OPEN → CLOSED | Within 90s of Ollama restart, chat requests succeed again |
| **PostgreSQL** | `unavailable`; all endpoints 503 | No data corruption; all in-flight ingest jobs re-enqueue (not lost); on recovery, job processing resumes |
| **AGE graph DB** | `no_graph`; vector+text path only | Queries that require graph return results via vector fallback; `graph_unavailable: true` in response |
| **Ingest worker (all replicas)** | Queue depth grows; jobs stay in stream | No data loss (Redis stream durability); on worker restart, all pending jobs processed |
| **Network partition (Redis ↔ API)** | Same as Redis kill | Handled identically |

**What must NOT happen in any scenario:**
- Unhandled Python exceptions returning 500 (all caught, mapped to error codes)
- Data written to wrong tenant (RLS holds even under degradation)
- DLQ entries accumulating silently without alert
- Circuit breaker state lost on API pod restart (state is in Redis, not in-process)

#### Phase 3 — Sustained Load & Resource Exhaustion

Goal: find the resource ceiling before it finds you.

| Test | Scenario | What we're looking for |
|---|---|---|
| **DB connection pool exhaustion** | 20 RPS sustained for 10 min (above HPA trigger) | Pool waiters observable in `/health`; requests queue rather than crash; P99 degrades gracefully |
| **Redis memory ceiling** | Fill semantic cache to `semantic_cache_max_rows` | Pruning job triggers correctly; no OOM; cache hit rate stable |
| **Embedding API rate limit** | Ingest 500 documents in 10 min | `RateLimitError` triggers backoff; no jobs lost to DLQ; total time < 3h (within P99 SLA) |
| **LLM context overflow** | Send 50 queries with 8000+ token context | Context trimming fires; `context_truncated: true` in response; no 500s |
| **DLQ depth** | Inject 20 permanently-failing ingest jobs | DLQ depth gauge increments; alert fires per entry; `/health` shows `dlq_depth > 0` as degraded |
| **Tenant budget exhaustion** | Exhaust Pro tier LLM budget mid-load-test | Budget guard fires at 100%; chat returns `402`; search continues; alert email sent |

#### Phase 4 — Regression Gate (CI)

Every PR that touches retrieval, generation, or caching runs an automated subset of the load tests:

```yaml
# .github/workflows/load-test.yml (runs on PR to main)
- name: Load regression test
  run: |
    locust -f tests/load/locustfile.py \
      --headless --users 5 --spawn-rate 1 --run-time 3m \
      --host $STAGING_URL \
      --csv tests/load/results/pr-${{ github.sha }} \
      --exit-code-on-error 1
  env:
    LOCUST_FAIL_ON_ERROR_RATE: "0.01"       # fail if > 1% errors
    LOCUST_FAIL_ON_P95_MS: "2000"           # fail if P95 > 2s
```

Result CSV is compared against the baseline; if P95 regresses > 200 ms the PR is blocked (same tolerance as evaluation system regression).

#### Observability During Load Tests

All load tests are run with Langfuse and Prometheus active. Key dashboards to watch:

- **Latency heatmap** by stage — identifies which stage is the bottleneck as load increases
- **Error rate by error code** — distinguishes infrastructure errors from policy rejections
- **Circuit breaker state** — monitors CLOSED/OPEN/HALF-OPEN transitions
- **DLQ depth** — should be zero at all times except during chaos scenarios
- **Cache hit rate** — should stabilise as the load test warms the cache
- **DB pool utilisation** — pool_used / pool_max; alert if > 80% sustained

Load test results are stored in `tests/load/results/` (git-ignored for large CSVs; summaries committed). A Markdown summary is posted as a PR comment by the CI workflow.

---

### Docling-Graph Evaluation Checklist

Validate these items in a spike branch before integrating into the full pipeline. The docling-graph API is `run_pipeline(PipelineConfig(...)) → PipelineContext` — there is no `PipelineOrchestrator` class.

- [ ] **Smoke test with generic ontology** — run `run_pipeline(PipelineConfig(source=sample_pdf, template=GenericDocument, backend="llm", inference="local", provider_override="ollama", model_override="llama3.2:3b", dump_to_disk=False))` and verify `context.knowledge_graph.number_of_nodes() > 0`
- [ ] **Staged contract with small model** — verify `extraction_contract="staged"` with `llama3.2:3b` extracts meaningful entities; compare quality against `extraction_contract="direct"` with the same model; staged should be clearly better
- [ ] **Async thread safety** — confirm `run_pipeline()` can run concurrently in 2 threads via `asyncio.to_thread()` without shared state corruption; check `PipelineConfig` is instantiated fresh per call (it is not a singleton)
- [ ] **AGE import round-trip** — call `age_store.import_docling_graph(context, corpus_id, document_id)` → verify `node_count > 0`; run `run_cypher_query("MATCH (n) RETURN n.name LIMIT 5", corpus_id)` → verify results; confirm **do NOT** use `CypherExporter` (it generates Neo4j syntax incompatible with AGE's `ag_catalog.cypher()` wrapper)
- [ ] **Ontology loader** — upload a custom domain ontology via `POST /v1/corpus/{id}/ontology`; verify `load_ontology()` returns the correct root class; verify extraction uses domain-specific entity types
- [ ] **Generic fallback** — set `corpus_config.graph_ontology_path = None`; verify `load_ontology(None)` returns `GenericDocument`; verify graph still populates
- [ ] **Corpus toggle** — set `enable_graph_extraction=False`; verify `graph_extractor.extract()` returns `None` immediately with no LLM calls
- [ ] **Soft failure** — kill Ollama mid-extraction; verify vector path still completes; verify `graph_extraction_failed: true` in chunk metadata; verify job does NOT go to DLQ
- [ ] **Parallel overhead** — measure wall-clock time for vector-only vs. vector+graph in `asyncio.gather()` on a 20-page PDF; confirm graph task does not extend overall ingest latency beyond 2× (both run in parallel)
- [ ] **VLM extraction** — test `backend="vlm"`, `inference="local"` with a scanned PDF; verify docling's vision pipeline is used; measure extraction quality vs. LLM mode; note VLM requires GPU
- [ ] **Memory footprint** — profile peak RSS with 2 concurrent ingest workers each running `run_pipeline()`; confirm total RSS < 8 GB (the ingest-worker container limit)
- [ ] **Graph query latency** — with 50k entities in AGE (ingested from test corpus), measure NL→Cypher retrieval P99 against the graph retriever

---

### Implementation Phases

> **Phase naming note:** The phases below use letter labels (A–L). The `TODO_implementation.md` file uses numbered phases (0–16) which cover the same work at higher granularity. Cross-reference:
>
> | Design phase | TODO phase | Description |
> |---|---|---|
> | A | 0 | Housekeeping |
> | B | In Progress section | Rate limiting, timeouts, retries |
> | C | 1 + 3 | Module skeleton, config, bus |
> | C2 | 7 | Validation + hooks |
> | D | 4 | Ingestion pipeline |
> | E | 5 | Retrieval + caching |
> | F | 9 | Security layer |
> | G | 8 | API routes |
> | H | 13 | Docker Compose + infra |
> | I | 15 | Cloud IaC (CI/CD, Helm, Terraform) |
> | J | 12 | Evaluation system |
> | K | (embedded in 5 + 6) | Confidence-based scoring |
> | L | 6 | Confidence-aware pipeline |

#### Phase A — Housekeeping (no new features, before any refactor)
- [x] Move `kg/legal/` → `misc/kg_legal_cuad/` (done)
- [x] Move `rag/legal/` → `misc/kg_legal_cuad/rag_data/` (done)
- [x] Move `rag/ingestion/cuad_ingestion.py` → `misc/kg_legal_cuad/` (done)
- [x] Move `rag/tests/ingestion/test_cuad_ingestion.py` → `misc/kg_legal_cuad/tests/` (done)
- [x] Move `rag/tests/knowledge_graph/` → `misc/kg_legal_cuad/tests/kg/` (done)
- [x] Delete `rag/retrieval/dead_code/` (done)
- [ ] Run `python -m pytest rag/tests/ -m "not integration" -v` — confirm no regressions after moves

#### Phase B — Rate Limiting, Timeouts, Retries (in-progress, see section below)
- Complete existing 4-step plan (see "In Progress — Rate Limiting, Timeouts & Retries" section at the bottom of this document) before starting module restructure

#### Phase C — Module Skeleton
- [ ] Create `knowledge/` package with empty modules (no logic yet)
- [ ] Port `settings.py` — add `corpus_configs`, `redis_url`, `jwt_*`, `jwe_*`, `cache_*`, `worker_*` fields
- [ ] Implement `knowledge/bus/` — Redis Streams publisher + consumer base class
- [ ] Write worker harness tests (mock Redis, verify ack/retry/DLQ logic)

#### Phase C2 — Validation + Hooks + Model Router Skeletons
- [ ] Implement `knowledge/hooks/registry.py` — `HookRegistry`, `HookPoint`, `HookContext`
- [ ] Register all built-in placeholder hooks (`audit_log`, `pii_redact`, `response_filter`, `metrics`)
- [ ] Implement `knowledge/validation/pipeline.py` — V1–V4 (schema, length, injection regex); stub V5 content policy
- [ ] Implement `knowledge/agent/model_router.py` — `QueryRouter` using nano model + fallback logic
- [ ] Wire validation → hook `PRE_VALIDATE`/`POST_VALIDATE`/`ON_VALIDATION_FAIL` → router in API request lifecycle
- [ ] Add model tier config fields to `settings.py`

#### Phase D — Ingestion Pipeline Port
- [ ] Port `DocumentIngestionPipeline` → `knowledge/ingestion/pipeline.py` (split into 3 classes per existing debt)
- [ ] Add `docling_processor.py` (thin wrapper; cached converter)
- [ ] Add `graph_extractor.py` (docling-graph wrapper; `asyncio.to_thread` + timeout)
- [ ] Implement parallel chunk + graph paths via `asyncio.gather`
- [ ] Add `knowledge/store/cache.py` (Redis L2 cache)
- [ ] Implement document fingerprint dedup using L2 cache

#### Phase E — Retrieval Port + Caching
- [ ] Port `Retriever` → `knowledge/retrieval/retriever.py` with corpus-scoped queries
- [ ] Implement `knowledge/retrieval/semantic_cache.py` + `semantic_cache` table migration
- [ ] Wire L1 (lru_cache on embedder) + L2 (Redis) + L3 (semantic cache) in retrieval path
- [ ] Add cache observability counters (Prometheus)

#### Phase F — Security Layer
- [ ] Implement `knowledge/api/auth.py` — `require_jwt` dependency, JWKS fetch + cache, RBAC check
- [ ] Implement JWE helpers (encrypt/decrypt answer blobs)
- [ ] Add `knowledge/api/middleware.py` — correlation ID, audit event emission
- [ ] Add input validation (length cap, prompt injection guard)
- [ ] Implement rate limiting (`slowapi`, per-user by JWT `sub`)

#### Phase G — API Port
- [ ] Port FastAPI routes to `knowledge/api/routes/`
- [ ] Add ingest job status routes (poll + SSE)
- [ ] Add corpus admin routes (list, cache invalidate)
- [ ] Add `/metrics` endpoint

#### Phase H — Docker Compose + Local TLS
- [ ] Write `docker-compose.yml` with all services
- [ ] Add `infra/nginx/nginx.conf` with TLS + proxy config
- [ ] Add `Makefile` targets: `make dev`, `make dev-obs`, `make test`, `make migrate`
- [ ] Document model preload in `ollama` container

#### Phase I — Cloud IaC Skeleton
- [ ] Helm chart scaffolding for `api`, `ingest-worker`, `retrieval-worker`
- [ ] GitHub Actions CI workflow (test → build → push → staging deploy)
- [ ] Terraform module for Aurora PG + ElastiCache Redis (or Pulumi, TBD)
- [ ] Secrets Manager integration (CSI driver + projected volumes)

#### Phase J — Evaluation System
- [ ] Define `GoldSample` format + load initial NeuralFlow gold dataset (JSONL in `knowledge/evaluation/data/`)
- [ ] Implement `metrics/retrieval.py` — HitRate, MRR, NDCG, Precision, Recall (port from existing `test_retrieval_metrics.py`)
- [ ] Implement `metrics/performance.py` — latency span recording, token counting, cost estimation
- [ ] Implement `metrics/faithfulness.py` — claim decomposition + NLI verification via nano model
- [ ] Implement `metrics/answer_relevance.py` — reverse-question generation + embedding similarity
- [ ] Implement `metrics/correctness.py` — BLEU, ROUGE, METEOR via `nltk`; BERTScore via `bert-score`
- [ ] Implement `runner.py` — publishes to `knowledge:eval` Redis stream; eval worker consumes
- [ ] Create DB migration for `gold_samples`, `eval_runs`, `eval_results`, `user_feedback`, `implicit_signals`, `token_usage`
- [ ] Add `POST /v1/evaluate/run`, `GET /v1/evaluate/run/{id}`, compare endpoint to API
- [ ] Add `POST /v1/feedback` + `POST /v1/signals` endpoints
- [ ] Implement `reporter.py` — regression detection + Markdown CI report
- [ ] Add `eval-worker` service to Docker Compose
- [ ] Wire regression check into GitHub Actions CI (block merge on regression)
- [ ] Add all eval Prometheus metrics + 7-row Grafana dashboard

#### Phase K — Confidence-Based Scoring
- [ ] Add `raw_score`, `raw_score_type`, `confidence` fields to `SearchResult` in `knowledge/ingestion/models.py`; deprecate bare `similarity`
- [ ] Update `CrossEncoderReranker.rerank()` to populate `confidence` via `sigmoid(logit)` on every result
- [ ] Update semantic standalone path: set `confidence = raw_score` (cosine similarity) when reranker is off
- [ ] Replace `search_type == "semantic"` guardrail in `Retriever` with mode-agnostic `confidence >= min_confidence_score` filter (post-rerank only)
- [ ] Add `min_confidence_score` and `confidence_warn_threshold` to `settings.py`
- [ ] Wire low-confidence flag into agent system prompt: prepend uncertainty notice when best chunk confidence < `confidence_warn_threshold`
- [ ] Add `confidence` field to `Citation` model; map from `SearchResult.confidence`
- [ ] Add `low_confidence_context: bool` to API response envelope
- [ ] Add `mean_confidence`, `min_confidence`, `low_confidence_flag` fields to `EvalResult`
- [ ] Update `metrics/retrieval.py` to record and report confidence distribution per eval run
- [ ] Port changes back to `rag/retrieval/retriever.py` and `rag/retrieval/rerankers.py` for current system (pre-`knowledge/` migration)

#### Phase L — Confidence-Aware Pipeline
- [ ] Implement `knowledge/agent/judge.py` — `LLMJudge`: structured output `JudgeResult(verdict, confidence, reasoning)`; uses nano model; escalates to small if nano verdict confidence < 0.5
- [ ] Implement `knowledge/agent/pipeline.py` — `ConfidenceAwarePipeline` with 3-layer gate logic; `PipelineStatus` enum; `RAGResponse` model with `citations`, `low_confidence_warning`, `abstention_layer`, `pipeline_latency_ms`
- [ ] Extend `Retriever.retrieve_with_confidence()` — compute aggregate confidence sum over top-K; return `[]` if below `retrieval_confidence_threshold`
- [ ] Update `knowledge/agent/agent.py` — structured generation output: `GenerationResult(answer, citations, citation_check: CitationCheck)`; enforce citation system prompt constraint
- [ ] Add `retrieval_confidence_threshold`, `judge_confidence_threshold`, `judge_k` to `settings.py`
- [ ] Wire all 3 gate outcomes into `HookRegistry` (`ON_VALIDATION_FAIL` for abstentions, `POST_LLM` for passes)
- [ ] Update `EvalResult` — add `pipeline_status`, `abstention_layer`, `retrieval_aggregate_confidence`, `citation_trustworthy`, `judge_verdict`, `judge_confidence`, `false_abstention`
- [ ] Add `knowledge/evaluation/metrics/pipeline.py` — abstention rate, false abstention rate, per-layer abstention share, partial answer rate
- [ ] Implement threshold calibration script: sweep `retrieval_confidence_threshold` 0.5→3.0 and `judge_confidence_threshold` 0.4→0.8 over gold dataset; output knee-point recommendations
- [ ] Add abstention rate + false abstention rate panels to Grafana dashboard

---

## In Progress — Rate Limiting, Timeouts & Retries

Settings fields added to `rag/config/settings.py`. Complete before Phase C.

- [ ] **Step 1 — Embedding timeouts + retries** (`rag/ingestion/embedder.py`)
  - Pass `timeout=openai.Timeout(connect=5, read=embedding_timeout_s)` to `AsyncOpenAI`
  - Add exponential-backoff retry on `RateLimitError`, `APIConnectionError`, `APITimeoutError`
  - Settings: `embedding_timeout_s`, `embedding_retry_attempts`, `embedding_retry_backoff_s`

- [ ] **Step 2 — DB query timeouts** (`rag/storage/vector_store/postgres.py`)
  - Pass `timeout=db_query_timeout_s` to every `conn.fetch()` / `conn.fetchrow()` / `conn.execute()`
  - Catch `asyncpg.exceptions.QueryCanceledError` specifically (not bare `Exception`)
  - Settings: `db_query_timeout_s`, `db_health_timeout_s`

- [ ] **Step 3 — LLM call timeout** (`rag/agent/rag_agent.py`, `rag/api/app.py`)
  - Wrap `traced_agent_run` in `asyncio.wait_for(..., timeout=llm_timeout_s)`
  - Return `504 Gateway Timeout` on deadline exceeded (not 500)
  - Settings: `llm_timeout_s`

- [ ] **Step 4 — Inbound API rate limiting** (`rag/api/app.py`)
  - Add `slowapi` middleware; rate-limit `/v1/chat` + `/v1/chat/stream` by IP
  - Return `429 Too Many Requests` with `Retry-After` header
  - Settings: `api_rate_limit_rpm`, `api_rate_limit_burst`

---

## Queued — Production Hardening

Deferred to Phase B/C.

- [ ] Replace bare `except Exception` with specific types across all files
- [ ] Fix `threading.Lock` in async code (`embedder.py`)
- [ ] Add `__aenter__`/`__aexit__` to `DocumentIngestionPipeline`
- [ ] Complete type annotations (`pipeline.py`, `embedder.py`, `rerankers.py`)
- [ ] Extract magic numbers to settings (`lists=100`, `k=60`, cache TTL)
- [ ] Structured logging with `extra={}` fields; correlation ID propagation
- [ ] Pool utilisation metrics in health endpoint

---

## Done

- [x] Move `kg/legal/` + CUAD assets to `misc/kg_legal_cuad/` (`2026-06-04`)
- [x] Delete `rag/retrieval/dead_code/` (`2026-06-04`)
- [x] Metadata filtering during retrieval — `MetadataFilter` model, all search legs, cache key, agent tool (`2026-06-04`)
- [x] Settings fields for rate-limiting/timeouts/retries — `rag/config/settings.py` (`2026-06-04`)
