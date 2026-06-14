# RAG v2 — Test Plan

> Complete test plan for the `backend/knowledge/` module. Covers what each test category verifies, what it requires to run, how to run it, and what constitutes a passing result. For metric formulas and thresholds, see [TEST_QA_REFERENCE.md](TEST_QA_REFERENCE.md).

---

## Table of Contents

- [1. Test Organization](#1-test-organization)
  - [1.1 Category Overview](#11-category-overview)
  - [1.2 Directory Layout](#12-directory-layout)
  - [1.3 Pytest Markers](#13-pytest-markers)
- [2. Unit Tests](#2-unit-tests)
  - [2.1 What They Test](#21-what-they-test)
  - [2.2 Requirements](#22-requirements)
  - [2.3 How to Run](#23-how-to-run)
  - [2.4 File Map](#24-file-map)
  - [2.5 Pass Criteria](#25-pass-criteria)
- [3. Integration Tests](#3-integration-tests)
  - [3.1 What They Test](#31-what-they-test)
  - [3.2 Requirements](#32-requirements)
  - [3.3 How to Run](#33-how-to-run)
  - [3.4 File Map](#34-file-map)
  - [3.5 Pass Criteria](#35-pass-criteria)
- [4. Retrieval Quality Tests](#4-retrieval-quality-tests)
  - [4.1 What They Test](#41-what-they-test)
  - [4.2 Requirements](#42-requirements)
  - [4.3 How to Run](#43-how-to-run)
  - [4.4 Gold Datasets](#44-gold-datasets)
  - [4.5 Pass Criteria](#45-pass-criteria)
- [5. Ingestion Tests](#5-ingestion-tests)
  - [5.1 What They Test](#51-what-they-test)
  - [5.2 Requirements](#52-requirements)
  - [5.3 How to Run](#53-how-to-run)
  - [5.4 Pass Criteria](#54-pass-criteria)
- [6. Agent and Generation Tests](#6-agent-and-generation-tests)
  - [6.1 What They Test](#61-what-they-test)
  - [6.2 Requirements](#62-requirements)
  - [6.3 How to Run](#63-how-to-run)
  - [6.4 Pass Criteria](#64-pass-criteria)
- [7. API Tests](#7-api-tests)
  - [7.1 What They Test](#71-what-they-test)
  - [7.2 Requirements](#72-requirements)
  - [7.3 How to Run](#73-how-to-run)
  - [7.4 Pass Criteria](#74-pass-criteria)
- [8. Evaluation System Tests](#8-evaluation-system-tests)
  - [8.1 Offline Eval Run](#81-offline-eval-run)
  - [8.2 Regression Gate](#82-regression-gate)
  - [8.3 How to Run](#83-how-to-run)
- [9. Load Tests](#9-load-tests)
  - [9.1 What They Test](#91-what-they-test)
  - [9.2 Requirements](#92-requirements)
  - [9.3 How to Run](#93-how-to-run)
  - [9.4 Pass Criteria](#94-pass-criteria)
- [10. Chaos Tests](#10-chaos-tests)
  - [10.1 What They Test](#101-what-they-test)
  - [10.2 How to Run](#102-how-to-run)
  - [10.3 Pass Criteria](#103-pass-criteria)
- [11. Phase Test Gates](#11-phase-test-gates)
- [12. CI Integration](#12-ci-integration)
- [13. Test Data Management](#13-test-data-management)

---

## 1. Test Organization

### 1.1 Category Overview

| Category | Location | External deps | Run in CI | Run time |
|----------|----------|--------------|-----------|---------|
| Unit | `tests/unit/` | None | Yes — every PR | < 30 s |
| Integration | `tests/integration/` | PostgreSQL + Redis | Yes — every PR (services via Docker) | 2–5 min |
| Retrieval quality | `tests/retrieval/` | PostgreSQL + Ollama + ingested data | Yes — every PR | 3–8 min |
| Ingestion | `tests/ingestion/` | PostgreSQL + Redis + Ollama | Yes — every PR | 5–10 min |
| Agent / generation | `tests/agent/` | PostgreSQL + Redis + Ollama | Yes — every PR | 5–10 min |
| API | `tests/api/` | Full stack (all services) | Yes — every PR | 2–4 min |
| Offline eval | `knowledge/evaluation/` | Full stack + gold dataset | Yes — every PR (non-blocking on first run) | 5–20 min |
| Load | `tests/load/` | Full staging stack | Manual / weekly | 30–60 min |
| Chaos | `tests/chaos/` | Full staging stack | Manual / pre-release | 1–2 h |

---

### 1.2 Directory Layout

```
backend/
└── tests/
    ├── conftest.py                    # shared fixtures: DB pool, Redis client, test corpus setup
    ├── unit/
    │   ├── test_backoff.py            # exponential_backoff() math and jitter
    │   ├── test_circuit_breaker.py    # CircuitBreaker state machine (fakeredis)
    │   ├── test_fusion.py             # RRF math, confidence assignment, confidence filter
    │   ├── test_validation.py         # V1–V4 validation chain; content policy stub
    │   ├── test_quota.py              # enforce_quota() Redis counter logic (fakeredis)
    │   ├── test_scheduler.py          # cron next-run computation, get_due_jobs logic
    │   ├── test_chunker.py            # DoclingHybridChunker with mock DoclingDocument
    │   ├── test_models.py             # Pydantic model validation, field constraints
    │   └── test_auth.py               # JWT decode, RBAC check, expired token handling
    ├── integration/
    │   ├── test_vector_store.py       # upsert, semantic search, text search, RRF, corpus isolation
    │   ├── test_cache.py              # Redis L2 cache: get/set/invalidate by corpus
    │   ├── test_semantic_cache.py     # L3 semantic cache: lookup, store, prune, threshold
    │   ├── test_ingestion_pipeline.py # end-to-end: file → Docling → chunks → DB
    │   ├── test_retrieval_pipeline.py # end-to-end: query → all cache layers → ranked results
    │   ├── test_agent.py              # confidence-aware pipeline: 3-layer gate, streaming
    │   └── test_api.py                # all REST endpoints: status codes, SSE, error envelopes
    ├── retrieval/
    │   ├── test_retrieval_metrics.py  # gold dataset eval: Hit Rate, MRR, NDCG, P@K, R@K
    │   └── test_legal_retrieval.py    # CUAD legal gold dataset + corpus isolation checks
    ├── ingestion/
    │   ├── test_docling_processor.py  # PDF vs standard converter routing; VLM toggle
    │   ├── test_incremental.py        # hash-based skip, modify, delete cycle
    │   └── test_audio.py              # Whisper ASR fallback behavior
    ├── agent/
    │   ├── test_rag_agent.py          # agent.run() on known queries; answer correctness spot check
    │   ├── test_streaming.py          # agent.run_stream(); SSE event sequence validation
    │   ├── test_judge.py              # LLMJudge: verdict paths, nano→small escalation
    │   └── test_pipeline.py           # ConfidenceAwarePipeline: abstention, citation gate, partial
    ├── api/
    │   ├── test_chat_endpoints.py     # POST /v1/chat, GET /v1/chat/stream (SSE)
    │   ├── test_ingest_endpoints.py   # POST /v1/ingest, status poll, SSE progress
    │   ├── test_search_endpoint.py    # POST /v1/search; cache hit/miss headers
    │   ├── test_corpus_endpoints.py   # GET /v1/corpus, cache invalidate
    │   ├── test_scheduler_endpoints.py# CRUD for scheduled jobs, run-now
    │   ├── test_auth_endpoints.py     # JWT login, refresh, expired token 401
    │   ├── test_health_endpoint.py    # /health: healthy, degraded, unhealthy states
    │   └── test_error_envelopes.py    # every error code returns correct HTTP status + body
    ├── load/
    │   ├── locustfile.py              # RAGUser with search/chat/ingest tasks
    │   └── results/                   # committed summaries; large CSVs git-ignored
    └── chaos/
        ├── test_redis_kill.py         # kill Redis; verify no_cache mode; verify recovery
        ├── test_ollama_kill.py        # kill Ollama; verify search_only mode; verify circuit
        ├── test_postgres_kill.py      # kill PostgreSQL; verify 503; verify job queue integrity
        └── test_worker_kill.py        # kill all ingest workers; verify queue persists; verify resume
```

---

### 1.3 Pytest Markers

Defined in `backend/pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "strict"
markers = [
    "unit: pure unit test, no external deps",
    "integration: requires PostgreSQL + Redis",
    "retrieval: requires PostgreSQL + Ollama + ingested data",
    "ingestion: requires PostgreSQL + Redis + Ollama",
    "agent: requires full stack (PostgreSQL + Redis + Ollama)",
    "api: requires full running API server",
    "load: load test, run manually only",
    "chaos: chaos test, run manually only",
]
```

Usage:

```bash
# Run only unit tests (fastest, no deps)
pytest tests/unit/ -v

# Run without load and chaos tests
pytest tests/ -m "not load and not chaos" -v

# Run only integration tests
pytest tests/integration/ -v

# Run retrieval quality gate
pytest tests/retrieval/ -v --log-cli-level=INFO
```

---

## 2. Unit Tests

### 2.1 What They Test

Pure logic with no I/O. These tests run against the actual code but mock or fake all external dependencies (database, Redis, LLM, Docling). They verify:

- Mathematical correctness of metric functions (RRF, NDCG, sigmoid calibration)
- State machine logic (circuit breaker transitions, consumer retry/DLQ)
- Validation chain rejection conditions
- Redis quota counter logic (using `fakeredis`)
- JWT decode and RBAC checks (using a test RSA keypair)
- Cron expression computation and `get_due_jobs` filtering
- Pydantic model validation and field constraints

### 2.2 Requirements

None. No running services needed. `fakeredis` is used for Redis-dependent tests.

```bash
pip install fakeredis  # already in dev-dependencies via uv
```

### 2.3 How to Run

```bash
# From backend/
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ -v --cov=knowledge --cov-report=term-missing
```

Expected runtime: < 30 seconds.

### 2.4 File Map

| File | What it tests |
|------|--------------|
| `test_backoff.py` | `exponential_backoff()`: schedule for 3 attempts; jitter within bounds; max cap |
| `test_circuit_breaker.py` | CLOSED→OPEN on 5 failures; OPEN→HALF-OPEN after probe interval; HALF-OPEN→CLOSED on success; HALF-OPEN→OPEN on failure; state in Redis (fakeredis) |
| `test_fusion.py` | RRF score formula `1/(60+rank)`; confidence = `sigmoid(logit)`; confidence filter removes results below threshold; standalone semantic path sets `confidence = cosine` |
| `test_validation.py` | V1: Pydantic rejects malformed body; V2: length > MAX_QUERY_CHARS rejected; V4: regex injection patterns rejected; V5 content policy: stub returns all three verdicts |
| `test_quota.py` | DAILY_QUOTA_EXCEEDED fires when counter > limit; RATE_LIMIT_EXCEEDED fires on RPM breach; LLM_NOT_ENABLED_ON_FREE_TIER fires for chat on free tier; headers set correctly |
| `test_scheduler.py` | `compute_next_run_at()` for daily/weekly/cron expressions; `get_due_jobs()` returns only past-due active jobs; `run-now` publishes immediately regardless of `next_run_at` |
| `test_chunker.py` | `DoclingHybridChunker` with a mock `DoclingDocument`; `contextualize()` output differs from `chunk.text`; fallback fires when no DoclingDocument; token counts within `max_tokens` |
| `test_models.py` | `SearchResult.confidence` must be 0–1 or `None`; `IngestJob.mode` must be `"full"` or `"incremental"`; `Citation.relevance_score` cannot be negative |
| `test_auth.py` | Valid JWT passes; expired JWT returns 401; JWT with wrong `kid` returns 401; correct role passes RBAC; wrong role returns 403 |

### 2.5 Pass Criteria

- 0 failures
- 0 warnings that indicate logic errors (deprecation warnings from third-party libs are acceptable)
- Line coverage ≥ 80% for `knowledge/bus/`, `knowledge/validation/`, `knowledge/api/auth.py`

---

## 3. Integration Tests

### 3.1 What They Test

End-to-end behaviour of individual layers against real services. No mocking of infrastructure. Each test brings up its fixtures, performs operations, and asserts database or cache state.

Key scenarios covered:

- Vector store upserts chunks; retrieves by corpus_id; confirms cross-corpus isolation (tenant A cannot see tenant B's chunks)
- Redis L2 cache: set embedding cache, get on second call, delete by corpus invalidates relevant keys only
- L3 semantic cache: store an answer; retrieve with cosine ≥ 0.95 threshold; miss on cosine < 0.95; prune fires when row count exceeds max
- Ingestion pipeline: ingest a Markdown file; verify chunk count, token counts, `corpus_id`, `tenant_id`, metadata fields
- Retrieval pipeline: query returns results with `confidence` populated; L2 cache miss on first call, hit on second identical call; low-confidence results filtered out
- Agent: `ConfidenceAwarePipeline.run()` returns `RAGResponse` with correct `status`; `agent.run_stream()` yields delta events

### 3.2 Requirements

- PostgreSQL running (with pgvector extension enabled)
- Redis running
- For agent tests: Ollama running with `llama3.2:3b` and `nomic-embed-text:latest` pulled

Start with Docker Compose:

```bash
cd backend && docker compose up postgres redis ollama -d
```

Or use the `conftest.py` fixtures which auto-skip when services are unreachable:

```python
# conftest.py
@pytest_asyncio.fixture
async def pg_pool():
    try:
        pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=3)
        yield pool
        await pool.close()
    except (asyncpg.exceptions.ConnectionFailure, OSError):
        pytest.skip("PostgreSQL not reachable")
```

### 3.3 How to Run

```bash
# All integration tests
pytest tests/integration/ -v --log-cli-level=INFO

# Skip agent tests (requires Ollama)
pytest tests/integration/ -v -m "integration and not agent"

# Single file
pytest tests/integration/test_vector_store.py -v
```

### 3.4 File Map

| File | What it tests |
|------|--------------|
| `test_vector_store.py` | upsert → semantic_search → text_search → hybrid RRF → corpus_id isolation → tenant RLS → delete_by_corpus |
| `test_cache.py` | L2 embedding cache hit/miss; search cache hit/miss; invalidate_corpus deletes only that corpus's keys |
| `test_semantic_cache.py` | Store + retrieve at cosine ≥ 0.95 threshold; miss at cosine < 0.95; JWE encrypt/decrypt round-trip; prune fires and row count drops |
| `test_ingestion_pipeline.py` | Ingest `rag/documents/` sample; verify chunk count, `has_context=True`, corpus_id, YAML frontmatter in metadata; incremental mode skips unchanged; modified file re-ingested |
| `test_retrieval_pipeline.py` | Query against ingested data; confidence values non-None after rerank; L2 cache miss then hit; corpus isolation |
| `test_agent.py` | `ConfidenceAwarePipeline.run()` on known query returns `status="answered"`; empty corpus returns `status="abstained_retrieval"`; `run_stream()` yields delta events |
| `test_api.py` | HTTP client against live API server; all routes return correct status codes; SSE endpoints stream events; error envelopes match schema |

### 3.5 Pass Criteria

- 0 failures
- All corpus isolation assertions pass (cross-tenant data leakage is a critical bug)
- L3 semantic cache JWE round-trip succeeds (encrypted answer decrypts correctly)

---

## 4. Retrieval Quality Tests

### 4.1 What They Test

Retrieval quality against gold datasets. These tests measure whether the retriever surfaces the right documents for real user queries. They run the full retrieval stack (embedding → hybrid search → rerank) and compute IR metrics.

Two gold datasets:

1. **NeuralFlow AI corpus** — 10 queries about a fictional AI company's documents
2. **CUAD legal corpus** — 10 queries about contract clause types (requires separate ingestion)

Tests also verify **corpus isolation**: a legal query must not surface NeuralFlow documents, and vice versa.

### 4.2 Requirements

- PostgreSQL running with both corpora ingested
- Ollama running with `nomic-embed-text:latest` (embeddings) and the configured reranker

Ingest the NeuralFlow corpus before running:

```bash
cd backend && python -m knowledge.ingestion.worker --mode full --source rag/documents/ --corpus-id neuralflow:default
```

### 4.3 How to Run

```bash
# Full retrieval quality suite with metric output
pytest tests/retrieval/ -v --log-cli-level=INFO --tb=short

# NeuralFlow corpus only
pytest tests/retrieval/test_retrieval_metrics.py -v --log-cli-level=INFO

# Legal corpus only (requires CUAD ingestion)
pytest tests/retrieval/test_legal_retrieval.py -v -m "integration" --log-cli-level=INFO
```

Output includes a metrics table logged to INFO:

```
====================================================================
  RETRIEVAL METRICS — hybrid search, NeuralFlow AI corpus
====================================================================
  Metric              K=1       K=3       K=5
-------------------------------------------------
  HIT_RATE@K         0.700     0.800     0.900
  MRR@K              0.700     0.733     0.733
  PRECISION@K        0.700     0.300     0.200
  RECALL@K           0.700     0.750     0.850
  NDCG@K             0.700     0.772     0.798
-------------------------------------------------
  Mean latency                           342ms
  P95  latency                           589ms
====================================================================
```

### 4.4 Gold Datasets

| Dataset | File | Queries | Corpus |
|---------|------|---------|--------|
| NeuralFlow AI | `knowledge/evaluation/data/neuralflow_default.jsonl` | 10 | `neuralflow:default` |
| CUAD Legal | `knowledge/evaluation/data/legal_cuad.jsonl` | 10 | `legal:cuad` |

For JSONL format, see [TEST_QA_REFERENCE.md — §8 Gold Dataset Format](TEST_QA_REFERENCE.md#8-gold-dataset-format).

The same pure metric functions (`hit_rate`, `mrr`, `ndcg_at_k`, etc.) are used in both test files and the evaluation harness. They live in `tests/retrieval/test_retrieval_metrics.py` and are imported by `test_legal_retrieval.py` and `knowledge/evaluation/metrics/retrieval.py`.

### 4.5 Pass Criteria

All of the following must pass for a PR to merge:

| Check | Threshold |
|-------|-----------|
| NeuralFlow Hit Rate@5 | ≥ 0.60 |
| NeuralFlow MRR@5 | ≥ 0.40 |
| NeuralFlow NDCG@5 | ≥ 0.40 |
| NeuralFlow Precision@5 | ≥ 0.15 |
| NeuralFlow Recall@5 | ≥ 0.40 |
| Retrieval P95 latency | ≤ 10,000 ms |
| Semantic Hit Rate@5 | ≥ 0.40 |
| Text Hit Rate@5 | ≥ 0.40 |
| Hybrid ≥ Semantic − 10pp | Hit Rate@5 |
| Corpus isolation (legal query) | 0 NeuralFlow docs in top-5 |
| Corpus isolation (company query) | 0 legal docs in top-5 |

For full metric definitions and formulas, see [TEST_QA_REFERENCE.md — §1](TEST_QA_REFERENCE.md#1-retrieval-metrics).

---

## 5. Ingestion Tests

### 5.1 What They Test

Correctness of the ingestion pipeline for each document type and ingestion mode.

| Test scenario | What is verified |
|--------------|-----------------|
| PDF ingestion (no VLM) | Chunks created; `chunk_method="hybrid"`; token counts ≤ `max_tokens` |
| PDF ingestion (VLM enabled) | Same as above + figure description text present in at least 1 chunk |
| DOCX ingestion | Standard converter used; no VLM call made |
| Audio ingestion | Whisper ASR output stored; `chunk_method="hybrid"` or fallback |
| Audio — missing FFmpeg | Error placeholder chunk stored; no exception propagated |
| Markdown with YAML frontmatter | Frontmatter fields present in chunk metadata |
| Incremental — unchanged file | `skipped=True`; DB row count unchanged; Redis fingerprint hit |
| Incremental — modified file | Old chunks deleted; new chunks inserted; `content_hash` updated |
| Incremental — deleted file | After file deletion + re-run, document and chunks removed from DB |
| Graph extraction enabled | Entity count > 0; `graph_extraction_failed` not set in metadata |
| Graph extraction disabled | No graph store calls; `enable_graph_extraction=False` is a no-op |
| Graph extraction timeout | Graph fails; vector path completes; `graph_extraction_failed: true` in metadata |
| Corpus scoping | All chunks have correct `corpus_id` and `tenant_id` |
| Contextualization | `contextualize()` output ≠ raw `chunk.text`; heading context prepended |

### 5.2 Requirements

- PostgreSQL running
- Redis running
- Ollama running with `nomic-embed-text:latest` (for embeddings)
- For VLM test: Ollama with `qwen2.5vl:7b` pulled and `VLM_ENABLED=true`
- For audio test: FFmpeg in PATH and Whisper installed (or test verifies graceful failure)

### 5.3 How to Run

```bash
# All ingestion tests
pytest tests/ingestion/ -v --log-cli-level=INFO

# Incremental ingestion tests only
pytest tests/ingestion/test_incremental.py -v

# Audio tests (graceful failure if deps missing)
pytest tests/ingestion/test_audio.py -v
```

### 5.4 Pass Criteria

- 0 empty chunks created (empty content is a processing error)
- All chunks have `token_count > 0` and `token_count <= max_tokens`
- Contextualized chunks: `contextualize(chunk)` output is longer than `chunk.text`
- Incremental skip: DB chunk count is identical before and after re-running on unchanged corpus
- Corpus isolation: no chunks have a different `corpus_id` than the job's `corpus_id`
- Graph extraction failure: job completes as `succeeded`; not promoted to DLQ; `graph_extraction_failed: true` in affected chunk metadata

---

## 6. Agent and Generation Tests

### 6.1 What They Test

The full generation pipeline: retrieval → confidence gate → agent → citation check → judge → response.

| Test scenario | What is verified |
|--------------|-----------------|
| Known query, data ingested | `status="answered"`; `citations` list non-empty; each citation has `confidence > 0` |
| Empty corpus (no data ingested) | `status="abstained_retrieval"`; `citations=None`; no LLM call made |
| Agent returns uncited claim | `status="abstained_citation"`; `citation_check.is_trustworthy=False` |
| Judge returns unsupported | `status="abstained_judge"`; `abstention_layer=3` |
| Judge returns partial | `status="answered"`; `low_confidence_warning=True`; uncertainty note appended |
| SSE streaming | `agent.run_stream()` yields `{"delta": "..."}` events; final event has `{"citations": [...], "done": true}` |
| Multi-turn conversation | `message_history` passed to `agent.run()`; answer reflects prior context |
| Langfuse trace | When `LANGFUSE_ENABLED=true`, a trace appears in Langfuse after `traced_agent_run()` |
| Model tier routing | `complexity="simple"` routes to nano; `complexity="complex"` routes to large |
| Cost guard | Tenant at 100% budget receives 402; system budget breach returns 503 |

### 6.2 Requirements

- PostgreSQL running with NeuralFlow corpus ingested
- Redis running
- Ollama running with `llama3.2:3b`, `nomic-embed-text:latest`, `qwen2.5:0.5b` (nano tier)
- For Langfuse test: Langfuse running (optional; skipped if not available)

### 6.3 How to Run

```bash
# All agent tests
pytest tests/agent/ -v --log-cli-level=INFO

# Streaming test only
pytest tests/agent/test_streaming.py -v

# Pipeline abstention tests
pytest tests/agent/test_pipeline.py -v
```

### 6.4 Pass Criteria

- Known-query test: `status="answered"` with at least 1 citation
- All citations have `relevance_score` (= `confidence`) between 0.0 and 1.0
- SSE streaming: events arrive in correct order: `delta` events, then `citations+done`
- No test produces an unhandled exception — all error states produce structured abstention responses
- Corpus isolation: agent does not use context from a corpus the test's JWT role cannot access

---

## 7. API Tests

### 7.1 What They Test

The HTTP surface: correct status codes, response envelopes, error codes, SSE event format, auth enforcement, and rate limiting headers.

All API tests run against a live FastAPI test client (`httpx.AsyncClient(app=app, base_url="http://test")`). No external services are mocked except at the integration boundary — DB and Redis must be running.

| Area | Tests |
|------|-------|
| Chat endpoints | `POST /v1/chat` returns `ChatResponse`; `GET /v1/chat/stream` streams SSE events; invalid JWT returns 401; wrong corpus role returns 403 |
| Ingest endpoints | `POST /v1/ingest` returns job_id; `GET /v1/ingest/{id}/status` returns correct status; SSE progress stream sends events |
| Search endpoint | `POST /v1/search` returns results with confidence; L2 cache hit sets `cache_hit: "l2"` in response |
| Corpus endpoints | `GET /v1/corpus` returns only corpora accessible to JWT role; `POST /v1/corpus/{id}/cache/invalidate` requires admin role |
| Scheduler endpoints | Full CRUD for scheduled jobs; `POST .../run-now` publishes to Redis stream |
| Health endpoint | Returns 200 when all healthy; 503 with degraded component info when any is down |
| Error envelopes | Every error returns `{"request_id": ..., "data": null, "error": {"code": ..., "message": ...}}`; 500 is never returned for expected errors |
| Rate limiting | Exceeding RPM returns 429 with `Retry-After` and `X-RateLimit-*` headers |
| Budget headers | `X-Budget-Warning: 0.80` appears when tenant is at 80% of budget |

### 7.2 Requirements

- PostgreSQL running
- Redis running
- Ollama running (for chat/search tests that go beyond cache)

### 7.3 How to Run

```bash
pytest tests/api/ -v --log-cli-level=INFO
```

### 7.4 Pass Criteria

- Every route returns the documented HTTP status code for its happy path
- Every expected error returns a structured `ErrorDetail` body (never a bare string)
- HTTP 500 never appears in any test (all expected errors map to 400–503)
- SSE endpoints: `Content-Type: text/event-stream` header present
- Auth: all non-health routes return 401 with no token; 403 with insufficient role

---

## 8. Evaluation System Tests

### 8.1 Offline Eval Run

The offline evaluation system runs the full pipeline (retrieval + generation + metrics) against the gold dataset and stores results in the DB. It is triggered via API and run as a worker job.

**What it produces:**
- Per-sample `EvalResult` rows in `eval_results` table
- Aggregated metrics in `eval_runs.report_json`
- Prometheus gauge updates (`eval_hit_rate{corpus, run_id}`, etc.)
- Markdown regression report (for GitHub PR comment)

**How to trigger:**

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/evaluate/run \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"corpus_id": "neuralflow:default", "k": 5, "model_tier": "small", "baseline_run_id": "<previous_run_id>"}'

# Poll status
curl http://localhost:8000/api/v1/evaluate/run/<run_id>

# Via CLI (for CI)
python -m knowledge.evaluation.runner \
  --corpus-id neuralflow:default \
  --baseline-run-id <previous_run_id> \
  --fail-on-regression
```

### 8.2 Regression Gate

A regression is declared when any metric drops beyond its tolerance vs. the baseline run. See [TEST_QA_REFERENCE.md — §9](TEST_QA_REFERENCE.md#9-regression-thresholds) for the full tolerance table.

**CI behaviour:**
- `--fail-on-regression` exits non-zero if any metric regresses → blocks PR merge
- On the first run for a new corpus (no baseline), eval runs but does not fail
- Regression report is posted as a GitHub PR comment by the CI workflow

### 8.3 How to Run

```bash
# Single eval run with regression check
python -m knowledge.evaluation.runner \
  --corpus-id neuralflow:default \
  --baseline-run-id $(cat .last_baseline_run_id) \
  --fail-on-regression

# View regression diff
curl "http://localhost:8000/api/v1/evaluate/compare?a=$BASELINE_ID&b=$CURRENT_ID"
```

---

## 9. Load Tests

### 9.1 What They Test

System throughput, latency under load, and resource consumption. Run against the staging environment with production-equivalent data (≥ 5,000 ingested chunks).

Load tests validate that the SLA numbers in [TEST_QA_REFERENCE.md — §6](TEST_QA_REFERENCE.md#6-scale-test-plan) are measurements, not assumptions.

### 9.2 Requirements

- Full staging stack running (all Docker Compose services)
- ≥ 5,000 chunks ingested
- Prometheus + Grafana running (`--profile observability`)
- Valid JWT for the test user

### 9.3 How to Run

```bash
cd backend/tests/load/

# Baseline search at 1 RPS for 5 min
locust -f locustfile.py \
  --headless --users 1 --spawn-rate 1 --run-time 5m \
  --host https://localhost \
  --csv results/baseline-search-$(date +%F)

# Sustained peak at 5 RPS for 30 min
locust -f locustfile.py \
  --headless --users 5 --spawn-rate 1 --run-time 30m \
  --host https://localhost \
  --csv results/sustained-peak-$(date +%F) \
  --exit-code-on-error 1

# Ramp test (find breaking point)
locust -f locustfile.py \
  --headless --users 20 --spawn-rate 2 --run-time 10m \
  --host https://localhost \
  --csv results/ramp-$(date +%F)
```

### 9.4 Pass Criteria

See [TEST_QA_REFERENCE.md — §6.2 Baseline Load Scenarios](TEST_QA_REFERENCE.md#62-baseline-load-scenarios) for the full table. Summary:

| Scenario | Pass condition |
|----------|--------------|
| Baseline search (1 RPS) | P95 < 600 ms; 0% errors |
| Baseline chat (1 RPS) | P95 < 2,000 ms; 0% errors |
| Sustained peak (5 RPS, 30 min) | P95 < 2,000 ms; error rate < 0.1% |
| Burst (0 → 15 RPS for 60 s) | Recovery within 2 min; 0 DLQ entries |

Commit `tests/load/results/<scenario>-<date>.md` (summary only, not raw CSV) after each run.

---

## 10. Chaos Tests

### 10.1 What They Test

System behaviour when individual components fail. Each test kills one service, applies 3 RPS load for 60 s, then restarts the service and verifies recovery.

Chaos tests verify the **graceful degradation matrix** from [TEST_QA_REFERENCE.md — §7](TEST_QA_REFERENCE.md#7-chaos-and-resilience-test-plan).

### 10.2 How to Run

```bash
# Kill Redis, run load for 60s, restart, verify recovery
make chaos-kill-redis

# Kill Ollama (LLM), run load, restart, verify circuit recovery
make chaos-kill-ollama

# Kill PostgreSQL, run load, restart, verify job queue integrity
make chaos-kill-postgres
```

Each `make chaos-*` target:
1. Stops the Docker service
2. Runs 3 RPS `locust` load for 60 s
3. Restarts the service
4. Runs verification assertions (correct degraded-mode headers, 0 HTTP 500s, recovery within SLA)

### 10.3 Pass Criteria

See [TEST_QA_REFERENCE.md — §7.2](TEST_QA_REFERENCE.md#72-recovery-acceptance-criteria) for the full list. Non-negotiable:

1. **No HTTP 500s** during failure — every error has a structured code
2. **`X-Degraded-Mode` header** present on every response during degradation
3. **No data corruption** — cross-tenant data never visible after any failure
4. **Alert email** sent to `rohan.vazirani@gmail.com` within 60 s of circuit opening
5. **DLQ depth = 0** after all services restored

---

## 11. Phase Test Gates

Each implementation phase must pass its test gate before the next phase begins. These are the checkpoints from `TODO_implementation.md`, stated as runnable commands.

| Phase | Gate command | What must pass |
|-------|-------------|----------------|
| 0 — Scaffold | `pytest rag/tests/ -m "not integration" -v` | Existing v1 tests: 0 regressions |
| 1 — Config + Migrations | `make migrate && psql -c "SELECT 1 FROM chunks LIMIT 1"` | All migrations apply cleanly; tables exist |
| 2 — Storage | `pytest tests/integration/test_vector_store.py tests/integration/test_cache.py -v` | upsert, search, RLS isolation all pass |
| 3 — Message Bus | `pytest tests/unit/test_backoff.py tests/unit/test_circuit_breaker.py tests/unit/test_consumer.py -v` | All worker lifecycle tests pass without live Redis |
| 4 — Ingestion | `pytest tests/integration/test_ingestion_pipeline.py tests/ingestion/ -v` | File → chunks → DB round-trip; incremental skip; graph disabled path |
| 5 — Retrieval | `pytest tests/integration/test_retrieval_pipeline.py tests/integration/test_semantic_cache.py -v` | Confidence populated; L2 hit on repeat; corpus isolation |
| 6 — Agent | `pytest tests/integration/test_agent.py tests/agent/test_pipeline.py -v` | answered + abstained_retrieval + streaming all work |
| 7 — Validation + Hooks | `pytest tests/unit/test_validation.py -v && pytest tests/integration/test_api.py::test_content_policy_rejection -v` | All V1–V5 rejection paths return correct status codes |
| 8 — API | `pytest tests/api/ -v` | All routes: correct status codes; error envelopes; SSE works |
| 9 — Security | `pytest tests/unit/test_auth.py tests/unit/test_quota.py tests/api/test_auth_endpoints.py -v` | JWT auth, RBAC, rate limiting all enforced |
| 10 — Scheduler | `pytest tests/unit/test_scheduler.py tests/api/test_scheduler_endpoints.py -v` | CRUD + run-now + cron computation |
| 11 — Observability | `curl http://localhost:8000/metrics \| grep rag_` | Prometheus scrape returns defined metrics |
| 12 — Evaluation | `python -m knowledge.evaluation.runner --corpus-id neuralflow:default` | Eval run completes; metrics written to DB |
| 13 — Docker + Infra | `docker compose up -d && curl -k https://localhost/health` | All services healthy; Nginx proxies correctly |
| 14 — Frontend | Manual: open browser → send chat → SSE streams → ingestion panel works | Visual verification; no console errors |
| 15 — CI/CD | GitHub Actions workflow runs green on a test PR | All CI steps pass |
| 16 — Load + Chaos | Baseline and sustained-peak load tests; Redis + Ollama chaos scenarios | All scenarios meet pass criteria from §9 and §10 |

---

## 12. CI Integration

### On every PR

```yaml
# .github/workflows/ci.yml (relevant steps)
steps:
  - name: Lint and type check
    run: make v2-check   # ruff fix + mypy backend/knowledge/

  - name: Unit tests
    run: pytest tests/unit/ -v --tb=short

  - name: Integration tests
    run: pytest tests/integration/ tests/retrieval/ tests/ingestion/ tests/agent/ tests/api/ -v --tb=short
    env:
      DATABASE_URL: ${{ secrets.CI_DATABASE_URL }}
      REDIS_URL: redis://localhost:6379
      OLLAMA_BASE_URL: http://localhost:11434/v1

  - name: Offline eval with regression gate
    run: |
      python -m knowledge.evaluation.runner \
        --corpus-id neuralflow:default \
        --baseline-run-id ${{ vars.BASELINE_EVAL_RUN_ID }} \
        --fail-on-regression

  - name: Load regression (lightweight)
    run: |
      locust -f tests/load/locustfile.py \
        --headless --users 5 --spawn-rate 1 --run-time 3m \
        --host $STAGING_URL \
        --exit-code-on-error 1
    env:
      LOCUST_FAIL_ON_ERROR_RATE: "0.01"
      LOCUST_FAIL_ON_P95_MS: "2000"
```

### Blocking conditions (PR cannot merge if any of these fail)

1. `ruff check` or `mypy` errors in `backend/knowledge/`
2. Any unit test failure
3. Any integration test failure
4. Any retrieval quality metric below threshold
5. Offline eval regression on any metric (tolerance in §9.1 of TEST_QA_REFERENCE.md)
6. Load test P95 > 2,000 ms or error rate > 1%

### Non-blocking (reported but not blocking on first run)

- Corpus isolation test when CUAD corpus is not ingested (skipped, not failed)
- Langfuse trace test when Langfuse is not configured
- VLM ingestion test when `VLM_ENABLED=false`

---

## 13. Test Data Management

### Document corpus for tests

The NeuralFlow AI documents in `rag/documents/` are the standard test corpus. All retrieval, agent, and generation tests assume these documents are ingested in the `neuralflow:default` corpus.

**Ingest before running integration/retrieval/agent tests:**

```bash
cd backend
python -m knowledge.ingestion.worker \
  --mode full \
  --source ../rag/documents/ \
  --corpus-id neuralflow:default \
  --tenant-id test-tenant
```

### Gold dataset files

Located in `backend/knowledge/evaluation/data/`. These are committed to the repo and version-controlled. Do not modify gold samples mid-sprint — changes to `relevant_doc_sources` or `ground_truth_answer` may invalidate the baseline eval run used for regression comparison.

Process for updating gold samples:
1. Create a new eval run with the updated dataset
2. Confirm the new run's metrics are acceptable
3. Set this run as the new `BASELINE_EVAL_RUN_ID` in CI vars
4. Commit the updated `.jsonl` file and the new baseline run ID together

### Test isolation

- Each test that writes to the DB uses a unique `tenant_id` and `corpus_id` generated in the test fixture
- All test data is cleaned up in fixture teardown (`ON DELETE CASCADE` handles chunks automatically)
- Redis keys use the test `tenant_id` prefix; `conftest.py` flushes keys with that prefix after each test session
- Tests never share a corpus — parallel test runs are safe because corpus_id namespaces the data
