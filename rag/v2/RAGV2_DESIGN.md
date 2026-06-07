# RAG v2 — System Design

## Table of Contents

- [Architecture Proposal — Enterprise RAG v2](#architecture-proposal--enterprise-rag-v2)
  - [Goals](#goals)
  - [System Design Constraints](#system-design-constraints)
  - [Module Layout](#module-layout)
  - [Knowledge Layer — Multi-Corpus Design](#knowledge-layer--multi-corpus-design)
  - [Ingestion Pipeline — Docling-Graph Parallel Paths](#ingestion-pipeline--docling-graph-parallel-paths)
  - [Redis Pub/Sub + Async Worker Model](#redis-pubsub--async-worker-model)
  - [Caching Architecture](#caching-architecture)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Confidence-Based Scoring](#confidence-based-scoring)
  - [Confidence-Aware Pipeline](#confidence-aware-pipeline)
  - [Model Tiering](#model-tiering)
  - [Query Validation & Hook System](#query-validation--hook-system)
  - [Guardrail Architecture — Key Principles](#guardrail-architecture--key-principles)
  - [Error Handling Strategy](#error-handling-strategy)
  - [Retry & Resilience Strategy](#retry--resilience-strategy)
  - [Security Layer — JWT, JWE, HTTPS, RBAC](#security-layer--jwt-jwe-https-rbac)
  - [API Layer](#api-layer)
  - [Docker Compose — Local Dev](#docker-compose--local-dev)
  - [Packaging & Developer Install](#packaging--developer-install)
  - [Cloud Deployment — Production](#cloud-deployment--production)
  - [SaaS Deployment Model](#saas-deployment-model)
  - [Evaluation System — Offline & Online Metrics](#evaluation-system--offline--online-metrics)
  - [Load & Chaos Testing Strategy](#load--chaos-testing-strategy)
  - [Docling-Graph Evaluation Checklist](#docling-graph-evaluation-checklist)
  - [Implementation Phases](#implementation-phases)
- [In Progress — Rate Limiting, Timeouts & Retries](#in-progress--rate-limiting-timeouts--retries)
- [Queued — Production Hardening](#queued--production-hardening)
- [Done](#done)

---

## Architecture Proposal — Enterprise RAG v2

### Goals

1. Single `knowledge/` module replacing `rag/` + domain-specific KG code.
2. Multi-corpus ingestion: any folder on disk (or remote source) becomes a corpus namespace.
3. Docling-graph integration: chunking and KG extraction run as parallel async tasks per document.
4. Redis pub/sub + async workers for all heavyweight I/O (ingestion, retrieval, LLM calls).
5. Multi-level caching: in-process LRU → Redis → semantic similarity cache.
6. Enterprise security baseline: JWT auth, JWE payload encryption, TLS 1.3, RBAC, audit log.
7. Docker Compose for local development; cloud-native deployment (K8s + managed services) for production.

---

### System Design Constraints

Two load-bearing workloads with fundamentally different SLA profiles: **retrieval** (interactive, latency-sensitive, user-blocking) and **ingestion** (batch, throughput-sensitive, async and non-user-blocking). The constraints below are derived from the 10 K DAU target and drive every capacity and cost decision in the architecture.

---

#### Load Model

| Parameter | Value | Derivation |
|---|---|---|
| Daily active users | **10,000** | given |
| Queries per user per day | **5** (median; range 1–20) | typical enterprise RAG usage |
| Total queries per day | **50,000** | 10K × 5 |
| Active window | **8 h** (business hours, UTC-normalised) | multi-tenant; overlapping TZs |
| Average RPS | **1.7 req/s** | 50K / (8 × 3,600) |
| Peak RPS (3× burst) | **5 req/s** | morning sync, post-lunch spike |
| Peak concurrency | **~10 in-flight** | Little's Law: 5 req/s × 2 s P95 latency |
| Documents ingested per day | **100–500** | background, async, no user impact |

**Cache offload assumption** (reduces LLM calls):

| Layer | Hit rate | What hits |
|---|---|---|
| L2 Redis exact match | ~10% | identical query + corpus within TTL (5 min) |
| L3 semantic cache (cosine ≥ 0.95) | ~30% | near-paraphrase of a recent popular query |
| **Queries reaching LLM (cache bypass)** | **~60%** | 30,000 queries/day reach the LLM; 40% served from cache (10% L2 + ~30% of remaining 90% via L3 ≈ 10% + 27% = 37%; rounded to ~40% for planning) |

The 0.95 semantic threshold is strict by design — serving a wrong cached answer is worse than a cache miss. Tune down to 0.92 per corpus once confidence distributions are measured.

---

#### Retrieval — SLA

Six distinct paths, each with its own latency contract. SLAs are end-to-end wall-clock from request receipt to first byte of response body.

| Path | P50 | P95 | P99 |
|---|---|---|---|
| **L2 Redis exact hit** | < 20 ms | < 40 ms | < 80 ms |
| **L3 semantic cache hit** | < 70 ms | < 140 ms | < 280 ms |
| **Search-only** (no generation) | < 250 ms | < 600 ms | < 1,200 ms |
| **Chat — small model** (`llama3.2:3b` / `claude-haiku-4-5`) | < 700 ms | < 2,000 ms | < 4,000 ms |
| **Chat — large model** (`llama3.1:70b` / `claude-opus-4-8`) | < 2,500 ms | < 6,000 ms | < 12,000 ms |
| **Streaming TTFT** (small model, SSE) | < 300 ms | < 800 ms | < 1,500 ms |

Span budget per stage (all P95, standard config):

| Stage | P95 budget | Alert threshold |
|---|---|---|
| Schema + length guard (V1–V2) | < 2 ms | > 10 ms |
| Content policy classifier V5 (nano) | < 50 ms | > 150 ms |
| Query routing (nano) | < 80 ms | > 250 ms |
| L2 Redis lookup | < 5 ms | > 20 ms |
| Query embedding | < 80 ms | > 250 ms |
| Hybrid retrieval (vector + text, parallel) | < 120 ms | > 400 ms |
| CrossEncoder rerank | < 200 ms | > 600 ms |
| L3 semantic cache lookup | < 40 ms | > 100 ms |
| LLM first token (small model) | < 600 ms | > 1,500 ms |
| LLM full generation (small, ~300 output tokens) | < 1,200 ms | > 3,000 ms |
| Judge gate (nano) | < 80 ms | > 250 ms |
| **Total — search-only** | **< 600 ms** | **> 1,200 ms** |
| **Total — chat small** | **< 2,000 ms** | **> 4,000 ms** |

PagerDuty alerts fire on:
- `chat_latency_p95 > 3 s` sustained 5 min
- `search_latency_p99 > 1.5 s`
- `streaming_ttft_p95 > 1,000 ms`
- `l3_cache_hit_rate < 15%` (cache cold or corpus recently invalidated)

---

#### Retrieval — Token Budget

Per-query token counts for each active stage. Stages are skipped when their feature flag is off.

| Stage | Model tier | Input tokens | Output tokens | Flag |
|---|---|---|---|---|
| Content policy (V5) | nano | 200 | 30 | `content_policy_enabled` |
| Query routing | nano | 150 | 30 | `model_routing_enabled` |
| Query embedding | embedding | 50 | — | always on |
| Retrieved context (top-5 reranked chunks, 200 tok avg each) | — | 1,000 | — | always on |
| LLM generation (system prompt 300 + context 1,000 + query 50) | small/large | 1,350 | 300 | always on |
| Judge gate (context + query + answer) | nano | 1,700 | 100 | `confidence_aware_pipeline` |

**Per-query totals by configuration:**

| Config | Input tokens | Output tokens | Total |
|---|---|---|---|
| Minimal (routing + generation, no judge, no V5) | 1,550 | 330 | **1,880** |
| Standard (routing + generation + judge) | 3,200 | 430 | **3,630** |
| Full (V5 + routing + generation + judge) | 3,400 | 460 | **3,860** |

**Daily token consumption** (50K queries/day; 30K reach full LLM after cache):

| Config | Daily input | Daily output | Monthly total |
|---|---|---|---|
| Minimal | 46.5M | 9.9M | **1.69B** |
| Standard | 96M | 12.9M | **3.27B** |
| Full | 102M | 13.8M | **3.47B** |

**Cost — local Ollama (small model on 1× A100 80 GB):**

| Item | Monthly cost |
|---|---|
| GPU instance (RunPod/Vast.ai, always-on A100) | $720–$1,440 |
| 5 req/s peak → 1 GPU sufficient at `llama3.2:3b` | single instance |
| Embedding (nomic-embed-text, same GPU) | included |

**Cost — cloud models (`claude-haiku-4-5`, $0.25/$1.25 per MTok in/out):**

| Config | Daily LLM cost | Monthly LLM cost |
|---|---|---|
| Minimal | $26.27 | **$788** |
| Standard | $40.13 | **$1,204** |
| Full | $42.75 | **$1,283** |

> Escalating from Haiku to Sonnet ($3/$15 per MTok) multiplies cost ~10×. Keep `large` tier only for genuinely complex queries; the router must enforce this.

---

#### Ingestion — SLA

Ingestion is fully async (Redis Streams). The user-visible SLA is job latency from submission to `status=completed`, observable via SSE or status poll. The retrieval path is never blocked by ingestion.

**End-to-end job latency by document type:**

| Document type | P50 | P95 | P99 |
|---|---|---|---|
| Plain text / Markdown (< 10 KB) | < 5 s | < 15 s | < 30 s |
| PDF, < 20 pages | < 30 s | < 90 s | < 3 min |
| PDF, 20–100 pages | < 2 min | < 6 min | < 12 min |
| DOCX / PPTX | < 20 s | < 60 s | < 2 min |
| Audio, 60 min (Whisper ASR) | < 5 min | < 12 min | < 20 min |
| Any type + graph extraction | +50–100% on all tiers | | |
| **Batch, 100 documents** | < 30 min | < 90 min | < 3 h |

**Sub-SLA per stage (10-page PDF baseline):**

| Step | P50 | P95 | Notes |
|---|---|---|---|
| API → Redis XADD (job accepted) | < 80 ms | < 150 ms | synchronous fast-path |
| Worker pickup (XREADGROUP) | < 1 s | < 5 s | depends on queue depth |
| Docling parse | < 8 s | < 20 s | CPU-bound; scales with page count |
| HybridChunker | < 1 s | < 3 s | pure Python |
| Embedding batch (65 chunks) | < 5 s | < 15 s | nomic-embed-text; GPU |
| Vector store upsert (asyncpg executemany) | < 2 s | < 5 s | |
| Graph extraction, optional (LLM, per chunk) | < 30 s | < 90 s | parallelised across chunks |
| Entity index upsert (GIN) | < 1 s | < 3 s | |

**Retry + DLQ policy:** 3 attempts with exponential backoff (5 s, 25 s, 125 s). After 3 failures, job promoted to `knowledge:ingest:dlq` + alert fired. Max acceptable DLQ depth: 0 sustained (every DLQ entry is an incident).

---

#### Ingestion — Token Budget

Baseline: 10-page PDF → ~13,000 body tokens → 65 chunks × 200 tokens average.

**Per-document token breakdown:**

| Step | Model | Input tokens | Output tokens | Notes |
|---|---|---|---|---|
| Embedding (all chunks) | `nomic-embed-text` | 13,000 | — | billed per input only; no output |
| Graph extraction (per chunk, optional) | small | 5,000 | 1,000 | entity + relationship extraction |
| **Total — vector only** | | **13,000** | **0** | |
| **Total — vector + graph** | | **18,000** | **1,000** | |

Scales linearly: 100-page PDF ≈ 10× above figures.

**Daily ingestion token budget (500 docs/day, 10-page average):**

| Mode | Daily embedding tokens | Daily graph LLM tokens | Monthly embedding | Monthly graph LLM |
|---|---|---|---|---|
| Vector only | 6.5M | 0 | 195M | 0 |
| Vector + graph | 6.5M | 3.5M in / 500K out | 195M | 105M in / 15M out |

**Cost — ingestion (cloud models):**

| Item | Daily | Monthly |
|---|---|---|
| Embedding (`text-embedding-3-small`, $0.02/MTok) | $0.13 | **$3.90** |
| Graph extraction (`claude-haiku-4-5`, 500 docs/day) | $1.25 | **$37.50** |
| **Total ingestion** | **$1.38** | **$41.40** |

Ingestion cost is ~3% of retrieval cost and is dominated by graph extraction. Disable graph extraction (`enable_graph_extraction=False` per corpus) on corpora where KG traversal is not needed.

---

#### Total System Cost Summary (10 K DAU, standard config)

| Component | Local GPU path | Cloud model path |
|---|---|---|
| Retrieval — GPU (A100, always-on) | $720–$1,440/month | — |
| Retrieval — LLM (`claude-haiku-4-5`) | — | $1,204/month |
| Ingestion — embedding | $0 (same GPU) | $4/month |
| Ingestion — graph extraction | $0 (same GPU) | $38/month |
| PostgreSQL + Redis (cloud managed) | $200–$600/month | $200–$600/month |
| **Total** | **$920–$2,040/month** | **$1,446–$1,846/month** |
| **Cost per query** | **$0.018–$0.041** | **$0.029–$0.037** |

> At 10 K DAU the two paths are cost-comparable. Local GPU wins on cost at high query volume but requires GPU ops expertise. Cloud wins on operational simplicity and latency consistency (no GPU saturation at peak).

---

#### Budget Controls & Cost Circuit Breakers

Cost controls are enforced at two levels: per-tenant soft and hard limits, and system-wide circuit breakers. These are not monitoring dashboards — they are enforcement mechanisms baked into the request path.

**Per-tenant monthly LLM budget** (stored in `TenantQuota.llm_budget_usd_per_month`):

| Budget state | Enforcement action |
|---|---|
| `cost < 80% of limit` | Normal operation |
| `80% ≤ cost < 100%` | Return `X-Budget-Warning: 0.80` header on every response; alert tenant admin |
| `cost ≥ 100%` | Block LLM calls; serve cache hits and search-only responses; return `402 Payment Required` on generation requests |
| Admin override | `quota_override: true` in tenant config bypasses limit (enterprise tier) |

Budget is tracked in Redis: `quota:{tenant_id}:cost_usd:{YYYY-MM}` as a `INCRBYFLOAT` counter. Flushed monthly. Authoritative value for billing is `token_usage` table (Redis is the fast-path guard; SQL is the source of truth).

**System-wide cost circuit breaker:**

Triggered when total daily spend across all tenants exceeds `SYSTEM_DAILY_COST_LIMIT_USD` (ops-configured). On breach:
1. All new cloud-model LLM calls blocked (local Ollama unaffected).
2. PagerDuty alert fired immediately.
3. Auto-recovery: circuit resets at midnight UTC.

```python
# knowledge/agent/cost_guard.py
async def check_cost_circuit_breaker(tenant_id: str, model_id: str) -> None:
    """Raise BudgetExceeded if tenant or system budget is exhausted."""
    # Fast path: check Redis counter
    monthly_cost = float(await redis.get(f"quota:{tenant_id}:cost_usd:{month}") or 0)
    tenant_limit = await get_tenant_budget(tenant_id)
    if tenant_limit > 0 and monthly_cost >= tenant_limit:
        raise TenantBudgetExceeded(tenant_id=tenant_id, spent=monthly_cost, limit=tenant_limit)

    system_daily = float(await redis.get("system:cost_usd:daily") or 0)
    if system_daily >= settings.system_daily_cost_limit_usd:
        raise SystemBudgetExceeded(spent=system_daily, limit=settings.system_daily_cost_limit_usd)
```

Called at `PRE_LLM` hook point — before every LLM call. Zero cost is incurred on cache hits (neither hook nor circuit breaker fires).

**Token budget per request** (separate from monthly limits):

```python
max_prompt_tokens: int = 8192    # hard cap per request; Pydantic AI enforces via model_settings
max_output_tokens: int = 1024    # prevents runaway generation
```

If a request would exceed `max_prompt_tokens` after context insertion, the retriever trims chunks from lowest-confidence to highest until it fits. Never silently truncate; always log and emit `context_truncated: true` in the response.

**Cost observability additions** (to Prometheus metrics):

```
cost_budget_utilization{tenant_id, month}       # gauge: 0.0–1.0+ (1.0 = limit reached)
cost_circuit_breaker_triggered_total{scope}     # counter; scope=tenant|system
cost_blocked_requests_total{tenant_id}          # counter: LLM calls blocked by budget
cache_cost_saved_usd_total{corpus, tier}        # counter: cost avoided by cache hits
```

Cache savings tracking: every L2/L3 hit records `estimated_cost_usd` that would have been spent. This makes cache ROI visible — a cache hit rate drop is both a latency and a cost event.

---

### Module Layout

Replace current `rag/` with a single `knowledge/` package. The `kg/` core (`age_graph_store.py`, `entity_index.py`) is absorbed into `knowledge/store/`. Legal CUAD code is in `misc/kg_legal_cuad/` (already moved).

```
knowledge/
├── config/
│   └── settings.py              # Pydantic-settings; adds corpus, cache, worker, JWT/JWE fields
├── api/
│   ├── app.py                   # FastAPI factory (lifespan, middleware stack)
│   ├── auth.py                  # JWT decode + RBAC dependency; JWE encrypt/decrypt helpers
│   ├── middleware.py            # CorrelationID, structured-log, audit-event emission
│   ├── quota.py                 # enforce_quota(): per-tenant rate limiting + budget enforcement
│   ├── timeout.py               # TimeoutBudget dataclass + per-stage sub-deadline helpers
│   ├── routes/
│   │   ├── auth.py              # POST /v1/auth/token, POST /v1/auth/refresh
│   │   ├── ingest.py            # POST /v1/ingest → publish job; GET /v1/ingest/{job_id}/status
│   │   ├── search.py            # POST /v1/search (sync fast path) + async via Redis
│   │   ├── chat.py              # POST /v1/chat, POST /v1/chat/stream (SSE — POST, not GET)
│   │   ├── corpus.py            # GET /v1/corpus, POST /v1/corpus/{id}/cache/invalidate
│   │   ├── evaluate.py          # POST /v1/evaluate/run, GET /v1/evaluate/run/{id}
│   │   ├── feedback.py          # POST /v1/feedback, POST /v1/signals
│   │   ├── scheduler.py         # CRUD for scheduled ingestion jobs
│   │   └── health.py            # GET /health (pool stats, Redis ping, worker heartbeat)
│   └── schemas.py               # Pydantic request/response models (versioned)
├── bus/
│   ├── publisher.py             # async Redis pub/sub + Redis Streams publisher
│   ├── consumer.py              # base async consumer loop (ack, dead-letter, backoff)
│   └── schemas.py               # IngestJob, SearchRequest, WorkerEvent message models
├── ingestion/
│   ├── worker.py                # Redis consumer → pipeline orchestrator
│   ├── pipeline.py              # per-document orchestrator: spawns chunker + graph_extractor concurrently
│   ├── docling_processor.py     # Docling DocumentConverter wrapper (cached instance per worker)
│   ├── chunker.py               # HybridChunker wrapper; returns List[ChunkData]
│   ├── graph_extractor.py       # docling-graph PipelineOrchestrator wrapper; returns entities + edges
│   └── embedder.py              # async OpenAI-compatible embedder (timeout + exponential backoff)
├── store/
│   ├── vector.py                # PostgresHybridStore: pgvector HNSW + tsvector GIN + RRF
│   ├── graph.py                 # AgeGraphStore: Apache AGE Cypher ops over asyncpg
│   ├── entity_index.py          # EntityIndex: tsvector shadow table for entity name search
│   └── cache.py                 # RedisCache: L2 query/embedding/doc-fingerprint cache
├── retrieval/
│   ├── worker.py                # Redis consumer → retrieval pipeline (for async search requests)
│   ├── retriever.py             # hybrid retriever: vector + text + optional graph traversal
│   ├── graph_retriever.py       # NL→Cypher query against AgeGraphStore
│   ├── fusion.py                # CrossEncoder reranker (default, always-on) + RRF fusion; optional LLM re-ranker
│   └── semantic_cache.py        # L3 semantic cache: pgvector cosine-sim lookup before LLM call
├── agent/
│   ├── pipeline.py              # ConfidenceAwarePipeline: 3-layer gate orchestrator (retrieval → citation → judge)
│   ├── agent.py                 # Pydantic AI agent; tools: search_knowledge_base, search_knowledge_graph, search_hybrid_kg, run_graph_query, nl_graph_query
│   ├── judge.py                 # LLMJudge: JudgeResult(verdict, confidence, reasoning); nano model with small escalation
│   ├── cost_guard.py            # check_cost_circuit_breaker(): tenant + system monthly budget enforcement
│   ├── model_router.py          # QueryRouter (nano model) → RoutingDecision
│   └── prompts.py
├── memory/
│   ├── mem0_store.py            # Tier 3: Mem0-backed user semantic memory (extraction, dedup, cosine search)
│   ├── conversation_store.py    # Tier 2: episodic — conversation + message CRUD; active window loader
│   ├── summarizer.py            # Tier 2: auto-summarize when turn_count > 20 (nano model)
│   ├── working_memory.py        # Tier 1: context assembly + token-budget trim (drop lowest confidence first)
│   └── pruning.py               # Background jobs: TTL eviction, LRU eviction, memory compaction
├── corpus/
│   └── registry.py              # CorpusRegistry: load corpus configs, enforce RBAC at query time
├── billing/
│   ├── metering.py              # BillingEvent emit + nightly Stripe flush cron
│   └── provisioner.py           # TenantProvisioner: onboard, offboard, GDPR erase
├── scheduler/
│   ├── job_store.py             # ScheduledJob CRUD in PostgreSQL (scheduled_jobs table)
│   ├── runner.py                # APScheduler integration: cron + interval triggers
│   └── schemas.py               # ScheduledJob, JobTrigger, JobStatus Pydantic models
├── hooks/
│   ├── registry.py              # HookRegistry, HookPoint enum, Hook type alias
│   ├── context.py               # HookContext dataclass
│   └── builtins.py              # placeholder hooks registered at app startup
├── validation/
│   └── pipeline.py              # V1–V6 validation chain; ContentPolicyResult schema
├── evaluation/
│   ├── harness.py               # orchestrates evaluation runs end-to-end
│   ├── datasets.py              # GoldDataset loader + GoldSample Pydantic model
│   ├── runner.py                # async runner; publishes to knowledge:eval stream
│   ├── reporter.py              # metric aggregation, regression detection, CI report
│   ├── schemas.py               # EvalRun, EvalResult, UserFeedback, ImplicitSignal models
│   └── metrics/
│       ├── retrieval.py         # HitRate@k, MRR@k, NDCG@k, Precision@k, Recall@k
│       ├── faithfulness.py      # claim extraction + NLI check (nano model)
│       ├── answer_relevance.py  # reverse-question embedding similarity (nano model)
│       ├── correctness.py       # BLEU-4, ROUGE-1/2/L, METEOR, BERTScore, semantic-sim
│       ├── performance.py       # latency span recording, token counting, cost estimation
│       ├── pipeline.py          # abstention rate, false abstention rate, per-layer share
│       └── online.py            # user feedback aggregation + implicit signal processing
└── observability/
    ├── langfuse.py              # Langfuse trace + span helpers
    └── metrics.py               # Prometheus counters/histograms via prometheus-client
```

---

### Knowledge Layer — Multi-Corpus Design

Each corpus is an independent namespace sharing the same PostgreSQL cluster. Corpus isolation is enforced at the storage layer via a `corpus_id` column on `documents` and `chunks`.

**Corpus config** (`knowledge/corpus/registry.py`):
```python
class CorpusConfig(BaseModel):
    id: str                          # slug, e.g. "hr-policies"
    display_name: str
    source_folders: list[Path]       # local paths scanned on ingest
    allowed_roles: list[str]         # RBAC: which JWT roles can read/write
    metadata_tags: dict[str, str]    # extra metadata attached to every chunk

    # Knowledge graph extraction (docling-graph)
    enable_graph_extraction: bool = False
    # Path to the Pydantic ontology template, relative to knowledge/corpus/ontologies/
    # If None, uses the generic default template (extracts entities/relations without domain specifics)
    graph_ontology_path: str | None = None
    # LLM backend provider — any LiteLLM-compatible provider; "ollama" for local
    graph_extraction_provider: str = "ollama"
    # Model for graph extraction (can differ from chat model; smaller is fine for entity extraction)
    graph_extraction_model: str = "llama3.2:3b"
    # Extraction contract:
    #   "direct"  — single LLM call per chunk; fastest; good for large models (≥ 70B)
    #   "staged"  — multi-pass ID → fill → quality gate; recommended for small models (≤ 8B)
    #   "delta"   — chunk-by-chunk with merge + dedup resolvers; best for long documents
    graph_extraction_contract: Literal["direct", "staged", "delta"] = "staged"
    # Processing mode:
    #   "many-to-one" — all chunks merged into one graph; best for most docs
    #   "one-to-one"  — page-by-page; best for forms and complex layouts
    graph_processing_mode: Literal["many-to-one", "one-to-one"] = "many-to-one"
    # VLM extraction for scanned/image-heavy PDFs (requires GPU)
    graph_extraction_backend: Literal["llm", "vlm"] = "llm"
```

**Schema change** (additive migration):
- `documents.corpus_id TEXT NOT NULL` + B-tree index
- `chunks.corpus_id TEXT NOT NULL` + B-tree index (for fast corpus-scoped search)
- All queries gain a `WHERE corpus_id = $1` predicate automatically

**Cross-corpus search**: allowed with explicit `corpus_ids: list[str]` in the search request, subject to JWT role check across all listed corpora.

---

### Ingestion Pipeline — Docling-Graph Parallel Paths

Per document, after Docling conversion, two async tasks run concurrently:

```
DocumentConverter.convert(path)
        │
        ▼
 DoclingDocument (in memory)
        │
   asyncio.gather(
     ├── chunker_task:
     │      HybridChunker → List[ChunkData]
     │      → embedder.embed_batch()
     │      → vector_store.upsert_chunks()
     │
     └── graph_task (if corpus.enable_graph_extraction):
            load ontology class from corpus.graph_ontology_path
            run_pipeline(PipelineConfig(template=OntologyClass, ...))
            → PipelineContext.knowledge_graph (NetworkX DiGraph)
            → age_graph_store.import_docling_graph(context, corpus_id, doc_id)
               ├── iterate graph.nodes(data=True) → upsert_vertex() per node
               └── iterate graph.edges(data=True) → add_edge() per edge
            → entity_index.upsert_batch(vertices)
   )
        │
        ▼
  publish IngestCompleteEvent to Redis
```

---

### Knowledge Graph Extraction — Ontology and docling-graph API

This section documents exactly how docling-graph is used. Read this before implementing `knowledge/ingestion/graph_extractor.py`.

#### The ontology is a Pydantic template

docling-graph extracts entities and relationships whose shape is defined entirely by a **Pydantic `BaseModel` subclass** (called a "template" in docling-graph terminology). The template IS the ontology — there is no separate schema format.

**Minimal template** (required structure every ontology file must follow):

```python
# knowledge/corpus/ontologies/my_domain.py
"""
HR policy ontology.
Extracts policies, benefits, people, and departments from HR documents.
"""
from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs: Any) -> Any:
    """Required helper — marks a field as a directed graph edge."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components (value objects — no stable graph identity) ---
class ContactInfo(BaseModel):
    model_config = ConfigDict(is_entity=False)
    email: str | None = Field(None, description="Email address. LOOK FOR: @ symbol. EXAMPLES: 'hr@company.com'")
    phone: str | None = Field(None, description="Phone number")

# --- Entities (unique, identifiable — get stable node IDs) ---
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["full_name"])   # stable ID from these fields
    full_name: str = Field(description="Full name. LOOK FOR: Names near job titles. EXAMPLES: 'Jane Smith'")
    title: str | None = Field(None, description="Job title")
    contact: ContactInfo | None = Field(None, description="Contact details")

class Department(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="Department name. EXAMPLES: 'Engineering', 'HR'")
    head: Person | None = edge(label="LED_BY", default=None, description="Department head")
    members: List[Person] = edge(label="HAS_MEMBER", default_factory=list, description="Staff in dept")

class Policy(BaseModel):
    model_config = ConfigDict(graph_id_fields=["policy_id"])
    policy_id: str = Field(description="Policy identifier. EXAMPLES: 'PTO-001', 'REMOTE-002'")
    title: str = Field(description="Policy title")
    description: str | None = Field(None, description="Policy text summary")
    applies_to: List[Department] = edge(label="APPLIES_TO", default_factory=list,
                                         description="Departments this policy covers")

# --- Root document model (last in file, captures the whole document) ---
class HRPolicyDocument(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_title"])
    document_title: str = Field(description="Document title. LOOK FOR: Title page heading.")
    policies: List[Policy] = edge(label="CONTAINS_POLICY", default_factory=list,
                                   description="Policies described in this document")
    departments: List[Department] = edge(label="REFERENCES_DEPT", default_factory=list,
                                          description="Departments mentioned")

HRPolicyDocument.model_rebuild()
```

**Key rules for every ontology template:**
1. The `edge()` helper MUST be defined identically in every template file — `Field(..., json_schema_extra={"edge_label": label}, **kwargs)`
2. **Entities** have `graph_id_fields` in `ConfigDict` — these fields create stable node IDs and enable cross-chunk deduplication
3. **Components** have `is_entity=False` — they are value objects embedded in entities, deduplicated by content
4. **List edges** MUST have `default_factory=list`
5. Field `description` follows `LOOK FOR / EXTRACT / EXAMPLES` pattern — this is the prompt the LLM sees; poor descriptions = poor extraction
6. The root model (last class in file) is what `PipelineConfig.template` points to
7. Call `Model.model_rebuild()` at file end when using forward references

#### Entities vs Components — decision rule

| Question | Entity | Component |
|----------|--------|-----------|
| Does it need a stable, reusable node ID? | Yes | No |
| Can two instances be "the same thing"? | Yes (dedup by `graph_id_fields`) | Yes (dedup by content) |
| Can it appear as a standalone node? | Yes | No — only embedded in entities |
| Example | Person, Department, Policy | Address, ContactInfo, MonetaryAmount |

#### Extraction contracts — which to use

| Contract | When to use | How it works |
|----------|-------------|-------------|
| `"direct"` | Large models (≥ 70B), simple schemas | One LLM call per chunk; fastest |
| `"staged"` | Small models (≤ 8B like llama3.2:3b), complex nested schemas | Multi-pass: ID discovery → fill pass → quality gate; recommended for Ollama |
| `"delta"` | Long documents (>50 pages), many entities of the same type | Chunk-by-chunk with incremental merge and semantic deduplication resolvers |

**Default for our system:** `"staged"` — we use `llama3.2:3b` via Ollama for graph extraction. Staged contract breaks complex templates into simpler multi-pass operations that smaller models handle reliably.

#### Actual API call

```python
# knowledge/ingestion/graph_extractor.py
from pathlib import Path
from docling_graph import PipelineConfig, run_pipeline
from docling_graph.pipeline.context import PipelineContext

async def extract_graph(
    doc_path: Path,
    ontology_class: type,          # loaded from corpus.graph_ontology_path
    corpus_config: CorpusConfig,
    settings: Settings,
) -> PipelineContext | None:
    """Run docling-graph extraction. Returns PipelineContext or None on failure.

    NOTE: Do NOT use CypherExporter. AGE uses ag_catalog.cypher() SQL wrapper
    syntax — not Neo4j-compatible raw Cypher. Feed the NetworkX DiGraph directly
    to AgeGraphStore.import_docling_graph() instead.
    """

    def _run_sync() -> PipelineContext:
        config = PipelineConfig(
            source=str(doc_path),
            template=ontology_class,
            backend=corpus_config.graph_extraction_backend,           # "llm" | "vlm"
            inference="local",
            provider_override=corpus_config.graph_extraction_provider, # "ollama"
            model_override=corpus_config.graph_extraction_model,       # "llama3.2:3b"
            processing_mode=corpus_config.graph_processing_mode,       # "many-to-one"
            extraction_contract=corpus_config.graph_extraction_contract, # "staged"
            use_chunking=True,
            chunk_max_tokens=settings.chunk_max_tokens,
            structured_output=True,
            dump_to_disk=False,    # API mode — no files on disk
        )
        return run_pipeline(config)   # returns PipelineContext, not a string

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_sync),
            timeout=settings.graph_extraction_timeout_s,
        )
    except TimeoutError:
        logger.warning("Graph extraction timed out for %s", doc_path.name)
        return None
    except Exception as exc:
        logger.error("Graph extraction failed for %s: %s", doc_path.name, exc)
        return None
```

Then in the pipeline orchestrator:
```python
context = await extract_graph(doc_path, ontology_class, corpus_config, settings)
if context:
    node_count, edge_count = await age_store.import_docling_graph(
        context, corpus_id=corpus_config.id, document_id=document_id
    )
    await entity_index.upsert_batch_from_graph(context.knowledge_graph, document_id)
else:
    chunk_metadata["graph_extraction_failed"] = True
```

---

### Apache AGE — Graph Store Design (`knowledge/store/graph.py`)

The v2 `AgeGraphStore` is a rewrite of `kg/age_graph_store.py` adapted for multi-corpus, multi-tenant use. The v1 implementation is hardwired to the CUAD legal ontology (label allowlist from `cuad_ontology.py`); v2 accepts any labels from the user's docling-graph template.

#### How AGE works with asyncpg

Apache AGE adds openCypher graph queries to PostgreSQL via a SQL function wrapper. Every Cypher statement must be wrapped:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (n:Person) RETURN n.name, n.uuid
$$) AS (name agtype, uuid agtype)
```

`agtype` columns are returned as strings by asyncpg (they look like `"Acme Corp"` with surrounding quotes). Strip with `s[1:-1]` if starts/ends with `"`.

Every connection must run two setup statements before any Cypher:
```python
await conn.execute("LOAD 'age'")
await conn.execute("SET search_path = ag_catalog, \"$user\", public")
```

Register this as an asyncpg pool `init` callback — AGE state is connection-local and gets reset by `RESET ALL` when connections return to the pool.

#### Graph name per corpus

Each corpus gets its own AGE graph: `f"{tenant_id}_{corpus_id}"` (e.g. `"acme_corp_hr_policies"`). This gives hard isolation — queries against one corpus never touch another's graph. The graph is created on first ingest:

```python
await conn.execute(f"SELECT create_graph('{graph_name}')")
```

Use `try/except` around creation — AGE raises `InvalidSchemaNameError` if the graph already exists.

#### Key method: `import_docling_graph()`

This is the primary write path from docling-graph. It iterates the NetworkX DiGraph from `PipelineContext` directly — **not** `CypherExporter`. AGE uses a SQL wrapper syntax that is incompatible with the raw Cypher `CREATE` statements that `CypherExporter` generates for Neo4j.

```python
async def import_docling_graph(
    self,
    context: "PipelineContext",   # from docling_graph.pipeline.context
    corpus_id: str,
    document_id: str,
) -> tuple[int, int]:
    """Import a docling-graph PipelineContext into Apache AGE.

    Iterates context.knowledge_graph (NetworkX DiGraph) directly.
    Do NOT use CypherExporter — its output is Neo4j syntax, incompatible with AGE.

    Returns (node_count, edge_count).
    """
    graph = context.knowledge_graph     # networkx.DiGraph
    graph_name = self._graph_name(corpus_id)

    node_id_map: dict[str, str] = {}    # NetworkX node_id → AGE vertex uuid

    # 1. Upsert all vertices
    for nx_id, attrs in graph.nodes(data=True):
        label = _sanitize_label(attrs.get("label", "Entity"))
        name  = str(attrs.get("name") or attrs.get("id") or nx_id)
        props = {k: str(v) for k, v in attrs.items()
                 if k not in ("label",) and v is not None}
        props["corpus_id"]   = corpus_id
        props["document_id"] = document_id

        uuid = await self._upsert_vertex(graph_name, nx_id, label, name, props)
        node_id_map[str(nx_id)] = uuid

    # 2. Upsert all edges
    edge_count = 0
    for src_nx, tgt_nx, edge_attrs in graph.edges(data=True):
        rel_type = _sanitize_rel_type(edge_attrs.get("label", "RELATED_TO"))
        src_uuid = node_id_map.get(str(src_nx))
        tgt_uuid = node_id_map.get(str(tgt_nx))
        if src_uuid and tgt_uuid:
            await self._add_edge(graph_name, src_uuid, tgt_uuid, rel_type,
                                  {"corpus_id": corpus_id, "document_id": document_id})
            edge_count += 1

    return len(graph.nodes), edge_count
```

#### Label and relationship-type sanitization (v2 — no hardcoded allowlist)

v1 validated labels against a hardcoded CUAD list. v2 accepts any label from the user's ontology template, only sanitizing characters:

```python
import re

def _sanitize_label(label: str) -> str:
    """Strip non-alphanumeric characters; ensure starts with uppercase letter."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", label)
    if not cleaned:
        return "Entity"
    return cleaned[0].upper() + cleaned[1:]

def _sanitize_rel_type(rel_type: str) -> str:
    """Uppercase + strip non-alphanumeric except underscore."""
    cleaned = re.sub(r"[^A-Z0-9_]", "", rel_type.upper())
    return cleaned or "RELATED_TO"
```

#### Vertex upsert (MERGE pattern)

```python
async def _upsert_vertex(
    self, graph_name: str, nx_id: str, label: str, name: str, props: dict
) -> str:
    """MERGE vertex by (nx_id, corpus_id); return AGE uuid."""
    vertex_uuid = str(uuid.uuid4())
    name_esc = name.replace('"', '\\"')
    nx_id_esc = str(nx_id).replace('"', '\\"')
    corpus_esc = props.get("corpus_id", "").replace('"', '\\"')

    # MERGE on stable identity: the docling-graph node ID + corpus
    cypher = (
        f'MERGE (v:{label} {{nx_id: "{nx_id_esc}", corpus_id: "{corpus_esc}"}}) '
        f'SET v.uuid = COALESCE(v.uuid, "{vertex_uuid}"), '
        f'v.name = "{name_esc}", '
        f'v.label = "{label}" '
        f'RETURN v.uuid'
    )
    async with self._conn() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS (uuid agtype)"
        )
    return _unquote_agtype(rows[0]["uuid"]) if rows else vertex_uuid
```

#### Read-only query (for the graph retriever)

```python
async def run_cypher_query(self, cypher: str, corpus_id: str) -> str:
    """Execute a read-only MATCH query scoped to corpus_id's graph."""
    if re.search(r"\b(CREATE|MERGE|SET|DELETE|DROP|DETACH|FOREACH)\b", cypher, re.I):
        return "Error: only MATCH queries are permitted."

    graph_name = self._graph_name(corpus_id)
    aliases = _parse_return_aliases(cypher)   # from v1; infer column names from RETURN clause
    as_clause = ", ".join(f"c{i} agtype" for i in range(len(aliases)))

    async with self._conn() as conn:
        try:
            rows = await conn.fetch(
                f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS ({as_clause})"
            )
        except Exception as exc:
            return f"Cypher error: {exc}"

    if not rows:
        return "No results."
    header = " | ".join(aliases)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(_unquote_agtype(row[f"c{i}"]) for i in range(len(aliases))))
    lines.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(lines)
```

#### Corpus-scoped delete (tenant offboarding)

```python
async def delete_corpus_graph(self, corpus_id: str) -> None:
    """Drop the entire AGE graph for a corpus — all vertices and edges."""
    graph_name = self._graph_name(corpus_id)
    async with self._conn() as conn:
        await conn.execute(f"SELECT drop_graph('{graph_name}', true)")

async def delete_document_vertices(self, corpus_id: str, document_id: str) -> None:
    """Remove all vertices (and their edges) for one document from a corpus graph."""
    graph_name = self._graph_name(corpus_id)
    cypher = f'MATCH (v {{document_id: "{document_id}"}}) DETACH DELETE v'
    async with self._conn() as conn:
        await conn.execute(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS (r agtype)"
        )
```

#### Entity index (shadow table in main PostgreSQL)

AGE does not support `tsvector` GIN indexes or `pgvector` — all CONTAINS scans in AGE are O(n). The `knowledge/store/entity_index.py` (ported from `kg/entity_index.py`) maintains a `kg_entity_index` shadow table in the main PostgreSQL database with:
- `age_uuid TEXT PRIMARY KEY` — maps back to the AGE vertex
- `name TEXT` + `name_tsv tsvector GENERATED` — GIN-indexed for BM25 search
- `label TEXT` — B-tree indexed for label filtering
- `corpus_id TEXT` + `document_id TEXT` — for scoped deletes
- `embedding vector(768)` — HNSW-indexed for semantic entity search

After each `import_docling_graph()`, call `entity_index.upsert_batch_from_graph(graph, document_id, corpus_id)` to sync vertex names into the shadow table.

#### Docker Compose — AGE runs separately from the main PostgreSQL

AGE cannot run in the same container as the main pgvector database (different extension sets, potential version conflicts). Use two separate PostgreSQL instances:

```yaml
postgres:
  image: pgvector/pgvector:pg16    # main DB: pgvector for vector search
  ports: ["5432:5432"]

age:
  image: apache/age:latest         # graph DB: Apache AGE for Cypher queries
  ports: ["5433:5432"]             # mapped to 5433 on host to avoid conflict
  environment:
    POSTGRES_DB: age_graph
    POSTGRES_USER: age
    POSTGRES_PASSWORD: ${AGE_DB_PASSWORD}
```

Settings:
```python
database_url: str         # main PostgreSQL (pgvector) — port 5432
age_database_url: str     # AGE PostgreSQL — port 5433
age_graph_prefix: str = "kg"  # graph names: f"{prefix}_{tenant_id}_{corpus_id}"
```

#### Ontology storage and loading (`knowledge/corpus/ontologies/`)

```
knowledge/corpus/ontologies/
├── __init__.py
├── loader.py          # load_ontology(path: str) → type[BaseModel]; LRU-cached
├── generic.py         # default ontology when no corpus-specific template provided
├── hr_policy.py       # example domain ontology
├── legal_contract.py  # example domain ontology
└── <user-defined>.py  # uploaded by admin via POST /v1/corpus/{id}/ontology
```

**Generic default ontology** (`generic.py`) — used when `corpus_config.graph_ontology_path is None`. Extracts named entities, organizations, dates, and generic relationships without domain specifics:

```python
class GenericEntity(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="Entity name. EXTRACT: The most specific identifier. EXAMPLES: 'Apple Inc', 'John Smith', 'ISO 27001'")
    entity_type: str = Field(description="Type of entity. EXAMPLES: 'Organization', 'Person', 'Location', 'Concept', 'Date', 'Product'")
    description: str | None = Field(None, description="Brief description from document context")
    related_to: List["GenericEntity"] = edge(label="RELATED_TO", default_factory=list,
        description="Entities this one is related to per document context")

class GenericDocument(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str = Field(description="Document title or best identifying label")
    entities: List[GenericEntity] = edge(label="MENTIONS", default_factory=list,
        description="All named entities mentioned in the document")

GenericEntity.model_rebuild()
GenericDocument.model_rebuild()
```

**Ontology loader** (`loader.py`) — loads a Python file from the ontologies directory and returns the root Pydantic class:

```python
import importlib.util, functools
from pathlib import Path
from pydantic import BaseModel

ONTOLOGIES_DIR = Path(__file__).parent

@functools.lru_cache(maxsize=32)
def load_ontology(ontology_path: str | None) -> type[BaseModel]:
    """Load ontology class from path relative to ontologies/. LRU-cached per worker."""
    if ontology_path is None:
        from knowledge.corpus.ontologies.generic import GenericDocument
        return GenericDocument

    full_path = ONTOLOGIES_DIR / ontology_path
    if not full_path.exists():
        raise FileNotFoundError(f"Ontology not found: {full_path}")

    spec = importlib.util.spec_from_file_location("_ontology", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executes the Python file

    # Root class = last BaseModel subclass defined in the file (by convention)
    root_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            root_class = obj  # last one wins
    if root_class is None:
        raise ValueError(f"No BaseModel subclass found in {full_path}")
    return root_class
```

#### Ontology management API

Admins can upload new ontologies per corpus via the API:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/v1/corpus/{id}/ontology` | `admin` | Get current ontology file for corpus |
| `POST` | `/v1/corpus/{id}/ontology` | `admin` | Upload Python ontology file; validates it is a valid Pydantic template |
| `DELETE` | `/v1/corpus/{id}/ontology` | `admin` | Remove custom ontology (reverts to generic default) |

On upload, the API:
1. Parses the Python file and verifies it contains a root `BaseModel` subclass
2. Checks the `edge()` helper is defined correctly
3. Saves to `knowledge/corpus/ontologies/{corpus_id}.py`
4. Updates `CorpusConfig.graph_ontology_path` in the corpus registry
5. Clears the `load_ontology` LRU cache so next extraction uses the new template

---

### Redis Pub/Sub + Async Worker Model

**Message bus** uses Redis Streams (`XADD` / `XREADGROUP`) rather than plain pub/sub — streams give persistent delivery, consumer groups, and dead-letter via `XPENDING`.

```
Streams:
  knowledge:ingest          # ingestion jobs
  knowledge:search          # async search requests
  knowledge:events          # worker lifecycle heartbeats, job completions

Consumer groups:
  ingest-workers            # N replicas, each XREADGROUP from knowledge:ingest
  retrieval-workers         # M replicas, XREADGROUP from knowledge:search

Dead-letter:
  knowledge:ingest:dlq      # jobs that failed MAX_RETRIES times
  knowledge:search:dlq
```

**Worker lifecycle** (`knowledge/bus/consumer.py`):
1. `XREADGROUP GROUP <group> <worker_id> COUNT 1 BLOCK 5000 STREAMS <stream> >`
2. Deserialize message → `IngestJob` | `SearchRequest`
3. Execute pipeline
4. `XACK` on success; increment retry counter + re-enqueue on transient failure
5. After `MAX_RETRIES` → move to DLQ, emit alert event
6. Heartbeat: `SET worker:<id>:heartbeat <ts> EX 30` every 10 s

**Job status** exposed via API:
- `GET /v1/ingest/{job_id}/status` → polls `HGETALL job:{job_id}` (hash: status, progress, error, corpus_id)
- `GET /v1/ingest/{job_id}/stream` → SSE subscription to `knowledge:events` filtered by job_id

**Fast-path search** (sync, < 200 ms budget):
- `POST /v1/search` hits the retriever directly in the API process for interactive queries
- Async path via Redis stream is for bulk/background search batches

---

### Caching Architecture

Three independent cache layers. Each layer has a distinct TTL and eviction strategy.

#### L1 — In-Process LRU (per worker)

- **Embedding cache**: `functools.lru_cache` on `embed(text: str) → list[float]`; max 1 000 entries; avoids round-trip to Ollama for repeated chunk texts during batch ingestion.
- **Document fingerprint cache**: `dict[sha256, bool]` per worker; skips re-ingesting already-processed files on incremental runs.
- **Settings/config cache**: corpus registry loaded once at worker startup.

#### L2 — Redis Distributed Cache

| Key pattern | Value | TTL | Purpose |
|---|---|---|---|
| `cache:embed:{sha256(text)}` | msgpack vector | 24 h | Embedding dedup across workers |
| `cache:search:{sha256(query+filters+corpus_ids)}` | msgpack SearchResult list | 5 min | Identical query short-circuit |
| `cache:doc_fingerprint:{sha256(file_content)}` | `"1"` | 7 days | Skip re-ingestion of unchanged files |
| `cache:health:{service}` | JSON | 30 s | Avoid DB health checks on every probe |

Cache invalidation:
- Ingest completion event → `DEL cache:search:*` for affected corpus (use Redis key pattern scan + pipeline delete — scoped, not full flush).
- Document delete → `DEL cache:doc_fingerprint:{sha256}`.
- Explicit corpus admin endpoint: `POST /v1/corpus/{id}/cache/invalidate`.

#### L3 — Semantic Query Cache (pgvector)

Separate `semantic_cache` table:
```sql
CREATE TABLE semantic_cache (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_ids  TEXT[] NOT NULL,
    query_text  TEXT NOT NULL,
    query_emb   vector(768) NOT NULL,
    answer_jwe  TEXT NOT NULL,      -- JWE-encrypted answer blob
    hit_count   INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX ON semantic_cache USING hnsw (query_emb vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

Lookup flow (`knowledge/retrieval/semantic_cache.py`):
1. Embed the incoming query (L1/L2 cache hit likely).
2. `SELECT id, answer_jwe, 1 - (query_emb <=> $1) AS sim FROM semantic_cache WHERE corpus_ids = $2 AND expires_at > NOW() ORDER BY query_emb <=> $1 LIMIT 1`.
3. If `sim >= SEMANTIC_CACHE_THRESHOLD` (default 0.95) → decrypt answer, increment `hit_count`, return.
4. On miss → run full retrieval + LLM pipeline → insert new cache row with `expires_at = NOW() + SEMANTIC_CACHE_TTL`.

Configuration knobs (in `settings.py`):
```python
semantic_cache_enabled: bool = True
semantic_cache_threshold: float = 0.95
semantic_cache_ttl_minutes: int = 60
semantic_cache_max_rows: int = 10_000   # prune oldest on exceed
```

#### Cache Observability

Every cache layer emits Prometheus counters:
- `cache_l1_hits_total{layer, operation}` / `cache_l1_misses_total`
- `cache_l2_hits_total{layer, operation}` / `cache_l2_misses_total`
- `cache_l3_hits_total` / `cache_l3_misses_total`
- `cache_l3_similarity_score` (histogram) — tracks score distribution; tune threshold

---

### Retrieval Pipeline

Three firm defaults (non-negotiable in this architecture):
- **Reranking is always on.** `reranker_enabled = True` out of the box. CrossEncoder (`BAAI/bge-reranker-base`) is the default — it runs locally via `sentence-transformers`, no API call. LLMReranker is an opt-in alternative.
- **Sources/citations are always included.** Every response carries `SearchResult.chunk_id`, `document_title`, `document_source`, and `similarity` score. The agent prompt mandates inline citation. Clients receive a structured `citations: list[Citation]` field alongside the answer text — never a bare string.
- **All models are local (Ollama).** No cloud LLM calls by default. Cloud model IDs in the tiering table are available but require explicit `cloud_models_enabled = True` in settings. Local model IDs are the defaults for every tier.

```
POST /v1/search
    │
    ├─► L3 semantic cache check (pgvector cosine sim)
    │       └── HIT → decrypt JWE → return { answer, citations } cached
    │
    ├─► L2 Redis cache check (exact query hash)
    │       └── HIT → return cached { chunks, citations }
    │
    ├─► hybrid retrieval (parallel):
    │       ├── vector_store.semantic_search(query_embedding, k × overfetch_factor)
    │       ├── vector_store.text_search(query_text, k × overfetch_factor)
    │       └── (optional) graph_retriever.query(query_text)   ← NL→Cypher → AGE
    │
    ├─► RRF fusion (k=60)
    │
    ├─► CrossEncoder rerank  ← ALWAYS ON (local BAAI/bge-reranker-base)
    │       trim to match_count; attach similarity + chunk_id to each result
    │
    ├─► score filter (min_relevance_score threshold)
    │
    ├─► Pydantic AI agent (search_knowledge_base tool)
    │       system prompt: "Always cite your sources using [chunk_id]."
    │       structured output: { answer: str, citations: list[Citation] }
    │
    ├─► populate L2 Redis cache  (async)
    └─► populate L3 semantic cache  (async, JWE-encrypted)

Citation model:
    class Citation(BaseModel):
        chunk_id: UUID
        document_title: str
        document_source: str      # file path or URL
        relevance_score: float    # = SearchResult.confidence (post-rerank sigmoid score, 0-1); never raw_score
        excerpt: str              # ≤ 200 chars of the supporting chunk
```

---

### Confidence-Based Scoring

#### Why the Current `similarity` Field Is Not a Confidence Score

The existing `SearchResult.similarity` field is a catch-all that holds fundamentally different values depending on search mode:

| Search mode | What `similarity` actually contains | Calibrated 0–1? |
|-------------|-------------------------------------|-----------------|
| `semantic` | `1 - pgvector cosine distance` | Yes — but cosine similarity ≠ relevance |
| `text` | `ts_rank` output | No — unbounded, no IDF, no length norm |
| `fuzzy` | `pg_trgm word_similarity` | Yes — but trigram overlap ≠ semantic relevance |
| `hybrid` (default) | RRF score `Σ 1/(60+rank)` | No — rank-based, max ~0.05 |

The `min_relevance_score` guardrail in `Retriever` only fires for `search_type == "semantic"` for exactly this reason — applying a threshold to an RRF score or `ts_rank` float would be meaningless. This means the guardrail is effectively dead in the default hybrid path.

After CrossEncoder reranking, scores are normalised to 0–1 and carry real signal — but they are used only for ordering and trimming, not for filtering. No chunk is ever dropped based on post-rerank confidence.

#### Design: Dual-Score `SearchResult`

The `knowledge/` module will separate raw search scores from calibrated confidence:

```python
class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_source: str
    content: str
    metadata: dict[str, Any]

    # Raw score from the search leg — scale varies by search_type
    raw_score: float
    raw_score_type: Literal["cosine_similarity", "ts_rank", "trigram_similarity", "rrf"]

    # Calibrated confidence — populated after reranking; None until then
    # Always 0-1; comparable across search types and corpus sizes
    confidence: float | None = None

# NOTE: SearchResult and Citation are defined in knowledge/ingestion/models.py for
# historical reasons (matching v1 layout). They are shared across ingestion, retrieval,
# agent, and API layers. If circular imports arise, move them to knowledge/models.py (root).
```

`Citation.relevance_score` maps to `confidence`, never `raw_score`. The agent and the API response only expose `confidence`.

#### Confidence Population

```
hybrid_search() → SearchResult[] with raw_score=rrf, raw_score_type="rrf", confidence=None
      │
      ▼
CrossEncoderReranker.rerank()
      │  scores all (query, chunk) pairs in one batch forward pass
      │  normalises raw logits → [0, 1] via sigmoid
      │
      └─► SearchResult.confidence = sigmoid(cross_encoder_logit)   # populated here
          SearchResult.raw_score  = rrf_score                      # unchanged

semantic_search() (standalone) → confidence = raw cosine similarity  # already 0-1
text_search()    (standalone) → confidence = None (ts_rank is not calibrated)
```

`confidence` is set on every result returned from the retriever whenever reranking is on (which is always, per design). For standalone semantic search without reranking, `confidence` falls back to the cosine similarity score.

#### Confidence Threshold Filter

Replace the current `search_type == "semantic"` guardrail with a mode-agnostic confidence filter applied post-rerank:

```python
# knowledge/retrieval/retriever.py
MIN_CONFIDENCE_THRESHOLD: float = settings.min_confidence_score  # default 0.10

results = [r for r in reranked if r.confidence is not None and r.confidence >= MIN_CONFIDENCE_THRESHOLD]
```

Settings additions:
```python
min_confidence_score: float = 0.10   # drop chunks with post-rerank confidence < this
confidence_warn_threshold: float = 0.40  # log warning if best chunk confidence < this
```

If the top result's `confidence < confidence_warn_threshold`, the agent receives a low-confidence context flag and the system prompt includes: *"The retrieved context has low confidence scores. State any uncertainty explicitly."*

#### Confidence in the Response

Every API response exposes per-citation confidence:

```json
{
  "answer": "The PTO policy allows ...",
  "citations": [
    {
      "chunk_id": "uuid",
      "document_title": "Employee Handbook",
      "document_source": "hr/handbook.pdf",
      "confidence": 0.87,
      "excerpt": "Employees accrue 15 days of PTO per year..."
    }
  ],
  "low_confidence_context": false
}
```

`low_confidence_context: true` is a flag clients can use to show a UI warning or trigger a human-review hook.

#### EvalResult Extension

Add `confidence` tracking to offline evaluation:

```python
class EvalResult(BaseModel):
    ...
    # Confidence distribution over retrieved chunks
    mean_confidence: float | None       # mean post-rerank confidence across top-K
    min_confidence: float | None        # lowest confidence chunk that was used
    low_confidence_flag: bool = False   # True if min_confidence < warn_threshold
```

This lets the Grafana dashboard correlate low-confidence retrieval with low faithfulness or poor user feedback — the primary signal for knowing when to improve the index or add more data to a corpus.

---

### Confidence-Aware Pipeline

The confidence-aware pipeline wraps the retriever, generator, and judge into a single orchestration function. At each of the three layers a hard gate either short-circuits to an abstention response or lets the request proceed. No answer reaches the user unless it clears all three gates.

Reference design (Microsoft Tech Community — "Confidence-Aware RAG: Teaching Your AI Pipeline to Acknowledge Uncertainty"):

```python
# NOTE: This is a synchronous reference sketch from an external source.
# The actual implementation — ConfidenceAwarePipeline — is fully async.
# See knowledge/agent/pipeline.py.
def confidence_aware_rag(user_query: str) -> dict:
    # Layer 1 — retrieve with confidence gating
    results = retrieve_with_confidence(user_query, threshold=1.5)
    if not results:
        return {"answer": "...", "status": "abstained_retrieval"}

    # Layer 2 — generate with citation requirements
    generation = generate_answer(user_query, context, results)
    if not generation["citation_check"]["is_trustworthy"]:
        return {"answer": "...", "status": "abstained_citation"}

    # Layer 3 — judge the answer
    judgement = judge_answer(user_query, context, generation["answer"])
    if judgement["verdict"] == "unsupported" or judgement["confidence"] < 0.6:
        return {"answer": "...", "status": "abstained_judge"}

    if judgement["verdict"] == "partial":
        generation["answer"] += "\n\nNote: This answer may be incomplete..."

    return {"answer": ..., "status": "answered", "confidence": ..., "sources": [...]}
```

#### Architecture Mapping

Each layer maps to a distinct component in the `knowledge/` module.

```
knowledge/agent/
├── pipeline.py        # ConfidenceAwarePipeline — top-level orchestrator
├── agent.py           # Layer 2: structured generation + citation check
├── judge.py           # Layer 3: LLM-as-judge (nano/small model)
└── model_router.py    # pre-pipeline: routes to correct model tier
```

#### Layer 1 — Retrieval Gate (`knowledge/retrieval/retriever.py`)

`retrieve_with_confidence` runs the standard hybrid retrieval + CrossEncoder rerank pipeline, then computes an **aggregate confidence score** over the top-K results. If the aggregate falls below `retrieval_confidence_threshold` the function returns an empty list and the pipeline short-circuits immediately — no LLM call is made.

**Aggregate score** — sum of `SearchResult.confidence` for the top-K reranked results:

```python
aggregate_confidence = sum(r.confidence for r in reranked_results[:k])
```

Why a sum rather than a mean: a single high-confidence chunk is insufficient if the query spans multiple topics; the sum rewards coverage. With K=5 and threshold=1.5 the system requires an average per-chunk confidence of 0.30 — a deliberately low floor that only blocks truly empty retrieval. Tighten `retrieval_confidence_threshold` per corpus as quality improves.

```python
# knowledge/config/settings.py additions
retrieval_confidence_threshold: float = 1.5   # aggregate sum of top-K confidences
judge_confidence_threshold: float = 0.60      # per judge_answer() call
judge_k: int = 5                              # top-K chunks fed to judge + generator
```

#### Layer 2 — Citation Gate (`knowledge/agent/agent.py`)

The Pydantic AI agent generates the answer as a structured output that includes an inline citation check. The LLM is required to ground every factual claim in a `chunk_id`; if it cannot, `is_trustworthy` is `False`.

```python
class CitationCheck(BaseModel):
    is_trustworthy: bool
    uncited_claims: list[str]   # claims the model couldn't attribute to a chunk

class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]       # Citation model from Retrieval Pipeline section
    citation_check: CitationCheck
```

System prompt constraint (always included):
> "Every factual statement in your answer MUST be supported by one of the provided source chunks, cited inline as [chunk_id]. If you cannot find a supporting chunk for a claim, omit that claim entirely. Do not invent information."

`is_trustworthy = len(uncited_claims) == 0`. If any claim is uncited, the pipeline returns `abstained_citation` without showing the answer.

This gate catches the failure mode where the LLM has memorised a plausible-sounding answer that happens to contradict or go beyond the retrieved context — independent of whether the retrieval score was high.

#### Layer 3 — Judge Gate (`knowledge/agent/judge.py`)

A separate LLM call (nano or small model tier, cheaper than the generation model) evaluates the answer against the context. The judge is deliberately independent: it receives only the query, context, and answer — not the citation metadata — so it cannot be fooled by a well-formatted but hallucinated citation.

```python
class JudgeResult(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"]
    confidence: float           # 0.0–1.0; judge's own confidence in its verdict
    reasoning: str              # short explanation (logged, not returned to user)

# Judge prompt (system):
# "You are an impartial evaluator. Given a question, a set of source passages,
#  and a generated answer, determine whether the answer is:
#  - supported: fully grounded in the passages
#  - partial: mostly grounded but missing or hedging on some aspects
#  - unsupported: contains claims not found in or contradicted by the passages
#  Return a JSON object with verdict, confidence (0-1), and reasoning."
```

Gate logic:
- `verdict == "unsupported"` OR `confidence < judge_confidence_threshold` → `abstained_judge`
- `verdict == "partial"` → answer proceeds but uncertainty note is appended
- `verdict == "supported"` AND `confidence >= judge_confidence_threshold` → `answered`

The judge uses the `nano` model tier by default. If the nano model's own `confidence` on the verdict is low (< 0.5), escalate the judge call to `small` — one level up. This avoids incorrect abstentions on ambiguous but answerable queries.

#### Pipeline Orchestrator (`knowledge/agent/pipeline.py`)

```python
class PipelineStatus(str, Enum):
    ANSWERED            = "answered"
    ABSTAINED_RETRIEVAL = "abstained_retrieval"   # Layer 1 gate
    ABSTAINED_CITATION  = "abstained_citation"    # Layer 2 gate
    ABSTAINED_JUDGE     = "abstained_judge"       # Layer 3 gate

class RAGResponse(BaseModel):
    answer: str
    status: PipelineStatus
    confidence: float | None           # judge confidence; None on abstentions
    citations: list[Citation] | None   # None on abstentions
    low_confidence_warning: bool       # True when verdict == "partial"
    pipeline_latency_ms: dict[str, int]  # {"retrieval": 120, "generation": 450, "judge": 80}
    # Cost fields — always populated; 0.0 for local Ollama models
    estimated_cost_usd: float          # total estimated cost for this request
    model_tier_used: str               # "nano" | "small" | "large" — what the router selected
    prompt_tokens: int                 # total input tokens across all LLM calls in pipeline
    completion_tokens: int             # total output tokens
    cache_hit: str | None              # "l2" | "l3" | None — which cache served this response
    # Observability
    request_id: str                    # UUID — correlates with logs, Langfuse trace, audit_events
    trace_url: str | None              # Langfuse trace URL (None when langfuse_enabled=False)
    # abstention fields (populated only on abstain)
    abstention_layer: int | None       # 1, 2, or 3
    abstention_reason: str | None
```

Abstention responses use fixed, corpus-configurable strings (not LLM-generated) — fast, deterministic, and safe from hallucination in the error path itself.

#### Hook Integration

Every gate fires its own hook point so observers and custom policies can intercept without touching pipeline logic:

| Gate outcome | Hook fired | HookContext additions |
|---|---|---|
| Layer 1 pass | `POST_RETRIEVE` | `aggregate_confidence`, `results` |
| Layer 1 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=1`, `aggregate_confidence` |
| Layer 2 pass | `POST_LLM` | `generation_result`, `citation_check` |
| Layer 2 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=2`, `uncited_claims` |
| Layer 3 pass | `POST_LLM` | `judge_result` |
| Layer 3 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=3`, `judge_verdict`, `judge_confidence` |
| Partial answer | `POST_LLM` | `judge_verdict="partial"`, note appended |

#### Evaluation Extension

Add to `EvalResult`:

```python
# Pipeline status tracking
pipeline_status: PipelineStatus
abstention_layer: int | None        # which layer gated (1/2/3)

# Per-layer confidence values (for tuning thresholds)
retrieval_aggregate_confidence: float
citation_trustworthy: bool | None
judge_verdict: str | None
judge_confidence: float | None

# Derived quality flags
false_abstention: bool   # pipeline abstained on a gold query that has a known GT answer
                         # = the system should have answered but didn't
```

Key eval metrics to track per corpus:

| Metric | Formula | Target |
|--------|---------|--------|
| Abstention rate | `abstained / total` | < 15% on gold dataset |
| False abstention rate | `abstained_on_answerable / answerable` | < 5% |
| Layer 1 abstention share | `abstained_layer1 / abstained` | diagnoses retrieval gaps |
| Layer 2 abstention share | `abstained_layer2 / abstained` | diagnoses citation/hallucination pressure |
| Layer 3 abstention share | `abstained_layer3 / abstained` | diagnoses judge threshold calibration |
| Partial answer rate | `partial / answered` | < 20% on gold dataset |

If `false_abstention_rate > 5%` → lower `retrieval_confidence_threshold` or `judge_confidence_threshold`. If `abstention_rate > 20%` on live traffic → likely a corpus coverage problem, not a threshold problem.

#### Threshold Calibration Workflow

Thresholds are not set once and forgotten. Calibrate per corpus using the gold dataset:

1. Run eval with `retrieval_confidence_threshold = 0` (disable Layer 1 gate) to get a baseline hit rate.
2. Sweep `retrieval_confidence_threshold` from 0.5 → 3.0; plot abstention rate vs. false abstention rate. Pick the knee point.
3. Repeat for `judge_confidence_threshold` from 0.4 → 0.8.
4. Re-run after every significant ingestion batch (new docs shift the confidence distribution).

Store calibration results alongside `eval_runs` in the `eval_runs.report_json` column.

---

### Model Tiering

Route queries to the cheapest model that can answer them. Saves VRAM, reduces latency, cuts cost.

#### Tier Definitions

All tiers default to local Ollama models. Cloud model IDs are listed for reference only and are gated behind `cloud_models_enabled = True` in settings (off by default).

| Tier | Local model (default) | Cloud model (opt-in) | Use cases |
|------|----------------------|----------------------|-----------|
| `nano` | `qwen2.5:0.5b` | `claude-haiku-4-5` | Input classification, intent detection, faithfulness/relevance evaluation, content policy check |
| `small` | `llama3.2:3b` | `claude-sonnet-4-6` | Standard RAG chat, document Q&A, summarisation, KG entity extraction (simple ontologies) |
| `large` | `llama3.1:70b` (q4) | `claude-opus-4-8` | Multi-hop reasoning, complex analysis, KG extraction on dense domains |

#### Routing Logic (`knowledge/agent/model_router.py`)

The router runs on the `nano` model so routing overhead is < 50 ms.

```
incoming query
    │
    ▼
QueryRouter (nano model, structured output)
    │  → complexity: "simple" | "moderate" | "complex"
    │  → requires_graph: bool
    │  → estimated_context_tokens: int
    │
    ├── "simple"   + context_tokens < 512  → Tier nano   (pure retrieval, no LLM rewrite)
    ├── "moderate" + context_tokens < 4096 → Tier small
    └── "complex"  OR requires_graph       → Tier large
```

**`QueryRouter` output schema**:
```python
class RoutingDecision(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    requires_graph: bool
    requires_multipass: bool   # triggers staged retrieval if True
    estimated_context_tokens: int
    rejected: bool             # True → query blocked before routing
    rejection_reason: str | None
```

**Forcing a tier**: clients may pass `model_tier: "small" | "large"` in the request body; the API honours it only if the JWT role includes `tier_override`. This lets power users or test harnesses bypass auto-routing.

**Fallback**: if the `nano` router call exceeds 3 s, default to `small`.

#### Configuration (`settings.py` additions)

```python
model_tier_nano: str = "qwen2.5:0.5b"
model_tier_small: str = "llama3.2:3b"
model_tier_large: str = "llama3.1:70b"
model_routing_enabled: bool = True
model_routing_timeout_s: float = 3.0
```

#### Observability

- `model_tier_selected_total{tier}` Prometheus counter — track tier distribution.
- `model_router_latency_seconds` histogram — ensure routing overhead stays < 100 ms P99.
- Log `routing_decision` as a structured field on every request trace.

---

### Query Validation & Hook System

All validation runs **before** the router and before any LLM or DB call. Reject fast.

#### Validation Pipeline (`knowledge/validation/pipeline.py`)

```
incoming request body
    │
    ├── [V1] Schema validation          — Pydantic model; type/length/format checks
    ├── [V2] Length guard               — reject if query > MAX_QUERY_CHARS (4096)
    ├── [V3] Language detection         — optional; reject if not in allowed_languages
    ├── [V4] Prompt injection detector  — regex + embedding-sim against known attack patterns
    ├── [V5] Content policy check       — nano-model classifier: "on_topic" | "off_topic" | "inappropriate"
    │         ├── "on_topic"       → pass
    │         ├── "off_topic"      → 422 Unprocessable Entity (polite decline)
    │         └── "inappropriate"  → 400 Bad Request + audit event flagged
    └── [V6] Corpus access check        — JWT roles vs. corpus RBAC (before any DB I/O)
```

**Content policy classifier** (`nano` model, structured output):
```python
class ContentPolicyResult(BaseModel):
    verdict: Literal["on_topic", "off_topic", "inappropriate"]
    confidence: float       # 0–1
    reason: str | None      # brief human-readable reason, logged but not returned to client
```

Corpus-specific topic scopes can be configured in `CorpusConfig.allowed_topics: list[str]`; the policy prompt includes them so the classifier rejects queries outside that domain.

#### Hook System (`knowledge/hooks/`)

Hooks are async callables invoked at named lifecycle points. They are **placeholders** — registered but no-ops until implemented. This gives extension points for custom policy, logging, or integration without touching core pipeline code.

```python
# knowledge/hooks/registry.py
class HookPoint(str, Enum):
    PRE_VALIDATE       = "pre_validate"        # before validation pipeline
    POST_VALIDATE      = "post_validate"       # after validation passes
    PRE_ROUTE          = "pre_route"           # before model router
    POST_ROUTE         = "post_route"          # after routing decision
    PRE_RETRIEVE       = "pre_retrieve"        # before retrieval
    POST_RETRIEVE      = "post_retrieve"       # after retrieval, before LLM
    PRE_LLM            = "pre_llm"             # before LLM call
    POST_LLM           = "post_llm"            # after LLM response
    PRE_INGEST         = "pre_ingest"          # before document ingestion
    POST_INGEST        = "post_ingest"         # after ingestion completes
    ON_CACHE_HIT       = "on_cache_hit"        # any cache layer hit
    ON_VALIDATION_FAIL = "on_validation_fail"  # query rejected
    ON_ERROR           = "on_error"            # unhandled exception in pipeline

Hook = Callable[[HookContext], Awaitable[HookContext | None]]

class HookRegistry:
    def register(self, point: HookPoint, fn: Hook, priority: int = 0) -> None: ...
    async def fire(self, point: HookPoint, ctx: HookContext) -> HookContext: ...
```

**`HookContext`** carries the full request state (query, corpus_id, user_id, routing_decision, retrieved_chunks, llm_response, error) and is passed through the hook chain. A hook can mutate context (e.g., redact PII from retrieved text) or raise `HookAbort` to short-circuit the pipeline with a custom response.

**Built-in placeholder hooks** (registered at app startup, body = `pass`):
- `audit_log_hook` at `POST_LLM` — emit audit event (stub; real impl in Phase F)
- `pii_redact_hook` at `POST_RETRIEVE` — placeholder for PII scrubbing before LLM sees context
- `response_filter_hook` at `POST_LLM` — placeholder for output filtering
- `metrics_hook` at every point — Prometheus counter increment (this one is real from Phase G)

---

### Guardrail Architecture — Key Principles

- **Layer 1 — Input guardrails** block ~90% of bad queries using cheap classifiers (nano model, regex, embedding-sim) before any retrieval or LLM call. Reject fast; pay the compute only on clean requests.
- **Layer 2 — Tool argument validation** checks tool call arguments, corpus permissions, and request scope before execution. No tool fires against a corpus the caller is not authorised to read.
- **Layer 3 — Execution monitoring** tracks agentic loop iteration counts, total tool calls per request, and access to sensitive resources (PII-tagged corpora, audit tables). Hard limits abort runaway loops before they cause damage or rack up cost.
- **Layer 4 — Output guardrails** check the generated response for toxicity, PII leakage, and factual grounding before it is returned to the client. Tied to the citation gate (Layer 2 of the Confidence-Aware Pipeline) and the judge gate (Layer 3).
- **Placement is a trade-off**: put expensive checks (LLM classifiers) late; put cheap checks (regex, schema validation, RBAC) early. Misplacing a slow guard on the hot path can add hundreds of milliseconds per request.
- **Multi-layer impact**: the combined approach targets ≥ 99% safe-output rate while reducing wasted compute by ~15% versus a single late-stage check — early rejection means no embedding call, no retrieval, no LLM invocation for invalid queries.
- **Measure before optimising**: always capture latency (per span), cost (tokens + infra), and quality (faithfulness, abstention rate) for each guard layer. Do not tighten or relax thresholds without a before/after eval run against the gold dataset.
- **Architecture and orchestration first**: the 4-layer guard structure, the Redis Streams worker model, and the confidence-aware pipeline matter more than the specific model selected at each tier. Swapping `qwen2.5:0.5b` for a different nano model should not require touching pipeline logic.
- **Production numbers matter**: target concrete SLAs — Layer 1 classifier < 50 ms P95, total validation chain < 100 ms P95, end-to-end search < 2 000 ms P95. Cite specific numbers in design reviews and postmortems; vague claims ("it's fast") are not actionable.
- **Ship a structured pipeline**: the deliverable is a complete, observable pipeline — architecture (module layout, data schemas), orchestration (worker lifecycle, hook system, confidence gates), and observability (Prometheus metrics, Langfuse traces, Grafana dashboards) — not just a working chat endpoint.

---

### Error Handling Strategy

Error handling is not defensive boilerplate — it is an explicit design decision for every failure mode. Every component in this architecture has a defined failure response that preserves system safety and gives the client actionable information.

#### Error Taxonomy

Errors are classified on two axes: **origin** (who caused it) and **recoverability** (can the system recover automatically).

| Class | Origin | Retriable | Examples |
|---|---|---|---|
| `CLIENT_ERROR` | Bad input, auth failure, quota | No | invalid query, expired JWT, budget exhausted |
| `TRANSIENT_ERROR` | Infrastructure blip | Yes (with backoff) | DB connection drop, Redis timeout, LLM overload (429) |
| `TIMEOUT_ERROR` | Deadline exceeded | Conditionally | embedding timeout, LLM generation exceeded SLA |
| `CAPACITY_ERROR` | System overloaded | No (return 503) | all DB pool slots in use, Redis OOM |
| `VALIDATION_FAILURE` | Policy rejection | No | content policy block, injection detected, RBAC deny |
| `ABSTENTION` | Deliberate pipeline gate | No | confidence gate, citation gate, judge gate |
| `PERMANENT_ERROR` | Unrecoverable failure | No (DLQ) | corrupt document, schema parse failure, auth misconfiguration |

#### Structured Error Response Schema

Every non-2xx response uses this envelope. The `error` field is never null on error, and `data` is always null on error.

```python
class ErrorDetail(BaseModel):
    code: str                    # machine-readable, SCREAMING_SNAKE_CASE
    message: str                 # human-readable; safe to show client
    details: dict[str, Any] = {} # structured context (field path, limit values, etc.)
    retry_after_s: int | None    # seconds; set only when retry is meaningful
    doc_url: str | None          # link to error documentation

class APIResponse(BaseModel):
    request_id: UUID
    data: Any | None             # None on error
    error: ErrorDetail | None    # None on success
    cache_hit: str | None        # "l2" | "l3" | None
```

**Example responses by error class:**

```json
// 429 — tenant budget exhausted
{
  "request_id": "...",
  "data": null,
  "error": {
    "code": "TENANT_BUDGET_EXHAUSTED",
    "message": "Monthly LLM budget exceeded. Search-only mode active until budget resets.",
    "details": { "budget_usd": 500.0, "spent_usd": 501.23, "resets_at": "2026-07-01T00:00:00Z" },
    "retry_after_s": null,
    "doc_url": "https://docs.example.com/errors/TENANT_BUDGET_EXHAUSTED"
  }
}

// 503 — LLM service unavailable (circuit breaker open)
{
  "request_id": "...",
  "data": null,
  "error": {
    "code": "LLM_CIRCUIT_OPEN",
    "message": "Generation service temporarily unavailable. Search results are still available.",
    "details": { "degraded_mode": "search_only" },
    "retry_after_s": 30
  }
}

// 422 — content policy rejection
{
  "request_id": "...",
  "data": null,
  "error": {
    "code": "CONTENT_POLICY_VIOLATION",
    "message": "Query was rejected by content policy.",
    "details": { "verdict": "inappropriate", "corpus_id": "hr-policies" },
    "retry_after_s": null
  }
}
```

#### HTTP Status Code Policy

| Status | When | Notes |
|---|---|---|
| `200` | Successful response, including abstentions | Abstentions are business logic, not errors; status field in body conveys outcome |
| `400` | Malformed request body (schema validation) | Pydantic `ValidationError` serialised into `error.details` |
| `401` | Missing or invalid JWT | Always return `WWW-Authenticate: Bearer` header |
| `403` | Valid JWT but insufficient role for corpus | Distinguish from 401; RBAC failure |
| `404` | Job ID / corpus ID not found | |
| `422` | Semantically invalid request (content policy, language mismatch) | Syntactically valid but rejected by policy |
| `429` | Rate limit or budget limit hit | Always set `Retry-After` and `X-Quota-Reset` headers |
| `500` | Unhandled exception in API process | Logged with full traceback; generic message to client |
| `502` | Worker unreachable (Redis Streams stale / worker crashed) | |
| `503` | Circuit breaker open; overload shed | Set `Retry-After`; specify degraded capability in body |
| `504` | Upstream timeout (LLM, embedding, DB) | Includes `details.timeout_stage` so client knows which component |

`500` is a bug. Any `500` in production is an incident and fires PagerDuty immediately.

#### Graceful Degradation Matrix

When a component is unavailable, the system degrades to its highest-quality remaining capability rather than failing completely. Degraded mode is declared in the response header `X-Degraded-Mode: <mode>`.

| Component down | Degraded mode | What still works | What fails |
|---|---|---|---|
| **Ollama / LLM** | `search_only` | Search, citations, cache hits | Generation, judge, model routing |
| **Redis** | `no_cache` | All queries served from DB; rate limiting uses DB counter | L2 cache, stream-based async ingest |
| **PostgreSQL** | `unavailable` | Nothing — primary datastore | Return 503 for all read/write paths |
| **Apache AGE** | `no_graph` | Vector + text retrieval | NL→Cypher graph traversal |
| **Embedding service** | `no_new_queries` | L2/L3 cache hits served | Any query requiring fresh embedding |
| **Reranker (CrossEncoder)** | `rrf_only` | Retrieval via RRF score; no reranking | Confidence-based gating; abstentions skip |
| **Langfuse** | `no_traces` | All queries served | Trace visibility; eval offline runs paused |

Degradation is detected per circuit breaker state. The health endpoint reports current degraded modes:

```json
// GET /health
{
  "status": "degraded",
  "degraded_modes": ["no_graph"],
  "components": {
    "postgres": "healthy",
    "redis": "healthy",
    "ollama": "healthy",
    "age_graph": "circuit_open",
    "langfuse": "healthy"
  }
}
```

#### Error Propagation — Worker Pipeline

Worker errors must not be silently swallowed. The propagation contract is:

```
TRANSIENT_ERROR in worker
    → retry with backoff (up to MAX_RETRIES)
    → on final failure: XACK job + publish to DLQ stream + update job hash:
        HSET job:{id} status "failed" error_code "..." error_msg "..." failed_at "..."
    → fire ON_ERROR hook → sends alert email + PagerDuty

PERMANENT_ERROR in worker
    → no retry: XACK + DLQ immediately
    → same alerting

API reads job hash on GET /v1/ingest/{id}/status
    → returns structured error in job status response body (not a 5xx — the API call itself succeeded)
```

Workers never raise unhandled exceptions to the consumer loop. Every `pipeline.run()` call is wrapped in `try/except BaseException` at the harness level — this is the one place where catching `BaseException` is correct, to prevent the consumer from crashing and losing the Redis `XPENDING` entry.

#### Alert Email Configuration

**All warnings and errors send email alerts to `rohan.vazirani@gmail.com`.** This is a mandatory deployment requirement — not optional, not production-only. Local development, staging, and production all alert to this address.

```yaml
# .env (required; scaffolded by install.sh / install.ps1)
ALERT_EMAIL=rohan.vazirani@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<sender>
SMTP_PASSWORD=<app_password>
SMTP_FROM=alerts@rag-system.local
```

Alert severity levels and delivery:

| Severity | Trigger | Channel |
|---|---|---|
| `CRITICAL` | 500 error, DLQ entry, circuit breaker opens, system budget breach | Email + PagerDuty |
| `WARNING` | P99 latency breach, cache hit rate < 20%, tenant budget at 80% | Email |
| `INFO` | Eval regression detected, new tenant provisioned, daily cost summary | Email (digest, 1×/day) |

Email is sent via `knowledge/observability/alerts.py` as a background `asyncio.Task` — never blocking the request path. Template:

```
Subject: [RAG] CRITICAL — LLM_CIRCUIT_OPEN on corpus hr-policies
Body:
  Time:     2026-06-06 14:32:01 UTC
  Severity: CRITICAL
  Code:     LLM_CIRCUIT_OPEN
  Corpus:   hr-policies
  Tenant:   acme-corp
  Request:  <request_id>
  Detail:   5 failures in 60s window. Circuit open. Retry probe in 30s.
  Trace:    https://langfuse.internal/trace/<trace_id>
```

Local dev alert delivery: if `SMTP_HOST` is not reachable, alerts are written to `logs/alerts.jsonl` and printed to stderr. Never silently dropped.

---

### Retry & Resilience Strategy

Retries are not a catch-all fallback. Every retry decision is explicit: what is retriable, how many times, with what backoff, and what happens when retries are exhausted.

#### Retriable vs Non-Retriable Classification

| Error | Retriable | Reason |
|---|---|---|
| `RateLimitError` (LLM / embedding API) | Yes | Transient; provider will accept request after backoff |
| `APIConnectionError` / `APITimeoutError` | Yes | Network blip; idempotent read or write |
| `asyncpg.ConnectionDoesNotExistError` | Yes | Pool connection died; pool will hand new connection |
| `asyncpg.TooManyConnectionsError` | Yes | Pool exhausted; backoff and retry |
| `asyncpg.QueryCanceledError` | Conditional | Retry only if `command_timeout` was set (our timeout); not if Postgres cancelled for lock |
| `redis.ConnectionError` | Yes | Redis transient; up to 3 attempts |
| `redis.TimeoutError` | Yes | |
| `AuthenticationError` (LLM / embedding) | No | Permanent misconfiguration; alert and fail |
| `InvalidRequestError` (bad prompt) | No | Permanent; retrying will produce same error |
| `ContentPolicyError` | No | Permanent; retrying is futile and wastes tokens |
| `pydantic.ValidationError` | No | Input data is malformed; retrying won't fix it |
| `asyncpg.IntegrityConstraintViolationError` | No | Duplicate insert; not a transient failure |
| `PermissionDeniedError` (RBAC) | No | Permanent |
| Ingest job — document parse failure (Docling) | No | Corrupt or unsupported file; DLQ immediately |
| Ingest job — embedding timeout | Yes | Transient; full backoff policy applies |
| Ingest job — graph extraction failure | Yes | LLM transient; up to 3 attempts; on final failure, skip graph path and proceed with vector-only |

Graph extraction has a dedicated soft-failure policy: after exhausting retries, the document is ingested as vector-only and `graph_extraction_failed: true` is set in chunk metadata. The job is not moved to DLQ — a partial ingest is better than no ingest.

#### Backoff Specification

```python
# knowledge/bus/backoff.py
import random

def exponential_backoff(
    attempt: int,           # 1-indexed
    base_s: float = 5.0,
    multiplier: float = 2.0,
    max_s: float = 125.0,
    jitter_factor: float = 0.15,
) -> float:
    """
    Backoff with partial jitter (15% of raw delay) to prevent thundering herd.
    Note: "full jitter" would be uniform(0, raw); this uses a smaller jitter window
    to bound worst-case delay while still preventing synchronised retry storms.
    attempt=1 → ~5s, attempt=2 → ~10s, attempt=3 → ~20s (capped at max_s).
    Jitter = uniform(0, jitter_factor × raw_backoff).
    """
    raw = min(base_s * (multiplier ** (attempt - 1)), max_s)
    jitter = random.uniform(0, jitter_factor * raw)
    return raw + jitter
```

Default backoff schedule (base=5s, 3 attempts):

| Attempt | Base | With jitter (typical) | Cumulative |
|---|---|---|---|
| 1st | 5 s | 5–5.75 s | 5 s |
| 2nd | 10 s | 10–11.5 s | 15 s |
| 3rd (final) | 20 s | 20–23 s | 35 s |
| → DLQ | — | — | — |

Embedding API uses shorter base (1s, max 15s) since it's a fast network call. DB retries use shorter base (0.5s, max 5s) since pool recovery is fast.

#### Circuit Breaker Design

One circuit breaker per external service. Implemented in `knowledge/bus/circuit_breaker.py`.

```
States:
  CLOSED   → normal; requests pass through; failure counter maintained
  OPEN     → all requests blocked immediately; probe timer running
  HALF-OPEN → one probe request allowed; success → CLOSED; failure → OPEN

Transitions:
  CLOSED → OPEN:       failure_count >= OPEN_THRESHOLD in last WINDOW_SECONDS
  OPEN → HALF-OPEN:    PROBE_INTERVAL_S elapsed since circuit opened
  HALF-OPEN → CLOSED:  CONSECUTIVE_SUCCESS_THRESHOLD successes in half-open
  HALF-OPEN → OPEN:    any failure in half-open state

Default thresholds:
  OPEN_THRESHOLD:               5 failures
  WINDOW_SECONDS:               60
  PROBE_INTERVAL_S:             30
  CONSECUTIVE_SUCCESS_THRESHOLD: 2
```

Circuit breakers are per-service, not per-tenant. A single slow LLM call does not trip the breaker; five failures in a minute does.

```python
# knowledge/bus/circuit_breaker.py
class CircuitBreaker:
    def __init__(self, name: str, redis: Redis, settings: CircuitBreakerSettings): ...

    async def call(self, coro: Awaitable[T]) -> T:
        state = await self._get_state()
        if state == "open":
            raise CircuitOpenError(service=self.name, retry_after_s=self._probe_remaining())
        try:
            result = await coro
            await self._record_success()
            return result
        except RETRIABLE_EXCEPTIONS as exc:
            await self._record_failure()
            raise
```

Circuit state is stored in Redis (`cb:{name}:state`, `cb:{name}:failures`, `cb:{name}:opened_at`) so all API pod replicas share the same view. A circuit that opens on one pod is immediately open on all pods.

When a circuit opens, it fires `ON_ERROR` hook → email alert to `rohan.vazirani@gmail.com` + PagerDuty.

#### Idempotency Design

**Ingest jobs**: identified by `sha256(file_content + corpus_id)`. Before processing, the worker checks `cache:doc_fingerprint:{sha256}` in Redis (or `documents.metadata->>'content_hash'` in PostgreSQL on cache miss). If already processed and unchanged, job is ACKed without re-ingestion. This makes ingest retries safe — re-enqueuing a job for a document that already succeeded is a no-op.

**Vector upserts**: `INSERT ... ON CONFLICT (source) DO UPDATE` — idempotent by design. Partial ingestion (worker crash mid-batch) is recovered by re-running; chunks are upserted, not duplicated.

**LLM calls**: not inherently idempotent. For the judge gate, if the LLM call times out, the default is **pessimistic abstention** — treat as `abstained_judge` rather than retrying and potentially returning a different verdict. For generation, the request is retried once within the SLA budget; a second timeout returns `GENERATION_TIMEOUT` to the client.

**Cache writes**: Redis writes use `SET key value EX ttl NX` (set-if-not-exists) where duplicate prevention matters. For L2 search cache, `SET ... NX` prevents two concurrent request completions from overwriting each other.

#### Cascading Timeout Budget

The API request deadline is the parent budget. Each downstream call carves a sub-deadline from the remaining parent budget.

```python
# knowledge/api/timeout.py
@dataclass
class TimeoutBudget:
    total_s: float = 30.0       # API hard deadline

    validation_s: float = 0.2
    routing_s: float = 3.0      # includes one retry within budget
    embedding_s: float = 5.0    # includes one retry within budget
    retrieval_s: float = 8.0
    rerank_s: float = 3.0
    semantic_cache_s: float = 1.0
    generation_s: float = 15.0  # streaming TTFT must start within this
    judge_s: float = 5.0

    # Remaining budget is slack / buffer for I/O overhead.
    # If any stage exceeds its sub-budget, the overall deadline propagates:
    # asyncio.wait_for(stage_coro, timeout=min(stage_s, remaining_parent_budget))
```

If `generation_s` is exhausted mid-stream, the SSE connection sends a `data: {"type": "error", "code": "GENERATION_TIMEOUT"}` event and closes. Partial streamed tokens are not truncated — the stream is left open until the budget expires, then closed with the error event.

#### Worker Retry Loop

```python
# knowledge/bus/consumer.py
async def consume_loop(stream: str, group: str, worker_id: str, handler: Handler) -> None:
    while True:
        messages = await xreadgroup(stream, group, worker_id, count=1, block_ms=5000)
        for msg_id, payload in messages:
            job = deserialize(payload)
            await _execute_with_retry(msg_id, job, handler)

async def _execute_with_retry(msg_id: str, job: Job, handler: Handler) -> None:
    attempt = job.attempt  # stored in job payload; incremented on re-enqueue
    try:
        await asyncio.wait_for(handler(job), timeout=JOB_TIMEOUT_S)
        await xack(msg_id)                        # success: ACK and done
    except NON_RETRIABLE_EXCEPTIONS as exc:
        await xack(msg_id)
        await move_to_dlq(job, exc, permanent=True)
        await fire_hook(ON_ERROR, error=exc, job=job)
    except (RETRIABLE_EXCEPTIONS, asyncio.TimeoutError) as exc:
        if attempt >= MAX_RETRIES:
            await xack(msg_id)
            await move_to_dlq(job, exc, permanent=False)
            await fire_hook(ON_ERROR, error=exc, job=job)
        else:
            backoff_s = exponential_backoff(attempt)
            await asyncio.sleep(backoff_s)
            await re_enqueue(job, attempt=attempt + 1)  # new XADD with incremented attempt
            await xack(msg_id)                          # ACK original; re-enqueued copy takes over
```

`MAX_RETRIES = 3`. After 3 failures, the job enters DLQ and an alert fires. The DLQ is never silently drained — every DLQ entry is an incident requiring human review.

---

### Security Layer — JWT, JWE, HTTPS, RBAC

#### Authentication — JWT (RS256)

- **Issuer**: dedicated auth service (or Auth0 / Cognito in cloud).
- **Algorithm**: RS256; private key signs tokens, public JWKS endpoint for verification.
- **Access token**: 15-minute TTL; contains `sub`, `roles: list[str]`, `tenant_id`.
- **Refresh token**: 7-day TTL; rotated on use; stored server-side in Redis (`SET rt:{jti} <user_id> EX 604800`).
- **API dependency** (`knowledge/api/auth.py`): `Depends(require_jwt)` on all non-health routes; extracts roles, checks corpus RBAC.
- **JWKS caching**: public keys cached in process for 1 hour to avoid hot-path network calls.

#### Payload Encryption — JWE (A256GCM)

- Sensitive search responses and cached answers stored as JWE compact-serialised blobs.
- Algorithm: `ECDH-ES+A256KW` (ephemeral ECDH key agreement + AES-256-KW) with `A256GCM` content encryption.
- Per-tenant encryption keys; key rotation via versioned key IDs in the JWE header.
- Semantic cache stores encrypted answers (`answer_jwe TEXT`) — decryption happens in-process only after JWT auth passes.
- Library: `python-jose` or `joserfc` (preferred, pure-Python, actively maintained).

#### Transport — HTTPS / TLS

- TLS 1.3 only; TLS 1.2 disabled at NGINX layer.
- HSTS header: `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`.
- Local dev: self-signed cert via `mkcert`; mounted into NGINX container.
- Cloud: ACM / GCP-managed certificates on ALB / Cloud Load Balancer; automatic renewal.
- Internal service-to-service: mTLS enforced via Istio sidecar proxies (cloud only).

#### RBAC

Roles embedded in JWT `roles` claim, checked against `CorpusConfig.allowed_roles`:
- `reader` — search + chat on allowed corpora.
- `writer` — ingest + delete documents.
- `admin` — corpus management, cache invalidation, audit log access.
- `service` — internal worker-to-API calls (machine identity tokens, short TTL).

#### Audit Log

Append-only `audit_events` table (never UPDATE or DELETE):
```sql
CREATE TABLE audit_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    action      TEXT NOT NULL,   -- "search", "ingest", "delete", "cache_invalidate"
    corpus_id   TEXT,
    query_text  TEXT,            -- hashed in production (SHA-256), not plaintext
    request_id  UUID NOT NULL,
    ip_address  INET,
    response_ms INTEGER
);
CREATE INDEX ON audit_events (user_id, ts DESC);
CREATE INDEX ON audit_events (tenant_id, ts DESC);
```

Emit from `knowledge/api/middleware.py` as a background task (non-blocking, fire-and-forget).

#### Input Validation

- Prompt injection guard: regex + embedding-similarity check against known injection patterns.
- Query length cap: `MAX_QUERY_CHARS = 4096` (configurable).
- File type allowlist enforced at ingest time; MIME sniffing, not just extension.
- Rate limiting: `slowapi` per-user (JWT `sub`) rather than per-IP in production.

---

### Memory Architecture

The system uses five distinct memory tiers mapping to cognitive science memory types. Full design details and pruning/eviction strategy are in `basics/rag/memory/MEMORY_DESIGN.md` — this section captures the decisions that affect the module layout, database schema, and API.

#### Five Memory Tiers

| Cognitive type | Tier | Storage | Lifespan |
|----------------|------|---------|----------|
| **Short-term / Working** | 1 | RAM (context window) | Per request |
| **Episodic** | 2 | PostgreSQL `conversations` + `messages` | 90 days (configurable) |
| **Semantic — user** | 3 | PostgreSQL + pgvector `user_memories` | Indefinite; user-controlled |
| **Semantic — world** | 4 | PostgreSQL + pgvector + Apache AGE | Until deleted |
| **Procedural** | 5 | Files + DB `system_prompts` | Indefinite; versioned |

#### Critical design decision: server-side conversation history

The current `ChatRequest` contains `message_history: list | None`. **This is wrong for production.** Passing history as a request field means multi-device fails, history is lost on tab close, and 30-turn conversations send 30× the payload.

```
Wrong:  ChatRequest { query, session_id, message_history: [...] }
Correct: ChatRequest { query, session_id }
         → server loads history from DB by session_id
```

`message_history` is removed from `ChatRequest`. The server loads the last 8 turns (or `summary + last 8` for long conversations) from the `messages` table on every request.

#### Module additions to knowledge/memory/

```
knowledge/memory/
├── __init__.py
├── mem0_store.py          # Tier 3: Mem0-backed user semantic memory (EXISTING — port from rag/)
├── conversation_store.py  # Tier 2: episodic conversation + message CRUD
├── summarizer.py          # Tier 2: auto-summarization trigger + nano model call
├── working_memory.py      # Tier 1: context assembly + token-budget trim logic
└── pruning.py             # Background jobs: Tier 2 TTL eviction, Tier 3 LRU + compaction
```

#### Schema additions (migration 008_memory.sql)

```sql
CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL UNIQUE,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,            -- SHA-256(sub + tenant_salt)
    corpus_ids      TEXT[] NOT NULL,
    title           TEXT,
    summary         TEXT,                     -- auto-set when turn_count > 20
    turn_count      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_turn_at    TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    deleted_at      TIMESTAMPTZ
);
CREATE INDEX ON conversations (user_id, last_turn_at DESC);

CREATE TABLE messages (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id   UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT NOT NULL,
    citations         JSONB,
    pipeline_status   TEXT,
    confidence        FLOAT,
    model_tier        TEXT,
    prompt_tokens     INT,
    completion_tokens INT,
    cost_usd          FLOAT,
    cache_hit         TEXT,
    request_id        UUID,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON messages (conversation_id, created_at);

CREATE TABLE user_memories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL,         -- SHA-256(sub + tenant_salt)
    tenant_id           TEXT NOT NULL,
    content             TEXT NOT NULL,
    embedding           vector(768),
    source_message_id   UUID,
    last_retrieved_at   TIMESTAMPTZ,           -- for LRU eviction
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON user_memories (user_id, tenant_id);
CREATE INDEX ON user_memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE system_prompts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    content     TEXT NOT NULL,
    version     INT NOT NULL DEFAULT 1,
    active      BOOLEAN NOT NULL DEFAULT FALSE,
    corpus_id   TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    created_by  TEXT NOT NULL
);
CREATE UNIQUE INDEX ON system_prompts (name, version);
```

#### Token budget trim order (Tier 1)

When assembled context exceeds 8,192 tokens, trim in this order — lower-priority items dropped first:

1. Drop lowest-confidence retrieved chunks (Tier 4)
2. Replace oldest message turns with conversation summary (Tier 2)
3. Reduce user memories to top-1 (Tier 3)
4. Emit `context_truncated: true` — never fail silently

Never trim: system prompt (Tier 5), current query.

#### Pruning, eviction, and compaction

Full algorithm in `basics/rag/memory/MEMORY_DESIGN.md §Memory Pruning`. Summary:

| Tier | Mechanism | Trigger |
|------|-----------|---------|
| 2 (Episodic) | TTL eviction: `DELETE WHERE expires_at < NOW()` | Nightly job |
| 2 (Episodic) | Compaction: summarize turns 1→(N-8), delete raw rows | When `turn_count > 20` |
| 3 (Semantic/user) | LRU eviction: drop memories not retrieved in 60 days | When count > 200 (hard cap) |
| 3 (Semantic/user) | Contradiction resolution | Mem0 on every `add()` |
| 3 (Semantic/user) | Compaction: merge similar memories (cosine ≥ 0.85) | Weekly background job |
| 4 (Knowledge) | Incremental delete by document_id | On re-ingest |
| 4 (Knowledge) | HNSW index rebuild | After > 20% of corpus deleted |

#### Framework: Mem0 for Tier 3 only

Mem0 (open-source, pgvector-backed) handles user semantic memory extraction, deduplication, contradiction resolution, and cosine retrieval. No other external memory framework is needed:

- **Zep** would handle Tier 2 but adds a service dependency — PostgreSQL is sufficient
- **Letta/MemGPT** is designed for autonomous agents, not RAG systems
- **LangMem** requires LangChain; not compatible with Pydantic AI stack

---

### API Layer

**Base URL**: `/api/v1` — all routes below are relative to this prefix (full path: `/api/v1/ingest`, `/api/v1/chat`, etc.)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/auth/token` | none | Issue JWT access + refresh tokens |
| `POST` | `/v1/auth/refresh` | none | Rotate refresh token; return new access token |
| `POST` | `/v1/ingest` | `writer` | Submit ingest job; returns `job_id` |
| `GET`  | `/v1/ingest/{job_id}/status` | `writer` | Poll job status |
| `GET`  | `/v1/ingest/{job_id}/stream` | `writer` | SSE job progress stream |
| `POST` | `/v1/search` | `reader` | Synchronous hybrid search (< 200 ms fast path) |
| `POST` | `/v1/chat` | `reader` | Agent chat (blocking) |
| `POST` | `/v1/chat/stream` | `reader` | Agent chat with SSE streaming (**POST**, not GET — body carries corpus_ids + message_history) |
| `GET`  | `/v1/corpus` | `reader` | List corpora accessible to JWT role |
| `POST` | `/v1/corpus/{id}/cache/invalidate` | `admin` | Flush L2+L3 caches for corpus |
| `GET`  | `/v1/scheduler/jobs` | `writer` | List scheduled ingestion jobs |
| `POST` | `/v1/scheduler/jobs` | `writer` | Create scheduled job |
| `PATCH`| `/v1/scheduler/jobs/{id}` | `writer` | Update trigger or source |
| `DELETE`| `/v1/scheduler/jobs/{id}` | `writer` | Cancel job |
| `POST` | `/v1/scheduler/jobs/{id}/run-now` | `writer` | Trigger immediate one-off run |
| `POST` | `/v1/evaluate/run` | `admin` | Trigger offline eval run |
| `GET`  | `/v1/evaluate/run/{id}` | `admin` | Poll run status + aggregated metrics |
| `GET`  | `/v1/evaluate/run/{id}/results` | `admin` | Per-sample results (paginated) |
| `GET`  | `/v1/evaluate/compare` | `admin` | Regression diff: `?a={id}&b={id}` |
| `POST` | `/v1/feedback` | `reader` | Submit explicit user feedback |
| `POST` | `/v1/signals` | `service` | Submit implicit behavioural signal |
| `GET`  | `/v1/conversations` | `reader` | List conversations for current user (newest first, paginated) |
| `GET`  | `/v1/conversations/{id}` | `reader` | Get conversation + messages |
| `DELETE` | `/v1/conversations/{id}` | `reader` | Delete conversation (GDPR erasure) |
| `GET`  | `/v1/memories` | `reader` | List user memories |
| `POST` | `/v1/memories` | `reader` | Manually add a memory |
| `DELETE` | `/v1/memories/{id}` | `reader` | Delete one memory |
| `DELETE` | `/v1/memories` | `reader` | Delete ALL memories (right to erasure) |
| `GET`  | `/v1/admin/system-prompts` | `admin` | List system prompt versions |
| `POST` | `/v1/admin/system-prompts` | `admin` | Create / activate a prompt version |
| `GET`  | `/health` | none | Liveness + readiness (pool, Redis, worker heartbeats) |
| `GET`  | `/metrics` | `service` | Prometheus metrics endpoint |

**Versioning**: URL prefix `/api/v1`; future breaking changes get `/api/v2`. Non-breaking changes are additive within v1.

**User identity on every request:**
- `user_id` — extracted from JWT `sub` claim by `require_jwt` dependency; never sent by the client directly
- `session_id` — required field in `ChatRequest` body; generated by the frontend as a UUID when a new conversation is created; identifies a conversation thread across multiple turns; used for Langfuse trace grouping, audit log, log correlation, and multi-turn history retrieval
- Both fields appear on every structured log line, every Langfuse trace, and every `audit_events` row

**Response envelope**:
```json
{
  "request_id": "uuid",
  "data": { ... },
  "error": null,
  "cache_hit": "l2" | "l3" | null
}
```

---

### Docker Compose — Local Dev

> **File:** `backend/docker-compose.yml` — this is the backend-only compose file. A top-level `docker-compose.yml` at the repo root extends it to add the `frontend` service. See TODO_implementation.md Phase 13 for the full file content.
>
> **Note on the `postgres` image:** `apache/age:latest` bundles Apache AGE but does **not** automatically include pgvector. Verify the image includes pgvector before using it, or use a custom image that installs both extensions. The existing `docker-compose.yml` at repo root maps AGE to port 5433; this design uses port 5432 — adjust if running both side by side.

```yaml
# backend/docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports: ["443:443", "80:80"]
    volumes: [./infra/nginx/nginx.conf, ./infra/certs:/certs:ro]
    depends_on: [api]

  api:
    build: .
    command: uvicorn knowledge.api.app:app --host 0.0.0.0 --port 8000 --workers 2
    env_file: .env
    depends_on: [postgres, redis, ollama]

  ingest-worker:
    build: .
    command: python -m knowledge.ingestion.worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]

  retrieval-worker:
    build: .
    command: python -m knowledge.retrieval.worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]

  postgres:
    image: apache/age:latest              # includes pgvector + Apache AGE
    environment: [POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]
    volumes: [pgdata:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    command: redis-server --save 60 1 --appendonly yes
    volumes: [redisdata:/data]
    ports: ["6379:6379"]

  ollama:
    image: ollama/ollama:latest
    volumes: [ollamamodels:/root/.ollama]
    ports: ["11434:11434"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]

  langfuse:        # optional observability profile
    image: langfuse/langfuse:latest
    profiles: [observability]
    depends_on: [langfuse-postgres]

  prometheus:
    image: prom/prometheus:latest
    profiles: [observability]
    volumes: [./infra/prometheus.yml:/etc/prometheus/prometheus.yml]

  grafana:
    image: grafana/grafana:latest
    profiles: [observability]
    depends_on: [prometheus]

volumes:
  pgdata:
  redisdata:
  ollamamodels:
```

**Profiles**:
- `docker compose up` — core services (api, workers, postgres, redis, ollama, nginx)
- `docker compose --profile observability up` — adds Langfuse, Prometheus, Grafana

---

### Packaging & Developer Install

The `knowledge/` module (and the current `rag/` module it replaces) is packaged as a standard Python project using **uv** and **hatchling**. The goal is a single command from a clean machine to a running system.

#### Package Manager — uv

uv replaces pip + virtualenv in one tool. Key commands:

```bash
uv sync                    # create .venv, install core deps from uv.lock
uv sync --extra all        # install every optional feature
uv sync --extra ingestion  # core + Docling ingestion only
uv run python -m rag.main  # run inside the managed venv (no activate needed)
```

The `uv.lock` file is committed to the repo. It pins every transitive dependency so any developer gets an identical environment regardless of when they clone.

#### Optional Extras Architecture

Core dependencies (always installed): Pydantic AI, asyncpg, pgvector, FastAPI, httpx. Heavy or optional features are gated behind named extras so a CI container or production image can install only what it needs.

| Extra | Key packages | When to include |
|-------|-------------|-----------------|
| `ingestion` | `docling`, `transformers` | Any node that runs `--ingest` |
| `audio` | `openai-whisper` | Audio ingestion only; also needs FFmpeg in PATH |
| `ui` | `streamlit` | Developer workstations + Streamlit deployments |
| `observability` | `langfuse` | Staging + production; not needed in CI unit tests |
| `mcp` | `mcp` | MCP server deployments only |
| `reranker` | `sentence-transformers` | API pods when `reranker_enabled = True` |
| `mem0` | `mem0ai` | API pods when `mem0_enabled = True` |
| `nl2sql` | `sqlglot` | NL-to-SQL service pods |
| `all` | everything | Local development (default) |

In Docker images, use targeted extras to keep image size down:

```dockerfile
# API image — no UI, no audio
RUN uv sync --extra ingestion --extra observability --extra reranker --extra mcp --no-dev

# Ingest-worker image
RUN uv sync --extra ingestion --extra audio --extra observability --no-dev
```

#### Install Script

Two scripts cover all platforms. Both do the same thing: install uv if missing, scaffold `.env`, run `uv sync --extra all`, and start the pgvector container.

```powershell
# Windows (PowerShell)
.\install.ps1

# Linux / macOS (Bash)
chmod +x install.sh && ./install.sh
```

After the script completes:
1. Edit `.env` — set `DATABASE_URL`, `LLM_*`, `EMBEDDING_*`
2. `ollama serve` — start Ollama
3. `ollama pull llama3.1:8b && ollama pull nomic-embed-text`
4. `uv run python -m rag.main --validate` — smoke-test the connection (v1 entrypoint; use `python -m knowledge.main --validate` once v2 is complete)
5. `uv run python -m rag.main --ingest --documents rag/documents` (v1 entrypoint; v2 uses the ingest worker via Redis)

#### Latency & Safety at Install Time

- **Guardrails and observability are off by default** (`langfuse_enabled = False`, `reranker_enabled = False`, `mem0_enabled = False`). Turn each on explicitly in `.env` once the required service is running. This prevents the install script from failing if Langfuse or a reranker endpoint isn't up yet.
- **Always measure before optimising**: run `uv run pytest rag/tests/core/ -v` first (no external deps, < 5 s). Only then run the integration suite once PostgreSQL and Ollama are confirmed healthy.
- **Specific numbers to target out of the box**: `uv sync --extra all` < 3 min on a fresh machine (dominated by Docling + Whisper downloads); `--validate` round-trip < 500 ms; first `--ingest` on the sample docs < 60 s on CPU-only Ollama.

#### pyproject.toml Key Fields (reference)

```toml
[project]
name = "rag-agent"
requires-python = ">=3.13"
# core deps here — see pyproject.toml for full list

[project.optional-dependencies]
ingestion    = ["docling>=2.14.0", "docling-core>=2.4.0", "transformers>=4.47.0"]
audio        = ["openai-whisper>=20240930"]
ui           = ["streamlit>=1.40.0"]
observability = ["langfuse>=2.0.0"]
mcp          = ["mcp>=1.0.0"]
reranker     = ["sentence-transformers>=3.0.0"]
mem0         = ["mem0ai>=0.1.0"]
nl2sql       = ["sqlglot>=25.0.0"]
all          = ["rag-agent[ingestion,audio,ui,observability,mcp,reranker,mem0,nl2sql]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# During migration: include both v1 (rag, kg) and v2 (knowledge) packages.
# After v2 reaches feature parity and v1 is retired: packages = ["knowledge", "nl2sql"]
packages = ["rag", "kg", "knowledge", "nl2sql"]

[tool.uv]
dev-dependencies = ["pytest>=8.3.0", "pytest-asyncio>=0.24.0", "ruff>=0.8.0", "mypy>=1.11.0"]
```

---

### Cloud Deployment — Production

#### Infrastructure Overview

```
Internet
   │
   ▼
WAF (AWS Shield / Cloudflare)
   │
   ▼
ALB / Cloud Load Balancer   ← TLS termination (ACM / GCP-managed cert)
   │
   ▼
EKS / GKE Cluster
├── Deployment: api                (2–10 pods, HPA on CPU/request rate)
├── Deployment: ingest-worker      (2–20 pods, HPA on Redis stream length)
├── Deployment: retrieval-worker   (2–10 pods, HPA on Redis stream length)
└── Istio sidecar mesh             (mTLS, traffic policies, circuit breakers)
   │
   ├── AWS Aurora PostgreSQL (Multi-AZ, pgvector enabled)
   │     └── Read replica for retrieval workers
   ├── ElastiCache Redis (Cluster Mode, 3 shards, Multi-AZ)
   ├── AGE-specific PostgreSQL (separate RDS instance or container if AGE not Aurora-compatible)
   └── S3 / GCS bucket (raw document storage, pre-signed upload URLs)
```

#### Secrets & Config

- Secrets: AWS Secrets Manager / GCP Secret Manager — DB passwords, JWT private keys, API keys.
- Config: Kubernetes ConfigMaps for non-secret settings; sealed-secrets for GitOps.
- Never pass secrets via environment variables in pod specs — use projected volumes from CSI secrets store driver.

#### Auth in Cloud

- JWT issuer: AWS Cognito User Pool (or Auth0 tenant).
- JWKS endpoint cached at API pods; key rotation handled by issuer.
- JWE keys: per-tenant RSA-OAEP keys stored in Secrets Manager; loaded at startup.
- mTLS: Istio-managed certificates (SPIFFE/SVID); zero-trust pod-to-pod.

#### Scaling Rules

| Component | Scale Trigger | Min | Max |
|---|---|---|---|
| `api` | CPU > 60% or req latency P99 > 500 ms | 2 | 10 |
| `ingest-worker` | Redis stream `knowledge:ingest` pending > 50 | 2 | 20 |
| `retrieval-worker` | Redis stream `knowledge:search` pending > 20 | 2 | 10 |
| PostgreSQL | Vertical + read replicas | — | — |
| Redis | ElastiCache shard add (manual / CloudWatch alarm) | 3 shards | 9 shards |

#### Observability Stack

- **Tracing**: OpenTelemetry SDK → AWS X-Ray / GCP Cloud Trace; Langfuse for LLM-specific traces.
- **Metrics**: Prometheus via `prometheus-client` → scrape by Grafana Cloud or CloudWatch Container Insights.
- **Logs**: structlog JSON → CloudWatch Logs / GCP Cloud Logging; correlation ID on every log line.
- **Alerts**: PagerDuty integration; alert on DLQ depth > 0, P99 search latency > 1 s, L3 cache hit rate < 20%.

See "Log Storage" section below for where logs land in local dev vs. production.

---

### Log Storage

#### What generates logs

Three distinct instrumentation layers, each stored differently:

| Layer | Tool | What it captures |
|-------|------|-----------------|
| **Structured request logs** | `structlog` (JSON) | Every HTTP request: request_id, user_id, corpus_id, route, latency_ms, status_code, cache_hit |
| **LLM + agent traces** | Pydantic AI (built-in) + Logfire | Every `agent.run()` / `agent.run_stream()` call: tool calls, token usage, model, latency — captured automatically |
| **Ingestion / worker logs** | `structlog` (JSON) | Job lifecycle: job_id, corpus_id, stage, duration_ms, chunk_count, error |
| **Alert events** | `knowledge/observability/alerts.py` | SMTP email + JSONL fallback |
| **Audit trail** | PostgreSQL `audit_events` table | Who queried what corpus when (for compliance; never deleted) |

#### Pydantic AI built-in usage tracking

Pydantic AI exposes token usage directly from every run — no manual token interception needed:

```python
# Blocking run
result = await agent.run("query", deps=state)
usage = result.usage()
# usage.request_tokens, usage.response_tokens, usage.total_tokens

# Streaming run — usage available after stream completes
async with agent.run_stream("query", deps=state) as streamed:
    async for delta in streamed.stream_text(delta=True):
        yield delta
usage = streamed.usage()
```

`RAGResponse.estimated_cost_usd` and `RAGResponse.prompt_tokens` / `completion_tokens` are populated from this usage object — not from manual token counting.

#### Pydantic AI + Langfuse tracing

We use **Langfuse** (self-hosted, open-source) for LLM traces — already in the Docker Compose setup. The `langfuse` Python SDK wraps each `agent.run()` call with a trace.

Token usage comes from Pydantic AI's built-in `result.usage()` — no manual interception, no third-party paid service:

```python
# knowledge/observability/langfuse.py
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()   # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env

@observe(name="rag_agent_run")
async def traced_agent_run(query: str, ...) -> RAGResponse:
    result = await agent.run(query, deps=state)
    usage = result.usage()
    langfuse_context.update_current_observation(
        usage={"input": usage.request_tokens, "output": usage.response_tokens},
        model=settings.model_tier_small,
    )
    return build_rag_response(result, usage)
```

`RAGResponse.trace_url` is the Langfuse trace URL for that specific request (e.g. `http://localhost:3001/trace/{trace_id}`), enabling one-click jump from the UI debug panel to the full tool-call trace.

#### Where logs are stored — by environment

**Local development (Docker Compose):**

All structured logs go to **stdout**, which Docker captures per-container. Access them with:

```bash
docker compose logs -f api            # API request logs
docker compose logs -f ingest-worker  # ingestion job logs
docker compose logs -f retrieval-worker
```

Or tail all services simultaneously:

```bash
docker compose logs -f 2>&1 | grep '"level":"ERROR"'   # errors only
docker compose logs -f 2>&1 | jq -r 'select(.request_id) | [.level, .request_id, .latency_ms, .route] | @tsv'
```

**Alert fallback (local dev only):** when `SMTP_HOST` is unreachable, alerts are additionally written to `backend/logs/alerts.jsonl`. This file is in the backend file tree (`backend/logs/` directory, git-ignored). It is only a safety net — never the primary log store.

**LLM traces (local dev):** Logfire / Langfuse are optional. Run the observability Docker profile to get them locally:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up langfuse
```

**Staging / Production:**

| Destination | Tool | Retention |
|-------------|------|-----------|
| Application logs (stdout) | CloudWatch Logs / GCP Cloud Logging | 30 days (configurable) |
| LLM + agent traces | Logfire cloud (or self-hosted Langfuse) | 90 days |
| Audit events | PostgreSQL `audit_events` table | 2 years |
| Token usage + billing | PostgreSQL `token_usage` + `billing_events` | 7 years |
| Metrics | Prometheus → Grafana Cloud | 13 months |

**Log format** — every structured log line is a JSON object on stdout:

```json
{
  "level": "INFO",
  "timestamp": "2026-06-07T09:23:41.123Z",
  "service": "api",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "sha256:abcd...",
  "session_id": "sess_7f3a9c2e",
  "tenant_id": "acme-corp",
  "corpus_id": "acme-corp:hr-policies",
  "route": "POST /api/v1/chat",
  "status_code": 200,
  "latency_ms": 843,
  "cache_hit": null,
  "model_tier": "small",
  "prompt_tokens": 1350,
  "completion_tokens": 287,
  "estimated_cost_usd": 0.000697,
  "pipeline_status": "answered"
}
```

The `request_id` field is the correlation key across logs, Langfuse/Logfire trace, Prometheus labels, and the `audit_events` table. It is returned in every API response as `RAGResponse.request_id` so the client (and the debug UI panel) can link directly to the trace.

#### `backend/logs/` directory

Added to the backend file tree for local dev use only:

```
backend/
└── logs/
    └── alerts.jsonl    # alert fallback when SMTP unreachable; git-ignored; rotated daily
```

Do not use `logs/` for application logs in production — use the container stdout pipeline.

#### Log Viewer API (UI-accessible)

The frontend provides a real-time log viewer page (`/logs`) for `admin` role users. This requires two API endpoints:

```
GET  /api/v1/logs         # query recent logs with filters (returns last N entries)
GET  /api/v1/logs/stream  # SSE stream of new log entries as they arrive (admin only)
```

**Storage:** structured log entries are written to a **Redis ring buffer** (`LPUSH knowledge:logs:recent + LTRIM 0 4999`) in addition to stdout. This gives the API fast access to the last 5,000 log lines without a DB query or file I/O. TTL is 24h per entry. The ring buffer is in Redis, not PostgreSQL — no schema migration needed.

```python
# knowledge/observability/metrics.py (addition to structlog processor chain)
class RedisLogProcessor:
    """structlog processor that mirrors each log entry to a Redis ring buffer."""
    async def __call__(self, logger, method, event_dict: dict) -> dict:
        entry = json.dumps(event_dict)
        await redis.lpush("knowledge:logs:recent", entry)
        await redis.ltrim("knowledge:logs:recent", 0, 4999)  # keep last 5000
        await redis.expire("knowledge:logs:recent", 86400)   # 24h TTL
        return event_dict
```

**Query endpoint** (`GET /v1/logs`) — on-demand, no streaming needed:

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Minimum log level to return |
| `service` | string | all | Filter by `api`, `ingest-worker`, `retrieval-worker` |
| `corpus_id` | string | all | Filter by corpus |
| `request_id` | UUID | — | Return all log entries for a single request (for drilling into a specific trace) |
| `limit` | int | 100 | Max entries (capped at 500) |
| `since` | ISO timestamp | 1h ago | Only entries after this time |

Response is a JSON array of log objects, newest first. Each entry includes a `trace_url` field when the log originated from an LLM call — links directly to the Langfuse trace.

Auth: `admin` JWT role required. Logs contain hashed user IDs and corpus names — not raw PII, but not public either.

---

#### CI/CD

```
git push → GitHub Actions
  ├── ruff check + mypy + pytest (unit+mocked only)
  ├── docker build → push to ECR / Artifact Registry
  ├── helm upgrade --install (staging namespace)
  ├── smoke tests against staging
  └── manual approval gate → helm upgrade (production namespace)
```

---

### SaaS Deployment Model

The cloud deployment section describes infrastructure. This section describes the business model layered on top of it: how tenants are isolated, provisioned, billed, and offboarded. These decisions are architectural — they affect schema design, Redis key namespacing, API auth, and the K8s resource model. They must be resolved before implementation, not bolted on later.

#### Tenant Isolation Model

**Decision: Row-Level Security (RLS) on a shared PostgreSQL cluster.**

Three options considered:

| Model | Isolation | Ops overhead | Data leak risk | Decision |
|---|---|---|---|---|
| Separate cluster per tenant | Complete | Very high (N clusters) | None | Enterprise tier only |
| Schema per tenant | Strong | Medium (N schemas, DDL migrations × N) | Low (Postgres RLS supplements) | Rejected |
| Shared tables + RLS | Moderate | Low (1 schema, 1 migration) | Low if RLS is correct | **Selected for Pro/Free** |

**RLS implementation:**

```sql
-- Every data table has tenant_id TEXT NOT NULL
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;

-- Policy: a connection may only see rows matching its set tenant_id
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id'));

-- API sets before every query (transaction-scoped):
SET LOCAL app.tenant_id = 'acme-corp';
```

`corpus_id` format: `{tenant_id}:{corpus_slug}` — tenant is always derivable from corpus_id, giving a second isolation layer without an extra join.

**Enterprise isolation**: dedicated PostgreSQL instance + dedicated Redis namespace. Provisioned via Terraform module; not self-service.

#### SLA Tiers

| Tier | Max users | Queries/day | Rate limit | Max corpora | Storage | LLM budget/month | Price |
|---|---|---|---|---|---|---|---|
| **Free** | 5 | 500 | 10 RPM, 100 RPD | 1 | 500 MB | $0 (search-only) | $0 |
| **Pro** | 100 | 10,000 | 60 RPM, 10K RPD | 5 | 10 GB | $200 | $299/mo |
| **Enterprise** | Unlimited | Custom | Custom | Unlimited | Custom | Custom | Custom |

Free tier: LLM generation disabled. Search + cache hits only. This controls cost while still providing value.

Tier enforcement at `PRE_VALIDATE` hook:
```python
class TenantQuota(BaseModel):
    tenant_id: str
    tier: Literal["free", "pro", "enterprise"]
    max_queries_per_day: int
    max_queries_per_minute: int
    max_corpus_count: int
    max_storage_gb: float
    llm_enabled: bool                      # False for free tier
    llm_budget_usd_per_month: float        # 0.0 = unlimited (enterprise with prepaid)
    max_prompt_tokens_per_request: int = 8192
    max_output_tokens_per_request: int = 1024
```

#### Tenant Onboarding Flow

Onboarding is automated end-to-end. No manual provisioning steps.

```
1. Customer signs up (Stripe checkout)
   └── Stripe webhook → POST /v1/webhooks/stripe → subscription.created event

2. TenantProvisioner.provision(tenant_id, tier):
   a. INSERT into tenants table (id, tier, created_at, billing_customer_id)
   b. INSERT into tenant_quotas (from tier template)
   c. Generate RS256 keypair → store private key in Secrets Manager
   d. Register JWKS endpoint: GET /v1/.well-known/jwks/{tenant_id}
   e. Create default corpus: {tenant_id}:default
   f. Seed audit_events: action="tenant_provisioned"
   g. Send welcome email to admin_email (via alerts.py SMTP)

3. Customer receives:
   - API base URL: https://api.ragv2.com/api/v1
   - API key (short-lived JWT signed by tenant private key, 90-day TTL)
   - Corpus ID: {tenant_id}:default
   - Quickstart documentation link
```

Provisioning is idempotent: re-running `provision()` for an existing `tenant_id` is a no-op (all steps are `INSERT ... ON CONFLICT DO NOTHING` or check-before-execute).

#### Quota Enforcement

Quota is enforced in Redis on the hot path. PostgreSQL is the audit trail — never the enforcement gate.

```python
# knowledge/api/quota.py
async def enforce_quota(tenant_id: str, request_type: str) -> None:
    """Check and increment quota counters. Raises QuotaExceeded on breach."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    month = datetime.now(UTC).strftime("%Y-%m")
    minute_key = f"quota:{tenant_id}:rpm:{int(time.time() // 60)}"
    # NOTE: use datetime.now(UTC) not datetime.utcnow() — utcnow() is deprecated in Python 3.12+

    pipe = redis.pipeline()
    pipe.incr(f"quota:{tenant_id}:queries:{today}")
    pipe.expire(f"quota:{tenant_id}:queries:{today}", 86400 + 3600)  # 25h buffer
    pipe.incr(minute_key)
    pipe.expire(minute_key, 120)  # 2 min sliding window
    daily_count, _, rpm_count, _ = await pipe.execute()

    quota = await get_tenant_quota(tenant_id)  # cached in L1 for 60s

    if daily_count > quota.max_queries_per_day:
        raise QuotaExceeded(
            code="DAILY_QUOTA_EXCEEDED",
            limit=quota.max_queries_per_day,
            resets_at=next_midnight_utc(),
        )
    if rpm_count > quota.max_queries_per_minute:
        raise QuotaExceeded(
            code="RATE_LIMIT_EXCEEDED",
            limit=quota.max_queries_per_minute,
            retry_after_s=60,
        )
    if not quota.llm_enabled and request_type == "chat":
        raise QuotaExceeded(code="LLM_NOT_ENABLED_ON_FREE_TIER")
```

Quota headers on every response (even when not exceeded):
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 47
X-RateLimit-Reset: 1749214680
X-Quota-Daily-Limit: 10000
X-Quota-Daily-Used: 3241
```

#### Billing & Metering

**Billing event** emitted after every successful LLM call (async, non-blocking):

```python
class BillingEvent(BaseModel):
    id: UUID
    tenant_id: str
    corpus_id: str
    request_id: UUID
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int          # provider-level cache hits (not our L2/L3)
    cost_usd: float
    timestamp: datetime
    cache_hit: str | None       # "l2" | "l3" | None — saves tracking for cost_saved
```

Stored in `billing_events` table. Stripe usage records created nightly:

```python
# knowledge/billing/metering.py — runs as a cron job at 00:05 UTC daily
async def flush_to_stripe(date: date) -> None:
    rows = await db.fetch(
        "SELECT tenant_id, SUM(cost_usd) FROM billing_events WHERE DATE(timestamp) = $1 GROUP BY tenant_id",
        date
    )
    for tenant_id, daily_cost in rows:
        subscription_id = await get_stripe_subscription(tenant_id)
        if subscription_id:  # Pro/Enterprise tenants only
            stripe.SubscriptionItem.create_usage_record(
                subscription_item_id=subscription_id,
                quantity=int(daily_cost * 100),  # cents
                timestamp=int(datetime.utcnow().timestamp()),
            )
```

Free tier never has `subscription_id` — costs are absorbed or hard-capped at $0 (search-only). Metering events are still written for analytics.

#### Tenant Offboarding & GDPR Compliance

Data deletion is a hard requirement, not an afterthought. The system supports right-to-erasure for any tenant or individual user.

**Tenant deletion** (`DELETE /v1/tenants/{id}` — admin-only):

```python
async def delete_tenant(tenant_id: str) -> None:
    # 1. Cancel Stripe subscription immediately
    await stripe.Subscription.cancel(tenant_subscription_id)

    # 2. Cascade delete all PostgreSQL data (FK cascade handles chunks, eval_results, etc.)
    await conn.execute("DELETE FROM documents WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM gold_samples WHERE corpus_id LIKE $1", f"{tenant_id}:%")
    # semantic_cache uses corpus_ids TEXT[] (array, no FK) — must be deleted explicitly
    await conn.execute("DELETE FROM semantic_cache WHERE corpus_ids && ARRAY(SELECT id FROM corpora WHERE tenant_id = $1)", tenant_id)
    # billing_events and token_usage have no FK cascade — delete explicitly
    await conn.execute("DELETE FROM billing_events WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM token_usage WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_id)

    # 3. Delete from Apache AGE (separate connection, graph vertices/edges)
    await age_store.delete_tenant_graph(tenant_id)

    # 4. Flush Redis keys for tenant
    keys = await redis.keys(f"quota:{tenant_id}:*")
    keys += await redis.keys(f"cache:*:{tenant_id}:*")
    if keys:
        await redis.delete(*keys)

    # 5. Rotate and delete JWT private key from Secrets Manager
    await secrets_manager.delete_secret(f"jwt_private_key/{tenant_id}")

    # 6. Audit event (append-only — this row is never deleted)
    await conn.execute(
        "INSERT INTO audit_events (user_id, tenant_id, action) VALUES ($1, $2, 'tenant_deleted')",
        "system", tenant_id
    )
    # 7. Alert
    await send_alert(severity="INFO", code="TENANT_DELETED", detail={"tenant_id": tenant_id})
```

Deletion is synchronous for the PostgreSQL cascade. AGE deletion and Redis flush are background tasks with their own retry policy.

**User-level right to erasure** (`POST /v1/users/{id}/erase`):

Individual user data — `audit_events.user_id`, `user_feedback.user_id`, `implicit_signals.user_id` — is stored as `SHA-256(user_id + tenant_salt)`. Erasing a user means replacing the stored hash with `SHA-256("ERASED" + tenant_salt)`. The row structure is preserved for analytics; the user is no longer identifiable.

**Data residency**: `CorpusConfig.data_region: Literal["us", "eu", "apac"]`. Multi-region PostgreSQL routing is a Phase I IaC concern — the schema supports it from day one.

**Retention policy**: `audit_events` rows older than `AUDIT_RETENTION_DAYS` (default 2 years) are pruned by a nightly job. `user_feedback` and `implicit_signals` are retained for 1 year. `token_usage` and `billing_events` are retained for 7 years (financial records).

#### Tenant Database Schema Additions

```sql
CREATE TABLE tenants (
    id              TEXT PRIMARY KEY,           -- slug, e.g. "acme-corp"
    display_name    TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'free',
    admin_email     TEXT NOT NULL,
    billing_customer_id TEXT,                  -- Stripe customer ID
    data_region     TEXT NOT NULL DEFAULT 'us',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ                -- soft delete; hard delete is async
);

CREATE TABLE tenant_quotas (
    tenant_id               TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    max_queries_per_day     INTEGER NOT NULL,
    max_queries_per_minute  INTEGER NOT NULL,
    max_corpus_count        INTEGER NOT NULL,
    max_storage_gb          FLOAT NOT NULL,
    llm_enabled             BOOLEAN NOT NULL DEFAULT false,
    llm_budget_usd_per_month FLOAT NOT NULL DEFAULT 0.0,
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE billing_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    corpus_id       TEXT NOT NULL,
    request_id      UUID NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        FLOAT NOT NULL,
    cache_hit       TEXT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON billing_events (tenant_id, timestamp DESC);
CREATE INDEX ON billing_events (timestamp DESC);   -- for daily flush job

-- Per-LLM-call token tracking (source of truth for cost; retained 7 years)
CREATE TABLE token_usage (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        UUID NOT NULL,
    corpus_id         TEXT NOT NULL,
    tenant_id         TEXT NOT NULL,
    model_tier        TEXT NOT NULL,     -- "nano" | "small" | "large"
    model_id          TEXT NOT NULL,     -- exact model name
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,  -- provider-level prompt cache hits
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON token_usage (tenant_id, timestamp DESC);
CREATE INDEX ON token_usage (corpus_id, timestamp DESC);
```

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
