# RAG v2 — Module Layout

## Table of Contents

- [Package Structure](#package-structure)
- [Subpackage Guide](#subpackage-guide)
  - [config](#config)
  - [api](#api)
  - [bus](#bus)
  - [ingestion](#ingestion)
  - [store](#store)
  - [retrieval](#retrieval)
  - [agent](#agent)
  - [memory](#memory)
  - [corpus](#corpus)
  - [billing](#billing)
  - [scheduler](#scheduler)
  - [hooks](#hooks)
  - [validation](#validation)
  - [evaluation](#evaluation)
  - [observability](#observability)

---

## Package Structure

The entire backend lives in the `knowledge/` Python package. Every subpackage has a single responsibility; cross-package dependencies flow in one direction (api → agent → retrieval → store).

```
knowledge/
├── config/
│   └── settings.py              # Pydantic-settings; reads .env; all tuneable knobs
├── api/
│   ├── app.py                   # FastAPI factory (lifespan, middleware stack)
│   ├── auth.py                  # JWT decode + RBAC dependency; JWE encrypt/decrypt helpers
│   ├── middleware.py            # CorrelationID, structured-log, audit-event emission
│   ├── quota.py                 # enforce_quota(): per-tenant rate limiting + budget enforcement
│   ├── timeout.py               # TimeoutBudget dataclass + per-stage sub-deadline helpers
│   ├── routes/
│   │   ├── auth.py              # POST /v1/auth/token, POST /v1/auth/refresh
│   │   ├── ingest.py            # POST /v1/ingest → publish job; GET /v1/ingest/{job_id}/status
│   │   ├── search.py            # POST /v1/search (sync fast path)
│   │   ├── chat.py              # POST /v1/chat, POST /v1/chat/stream (SSE)
│   │   ├── corpus.py            # GET /v1/corpus, POST /v1/corpus/{id}/cache/invalidate
│   │   ├── evaluate.py          # POST /v1/evaluate/run, GET /v1/evaluate/run/{id}
│   │   ├── feedback.py          # POST /v1/feedback, POST /v1/signals
│   │   ├── scheduler.py         # CRUD for scheduled ingestion jobs
│   │   └── health.py            # GET /health (pool stats, Redis ping, worker heartbeat)
│   └── schemas.py               # Pydantic request/response models (versioned)
├── bus/
│   ├── publisher.py             # async Redis Streams XADD publisher
│   ├── consumer.py              # base async XREADGROUP consumer loop (ack, dead-letter, backoff)
│   └── schemas.py               # IngestJob, EvalJob, WorkerEvent message models
├── ingestion/
│   ├── worker.py                # Redis consumer → pipeline orchestrator
│   ├── pipeline.py              # per-document: spawns chunker + graph_extractor concurrently
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
│   ├── retriever.py             # hybrid retriever: vector + text + optional graph traversal
│   ├── graph_retriever.py       # NL→Cypher query against AgeGraphStore
│   ├── fusion.py                # CrossEncoder reranker (always-on) + RRF fusion
│   ├── semantic_cache.py        # L3 semantic cache: pgvector cosine-sim lookup before LLM call
│   └── worker.py                # Redis consumer stub (reserved for bulk search batches — not yet wired)
├── agent/
│   ├── pipeline.py              # ConfidenceAwarePipeline: 3-layer gate (retrieval → citation → judge)
│   ├── agent.py                 # Pydantic AI agent; 5 tools: search_kb, search_kg, search_hybrid_kg, run_graph_query, nl_graph_query
│   ├── judge.py                 # LLMJudge: JudgeResult(verdict, confidence, reasoning); nano→small escalation
│   ├── cost_guard.py            # check_cost_circuit_breaker(): tenant + system budget enforcement
│   ├── model_router.py          # QueryRouter (nano model) → RoutingDecision
│   └── prompts.py               # all system prompts (see docs/PROMPTS.md)
├── memory/
│   ├── working_memory.py        # Tier 1: context assembly + token-budget trim
│   ├── conversation_store.py    # Tier 2: episodic — conversation + message CRUD; active window loader
│   ├── mem0_store.py            # Tier 3: Mem0-backed user semantic memory (extraction, dedup, cosine search)
│   ├── summarizer.py            # Tier 2: auto-summarize when turn_count > 20 (nano model)
│   └── pruning.py               # Background jobs: TTL eviction, LRU eviction, memory compaction
├── corpus/
│   ├── registry.py              # CorpusRegistry: load corpus configs, enforce RBAC at query time
│   └── ontologies/              # Pydantic ontology templates for KG extraction
│       ├── loader.py            # load_ontology(path) → type[BaseModel]; LRU-cached
│       ├── generic.py           # default ontology (entities + relations, no domain specifics)
│       └── *.py                 # domain-specific ontologies (hr_policy, legal_contract, …)
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

## Subpackage Guide

### config

Single `settings.py` using `pydantic-settings`. All configuration comes from environment variables (`.env`). No config files, no CLI flags — one source of truth. Every other subpackage imports `load_settings()`.

### api

FastAPI application. Routes are thin — they validate the request, call the appropriate service (pipeline, retriever, store), and return the response. No business logic lives in routes. See [REST_API.md](../REST_API.md) for the full endpoint reference.

### bus

Redis Streams publish/consume primitives. `publisher.py` is used by `api/routes/ingest.py` and `api/routes/evaluate.py` to enqueue jobs. `consumer.py` is the base class for all workers — handles `XREADGROUP`, `XACK`, retry counting, and DLQ promotion.

### ingestion

The ingest-worker process. `worker.py` is the entry point (Redis consumer). `pipeline.py` runs `asyncio.gather` over the chunker task and the (optional) graph extraction task per document. CPU-bound Docling work runs via `asyncio.to_thread`.

### store

All database access. Three backends: `vector.py` (PostgreSQL + pgvector), `graph.py` (Apache AGE), `entity_index.py` (pgvector shadow table for entity search). `cache.py` wraps Redis for L2 key/value caching. No business logic — pure read/write.

### retrieval

The sync retrieval pipeline called directly from `api/routes/search.py` and `api/routes/chat.py`. `retriever.py` runs vector + text + graph search in parallel via `asyncio.gather`, fuses with RRF, reranks with CrossEncoder, and returns confidence-scored results. See [design/RETRIEVAL.md](RETRIEVAL.md).

### agent

The LLM layer. `pipeline.py` is the 3-layer confidence gate that orchestrates retrieval → agent → judge. `agent.py` holds the Pydantic AI agent with its 5 tools. `judge.py` is a passthrough by default (`judge_enabled=False` in settings). `cost_guard.py` enforces tenant and system budget limits before every LLM call.

### memory

Five-tier context system assembled per request:

| Tier | File | What it stores |
|------|------|---------------|
| 1 | `working_memory.py` | Assembled context for the current request (transient) |
| 2 | `conversation_store.py` | Full conversation history in PostgreSQL |
| 3 | `mem0_store.py` | Long-term user facts (Mem0, pgvector-backed) |
| 4 | _(retrieval result)_ | Retrieved chunks injected into context |
| 5 | _(settings / DB)_ | System prompts from `system_prompts` table |

### corpus

`registry.py` loads `CorpusConfig` objects from settings and enforces RBAC. `ontologies/` holds Pydantic templates that define the KG extraction schema per corpus — see [design/INGESTION.md](INGESTION.md) for the ontology API.

### billing

Token usage metering and tenant quota management. `metering.py` inserts into `token_usage` and increments the Redis cost counter after each LLM call. `provisioner.py` handles tenant onboarding and GDPR erasure.

### scheduler

APScheduler-backed cron jobs for periodic ingestion. Jobs are stored in PostgreSQL (`scheduled_jobs` table) so they survive restarts. `runner.py` fires `publisher.publish_ingest_job()` on schedule.

### hooks

Lightweight event system. `HookPoint` enum defines named points in the request lifecycle (`PRE_LLM`, `POST_RETRIEVE`, `POST_LLM`, `ON_VALIDATION_FAIL`). Hooks are registered at startup and fired asynchronously. Used for cost guards, audit events, and observability.

### validation

Six-stage input validation chain (V1 schema → V2 length → V3 language → V4 injection → V5 content policy → V6 RBAC). All stages run before any LLM call. See [design/RELIABILITY.md](RELIABILITY.md).

### evaluation

Offline eval harness. Triggered via `POST /v1/evaluate/run`, which publishes to `knowledge:eval`. The eval runner measures retrieval quality (Hit Rate, MRR, NDCG) and answer quality (faithfulness, relevance, correctness) against a gold dataset. See [design/EVALUATION.md](EVALUATION.md).

### observability

Langfuse tracing (`@observe` decorator on LLM calls) and Prometheus metrics (request counters, latency histograms, cache hit rates, DLQ depth). Scraped by Grafana at `/metrics`.
