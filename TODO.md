# TODO

## Table of Contents

- [Architecture Proposal — Enterprise RAG v2](#architecture-proposal--enterprise-rag-v2)
  - [Goals](#goals)
  - [Module Layout](#module-layout)
  - [Knowledge Layer — Multi-Corpus Design](#knowledge-layer--multi-corpus-design)
  - [Ingestion Pipeline — Docling-Graph Parallel Paths](#ingestion-pipeline--docling-graph-parallel-paths)
  - [Redis Pub/Sub + Async Worker Model](#redis-pubsub--async-worker-model)
  - [Caching Architecture](#caching-architecture)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Model Tiering](#model-tiering)
  - [Query Validation & Hook System](#query-validation--hook-system)
  - [Security Layer — JWT, JWE, HTTPS, RBAC](#security-layer--jwt-jwe-https-rbac)
  - [API Layer](#api-layer)
  - [Docker Compose — Local Dev](#docker-compose--local-dev)
  - [Cloud Deployment — Production](#cloud-deployment--production)
  - [Evaluation System — Offline & Online Metrics](#evaluation-system--offline--online-metrics)
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
│   ├── routes/
│   │   ├── ingest.py            # POST /v1/ingest → publish job; GET /v1/ingest/{job_id}/status
│   │   ├── search.py            # POST /v1/search (sync fast path) + async via Redis
│   │   ├── chat.py              # POST /v1/chat, GET /v1/chat/stream (SSE)
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
│   ├── vector.py                # PostgresHybridStore: pgvector IVFFlat + tsvector GIN + RRF
│   ├── graph.py                 # AgeGraphStore: Apache AGE Cypher ops over asyncpg
│   ├── entity_index.py          # EntityIndex: tsvector shadow table for entity name search
│   └── cache.py                 # RedisCache: L2 query/embedding/doc-fingerprint cache
├── retrieval/
│   ├── worker.py                # Redis consumer → retrieval pipeline (for async search requests)
│   ├── retriever.py             # hybrid retriever: vector + text + optional graph traversal
│   ├── graph_retriever.py       # NL→Cypher query against AgeGraphStore
│   ├── fusion.py                # Reciprocal Rank Fusion + optional LLM re-ranker
│   └── semantic_cache.py        # L3 semantic cache: pgvector cosine-sim lookup before LLM call
├── agent/
│   ├── agent.py                 # Pydantic AI agent; tools: search_knowledge_base, search_graph
│   ├── model_router.py          # QueryRouter (nano model) → RoutingDecision
│   └── prompts.py
├── memory/
│   └── mem0_store.py            # Mem0Store (pgvector-backed per-user memory)
├── corpus/
│   └── registry.py              # CorpusRegistry: load corpus configs, enforce RBAC at query time
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
    enable_graph_extraction: bool    # toggle docling-graph per corpus
    graph_extraction_backend: str    # "ollama" | "openai" | "mistral"
    metadata_tags: dict[str, str]    # extra metadata attached to every chunk
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
            docling-graph PipelineOrchestrator
            → LLM/VLM entity + relationship extraction
            → age_graph_store.upsert_entities()
            → entity_index.upsert()
   )
        │
        ▼
  publish IngestCompleteEvent to Redis
```

**Docling-graph integration notes** (evaluation items below):
- `PipelineOrchestrator` is sync; wrap in `asyncio.to_thread()` for the worker.
- Use `mode="api"` so it does not dump intermediate files to disk.
- Configure LiteLLM backend to point at local Ollama (same model as RAG chat LLM).
- The `CypherExporter` output can be fed directly to `AgeGraphStore.run_cypher()`.
- Chunking in docling-graph uses `DocumentChunker` (same HybridChunker under the hood) — configure `chunk_max_tokens` consistently with the embedding model's context window.

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
CREATE INDEX ON semantic_cache USING ivfflat (query_emb vector_cosine_ops) WITH (lists = 50);
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
        relevance_score: float    # post-rerank score
        excerpt: str              # ≤ 200 chars of the supporting chunk
```

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

#### Validation Pipeline (`knowledge/api/validation.py`)

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

### API Layer

**Base URL**: `/api/v1`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/ingest` | `writer` | Submit ingest job; returns `job_id` |
| `GET`  | `/v1/ingest/{job_id}/status` | `writer` | Poll job status |
| `GET`  | `/v1/ingest/{job_id}/stream` | `writer` | SSE job progress stream |
| `POST` | `/v1/search` | `reader` | Synchronous hybrid search (< 200 ms fast path) |
| `POST` | `/v1/chat` | `reader` | Agent chat (blocking) |
| `GET`  | `/v1/chat/stream` | `reader` | Agent chat with SSE streaming |
| `GET`  | `/v1/corpus` | `admin` | List registered corpora |
| `POST` | `/v1/corpus/{id}/cache/invalidate` | `admin` | Flush L2+L3 caches for corpus |
| `GET`  | `/health` | none | Liveness + readiness (pool, Redis, worker heartbeats) |
| `GET`  | `/metrics` | `service` | Prometheus metrics endpoint |

**Versioning**: URL prefix `/api/v1`; future breaking changes get `/api/v2`. Non-breaking changes are additive within v1.

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

```yaml
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
    baseline_run_id UUID REFERENCES eval_runs(id)
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
    cache_tier_hit      TEXT
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

### Docling-Graph Evaluation Checklist

Before committing to full integration, validate these items in a spike branch:

- [ ] **Async wrapper**: confirm `PipelineOrchestrator.run()` can run in `asyncio.to_thread()` without thread-safety issues; measure baseline CPU + memory per document.
- [ ] **Ollama backend**: verify LiteLLM routes correctly to local Ollama; check JSON schema enforcement works with Llama 3.1/Mistral.
- [ ] **Chunking parity**: compare `DocumentChunker` output to existing `HybridChunker` usage; confirm `chunk_max_tokens` maps cleanly to `nomic-embed-text` 512-token context.
- [ ] **Cypher output**: test `CypherExporter` → `AgeGraphStore.run_cypher()` round-trip on 3 sample documents.
- [ ] **Parallel overhead**: measure wall-clock time for vector path alone vs. vector + graph in parallel on a 50-page PDF; establish acceptable graph-extraction timeout budget.
- [ ] **Corpus toggle**: confirm `enable_graph_extraction=False` in `CorpusConfig` fully skips graph path with no side-effects.
- [ ] **VLM fallback**: test `mode=vlm` with Ollama LLaVA for scanned PDFs; measure extraction quality vs. LLM mode.
- [ ] **Memory footprint**: profile peak RSS when 2 ingest workers run concurrently; ensure fits within 8 GB container limit.
- [ ] **Graph query latency**: with 50k entities in AGE, measure NL→Cypher retrieval P99.

---

### Implementation Phases

#### Phase A — Housekeeping (no new features, before any refactor)
- [x] Move `kg/legal/` → `misc/kg_legal_cuad/` (done)
- [x] Move `rag/legal/` → `misc/kg_legal_cuad/rag_data/` (done)
- [x] Move `rag/ingestion/cuad_ingestion.py` → `misc/kg_legal_cuad/` (done)
- [x] Move `rag/tests/ingestion/test_cuad_ingestion.py` → `misc/kg_legal_cuad/tests/` (done)
- [x] Move `rag/tests/knowledge_graph/` → `misc/kg_legal_cuad/tests/kg/` (done)
- [x] Delete `rag/retrieval/dead_code/` (done)
- [ ] Run `python -m pytest rag/tests/ -m "not integration" -v` — confirm no regressions after moves

#### Phase B — Rate Limiting, Timeouts, Retries (in-progress, see section below)
- Complete existing 4-step plan before starting module restructure

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
