# RAG v2 — System Design

## Table of Contents

### Overview
- [System Diagram](#system-diagram)
- [Why Chat Needs No Message Queue](#why-chat-needs-no-message-queue)
- [Why SSE and Not WebSockets](#why-sse-and-not-websockets)
- [User Query Data Flow](#user-query-data-flow)

### Architecture & Constraints (moved to design/)
- [Goals & Multi-Corpus Design](design/ARCHITECTURE_PROPOSAL.md)
- [System Design Constraints](design/SYSTEM_DESIGN_CONSTRAINTS.md) — load model, SLAs, token budgets, cost
- [Module Layout](design/MODULE_LAYOUT.md)

### Further Reading (design/)

- [REST_API.md](design/REST_API.md) — All /api/v2 endpoints with request/response shapes
- [INGESTION.md](design/INGESTION.md) — Ingestion pipeline, KG extraction, AGE store
- [REDIS_STREAMS.md](design/REDIS_STREAMS.md) — Message bus, async workers, DLQ
- [CACHING.md](design/CACHING.md) — L1/L2/L3 cache layers, TTLs, invalidation
- [RETRIEVAL.md](design/RETRIEVAL.md) — Hybrid search, RRF, confidence pipeline, model tiering
- [RELIABILITY.md](design/RELIABILITY.md) — Query validation, guardrails, error handling, retry
- [SECURITY.md](design/SECURITY.md) — JWT/JWE/RBAC, memory architecture
- [DEPLOYMENT.md](design/DEPLOYMENT.md) — Docker Compose, packaging, cloud, SaaS model
- [EVALUATION.md](design/EVALUATION.md) — Eval system, load/chaos testing, implementation phases

---

## System Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                              RAG v2 — System Overview                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

 Browser
       │
       │  HTTPS (TLS 1.3)  — two types of requests, same origin:
       │  ①  /*           page load (HTML, JS bundle, assets)
       │  ②  /api/v2/*   REST + SSE calls from browser JS (Authorization: Bearer)
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Nginx  (port 443)                                                          │
│  ├── /api/v2/*  → proxy_pass api:8000        (② REST + SSE calls)          │
│  │              SSE routes: proxy_buffering off, proxy_read_timeout 3600s   │
│  └── /*         → proxy_pass frontend:3000   (① page load only)            │
└────────────────┬──────────────────────────────────┬────────────────────────┘
                 │ ①                                │ ②
                 ▼                                  ▼
  ┌──────────────────────────────┐    ┌─────────────────────────────────────┐
  │  Frontend  (Next.js 15)      │    │  API  (Gunicorn + UvicornWorker)    │
  │  port 3000                   │    │  knowledge.api.app   port 8000      │
  │                              │    │                                     │
  │  Serves HTML + JS bundle     │    │  Middleware:                        │
  │  to the browser once.        │    │  CorrelationID, AuditEmitter,       │
  │                              │    │  RateLimiter (slowapi), JWT RBAC    │
  │  After that, ALL API calls   │    │                                     │
  │  come from browser JS via    │    │  Routes:                            │
  │  path ② (same origin):       │    │  POST /auth/token  /auth/refresh    │
  │                              │    │  POST /chat        /chat/stream     │
  │  src/lib/api.ts              │    │  POST /search      /ingest          │
  │    fetch('/api/v2/chat')     │───►│  GET  /corpus      /conversations   │
  │    fetch('/api/v2/search')   │    │  GET  /memories    /scheduler/jobs  │
  │    fetch('/api/v2/memories') │    │  POST /evaluate/run  /feedback      │
  │    …all endpoints…           │    │  GET  /logs        /health          │
  │                              │◄───│                                     │
  │  src/lib/sse.ts              │    │  Returns:                           │
  │    POST /chat/stream         │    │  JSON: { request_id, data, error }  │
  │    → ReadableStream chunks   │    │  SSE:  data: {"delta":"..."}\n\n    │
  │    yields delta/done/error   │    │        data: {"done":true,...}\n\n  │
  │                              │    └─────────────────────────────────────┘
  │  src/lib/auth.ts             │
  │    access token → memory     │    Local dev only (npm run dev):
  │    refresh → httpOnly cookie │    next.config.ts rewrites
  └──────────────────────────────┘    /api/v2/* → localhost:8000
                                      so browser still calls :3000 (same origin)
                                 │
                ─────────────────┴────────────────────────────────────
                SYNC PATH                         ASYNC PATH
                (chat, search, auth, health)       (ingest, bulk eval)
                ─────────────────────────────────  ─────────────────────
                                 │                         │
                                 │                         │ XADD {job_id, path,
                                 │                         │       corpus_id, …}
                                 │                         ▼
                                 │           ┌─────────────────────────────────────┐
                                 │           │  Redis Streams  (Message Bus)        │
                                 │           │                                     │
                                 │           │  knowledge:ingest     ← ingest jobs │
                                 │           │  knowledge:eval       ← eval runs   │
                                 │           │  knowledge:events     ← completions │
                                 │           │  knowledge:ingest:dlq ← failed ×3  │
                                 │           │                                     │
                                 │           │  L2 Cache keys:                     │
                                 │           │  cache:embed:*     24h TTL          │
                                 │           │  cache:search:*     5min TTL        │
                                 │           │  cache:doc_fingerprint:*  7d TTL   │
                                 │           │  job:{id}          status hash      │
                                 │           │  quota:*   cb:*   logs:*            │
                                 │           └──────────┬──────────────────────────┘
                                 │                      │ XREADGROUP
                                 │                      │ (consumer group:
                                 │                      │  ingest-workers × N)
                                 │                      ▼
                                 │           ┌─────────────────────────────────────┐
                                 │           │  Ingestion Worker (worker.py × N)   │
                                 │           │  XREADGROUP knowledge:ingest        │
                                 │           │  → DoclingProcessor → chunk         │
                                 │           │  → embed → upsert (vector + graph)  │
                                 │           │  → XACK on success                  │
                                 │           │  → retry ×3 → DLQ on failure        │
                                 │           │  → XADD knowledge:events (complete) │
                                 │           │  Heartbeat: job:{id} hash in Redis  │
                                 │           └─────────────────────────────────────┘
                                 │
        ┌────────────────────────┴───────────┐   ┌──────────────────────┐
        │  Validation Pipeline               │   │  Memory System       │
        │  V1 Schema                         │   │  (knowledge/memory/) │
        │  V2 Length guard                   │   │                      │
        │  V3 Language detect                │   │  Tier 1: assemble()  │
        │  V4 Injection detect               │   │  Tier 2: conv store  │
        │  V5 Content policy (nano model)    │   │  Tier 3: Mem0        │
        │  V6 RBAC check                     │   │  Tier 5: sys prompts │
        └────────────────────┬───────────────┘   └──────────────────────┘
                             │
                             ▼
               ┌────────────────────────┐
               │  Model Router          │
               │  (nano: qwen2.5:0.5b) │
               │  → RoutingDecision     │
               │    complexity          │
               │    requires_graph      │
               │    model_tier          │
               └────────────┬───────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────┐
  │  Confidence-Aware Pipeline                         │
  │  (knowledge/agent/pipeline.py)                     │
  │                                                    │
  │  Layer 1: retrieve_with_confidence()               │
  │    └── Σ(confidence top-K) < threshold → ABSTAIN  │
  │                                                    │
  │  Layer 2: agent.run() → GenerationResult           │
  │    └── uncited_claims > 0 → ABSTAIN               │
  │                                                    │
  │  Layer 3: judge() → JudgeResult                   │
  │    └── verdict=unsupported → ABSTAIN              │
  │                                                    │
  │  → RAGResponse { answer, citations, confidence,   │
  │                  cost_usd, trace_url, request_id }│
  └────────────┬───────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────────────┐
           │                  │                          │
           ▼                  ▼                          ▼
  ┌────────────────┐ ┌────────────────────┐  ┌───────────────────────────────┐
  │  Retriever     │ │  RAG Agent         │  │  LLM Judge                    │
  │  (retrieval/)  │ │  (agent/agent.py)  │  │  (agent/judge.py)             │
  │                │ │                    │  │                               │
  │  L3 semantic   │ │  PydanticAgent     │  │  nano → small escalation      │
  │  cache check   │ │  5 tools:          │  │                               │
  │                │ │  search_kb         │  │  JudgeResult:                 │
  │  asyncio.gather│ │  search_kg         │  │  supported/partial/unsupported│
  │  ├─ vec search │ │  search_hybrid_kg  │  │                               │
  │  ├─ text search│ │  run_graph_query   │  │  PASSTHROUGH when             │
  │  └─ graph      │ │  nl_graph_query    │  │  judge_enabled=False:         │
  │                │ │                    │  │  Layer 3 is skipped and       │
  │  CrossEncoder  │ │  result.usage()    │  │  response always passes       │
  │  rerank        │ │  → cost tracking   │  │  as status="answered"         │
  │  → confidence  │ └────────────────────┘  └───────────────────────────────┘
  └────────────────┘
                     (token counts — Pydantic AI built-in, no manual counting)
           │
           │  reads from
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Storage Layer  (knowledge/store/)                                           │
│                                                                              │
│  ┌─────────────────────────────────┐   ┌──────────────────────────────────┐ │
│  │  Main PostgreSQL  (port 5432)   │   │  Apache AGE  (port 5433)         │ │
│  │  pgvector/pgvector:pg16         │   │  apache/age:latest               │ │
│  │                                  │   │                                  │ │
│  │  documents        (source store) │   │  Per-corpus graphs:              │ │
│  │  chunks           HNSW + GIN     │   │  kg_{tenant}_{corpus}            │ │
│  │  kg_entity_index  HNSW + GIN     │   │                                  │ │
│  │  semantic_cache   HNSW           │   │  Vertices: entity types          │ │
│  │  audit_events     append-only    │   │  from docling-graph ontology     │ │
│  │  conversations    (Tier 2)       │   │                                  │ │
│  │  messages         GIN            │   │  Edges: EMPLOYS, APPLIES_TO,     │ │
│  │  user_memories    HNSW + GIN     │   │  HAS_MEMBER, etc. from edge()    │ │
│  │  system_prompts   (Tier 5)       │   │                                  │ │
│  │  gold_samples     (eval)         │   │  No GIN/HNSW in AGE — use        │ │
│  │  eval_runs        (eval)         │   │  kg_entity_index for entity      │ │
│  │  eval_results     (eval)         │   │  search instead                  │ │
│  │  token_usage      (billing)      │   │                                  │ │
│  │  tenants/quotas   (billing)      │   │  Cypher via ag_catalog.cypher()  │ │
│  │  scheduled_jobs   (scheduler)    │   │  SQL wrapper — not raw Cypher    │ │
│  └─────────────────────────────────┘   └──────────────────────────────────┘ │
│                                                                              │
│  RLS: SET LOCAL app.tenant_id before every query                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Ingestion Workers  (knowledge/ingestion/worker.py × N)                      │
│                                                                              │
│  Redis XREADGROUP knowledge:ingest                                           │
│        │                                                                     │
│        ▼                                                                     │
│  DoclingProcessor.process(path)    ← asyncio.to_thread (CPU-bound)          │
│  ├── PDF → _get_pdf_converter()    (VLM optional: PictureDescriptionApiOpts) │
│  ├── DOCX/MD → _get_standard_converter()                                    │
│  └── Audio → ASR pipeline (Whisper Turbo via Docling)                       │
│        │                                                                     │
│        ▼                                                                     │
│  asyncio.gather(                                                             │
│    ├── chunker_task:                                                         │
│    │     DoclingHybridChunker.chunk_document()   ← contextualize() each chunk│
│    │     → embedder.embed_batch()               ← AsyncOpenAI, L1 lru_cache │
│    │     → vector_store.upsert_chunks()                                      │
│    │                                                                         │
│    └── graph_task (if enable_graph_extraction):                              │
│          load_ontology(corpus.graph_ontology_path)  ← LRU-cached            │
│          run_pipeline(PipelineConfig(template=OntologyClass, ...))           │
│             ← asyncio.to_thread; docling-graph via LiteLLM → Ollama         │
│          → PipelineContext.knowledge_graph  (NetworkX DiGraph)               │
│          → age_store.import_docling_graph()  ← iterates nodes/edges directly │
│             (NOT CypherExporter — AGE uses SQL wrapper syntax)               │
│          → entity_index.upsert_batch_from_graph()                           │
│  )                                                                           │
│        │                                                                     │
│        └── Publish IngestCompleteEvent → knowledge:events                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Observability                                                               │
│                                                                              │
│  structlog JSON → stdout → Docker logs (docker compose logs -f api)          │
│  RedisLogProcessor → knowledge:logs:recent (ring buffer, 5k entries, 24h)   │
│  Langfuse (self-hosted) → LLM traces via @observe decorator                 │
│  Prometheus → /metrics scrape by Grafana (7-row dashboard)                  │
│  SMTP alerts → rohan.vazirani@gmail.com on circuit open / DLQ / budget      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Cache Layers                                                                │
│                                                                              │
│  L1  functools.lru_cache    in-process per worker   embedding dedup         │
│  L2  Redis cache:embed:*    24h TTL                  embedding dedup        │
│  L2  Redis cache:search:*   5min TTL                 exact query cache      │
│  L3  PostgreSQL semantic_cache  60min TTL cosine≥0.95  JWE-encrypted answer │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Chat Needs No Message Queue

A common question when reading this architecture: ingestion uses Redis Streams — why doesn't chat?

**The answer: a chat request is a synchronous HTTP connection. The browser holds it open and waits. There is nothing to queue.**

```
Browser                    Nginx                  API (Uvicorn)
  │                          │                        │
  │  POST /api/v2/chat/stream│                        │
  │─────────────────────────►│                        │
  │                          │  proxy_pass (keep-alive)│
  │                          │───────────────────────►│
  │                          │                        │ coroutine starts:
  │                          │                        │ validate → retrieve
  │                          │                        │ → LLM stream (awaiting
  │                          │                        │   tokens from Ollama)
  │◄─────────────────────────────────────────────────│
  │  data: {"delta": "The "}        (SSE chunk)       │
  │◄─────────────────────────────────────────────────│
  │  data: {"delta": "PTO policy"}                    │
  │◄─────────────────────────────────────────────────│
  │  data: {"done": true, "citations": [...]}         │
  │                          │                        │ coroutine ends
```

The SSE connection is just a long-lived HTTP response. Nginx keeps it open (`proxy_buffering off`, `proxy_read_timeout 3600s`). The browser streams tokens to the UI as they arrive from Ollama. No polling, no queue, no separate worker — one coroutine, one connection, start to finish.

**Why a queue would make things worse here:**

| | Ingestion (needs queue) | Chat (no queue) |
|---|---|---|
| Client waiting? | No — fire and forget | Yes — user is watching the screen |
| Duration | Minutes (Docling + embed + graph) | 1–4 seconds |
| Can run in the request? | No — would time out nginx | Yes — SSE keeps the connection alive |
| Failure recovery | Retry + DLQ needed | HTTP 500, client retries immediately |
| Work profile | CPU-bound (blocks event loop) | Async I/O throughout |
| Queue pickup delay | ~1–5 s — acceptable for batch | ~1–5 s — exceeds the entire response SLA |

**Why it doesn't block under concurrent load:** every step in the pipeline `await`s — the pgvector query, the Redis lookup, the Ollama HTTP stream. While one request is waiting on I/O, the asyncio event loop runs other requests. A single Uvicorn worker handles many concurrent sessions this way. See [design/REDIS_STREAMS.md](design/REDIS_STREAMS.md) for the concurrency math.

**The one async piece:** after the response is sent, `asyncio.create_task` fires background tasks (store conversation turn, extract memories, update billing). These run after the user already has their answer — non-blocking, no impact on response latency.

---

## Why SSE and Not WebSockets

The streaming path uses Server-Sent Events (SSE) over a plain HTTP POST, not WebSockets. This is a deliberate choice.

**Chat is unidirectional by nature.** The user sends one message; the server streams back one response. Once the user hits send, there is nothing more for the client to transmit until the full response arrives. That's exactly what SSE is designed for — a single long-lived HTTP response that the server writes to over time. WebSockets provide a bidirectional channel, which is the right tool for collaborative editing, live cursors, or multiplayer state — not for a request/response chatbot.

| Dimension | SSE (chosen) | WebSockets |
|-----------|-------------|------------|
| Communication direction | Server → client only | Bidirectional |
| Matches chat model? | Yes — one request, one streamed reply | Overkill — client has nothing to send mid-stream |
| Auth | Bearer token on every HTTP request | Token sent once at upgrade; harder to revoke mid-session |
| Nginx config | `proxy_buffering off` — one line | Requires `proxy_http_version 1.1`, `Upgrade`, `Connection` headers |
| Load balancer / CDN support | Native HTTP — works everywhere | Requires WS-aware proxy; some CDNs don't support it |
| Reconnection on drop | Browser `EventSource` auto-reconnects | Must implement in client code |
| Server-side state | Stateless — each stream is a fresh coroutine | Stateful — server must track open connections |
| FastAPI implementation | `StreamingResponse` + `async generator` | Separate `websocket` handler + connection manager |

**Multi-turn conversations work fine with SSE.** Users naturally ask follow-up questions on the same topic — "what about the parental leave policy?", "can contractors claim it too?". Each follow-up is simply a new `POST /chat/stream` request, carrying the same `session_id`. The API loads the conversation history for that session from PostgreSQL (Tier 2 episodic memory), prepends it to the context window, and streams the next response. No persistent connection is needed between turns — SSE handles one turn at a time, which is exactly how human conversation works.

**The one thing WebSockets would enable that SSE cannot:** the client interrupting a generation mid-stream to send a follow-up. With SSE the client can only close the connection (which cancels the coroutine server-side via `asyncio` task cancellation). In practice, users don't interrupt — they wait for the response. If that changes, WebSockets are a straightforward upgrade; the retrieval and agent logic doesn't change, only the transport layer.

---

## User Query Data Flow

**Quick reference:**

> ① Auth + quota → V1–V6 validation → model router → cost guard → PRE\_RETRIEVE (user memories) → retrieval (L2 cache → embed → L3 cache → parallel hybrid search → RRF → CrossEncoder → confidence filter) → Layer 1 gate → working memory assembly (all five tiers) → RAG agent (with tool calls) → Layer 2 gate (citation check) → judge (nano→small escalation) → async background tasks (L3 cache, episodic storage, memory extraction, billing) → RAGResponse → browser (SSE deltas or blocking JSON) → UI (MessageBubble, CitationPanel, CostBadge, DebugPanel)

Step-by-step trace of a single chat request from browser keypress to rendered response.
Each box is a component; arrows show what data moves between them; ✗ branches are abstention exits.

```
USER TYPES A QUERY AND HITS SEND
─────────────────────────────────────────────────────────────────────────────────────────

  Browser (React)
  │  chatStore.sendMessage(query, session_id, corpus_ids, model_tier)
  │  api.ts: POST /api/v2/chat/stream   { query, session_id, corpus_ids }
  │          Authorization: Bearer <access_token>
  ▼

  Nginx (port 443)
  │  /api/v2/* → proxy_pass api:8000
  │  proxy_buffering off  (SSE route)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  API  (knowledge/api/routes/chat.py)
─────────────────────────────────────────────────────────────────────────────────────────

  ① Auth & quota
  │  require_jwt()           → extracts user_id, tenant_id, roles from JWT
  │  enforce_quota()         → Redis INCR daily counter + RPM sliding window
  │                            → 429 if limit hit
  ▼

  ② Load memory (background-parallel with validation)
  │  conversation_store.load_active_window(session_id)
  │    → SELECT last 8 messages WHERE conversation_id = ...   [Tier 2: PostgreSQL]
  │    → if turn_count > 20: prepend conversations.summary
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  Validation Pipeline  (knowledge/validation/pipeline.py)
─────────────────────────────────────────────────────────────────────────────────────────

  V1  Schema check         Pydantic model — rejects malformed body → 400
  │
  V2  Length guard         len(query) > MAX_QUERY_CHARS → 422
  │
  V3  Language detect      optional — 422 if not in allowed_languages
  │
  V4  Injection detector   regex + embedding-sim against known attack patterns → 422
  │
  V5  Content policy       nano model (qwen2.5:0.5b)
  │   ContentPolicyResult { verdict, confidence, reason }
  │   on_topic   → continue
  │   off_topic  → 422
  │   inappropriate → 400 + audit flag
  │
  V6  RBAC check           JWT roles vs CorpusConfig.allowed_roles → 403
  │
  ▼  all pass

─────────────────────────────────────────────────────────────────────────────────────────
  Model Router  (knowledge/agent/model_router.py)
─────────────────────────────────────────────────────────────────────────────────────────

  nano model → RoutingDecision
  │  complexity: simple | moderate | complex
  │  requires_graph: bool
  │  model_tier: nano | small | large
  │  (3s timeout → default small)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  PRE_LLM hook → Cost guard  (knowledge/agent/cost_guard.py)
─────────────────────────────────────────────────────────────────────────────────────────

  Redis INCRBYFLOAT quota:{tenant_id}:cost_usd:{month}
  │  ≥ tenant_budget  → 402 TenantBudgetExceeded
  │  system breach    → 503 SystemBudgetExceeded
  ▼  within budget

─────────────────────────────────────────────────────────────────────────────────────────
  PRE_RETRIEVE hook → Inject user memories  (knowledge/memory/mem0_store.py)
─────────────────────────────────────────────────────────────────────────────────────────

  hybrid_search(query, user_id, k=3)
  │  tsvector BM25 + pgvector cosine → RRF(k=60)   [user_memories table]
  │  top-3 facts prepended to system prompt
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  Retrieval Pipeline  (knowledge/retrieval/retriever.py)
─────────────────────────────────────────────────────────────────────────────────────────

  ③ L2 cache check
  │  Redis GET cache:search:{sha256(query+corpus+filters)}
  │  HIT  → skip retrieval, skip LLM → return cached RAGResponse
  │  MISS ↓

  ④ Embed query
  │  AsyncOpenAI (nomic-embed-text) → vector(768)
  │  L1 lru_cache hit → skip embed call

  ⑤ L3 semantic cache check
  │  SELECT … WHERE query_emb <=> $vec ORDER BY cosine LIMIT 1
  │  cosine ≥ 0.95 and not expired
  │  HIT  → decrypt JWE → return cached answer + citations
  │  MISS ↓

  ⑥ Hybrid retrieval  (asyncio.gather — all three in parallel)
  │  ├── semantic_search()   pgvector HNSW  embedding <=> query_emb
  │  ├── text_search()       tsvector GIN   content_tsv @@ websearch_to_tsquery
  │  └── graph_retrieval()   AGE Cypher     [if requires_graph, circuit-broken]
  │
  ⑦ RRF fusion  score = Σ 1/(60 + rank_i)  across search legs
  │
  ⑧ CrossEncoder rerank
  │  BAAI/bge-reranker-base (local)
  │  confidence = sigmoid(cross_encoder_logit)   ← calibrated 0–1
  │
  ⑨ Confidence filter
  │  drop results where confidence < min_confidence_score (default 0.10)
  │
  ⑩ Populate L2 Redis cache (async, non-blocking)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 1 GATE  (knowledge/agent/pipeline.py)
─────────────────────────────────────────────────────────────────────────────────────────

  aggregate = Σ confidence for top-K results
  │
  aggregate < retrieval_confidence_threshold (default 1.5)
  │   ✗ → status = "abstained_retrieval"
  │         return RAGResponse immediately  (no LLM call)
  │
  ▼  pass

─────────────────────────────────────────────────────────────────────────────────────────
  Assemble working memory  (knowledge/memory/working_memory.py)  [Tier 1]
─────────────────────────────────────────────────────────────────────────────────────────

  system_prompt          [Tier 5 — from system_prompts table or prompts.py]
  + user_memory_context  [Tier 3 — top-3 facts from PRE_RETRIEVE]
  + conversation_history [Tier 2 — last 8 turns or summary + last 8]
  + retrieved_chunks     [Tier 4 — top-K confidence-filtered chunks with [chunk_id]]
  + current_query
  ↓
  trim_to_budget(8192 tokens)
  drop order: lowest-confidence chunks → oldest turns → user memories
  set context_truncated: true if trimming was needed
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  RAG Agent  (knowledge/agent/agent.py)   model tier from router
─────────────────────────────────────────────────────────────────────────────────────────

  agent.run(query, message_history=history, deps=state)
  │
  │  LLM may call tools (each is a round-trip to the model):
  │  ├── search_knowledge_base()   → retriever.retrieve()  [additional searches]
  │  ├── search_knowledge_graph()  → AgeGraphStore entity search
  │  ├── search_hybrid_kg()        → parallel semantic + graph, then fuse
  │  ├── run_graph_query()         → direct Cypher MATCH → AGE
  │  └── nl_graph_query()          → NL→Cypher (small model) → AGE
  │
  │  Returns GenerationResult:
  │  { answer: str,
  │    citations: list[Citation],       ← each has chunk_id, relevance_score
  │    citation_check: CitationCheck }  ← is_trustworthy, uncited_claims
  │
  │  token usage: result.usage()  ← Pydantic AI built-in, no manual counting
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 2 GATE — Citation check
─────────────────────────────────────────────────────────────────────────────────────────

  citation_check.is_trustworthy == False  (any claim lacks a [chunk_id])
  │   ✗ → status = "abstained_citation"
  │         return RAGResponse immediately
  │
  ▼  pass

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 3 GATE — LLM Judge  (knowledge/agent/judge.py)
─────────────────────────────────────────────────────────────────────────────────────────

  nano model sees:  query + retrieved passages + answer  (NO chunk_ids)
  JudgeResult { verdict, confidence, reasoning }
  │
  ├── verdict = "unsupported"               → status = "abstained_judge"
  ├── confidence < judge_confidence_threshold → status = "abstained_judge"
  ├── verdict = "partial"                   → status = "answered"
  │                                           + append uncertainty note
  └── verdict = "supported"                 → status = "answered"
  │
  nano confidence < 0.5  → escalate to small model (one retry)
  ▼  answered

─────────────────────────────────────────────────────────────────────────────────────────
  POST_LLM hook — async background tasks  (asyncio.create_task — non-blocking)
─────────────────────────────────────────────────────────────────────────────────────────

  ┌── Populate L3 semantic cache
  │     JWE-encrypt RAGResponse → INSERT INTO semantic_cache (expires 60min)
  │
  ├── Store episodic turn  [Tier 2]
  │     INSERT INTO messages (conversation_id, role='assistant', content, citations, ...)
  │     UPDATE conversations SET turn_count++, last_turn_at=NOW()
  │     IF turn_count == 20: trigger summarizer (nano model, background)
  │
  ├── Extract user memories  [Tier 3]
  │     nano model: extract_facts(query, answer, recent_turns)
  │     Mem0.add(facts, user_id)  ← dedup + contradiction resolution
  │
  └── Record token usage + billing
        INSERT INTO token_usage (prompt_tokens, completion_tokens, model_id, ...)
        INCRBYFLOAT quota:{tenant_id}:cost_usd:{month}

─────────────────────────────────────────────────────────────────────────────────────────
  RAGResponse returned to API route
─────────────────────────────────────────────────────────────────────────────────────────

  {
    answer:              str,
    status:              "answered" | "abstained_retrieval" | "abstained_citation" | "abstained_judge",
    citations:           list[Citation],     ← chunk_id, document_title, relevance_score, excerpt
    confidence:          float,              ← judge confidence
    low_confidence_warning: bool,            ← True when verdict = "partial"
    pipeline_latency_ms: { retrieval, rerank, generation, judge },
    estimated_cost_usd:  float,
    model_tier_used:     str,
    cache_hit:           "l2" | "l3" | null,
    request_id:          UUID,              ← links to logs + Langfuse trace
    trace_url:           str | null         ← Langfuse trace URL
  }

─────────────────────────────────────────────────────────────────────────────────────────
  Nginx → Browser
─────────────────────────────────────────────────────────────────────────────────────────

  Streaming path  (POST /chat/stream):
    data: {"delta": "The PTO policy"}
    data: {"delta": " allows 15 days"}
    ...
    data: {"citations": [...], "done": true}

  Blocking path  (POST /chat):
    200 OK  application/json   { "request_id": ..., "data": RAGResponse }

─────────────────────────────────────────────────────────────────────────────────────────
  Browser (React)
─────────────────────────────────────────────────────────────────────────────────────────

  Streaming: sse.ts yields events → useChat.appendToken() per delta
             final "done" event  → useChat.setCitations()

  Blocking:  api.ts returns RAGResponse → useChat stores in chatStore

  UI updates:
  ├── MessageBubble renders answer (markdown)
  ├── CitationPanel populates with Citations + ConfidenceBadge
  ├── CostBadge shows: $0.0007 · 1,637 tok · small · 843ms
  ├── PipelineStatusBadge: "Answered" | "Abstained — retrieval gap"
  └── DebugPanel (if ?debug=1): latency breakdown, model tier, cache hit, trace link
```

---

## Architecture Proposal & Constraints

These sections have been moved to dedicated documents:

- **[design/ARCHITECTURE_PROPOSAL.md](design/ARCHITECTURE_PROPOSAL.md)** — Goals and multi-corpus design
- **[design/SYSTEM_DESIGN_CONSTRAINTS.md](design/SYSTEM_DESIGN_CONSTRAINTS.md)** — Load model, SLAs, token budgets, cost model, circuit breakers
- **[design/MODULE_LAYOUT.md](design/MODULE_LAYOUT.md)** — Full `knowledge/` package layout with per-subpackage guide

---

## Further Reading

**Architecture & Constraints**

| Document | What it covers |
|----------|---------------|
| [design/ARCHITECTURE_PROPOSAL.md](design/ARCHITECTURE_PROPOSAL.md) | Goals, multi-corpus design |
| [design/SYSTEM_DESIGN_CONSTRAINTS.md](design/SYSTEM_DESIGN_CONSTRAINTS.md) | Load model, SLAs, token budgets, cost model, circuit breakers |
| [design/MODULE_LAYOUT.md](design/MODULE_LAYOUT.md) | Full `knowledge/` package tree with per-subpackage guide |

**Subsystem Deep-Dives**

| Document | What it covers |
|----------|---------------|
| [design/INGESTION.md](design/INGESTION.md) | Ingestion pipeline, KG extraction, AGE store, ontology API |
| [design/REDIS_STREAMS.md](design/REDIS_STREAMS.md) | Message bus, async worker lifecycle, DLQ, job status |
| [design/CACHING.md](design/CACHING.md) | L1/L2/L3 cache layers, TTLs, invalidation guide |
| [design/RETRIEVAL.md](design/RETRIEVAL.md) | Hybrid search, RRF, reranking, confidence-aware pipeline, model tiering |
| [design/RELIABILITY.md](design/RELIABILITY.md) | Query validation, guardrails, error handling, retry strategy |
| [design/SECURITY.md](design/SECURITY.md) | JWT/JWE/RBAC, memory architecture |
| [design/DEPLOYMENT.md](design/DEPLOYMENT.md) | Docker Compose, packaging, cloud deployment, SaaS model |
| [design/EVALUATION.md](design/EVALUATION.md) | Eval system, load/chaos testing, implementation phases |

**Reference**

| Document | What it covers |
|----------|---------------|
| [REST_API.md](design/REST_API.md) | All API endpoints with request/response shapes |
| [DATASTORE.md](DATASTORE.md) | Complete datastore reference |
