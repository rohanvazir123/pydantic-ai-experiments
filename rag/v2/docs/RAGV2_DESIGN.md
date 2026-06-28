# RAG v2 — System Design

## Table of Contents

### Overview
- [System at a Glance](#system-at-a-glance)
- [Ingestion Path — Detailed](#ingestion-path--detailed)
- [Retrieval Path — Detailed](#retrieval-path--detailed)
- [Shared Storage](#shared-storage)
- [Ingestion Flow](#ingestion-flow)
- [User Query Data Flow](#user-query-data-flow)

### How the System Works
- [Agentic RAG — The Pydantic AI Loop](#agentic-rag--the-pydantic-ai-loop)
- [How Chat Requests Are Staged and Cancelled](#how-chat-requests-are-staged-and-cancelled)
- [Queues in the System](#queues-in-the-system)
- [Conversation Isolation Across Users](#conversation-isolation-across-users)
- [Why SSE and Not WebSockets](#why-sse-and-not-websockets)

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
- [OBSERVABILITY.md](design/OBSERVABILITY.md) — Logs, metrics, Langfuse traces, alerts

---

## System at a Glance

Two independent data flows share the same storage layer.

```
╔══════════════════════════════════════════════════════════════════════╗
║                  RAG v2 — Two Flows, One Store                       ║
╚══════════════════════════════════════════════════════════════════════╝

  INGESTION FLOW (async, minutes per doc)
  ────────────────────────────────────────────────────────────────────
  Upload API  →  Redis Stream (knowledge:ingest)
             →  Ingestion Worker  →  Docling  →  Chunk + Embed
                                  →  KG Extract (parallel)
                                  →  PostgreSQL (vectors) + AGE (graph)

  RETRIEVAL FLOW (sync, 1–4 s per query)
  ────────────────────────────────────────────────────────────────────
  Chat API  →  Validate  →  Cache check (L2 Redis → L3 semantic)
           →  Hybrid Search (pgvector + tsvector + AGE, parallel)
           →  RRF + CrossEncoder rerank
           →  Pydantic AI Agent loop (LLM + optional tool calls)
           →  3-gate confidence pipeline
           →  SSE stream  →  Browser

  SHARED STORAGE
  ────────────────────────────────────────────────────────────────────
  PostgreSQL (port 5432)  pgvector chunks, conversations, memory, cache
  Apache AGE  (port 5433)  entity/relationship graph per corpus
  Redis       (port 7500)  job queues, embedding cache, query cache
```

---

## Ingestion Path — Detailed

```
  Browser / API client
        │
        │  POST /api/v2/ingest   multipart (file + corpus_id)
        │  Authorization: Bearer <JWT>
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  API  (Uvicorn, port 7100)                              │
  │  Validate JWT → check quota → save file → gen job_id   │
  │  SET job:{job_id} → { status: "queued" }  [Redis hash] │
  └──────────────────────────┬──────────────────────────────┘
                             │  XADD knowledge:ingest
                             │  { job_id, tenant_id, corpus_id,
                             │    file_path, file_type, enable_graph }
                             │
                             │  ◄── 202 Accepted { job_id }
                             │       client polls GET /ingest/{job_id}
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Redis Streams                                          │
  │  knowledge:ingest       pending jobs                    │
  │  knowledge:ingest:dlq   failed after 3 retries         │
  │  knowledge:events       completion notifications        │
  │  job:{id}               status hash (queued/processing/ │
  │                         complete/failed)                │
  └──────────────────────────┬──────────────────────────────┘
                             │  XREADGROUP
                             │  consumer group: ingest-workers
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Ingestion Worker  (knowledge/ingestion/worker.py × N)  │
  │                                                         │
  │  SET job:{job_id} → { status: "processing" }           │
  │                                                         │
  │  DoclingProcessor.process(file_path)                    │
  │    asyncio.to_thread(converter.convert)  ← CPU-bound   │
  │    ├── PDF  → _get_pdf_converter()                      │
  │    ├── DOCX / MD → _get_standard_converter()           │
  │    └── Audio → Whisper ASR pipeline                     │
  │                                                         │
  │  asyncio.gather(                                        │
  │    ┌─────────────────────────┐  ┌─────────────────────┐│
  │    │  VECTOR PATH            │  │  GRAPH PATH         ││
  │    │                         │  │  (if enable_graph)  ││
  │    │  chunk_document()       │  │                     ││
  │    │   contextualize() each  │  │  load_ontology()    ││
  │    │   chunk (section head)  │  │   LRU-cached        ││
  │    │                         │  │                     ││
  │    │  embed_batch()          │  │  run_pipeline()     ││
  │    │   AsyncOpenAI           │  │   asyncio.to_thread ││
  │    │   nomic-embed-text 768d │  │   LiteLLM → Ollama  ││
  │    │   L1 lru_cache dedup   │  │   → NetworkX DiGraph││
  │    │   L2 Redis cache dedup  │  │                     ││
  │    │                         │  │  import_graph()     ││
  │    │  upsert_chunks()        │  │   nodes → AGE       ││
  │    │   asyncpg executemany   │  │   edges → AGE       ││
  │    │   → chunks table (HNSW) │  │                     ││
  │    │                         │  │  upsert_batch()     ││
  │    └──────────┬──────────────┘  │   → entity_index    ││
  │               │                 └──────────┬──────────┘│
  │               └──────────┬─────────────────┘           │
  │                          │                             │
  │  XACK  (removes from pending entries)                  │
  │  SET job:{job_id} → { status: "complete" }            │
  │  XADD knowledge:events  IngestCompleteEvent            │
  │                                                         │
  │  On error: retry ×3 with backoff                       │
  │    attempt 3 → XADD knowledge:ingest:dlq               │
  │              → SET job:{job_id} { status: "failed" }  │
  └──────────┬──────────────────────────┬──────────────────┘
             │                          │
             ▼                          ▼
  ┌──────────────────────┐   ┌──────────────────────────────┐
  │  PostgreSQL           │   │  Apache AGE  (port 5433)     │
  │  chunks  (HNSW+GIN)  │   │  kg_{tenant}_{corpus} graph  │
  │  documents            │   │  Vertices: entity types      │
  │  kg_entity_index      │   │  Edges: EMPLOYS, APPLIES_TO  │
  │  (HNSW + GIN)        │   │  via SQL wrapper (ag_catalog) │
  └──────────────────────┘   └──────────────────────────────┘
```

---

## Retrieval Path — Detailed

```
  Browser  (Vite + React 19, port 7200)
        │
        │  POST /api/v2/chat/stream
        │  { query, session_id, corpus_ids, model_tier }
        │  Authorization: Bearer <JWT>
        │  AbortController.signal  ← Stop button wires here
        ▼
  Nginx (port 443)  proxy_pass api:7100  proxy_buffering off
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  API  (Uvicorn, port 7100)                              │
  │  CorrelationID → StructuredLog → CORS → RateLimiter     │
  │                                                         │
  │  ① JWT auth      extract user_id, tenant_id, roles     │
  │  ① Quota check   Redis INCR daily + RPM → 429 if over  │
  │  ② Conv history  load last 8 turns for session_id      │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Validation Pipeline  (knowledge/validation/pipeline.py)│
  │                                                         │
  │  V1  Schema      Pydantic model → 400 if malformed      │
  │  V2  Length      len > MAX_QUERY_CHARS → 422            │
  │  V3  Language    not in allowed_languages → 422         │
  │  V4  Injection   regex + embedding-sim → 422            │
  │  V5  Policy      nano LLM → on_topic / off_topic / bad  │
  │  V6  RBAC        JWT roles vs corpus.allowed_roles → 403│
  └──────────────────────────┬──────────────────────────────┘
                             │  all pass
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Model Router  (nano: qwen2.5:0.5b, 3s timeout)        │
  │  → complexity: simple / moderate / complex              │
  │  → requires_graph: bool  (drives parallel AGE leg)      │
  │  → model_tier: nano / small / large                     │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PRE_RETRIEVE hook                                      │
  │  Cost guard  → Redis budget check → 402 if exceeded    │
  │  Mem0 search → top-3 user facts injected into prompt   │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Cache Check                                            │
  │                                                         │
  │  L2  Redis GET cache:search:{sha256(query+corpus)}      │
  │       HIT ──────────────────────────────────────────►  │
  │       MISS ↓                                       SSE  │
  │                                                    or   │
  │  Embed query (nomic-embed-text, 768d)             JSON  │
  │       L1 lru_cache hit → skip embed call          resp  │
  │                                                         │
  │  L3  pgvector cosine ≥ 0.95 on semantic_cache          │
  │       HIT → decrypt JWE ───────────────────────────►  │
  │       MISS ↓                                            │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Hybrid Search  (asyncio.gather — all three parallel)   │
  │                                                         │
  │  ├── pgvector HNSW   embedding <=> query_emb           │
  │  ├── tsvector GIN    content_tsv @@ websearch_query    │
  │  └── AGE Cypher      entity search  [if requires_graph] │
  │                                                         │
  │  RRF fusion   score = Σ 1/(60 + rank_i)               │
  │  CrossEncoder rerank  (BAAI/bge-reranker-base)         │
  │  confidence = sigmoid(cross_encoder_logit)              │
  │  drop chunks where confidence < min_confidence          │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  GATE 1 — Retrieval confidence                          │
  │  Σ confidence(top-K) < threshold                        │
  │       ✗ → abstained_retrieval  (no LLM call)            │
  └──────────────────────────┬──────────────────────────────┘
                             │  pass
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Working Memory Assembly  (working_memory.py)           │
  │                                                         │
  │  system_prompt       [Tier 5]                           │
  │  + user_memory_facts [Tier 3]  top-3 Mem0 facts         │
  │  + conv_history      [Tier 2]  last 8 turns / summary   │
  │  + retrieved_chunks  [Tier 4]  top-K with [chunk_id]    │
  │  + current_query                                        │
  │                                                         │
  │  trim_to_budget(8192 tokens)                            │
  │  drop: low-confidence chunks → old turns → user facts   │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Pydantic AI Agent Loop  (agent/agent.py)               │
  │                                                         │
  │  agent.run(query, message_history=assembled_context)    │
  │                                                         │
  │  Turn 1: LLM reads full context                         │
  │    ├── enough context → GenerationResult  (most cases)  │
  │    └── gap found → tool call                            │
  │         search_knowledge_base(targeted_sub_query)       │
  │         search_knowledge_graph(entity_or_relation)      │
  │                                                         │
  │  Turn 2+: LLM reads context + tool result              │
  │    └── more tools or GenerationResult                   │
  │                                                         │
  │  GenerationResult { answer, citations, citation_check } │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  GATE 2 — Citation check                                │
  │  citation_check.is_trustworthy == False                 │
  │       ✗ → abstained_citation                            │
  └──────────────────────────┬──────────────────────────────┘
                             │  pass
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  GATE 3 — LLM Judge  (nano → small escalation)         │
  │  sees: query + passages (no chunk_ids) + answer         │
  │  verdict: supported / partial / unsupported             │
  │       unsupported → abstained_judge                     │
  │       partial     → answered + uncertainty note         │
  │       supported   → answered                            │
  └──────────────────────────┬──────────────────────────────┘
                             │  answered
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  POST_LLM async background tasks (asyncio.create_task)  │
  │  (fire-and-forget — do not block response)              │
  │                                                         │
  │  ├── L3 semantic cache write  (JWE encrypt → PG)        │
  │  ├── Episodic store  INSERT INTO messages               │
  │  ├── Summarizer      if turn_count == 20  (nano)        │
  │  ├── Fact extract    Mem0.add(user facts)  (nano)       │
  │  └── Billing         INSERT token_usage + Redis INCR    │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  SSE stream → Nginx → Browser
  data: {"delta": "..."}   (one per token)
  data: {"citations": [...], "done": true}

  Stop button pressed at any point:
  AbortController.abort() → fetch closes → asyncio Task cancelled
  background tasks already queued are NOT cancelled
```

---

## Shared Storage

Both flows read and write the same storage layer.

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  PostgreSQL  (port 5432)              Apache AGE  (port 5433)             │
  │                                                                           │
  │  chunks           HNSW + GIN          kg_{tenant}_{corpus}  graph        │
  │  documents        source records      Vertices: entity types              │
  │  kg_entity_index  HNSW + GIN          Edges: EMPLOYS, APPLIES_TO, etc.   │
  │  semantic_cache   HNSW (L3 cache)     Cypher via ag_catalog SQL wrapper   │
  │  conversations    Tier 2 episodic                                         │
  │  messages         GIN                 Redis  (port 7500)                  │
  │  user_memories    HNSW + GIN                                              │
  │  system_prompts   Tier 5              knowledge:ingest   job queue        │
  │  token_usage      billing             knowledge:events   completions      │
  │  scheduled_jobs   scheduler           knowledge:ingest:dlq  dead letter  │
  │  audit_events     append-only         cache:embed:*      24h TTL         │
  │                                       cache:search:*     5min TTL        │
  │  RLS: SET LOCAL app.tenant_id         job:{id}           status hash     │
  │  before every query                   quota:*  cb:*  logs:*              │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Flow

### Summary

```
  User uploads file
        │
        ▼
  POST /api/v2/ingest  ──► XADD knowledge:ingest  ──► 202 Accepted + job_id
                                    │
                                    │  XREADGROUP (consumer group: ingest-workers)
                                    ▼
                          Ingestion Worker (worker.py)
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
                  Chunk + Embed           KG Extract
                  (vector path)          (graph path)
                        │                       │
                        ▼                       ▼
                 PostgreSQL chunks        Apache AGE
                 (HNSW vectors)          (entity graph)
                                    │
                                    ▼
                          XADD knowledge:events (IngestCompleteEvent)
```

### Detailed Flow

```
CLIENT
  │  POST /api/v2/ingest
  │  multipart/form-data: file=<bytes>, corpus_id=<id>
  │  Authorization: Bearer <JWT>
  ▼

API (ingest route)
  │  Validates JWT → extracts tenant_id
  │  Checks quota (Redis INCR)
  │  Saves file bytes → local temp path or object store
  │  Generates job_id (UUID)
  │
  │  XADD knowledge:ingest {
  │    job_id, tenant_id, corpus_id,
  │    file_path, file_type, original_name,
  │    enable_graph_extraction: bool
  │  }
  │
  │  SET job:{job_id} → { status: "queued", created_at: ... }   [Redis hash]
  │
  └─► 202 Accepted { job_id }   ← client polls GET /ingest/{job_id} for status

REDIS STREAMS  (knowledge:ingest)
  │
  │  XREADGROUP (group: ingest-workers, consumer: worker-{N})
  │  At most one worker picks up each message — guaranteed once processing.
  ▼

INGESTION WORKER (knowledge/ingestion/worker.py)
  │
  │  SET job:{job_id} → { status: "processing" }
  │
  │  DoclingProcessor.process(file_path, file_type)
  │     └── asyncio.to_thread(converter.convert)   ← CPU-bound, off event loop
  │         ├── PDF:   _get_pdf_converter()   (optional VLM for tables/figures)
  │         ├── DOCX:  _get_standard_converter()
  │         ├── MD:    _get_standard_converter()
  │         └── Audio: ASR via Docling Whisper pipeline
  │
  │  asyncio.gather(
  │    ├── VECTOR PATH:
  │    │     DoclingHybridChunker.chunk_document()
  │    │       └── contextualize() — prepends section heading to each chunk
  │    │     embedder.embed_batch(chunks)
  │    │       └── AsyncOpenAI(nomic-embed-text)  768-dim
  │    │           L1 lru_cache dedup → L2 Redis cache:embed:* dedup
  │    │     vector_store.upsert_chunks()
  │    │       └── asyncpg executemany → chunks table (HNSW index)
  │    │
  │    └── GRAPH PATH (if enable_graph_extraction):
  │          load_ontology(corpus.graph_ontology_path)   ← LRU-cached
  │          run_pipeline(PipelineConfig(template=OntologyClass, ...))
  │            └── asyncio.to_thread(docling-graph via LiteLLM → Ollama)
  │          PipelineContext.knowledge_graph   (NetworkX DiGraph)
  │          age_store.import_docling_graph()
  │            └── iterates nodes/edges → INSERT via SQL wrapper (not raw Cypher)
  │          entity_index.upsert_batch_from_graph()
  │            └── kg_entity_index table (HNSW + GIN for entity lookup)
  │  )
  │
  │  XACK knowledge:ingest <message-id>   ← removes from pending entries
  │  SET job:{job_id} → { status: "complete", ... }
  │  XADD knowledge:events IngestCompleteEvent { job_id, chunk_count, ... }
  │
  │  On error (up to 3 retries with exponential backoff):
  │    attempt 1,2: XCLAIM (retry same message)
  │    attempt 3:   XADD knowledge:ingest:dlq   ← dead letter; alert fires
  │    SET job:{job_id} → { status: "failed", error: "..." }
```

---

## Agentic RAG — The Pydantic AI Loop

### Is This Agentic RAG?

Yes — partially. The system uses a **two-phase hybrid**:

| Phase | Where | What |
|-------|-------|------|
| Pre-retrieval | `pipeline.py` before `agent.run()` | Deterministic hybrid search (pgvector + tsvector + AGE) → RRF → CrossEncoder |
| Agentic loop | Inside `agent.run()` | LLM decides whether to call tools for additional searches, then produces final answer |

The pre-retrieval step is not optional — it always runs and provides the initial context. The agentic loop runs on top of it. This is intentional: pure agentic RAG (no pre-retrieval) is slower and less predictable; pre-retrieval ensures the LLM always starts with something relevant.

### The Agent Loop Step by Step

```
pipeline.py calls:
  result = await traced_agent_run(query, state, message_history)
       │
       │  This calls: PydanticAgent.run(query, deps=state, message_history=...)
       │
       ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  Pydantic AI internal agentic loop                                          │
│                                                                             │
│  Turn 1:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM INPUT (assembled by working_memory.py):                        │   │
│  │  • System prompt (instructions, citation rules, abstain policy)     │   │
│  │  • User memory facts (top-3 from Mem0, pre-injected)               │   │
│  │  • Conversation history (last 8 turns)                              │   │
│  │  • Retrieved chunks as text with [chunk_id] anchors (top-K)        │   │
│  │  • Current query                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        │  LLM response — one of two outcomes:                              │
│        │                                                                    │
│        ├── A) Tool call — LLM decides it needs more information:           │
│        │    tool: search_knowledge_base(query="...", match_count=5)        │
│        │    tool: search_knowledge_graph(query="...", entity_type="...")    │
│        │                                                                    │
│        │    Pydantic AI executes the tool (async Python function):         │
│        │    → retriever.retrieve() or age_store.query()                    │
│        │    → result text returned to LLM as next turn                     │
│        │                                                                    │
│        │    Turn 2: LLM sees tool result, decides again:                   │
│        │    → may call another tool, or produce final answer               │
│        │                                                                    │
│        └── B) Final answer — LLM produces GenerationResult:               │
│             {                                                               │
│               answer: str,                  ← prose response               │
│               citations: [Citation],        ← each references a [chunk_id] │
│               citation_check: {                                             │
│                 is_trustworthy: bool,        ← are all claims cited?       │
│                 uncited_claims: [str]        ← claims without a chunk_id   │
│               }                                                             │
│             }                                                               │
│                                                                             │
│  Loop terminates when:                                                      │
│  • LLM produces GenerationResult (structured output via output_type=)      │
│  • Max tool call iterations reached (Pydantic AI default: 5)               │
│  • LLM timeout fires (model_settings: max_tokens exceeded)                 │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │  result.output  → GenerationResult
       │  result.usage() → { request_tokens, response_tokens }  (no manual counting)
       ▼
  pipeline.py continues to Layer 2 gate (citation check) and Layer 3 (judge)
```

### When Does the LLM Call a Tool?

The system prompt instructs the LLM to call `search_knowledge_base` before omitting any claim it cannot support from the provided context. Concretely, tool calls fire when a claim in the LLM's nascent answer lacks a `[chunk_id]` anchor from the pre-retrieved chunks. Common triggers:

| Trigger | Which tool | Why |
|---------|-----------|-----|
| Multi-hop: chunks mention "Project X" but don't name its members | `search_knowledge_base("Project X team members")` | The initial query embedded the full question; a decomposed sub-query hits the right section |
| Cross-reference: a chunk says "see the Benefits Addendum" | `search_knowledge_base("Benefits Addendum")` | The addendum itself wasn't in top-K results |
| Relationship question: "who reports to the CTO?" | `search_knowledge_graph("CTO direct reports")` | Relationships live in the graph, not in text chunks |
| Router missed graph signal but LLM sees entity question | `search_knowledge_graph(query, entity_type=...)` | LLM overrides router's `requires_graph=False` decision |

The key instruction in the system prompt: **"Use a more targeted or decomposed query than the original question."** Repeating the full original query would hit the same chunks already provided. The tool call is only useful if the query narrows down to a specific policy name, section, sub-topic, or entity name.

Tool calls are **round-trips to the LLM**: Pydantic AI appends the tool result as a new message in the conversation, then re-invokes the model. The loop continues until the LLM produces a `GenerationResult` or the iteration cap (5 by default) is reached. In practice most queries take 1 turn (no tool calls); multi-hop or cross-referencing queries take 2–3 turns.

### Streaming Path

The streaming path (`POST /chat/stream`) uses `stream_agent` — a separate agent with `output_type=str` instead of `GenerationResult`. This allows `agent.run_stream()` to yield tokens to the client as they are generated. The trade-off: citations are not extracted in the streaming path (the answer is streamed as plain text). Layer 2 (citation gate) and Layer 3 (judge) are skipped; only Layer 1 (retrieval confidence gate) runs.

---

## How Chat Requests Are Staged and Cancelled

### Where Is the Request Staged?

Chat requests are **not queued** — they are handled directly by an asyncio coroutine that runs for the lifetime of the HTTP connection. The "staging" happens implicitly inside the asyncio event loop.

```
Browser sends POST /api/v2/chat/stream
       │
       ▼
Uvicorn accepts the TCP connection
  → creates an asyncio Task for the request coroutine
  → Task is scheduled on the event loop

Event loop runs the coroutine:
  while the coroutine awaits I/O (pgvector query, Redis lookup, Ollama tokens),
  the event loop runs other coroutines for other concurrent requests.

  This is NOT blocking — 50 concurrent requests means 50 concurrent asyncio Tasks,
  each making progress whenever their I/O completes.
```

There is no Redis Stream for chat. Ingestion uses a queue because it runs for minutes and is CPU-bound (Docling conversion). Chat is async I/O throughout and completes in 1–4 seconds — adding a queue would introduce 1–5 seconds of pickup delay, exceeding the entire response SLA.

| | Ingestion (needs queue) | Chat (no queue) |
|---|---|---|
| Client waiting? | No — fire and forget | Yes — user is watching |
| Duration | Minutes (Docling + embed) | 1–4 seconds |
| CPU-bound? | Yes (Docling blocks event loop) | No (async I/O throughout) |
| Failure recovery | Retry + DLQ | HTTP 500, client retries |

After the response is sent, `asyncio.create_task` fires background tasks (cache write, episodic storage, memory extraction, billing). These run after the user already has their answer.

### Cancel Button and Multiple Queries

**There is a cancel button.** The Send button in `InputBar.tsx` switches to a
spinning stop icon while a stream is in progress. Clicking it calls `stop()` in
`useChat.ts`, which fires `AbortController.abort()`.

```
User clicks Send                User clicks Stop (same button, different state)
     │                                    │
     ▼                                    ▼
ChatPage.handleSend(query)       useChat.stop()
  setLoading(true)                 abortRef.current.abort()
  await sendMessage(query)              │
       │                               ▼
       │  useChat.sendMessage:    fetch() cancelled → HTTP connection closes
       │    abortRef.abort()      → async generator stops yielding
       │    (cancels any          → Uvicorn detects disconnect
       │    previous in-flight)   → asyncio Task cancelled
       │    abortRef = new        → agent.run() receives CancelledError
       │    AbortController()
       │    fetch(..., signal)
       ▼
  finally: setLoading(false)     finally: setLoading(false)
  button returns to Send icon    button returns to Send icon
```

**UI serialization is enforced, not just recommended.** When `loading=true`:
- The `submit()` function returns early (`if (!q || loading) return`)
- Enter key is a no-op
- Suggested question buttons are `disabled`
- The only action available is clicking Stop

So users cannot stack queries. They can only stop the current one, wait for the
button to return to Send, then type the next query.

**Sending a new query also cancels the previous one** — `sendMessage` calls
`abortRef.current?.abort()` before creating a new `AbortController`. This is a
belt-and-suspenders guard for any code path that bypasses the `loading` check.

**Abort is silent on the client.** The `catch` block in `useChat` checks
`err?.name !== 'AbortError'` — an `AbortError` is swallowed with no error
message shown. The streaming message is left in whatever partial state it was
in when Stop was pressed. A `finally` block always runs `setLoading(false)`
regardless of how the stream ended.

### Cancellation Propagation — What Actually Happens Server-Side

Cancellation propagates differently at each layer. Not everything is cancelled,
and not everything should be.

#### 1. In-flight DB queries — cancelled at PostgreSQL level

When the asyncio Task is cancelled, `CancelledError` propagates into whichever
`await` is currently running. asyncpg intercepts it and sends a PostgreSQL
`CancelRequest` message to the backend process. The running query is killed
server-side before it completes.

This applies to all three legs of the hybrid search, which run under
`asyncio.gather` (retriever.py):

```python
sem_results, text_results, graph_results = await asyncio.gather(
    sem_task, text_task, graph_task   # no return_exceptions=True
)
```

When the parent Task is cancelled, `CancelledError` propagates into the
`gather`, which cancels all three child coroutines. All three DB queries
(pgvector HNSW scan, tsvector GIN query, AGE Cypher query) receive a
PostgreSQL `CancelRequest` simultaneously.

#### 2. `CancelledError` is not accidentally swallowed

The streaming pipeline has a broad `except` block (pipeline.py):

```python
try:
    async with stream_agent.run_stream(...) as streamed:
        async for delta in streamed.stream_text(delta=True):
            yield _sse({"delta": delta})
        ...
    await registry.fire(HookPoint.POST_LLM, ctx)   # ← never reached on cancel
except Exception as exc:                            # ← does NOT catch CancelledError
    yield _sse({"error": str(exc)})
finally:
    await state.close()                             # ← always runs
```

In Python 3.8+, `asyncio.CancelledError` inherits from `BaseException`, not
`Exception`. So `except Exception` correctly lets `CancelledError` through.
The `finally` block always runs — `state.close()` properly returns asyncpg
connections to the pool.

#### 3. `POST_LLM` hook never fires on cancel — background tasks are not created

`POST_LLM` is reached only after the stream completes normally. If the user
cancels mid-stream, `CancelledError` is raised inside `async for delta`,
exits the `async with` context manager, skips past `POST_LLM`, and the
`except Exception` block does not catch it.

| Task | Fires on cancel? | Notes |
|------|-----------------|-------|
| L3 semantic cache write | **No** | Lives in `POST_LLM` hook |
| `INSERT INTO messages` (episodic) | **No** | Lives in `POST_LLM` hook |
| Conversation summarizer | **No** | Lives in `POST_LLM` hook |
| Mem0 fact extraction | **No** | Lives in `POST_LLM` hook |
| Billing `INSERT token_usage` | **No** | Lives in `POST_LLM` hook |
| L2 Redis search cache write | **Yes** | Fired as `create_task` immediately after retrieval — before the agent runs; already in the event loop if cancel happens during LLM generation |

#### 4. Practical consequences

**Billing gap:** if the LLM generated 800 tokens before the user cancelled,
those tokens are not recorded. Quota enforcement can be evaded by spamming
cancel. Fix: track token usage inside the streaming loop and flush in the
`finally` block rather than relying on `POST_LLM`.

**Conversation history is clean:** cancelled turns are never stored. The
partial streamed text the user saw is lost on reload — this is correct
behaviour; storing a half-answer would be confusing.

**L2 cache is populated for cancelled requests:** if retrieval completed
before the cancel arrived during LLM generation, the search results are
cached in Redis for 5 minutes. A repeated query immediately after cancel
will hit L2 and skip retrieval. This is desirable — retrieval already
succeeded; there is no reason to redo it.

**LLM call to Ollama is terminated:** `CancelledError` closes the underlying
`httpx` connection inside Pydantic AI's `run_stream`. Ollama sees the
connection drop and stops generating. Partial tokens already sent are lost.

---

## Queues in the System

All queues are Redis Streams. Redis Streams provide consumer groups (at-most-once delivery per group), persistent log (survives Redis restart with AOF), and redelivery on crash (`XAUTOCLAIM`).

```
Stream name                  Purpose                              Consumer group
─────────────────────────────────────────────────────────────────────────────────
knowledge:ingest             Ingestion jobs (one per file upload) ingest-workers
knowledge:eval               Evaluation run requests              eval-workers
knowledge:events             Completion notifications             (no consumer —
                             (IngestCompleteEvent, EvalDone)       broadcast only)
knowledge:ingest:dlq         Failed ingestion jobs after 3        manual review /
                             retries                               alert-only
knowledge:logs:recent        Structured log ring buffer           (no consumer —
                             5,000 entries, 24h TTL               UI reads directly)
```

**There is no chat queue.** See [How Chat Requests Are Staged](#how-chat-requests-are-staged-and-cancelled).

### Job Status

Every async job (ingestion, eval) writes a status hash in Redis:

```
job:{job_id}  →  { status: "queued" | "processing" | "complete" | "failed",
                   created_at, started_at, completed_at, error, chunk_count }
```

The API exposes `GET /ingest/{job_id}` and `GET /evaluate/{run_id}` so clients can poll for completion without holding a long HTTP connection.

---

## Conversation Isolation Across Users

### How Users and Conversations Are Identified

Every request carries a JWT (RS256). The middleware extracts three values from it:

| Field | Source | Purpose |
|-------|--------|---------|
| `user_id` | JWT `sub` claim | Identifies the individual user |
| `tenant_id` | JWT `tenant_id` claim | Identifies the organisation (multi-tenancy) |
| `session_id` | Request body | Identifies the conversation thread |

`session_id` is generated by the frontend when the user starts a new chat. It is passed in every `POST /chat` or `POST /chat/stream` body. The server never generates it — the client controls conversation boundaries.

### Storage Isolation

```
PostgreSQL tables — per-conversation scoping:

  conversations
    id = session_id (UUID, from client)
    tenant_id
    user_id
    summary (auto-generated at turn 20 by nano model)
    turn_count

  messages
    conversation_id → conversations.id
    role: user | assistant
    content, citations, model_tier, prompt_tokens, ...

  user_memories    ← Tier 3 (Mem0)
    user_id        ← scoped per user, not per conversation
    tenant_id
    content (fact extracted from conversation)
    embedding (HNSW)
    tsvector (GIN)
```

### Row-Level Security

Every asyncpg query is preceded by:

```sql
SET LOCAL app.tenant_id = '<tenant_id>';
```

PostgreSQL RLS policies on `conversations`, `messages`, `chunks`, etc. enforce:

```sql
USING (tenant_id = current_setting('app.tenant_id'))
```

This means even if a bug passes the wrong `tenant_id` in application code, the database itself rejects cross-tenant reads. User A cannot see User B's conversations even if they share a tenant.

### Conversation History Loading

At the start of each request, `conversation_store.load_active_window(session_id)` fetches:

```sql
SELECT role, content FROM messages
WHERE conversation_id = $1
ORDER BY created_at DESC
LIMIT 8
```

After 20 turns, a nano-model summarizer runs asynchronously and writes a `summary` to `conversations`. Subsequent loads return `summary + last 8 turns` to keep context within the token budget.

User memories (`user_memories`) are fetched separately in the `PRE_RETRIEVE` hook — they are keyed by `user_id`, not `session_id`, so they persist across all of a user's conversations.

### Concurrency: Two Users, Same Time

```
User A: POST /chat { session_id: "A1", query: "PTO policy?" }
User B: POST /chat { session_id: "B1", query: "expense policy?" }

Both arrive simultaneously → two asyncio Tasks, running concurrently.

Each Task:
  • loads its own session history (different session_id → different rows)
  • runs its own retriever instance
  • calls agent.run() independently (no shared state between runs)
  • writes to its own message row

No locking. No serialization. Isolation is at the DB level (RLS + session_id scope).
```

---

## Why SSE and Not WebSockets

The streaming path uses Server-Sent Events over a plain HTTP POST, not WebSockets. This is a deliberate choice.

**Chat is unidirectional by nature.** The user sends one message; the server streams back one response. Once the user hits send, there is nothing more to transmit until the full response arrives. SSE is designed for exactly this — a single long-lived HTTP response that the server writes to over time.

| Dimension | SSE (chosen) | WebSockets |
|-----------|-------------|------------|
| Direction | Server → client only | Bidirectional |
| Matches chat model? | Yes — one request, one streamed reply | Overkill |
| Auth | Bearer token on every HTTP request | Token sent once at upgrade; harder to revoke |
| Nginx config | `proxy_buffering off` — one line | Requires Upgrade + Connection headers |
| CDN / LB support | Native HTTP | Requires WS-aware proxy |
| Reconnection on drop | Browser auto-reconnects (EventSource) | Must implement in client code |
| Server-side state | Stateless — fresh coroutine per stream | Stateful — must track open connections |

**Note:** we use `fetch + ReadableStream` rather than the browser's built-in `EventSource`, because `EventSource` is GET-only and cannot carry a JSON body. Our streaming endpoint is `POST /chat/stream` — it needs `session_id`, `corpus_ids`, and `model_tier` in the body. The `streamSSE` helper in `src/lib/sse.ts` replicates the SSE framing (`data: ...\n\n` parsing) on top of `fetch`.

**Multi-turn conversations work fine with SSE.** Each follow-up is a new `POST /chat/stream` carrying the same `session_id`. The API loads conversation history for that session from PostgreSQL, prepends it to context, and streams the next response.

**The one thing WebSockets would enable that SSE cannot:** the client interrupting a generation mid-stream to immediately send a new query. With SSE, the client must abort the current fetch first (which cancels the server coroutine), then send the new request. In practice users don't interrupt — they wait. If that changes, the transport layer is the only thing that needs to change; all retrieval and agent logic stays identical.

---

## User Query Data Flow

### Summary

```
  Browser  →  Nginx  →  API
                          │
                     ┌────▼────────────┐
                     │  Auth + Quota   │  reject: 401/429
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  Validation     │  reject: 400/422/403
                     │  V1–V6          │
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  Model Router   │  nano → tier decision
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  Cache check    │  HIT → return immediately
                     │  L2 Redis       │
                     │  L3 semantic    │
                     └────┬────────────┘
                          │ MISS
                     ┌────▼────────────┐
                     │  Hybrid Search  │  pgvector + tsvector + AGE (parallel)
                     │  RRF + rerank   │
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  GATE 1         │  low confidence → abstain (no LLM)
                     │  confidence Σ   │
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  Pydantic AI    │  LLM reads context
                     │  agent loop     │  → optional tool calls
                     │                 │  → GenerationResult
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  GATE 2         │  uncited claims → abstain
                     │  citation check │
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  GATE 3         │  unsupported → abstain
                     │  LLM judge      │
                     └────┬────────────┘
                          │
                     ┌────▼────────────┐
                     │  SSE stream     │  tokens → browser
                     │  + background   │  cache / billing / memory
                     └─────────────────┘
```

### Detailed Step-by-Step

> Quick reference: Auth + quota → V1–V6 validation → model router → cost guard → PRE\_RETRIEVE (user memories) → retrieval (L2 cache → embed → L3 cache → parallel hybrid search → RRF → CrossEncoder → confidence filter) → Layer 1 gate → working memory assembly (all five tiers) → RAG agent (LLM reads context + optional tool calls → GenerationResult) → Layer 2 gate (citation check) → judge (nano→small escalation) → async background tasks (L3 cache, episodic storage, memory extraction, billing) → RAGResponse → SSE stream → browser

```
USER TYPES A QUERY AND HITS SEND
─────────────────────────────────────────────────────────────────────────────────────────

  Browser (React + Zustand)
  │  chatStore.sendMessage(query, session_id, corpus_ids)
  │  sse.ts: POST /api/v2/chat/stream   { query, session_id, corpus_ids, model_tier }
  │          Authorization: Bearer <access_token>
  │          AbortController.signal  ← for cancel support
  ▼

  Nginx (port 443)
  │  /api/v2/* → proxy_pass api:7100
  │  proxy_buffering off   proxy_read_timeout 3600s  (SSE route)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  API  (knowledge/api/routes/chat.py)
─────────────────────────────────────────────────────────────────────────────────────────

  ① Auth & quota
  │  require_jwt()           → extracts user_id, tenant_id, roles from JWT
  │  enforce_quota()         → Redis INCR daily counter + RPM sliding window
  │                            → 429 if limit hit
  ▼

  ② Load conversation history (parallel with validation)
  │  conversation_store.load_active_window(session_id)
  │    → SELECT last 8 messages WHERE conversation_id = session_id  [PostgreSQL]
  │    → if turn_count > 20: prepend conversations.summary
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  Validation Pipeline  (knowledge/validation/pipeline.py)
─────────────────────────────────────────────────────────────────────────────────────────

  V1  Schema check         Pydantic model — rejects malformed body → 400
  V2  Length guard         len(query) > MAX_QUERY_CHARS → 422
  V3  Language detect      optional — 422 if not in allowed_languages
  V4  Injection detector   regex + embedding-sim against known attack patterns → 422
  V5  Content policy       nano model (qwen2.5:0.5b)
  │   ContentPolicyResult { verdict, confidence, reason }
  │   on_topic → continue  │  off_topic → 422  │  inappropriate → 400 + audit
  V6  RBAC check           JWT roles vs CorpusConfig.allowed_roles → 403
  ▼   all pass

─────────────────────────────────────────────────────────────────────────────────────────
  Model Router  (knowledge/agent/model_router.py)
─────────────────────────────────────────────────────────────────────────────────────────

  nano model → RoutingDecision
  │  complexity: simple | moderate | complex
  │  requires_graph: bool
  │  model_tier: nano | small | large
  │  (3s timeout → fallback: small)
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
  │  tsvector BM25 + pgvector cosine → RRF(k=60)  [user_memories table]
  │  top-3 facts prepended to system prompt
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  Retrieval Pipeline  (knowledge/retrieval/retriever.py)
─────────────────────────────────────────────────────────────────────────────────────────

  ③ L2 cache check
  │  Redis GET cache:search:{sha256(query+corpus+filters)}
  │  HIT → return cached RAGResponse immediately (no embed, no LLM)
  │  MISS ↓

  ④ Embed query
  │  AsyncOpenAI (nomic-embed-text) → vector(768)
  │  L1 lru_cache hit → skip embed call

  ⑤ L3 semantic cache check
  │  SELECT … ORDER BY query_emb <=> $vec LIMIT 1
  │  cosine ≥ 0.95 and not expired
  │  HIT → decrypt JWE → return cached answer + citations
  │  MISS ↓

  ⑥ Hybrid retrieval  (asyncio.gather — all three in parallel)
  │  ├── semantic_search()   pgvector HNSW   embedding <=> query_emb
  │  ├── text_search()       tsvector GIN    content_tsv @@ websearch_to_tsquery
  │  └── graph_retrieval()   AGE Cypher      [if requires_graph, circuit-broken]

  ⑦ RRF fusion   score = Σ 1/(60 + rank_i) across search legs

  ⑧ CrossEncoder rerank
  │  BAAI/bge-reranker-base (local)
  │  confidence = sigmoid(cross_encoder_logit)  ← calibrated 0–1

  ⑨ Confidence filter
  │  drop results where confidence < min_confidence_score (default 0.10)

  ⑩ Populate L2 Redis cache (async, non-blocking)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 1 GATE  (knowledge/agent/pipeline.py)
─────────────────────────────────────────────────────────────────────────────────────────

  aggregate = Σ confidence for top-K results
  │
  aggregate < retrieval_confidence_threshold (default 1.5)
  │   ✗ → status = "abstained_retrieval"   (no LLM call at all)
  │         return RAGResponse immediately
  ▼  pass

─────────────────────────────────────────────────────────────────────────────────────────
  Assemble working memory  (knowledge/memory/working_memory.py)
─────────────────────────────────────────────────────────────────────────────────────────

  system_prompt          [Tier 5 — from system_prompts table or prompts.py]
  + user_memory_context  [Tier 3 — top-3 facts from PRE_RETRIEVE]
  + conversation_history [Tier 2 — last 8 turns or summary + last 8]
  + retrieved_chunks     [Tier 4 — top-K confidence-filtered chunks with [chunk_id]]
  + current_query
  ↓
  trim_to_budget(8192 tokens)
  drop order: lowest-confidence chunks → oldest turns → user memories
  set context_truncated: True if trimming was needed
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  RAG Agent  (knowledge/agent/agent.py) — Pydantic AI agentic loop
─────────────────────────────────────────────────────────────────────────────────────────

  agent.run(query, message_history=assembled_context, deps=state)
  │
  │  Turn 1: LLM receives full context (chunks + history + system prompt + query)
  │  │
  │  ├── LLM produces tool call → Pydantic AI executes tool → result returned to LLM
  │  │   tool: search_knowledge_base(query, match_count, search_type)
  │  │     → retriever.retrieve()  [additional hybrid search]
  │  │   tool: search_knowledge_graph(query, entity_type, limit)
  │  │     → age_store.query()     [entity/relationship lookup]
  │  │
  │  │  Turn 2+: LLM sees tool results, decides to call another tool or answer
  │  │
  │  └── LLM produces final GenerationResult (structured output):
  │       {
  │         answer:         str,
  │         citations:      [{ chunk_id, relevance_score }],
  │         citation_check: { is_trustworthy, uncited_claims }
  │       }
  │
  │  result.usage() → { request_tokens, response_tokens }  (Pydantic AI built-in)
  ▼

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 2 GATE — Citation check
─────────────────────────────────────────────────────────────────────────────────────────

  citation_check.is_trustworthy == False  (any claim lacks a [chunk_id])
  │   ✗ → status = "abstained_citation"
  │         return RAGResponse immediately
  ▼  pass

─────────────────────────────────────────────────────────────────────────────────────────
  LAYER 3 GATE — LLM Judge  (knowledge/agent/judge.py)
─────────────────────────────────────────────────────────────────────────────────────────

  nano model sees:  query + retrieved passages + answer  (no chunk_ids)
  JudgeResult { verdict, confidence, reasoning }
  │
  ├── verdict = "unsupported"               → status = "abstained_judge"
  ├── confidence < judge_confidence_threshold → status = "abstained_judge"
  ├── verdict = "partial"                   → status = "answered" + uncertainty note
  └── verdict = "supported"                 → status = "answered"
  │
  nano confidence < 0.5 → escalate to small model (one retry)
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
    answer:               str,
    status:               "answered" | "abstained_retrieval" | "abstained_citation" | "abstained_judge",
    citations:            [{ chunk_id, document_title, relevance_score, excerpt }],
    confidence:           float,
    low_confidence_warning: bool,
    pipeline_latency_ms:  { retrieval, rerank, generation, judge },
    estimated_cost_usd:   float,
    model_tier_used:      str,
    cache_hit:            "l2" | "l3" | null,
    request_id:           UUID,
    trace_url:            str | null
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
  Browser (React + Zustand)
─────────────────────────────────────────────────────────────────────────────────────────

  Streaming: sse.ts async generator yields events → chatStore.appendToken() per delta
             final "done" event → chatStore.setCitations()

  Blocking:  api.ts returns RAGResponse → chatStore stores full response

  UI updates:
  ├── MessageBubble renders answer (react-markdown + remark-gfm)
  ├── CitationPanel populates with Citations + ConfidenceBadge
  ├── CostBadge shows: $0.0007 · 1,637 tok · small · 843ms
  ├── PipelineStatusBadge: "Answered" | "Abstained — retrieval gap"
  └── DebugPanel (if ?debug=1): latency breakdown, model tier, cache hit, trace link
```

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
| [design/REST_API.md](design/REST_API.md) | All API endpoints with request/response shapes |
| [DATASTORE.md](DATASTORE.md) | Complete datastore reference |
