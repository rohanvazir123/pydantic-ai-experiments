# RAG v2 — Reliability & Safety

## Table of Contents

- [Query Validation & Hook System](#query-validation--hook-system)
  - [Validation Pipeline (`knowledge/validation/pipeline.py`)](#validation-pipeline-knowledgevalidationpipelinepy)
  - [Hook System (`knowledge/hooks/`)](#hook-system-knowledgehooks)
- [Guardrail Architecture — Key Principles](#guardrail-architecture--key-principles)
- [Error Handling Strategy](#error-handling-strategy)
  - [Error Taxonomy](#error-taxonomy)
  - [Structured Error Response Schema](#structured-error-response-schema)
  - [HTTP Status Code Policy](#http-status-code-policy)
  - [Graceful Degradation Matrix](#graceful-degradation-matrix)
  - [Error Propagation — Worker Pipeline](#error-propagation--worker-pipeline)
  - [Alert Email Configuration](#alert-email-configuration)
- [Retry & Resilience Strategy](#retry--resilience-strategy)
  - [Retriable vs Non-Retriable Classification](#retriable-vs-non-retriable-classification)
  - [Backoff Specification](#backoff-specification)
  - [Circuit Breaker Design](#circuit-breaker-design)
  - [Idempotency Design](#idempotency-design)
  - [Cascading Timeout Budget](#cascading-timeout-budget)
  - [Worker Retry Loop](#worker-retry-loop)

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

