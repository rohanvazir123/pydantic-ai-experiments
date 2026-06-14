# RAG v2 — REST API Reference

## Table of Contents

- [Overview](#overview)
- [Auth](#auth)
- [Chat](#chat)
- [Search](#search)
- [Ingest](#ingest)
- [Corpus](#corpus)
- [Conversations](#conversations)
- [Memories](#memories)
- [Scheduler](#scheduler)
- [Evaluate](#evaluate)
- [Feedback](#feedback)
- [Ops](#ops)

---

## Overview

**Base URL:** `http://localhost:8001/api/v2` (dev) · `https://<host>/api/v2` (prod via Nginx)

**Envelope:** every response (success or error) is wrapped in `APIResponse[T]`:

```json
{
  "request_id": "uuid",
  "data":       { ... } | null,
  "error":      { "code": "...", "message": "...", "details": {} } | null,
  "cache_hit":  "l2" | "l3" | null
}
```

**Auth:** all routes except `/health`, `/metrics`, and `/auth/*` require `Authorization: Bearer <access_token>`.

**Status codes:**

| Code | Meaning |
|------|---------|
| 200  | Success |
| 400  | Bad request / content policy violation |
| 401  | Missing or invalid JWT |
| 402  | Tenant budget exhausted |
| 403  | Insufficient JWT role |
| 404  | Resource not found |
| 422  | Validation error |
| 429  | Rate limit hit |
| 503  | Dependency not initialised |

---

## Auth

### `POST /api/v2/auth/token`

Issue a JWT access token.

**Request**
```json
{ "email": "user@example.com", "password": "secret" }
```

**Response**
```json
{
  "access_token": "eyJ...",
  "token_type":   "bearer",
  "expires_in":   900
}
```

> **Status:** stub — accepts any credentials. Phase 9 adds RS256 signing + credential verification.

---

### `POST /api/v2/auth/refresh`

Rotate the refresh token and return a new access token. Reads the httpOnly refresh cookie set at login.

**Response**
```json
{
  "access_token": "eyJ...",
  "token_type":   "bearer",
  "expires_in":   900
}
```

---

## Chat

### `POST /api/v2/chat`

Blocking chat — runs the full 3-gate pipeline (retrieval → agent → judge) and returns a single JSON response.

**Request**
```json
{
  "query":      "What is the PTO policy?",
  "corpus_ids": ["hr-policies"],
  "session_id": "uuid",
  "model_tier": "auto"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | User question |
| `corpus_ids` | string[] | required | One or more corpus slugs to search |
| `session_id` | string | required | UUID — links turns into a conversation |
| `model_tier` | `"auto"` \| `"small"` \| `"large"` | `"auto"` | Override model routing |
| `message_history` | object[] \| null | null | Prior turns for multi-turn context |

**Response**
```json
{
  "answer":                 "Employees accrue 15 days of PTO per year...",
  "status":                 "answered",
  "confidence":             0.87,
  "citations": [
    {
      "chunk_id":        "uuid",
      "document_title":  "Team Handbook",
      "document_source": "documents/team-handbook.md",
      "relevance_score": 0.91,
      "excerpt":         "Employees accrue 15 days of PTO..."
    }
  ],
  "low_confidence_warning": false,
  "pipeline_latency_ms":    { "retrieval": 120, "generation": 850, "judge": 65 },
  "estimated_cost_usd":     0.0007,
  "model_tier_used":        "small",
  "prompt_tokens":          1420,
  "completion_tokens":      312,
  "cache_hit":              null,
  "request_id":             "uuid",
  "trace_url":              null
}
```

`status` values:

| Status | Meaning |
|--------|---------|
| `answered` | Response generated and passed all gates |
| `abstained_retrieval` | Layer 1 — aggregate retrieval confidence too low; no LLM call |
| `abstained_citation` | Layer 2 — answer contained uncited claims |
| `abstained_judge` | Layer 3 — judge rated answer as unsupported (only when `judge_enabled=True`) |

When `status` is an abstention, `answer` contains a corpus-safe fallback message (never LLM-generated).

---

### `POST /api/v2/chat/stream`

SSE streaming chat — Layer 1 gate only; judge is skipped for latency. Yields token deltas as Server-Sent Events.

**Request:** same body as `POST /chat`.

**SSE event stream:**
```
data: {"delta": "Employees accrue "}
data: {"delta": "15 days of PTO per year"}
...
data: {"done": true, "citations": [...], "prompt_tokens": 1420, "completion_tokens": 312}
```

On Layer 1 abstention:
```
data: {"abstained": true, "layer": 1, "reason": "no_retrieval"}
```

On error:
```
data: {"error": "Internal server error"}
```

---

## Search

### `POST /api/v2/search`

Synchronous hybrid search — retrieval only, no LLM. Returns ranked chunks with confidence scores. P95 target < 600ms.

> Search runs directly in the API process (not via Redis Streams). See [RAGV2_DESIGN.md](RAGV2_DESIGN.md) for the sync vs async path decision.

**Request**
```json
{
  "query":       "remote work policy",
  "corpus_ids":  ["hr-policies"],
  "k":           5,
  "search_type": "hybrid",
  "include_graph": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search query |
| `corpus_ids` | string[] | required | Corpora to search |
| `k` | int (1–50) | 5 | Max results |
| `search_type` | `"hybrid"` \| `"semantic"` \| `"text"` | `"hybrid"` | Search mode |
| `include_graph` | bool | false | Include graph traversal results |
| `metadata_filter` | object \| null | null | Key/value metadata filters |

**Response**
```json
{
  "results": [
    {
      "chunk_id":        "uuid",
      "document_title":  "Team Handbook",
      "document_source": "documents/team-handbook.md",
      "content":         "Full chunk text...",
      "confidence":      0.91,
      "excerpt":         "First 200 chars..."
    }
  ],
  "query": "remote work policy",
  "k": 5
}
```

---

## Ingest

Ingestion is async — the API publishes a job to Redis Streams and returns immediately. The ingest-worker processes it in the background.

### `POST /api/v2/ingest`

Submit an ingestion job.

**Request**
```json
{
  "corpus_id":               "hr-policies",
  "source_path":             "/path/to/document.pdf",
  "enable_graph_extraction": false,
  "mode":                    "incremental"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `corpus_id` | string | required | Target corpus |
| `source_path` | string \| null | null | Local path to ingest |
| `source_url` | string \| null | null | URL to fetch and ingest |
| `enable_graph_extraction` | bool | false | Run KG extraction alongside chunking |
| `mode` | `"full"` \| `"incremental"` | `"incremental"` | `incremental` skips files already fingerprinted |

**Response**
```json
{
  "job_id":       "uuid",
  "status":       "queued",
  "corpus_id":    "hr-policies",
  "submitted_at": "2026-06-14T12:00:00Z"
}
```

---

### `GET /api/v2/ingest/{job_id}/status`

Poll job status from the Redis job hash.

**Response**
```json
{
  "job_id":          "uuid",
  "status":          "completed",
  "progress":        100,
  "corpus_id":       "hr-policies",
  "chunks_ingested": 42,
  "error":           null,
  "submitted_at":    "2026-06-14T12:00:00Z",
  "completed_at":    "2026-06-14T12:01:30Z"
}
```

`status` values: `queued` · `running` · `completed` · `failed`

---

### `GET /api/v2/ingest/{job_id}/stream`

SSE stream of job progress events. Closes when the job reaches `job_completed` or `job_failed`.

```
data: {"job_id": "uuid", "event_type": "chunk_stored", "progress": 50}
data: {"job_id": "uuid", "event_type": "job_completed", "chunks_ingested": 42}
```

---

## Corpus

### `GET /api/v2/corpus`

List all corpora accessible to the current JWT role.

**Response**
```json
[
  {
    "id":                      "hr-policies",
    "display_name":            "HR Policies",
    "source_folders":          ["/data/hr"],
    "allowed_roles":           ["employee", "admin"],
    "enable_graph_extraction": false,
    "graph_ontology_path":     null
  }
]
```

---

### `POST /api/v2/corpus/{corpus_id}/cache/invalidate`

Flush all L2 (Redis) and L3 (pgvector semantic) cache entries for a corpus. Use after bulk updates.

**Response**
```json
{ "corpus_id": "hr-policies", "cache_keys_deleted": 17 }
```

---

### `GET /api/v2/corpus/{corpus_id}/ontology`

Return the current ontology Python source for a corpus (or the generic default if none is set).

**Response**
```json
{
  "corpus_id":  "hr-policies",
  "source":     "class HRPolicyDocument(BaseModel): ...",
  "is_default": false
}
```

---

### `POST /api/v2/corpus/{corpus_id}/ontology`

Upload a Python ontology file. The file is validated (must contain a root `BaseModel` subclass) before saving. Clears the LRU ontology cache.

**Request:** `multipart/form-data` with a single `file` field (`.py` file).

**Response**
```json
{ "corpus_id": "hr-policies", "root_class": "HRPolicyDocument", "saved": true }
```

Returns `422` if the file is invalid.

---

### `DELETE /api/v2/corpus/{corpus_id}/ontology`

Remove the custom ontology. Next extraction uses the generic default.

**Response**
```json
{ "corpus_id": "hr-policies", "deleted": true, "reverted_to": "generic" }
```

---

## Conversations

Conversation history is Tier 2 episodic memory — scoped to the current user via JWT.

### `GET /api/v2/conversations`

List conversations, newest first.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `limit` | int (1–100) | 20 | Max results |
| `cursor` | string | null | Pagination cursor |

**Response**
```json
[
  {
    "id":          "uuid",
    "session_id":  "uuid",
    "title":       "PTO policy questions",
    "summary":     "User asked about...",
    "turn_count":  5,
    "last_turn_at": "2026-06-14T11:30:00Z"
  }
]
```

---

### `GET /api/v2/conversations/{conversation_id}`

Get conversation metadata and all messages.

**Response**
```json
{
  "id":       "uuid",
  "messages": [
    { "role": "user",      "content": "What is the PTO policy?" },
    { "role": "assistant", "content": "Employees accrue 15 days..." }
  ]
}
```

---

### `DELETE /api/v2/conversations/{conversation_id}`

Soft-delete. Hard delete occurs after the 7-day GDPR grace period.

**Response**
```json
{ "deleted": true }
```

---

## Memories

User memories are Tier 3 semantic memory (Mem0) — long-term facts extracted from conversations, scoped to the current user.

### `GET /api/v2/memories`

List all memories for the current user.

**Response**
```json
[
  { "id": "uuid", "content": "User prefers concise answers", "created_at": "..." }
]
```

---

### `POST /api/v2/memories`

Manually add a memory.

**Request**
```json
{ "content": "User works in the London office" }
```

**Response**
```json
{ "id": "uuid", "content": "User works in the London office" }
```

---

### `DELETE /api/v2/memories/{memory_id}`

Delete one memory. Immediate hard delete (GDPR requirement).

**Response**
```json
{ "deleted": true }
```

---

### `DELETE /api/v2/memories`

Delete ALL memories for the current user (right to erasure).

**Response**
```json
{ "deleted_count": 12 }
```

---

## Scheduler

Scheduled ingestion jobs run on a cron schedule, submitting ingest jobs automatically.

### `GET /api/v2/scheduler/jobs`

List scheduled jobs for the current tenant.

**Response**
```json
[
  {
    "id":          "uuid",
    "name":        "nightly-hr-sync",
    "corpus_id":   "hr-policies",
    "cron_expr":   "0 2 * * *",
    "mode":        "incremental",
    "is_active":   true,
    "next_run_at": "2026-06-15T02:00:00Z",
    "last_run_at": "2026-06-14T02:00:00Z",
    "last_status": "completed"
  }
]
```

---

### `POST /api/v2/scheduler/jobs`

Create a new scheduled ingestion job.

**Request**
```json
{
  "name":                    "nightly-hr-sync",
  "source_type":             "local",
  "source_config":           { "path": "/data/hr" },
  "corpus_id":               "hr-policies",
  "cron_expr":               "0 2 * * *",
  "mode":                    "incremental",
  "enable_graph_extraction": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `source_type` | `"local"` \| `"url"` \| `"s3"` \| `"gcs"` | Source backend |
| `source_config` | object | Backend-specific config (e.g. `{"path": "..."}`, `{"bucket": "..."}`) |
| `cron_expr` | string | Standard 5-field cron (UTC) |

**Response:** same shape as list items above.

---

### `DELETE /api/v2/scheduler/jobs/{job_id}`

Cancel and remove a scheduled job.

**Response**
```json
{ "deleted": true }
```

---

### `POST /api/v2/scheduler/jobs/{job_id}/run-now`

Trigger an immediate one-off ingest run for a scheduled job outside its cron schedule.

**Response**
```json
{ "job_id": "uuid", "triggered": true }
```

---

## Evaluate

Offline evaluation — measures retrieval quality (Hit Rate, MRR, NDCG) against a gold dataset.

> **Status:** Phase 12 (in progress). `GET /run/{id}` and `GET /compare` return placeholder responses.

### `POST /api/v2/evaluate/run`

Trigger an evaluation run against the corpus's gold dataset.

**Request**
```json
{
  "corpus_id":       "hr-policies",
  "k":               5,
  "model_tier":      "small",
  "search_type":     "hybrid",
  "baseline_run_id": null
}
```

**Response**
```json
{ "run_id": "uuid", "corpus_id": "hr-policies", "status": "queued", "sample_count": 0 }
```

---

### `GET /api/v2/evaluate/run/{run_id}`

Poll eval run status and aggregated metrics.

---

### `GET /api/v2/evaluate/compare?a={run_id}&b={run_id}`

Regression diff between two eval runs.

---

## Feedback

### `POST /api/v2/feedback`

Submit explicit user feedback on a response.

**Request**
```json
{
  "request_id": "uuid",
  "thumbs":     true,
  "rating":     4,
  "correction": null,
  "tags":       ["accurate", "concise"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `thumbs` | bool \| null | Thumbs up/down |
| `rating` | int (1–5) \| null | Star rating |
| `correction` | string \| null | Free-text correction |
| `tags` | string[] | Free-form quality tags |

**Response**
```json
{ "received": true }
```

---

### `POST /api/v2/signals`

Submit an implicit behavioural signal. Intended for service-to-service calls from the frontend.

**Request**
```json
{
  "session_id":  "uuid",
  "signal_type": "copy_action",
  "request_id":  "uuid"
}
```

`signal_type` values: `query_reformulation` · `follow_up_question` · `session_abandoned` · `copy_action` · `escalation`

**Response**
```json
{ "received": true }
```

---

## Ops

### `GET /health`

Liveness + readiness check. No auth required.

**Response**
```json
{
  "status":         "healthy",
  "degraded_modes": [],
  "components": {
    "postgres": "healthy",
    "redis":    "healthy",
    "ollama":   "healthy"
  },
  "dlq_depth": 0
}
```

`status` values: `healthy` · `degraded` · `unhealthy`

`degraded_modes` values: `unavailable` (postgres down) · `no_cache` (Redis down) · `search_only` (Ollama down — chat disabled)

Returns `503` when `status` is not `healthy`.

---

### `GET /metrics`

Prometheus metrics in text format. No auth required. Scraped by Grafana.

```
# HELP rag_requests_total ...
# TYPE rag_requests_total counter
rag_requests_total{method="POST",route="/api/v2/chat",status="200"} 1234
...
```

---

## `GET /api/v2/logs`

Return recent log entries from the Redis ring buffer (`knowledge:logs:recent`, last 5,000 entries, 24h TTL). Admin role required.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `level` | string | `INFO` | Minimum log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `service` | string | null | Filter by service name |
| `corpus_id` | string | null | Filter by corpus |
| `request_id` | string | null | Filter by request ID |
| `limit` | int (1–500) | 100 | Max entries |

**Response**
```json
{
  "data": [
    {
      "timestamp":  "2026-06-14T12:00:01Z",
      "level":      "INFO",
      "service":    "api",
      "message":    "chat request completed",
      "request_id": "uuid",
      "corpus_id":  "hr-policies",
      "latency_ms": 843
    }
  ]
}
```
