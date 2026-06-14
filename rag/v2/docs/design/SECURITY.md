# RAG v2 — Security & Memory

## Table of Contents

- [Security Layer — JWT, JWE, HTTPS, RBAC](#security-layer--jwt-jwe-https-rbac)
  - [Authentication — JWT (RS256)](#authentication--jwt-rs256)
  - [Payload Encryption — JWE (A256GCM)](#payload-encryption--jwe-a256gcm)
  - [Transport — HTTPS / TLS](#transport--https--tls)
  - [RBAC](#rbac)
  - [Audit Log](#audit-log)
  - [Input Validation](#input-validation)
- [Memory Architecture](#memory-architecture)
  - [Five Memory Tiers](#five-memory-tiers)
  - [Critical design decision: server-side conversation history](#critical-design-decision-server-side-conversation-history)
  - [Module additions to knowledge/memory/](#module-additions-to-knowledgememory)
  - [Schema additions (migration 008_memory.sql)](#schema-additions-migration-008_memorysql)
  - [Token budget trim order (Tier 1)](#token-budget-trim-order-tier-1)
  - [Pruning, eviction, and compaction](#pruning-eviction-and-compaction)
  - [Framework: Mem0 for Tier 3 only](#framework-mem0-for-tier-3-only)
- [API Layer](#api-layer)

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

**Full design reference:** [`basics/rag/memory/MEMORY_DESIGN.md`](../../basics/rag/memory/MEMORY_DESIGN.md) — covers all five cognitive memory types, tsvector + pgvector hybrid search pattern for memory tables, Mem0/Zep/Letta framework assessment, and complete pruning/eviction/compaction algorithms. This section captures only the decisions that affect the module layout, database schema, and API surface of `knowledge/`.

The system uses five distinct memory tiers mapping to cognitive science memory types.

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
    content_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
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
CREATE INDEX ON messages USING GIN (content_tsv);  -- full-text search within conversation history

CREATE TABLE user_memories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL,         -- SHA-256(sub + tenant_salt)
    tenant_id           TEXT NOT NULL,
    content             TEXT NOT NULL,
    content_tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding           vector(768),
    source_message_id   UUID,
    last_retrieved_at   TIMESTAMPTZ,           -- for LRU eviction
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON user_memories (user_id, tenant_id);
CREATE INDEX ON user_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON user_memories USING GIN (content_tsv);   -- BM25 leg of hybrid search (same RRF pattern as chunks + entity_index)

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

