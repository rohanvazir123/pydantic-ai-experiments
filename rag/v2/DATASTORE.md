# Datastore Reference

Complete reference for all data stores used by the RAG v2 knowledge system:
two PostgreSQL databases, Redis, and the Apache AGE graph database.

## Table of Contents

1. [Overview](#1-overview)
2. [Entity Diagram](#2-entity-diagram)
3. [Main PostgreSQL Database](#3-main-postgresql-database)
   - 3.1 [Connection](#31-connection)
   - 3.2 [Table: `documents`](#32-table-documents)
   - 3.3 [Table: `chunks`](#33-table-chunks)
   - 3.4 [Table: `kg_entity_index`](#34-table-kg_entity_index)
   - 3.5 [Table: `semantic_cache`](#35-table-semantic_cache)
   - 3.6 [Table: `audit_events`](#36-table-audit_events)
   - 3.7 [Table: `conversations`](#37-table-conversations)
   - 3.8 [Table: `messages`](#38-table-messages)
   - 3.9 [Table: `user_memories`](#39-table-user_memories)
   - 3.10 [Table: `system_prompts`](#310-table-system_prompts)
   - 3.11 [Table: `gold_samples`](#311-table-gold_samples)
   - 3.12 [Table: `eval_runs`](#312-table-eval_runs)
   - 3.13 [Table: `eval_results`](#313-table-eval_results)
   - 3.14 [Table: `user_feedback`](#314-table-user_feedback)
   - 3.15 [Table: `implicit_signals`](#315-table-implicit_signals)
   - 3.16 [Table: `token_usage`](#316-table-token_usage)
   - 3.17 [Table: `tenants`](#317-table-tenants)
   - 3.18 [Table: `tenant_quotas`](#318-table-tenant_quotas)
   - 3.19 [Table: `billing_events`](#319-table-billing_events)
   - 3.20 [Table: `scheduled_jobs`](#320-table-scheduled_jobs)
   - 3.21 [Indexes](#321-indexes)
   - 3.22 [Row-Level Security](#322-row-level-security)
4. [Apache AGE Graph Database](#4-apache-age-graph-database)
   - 4.1 [Connection](#41-connection)
   - 4.2 [Graph naming](#42-graph-naming)
   - 4.3 [Vertex and edge structure](#43-vertex-and-edge-structure)
   - 4.4 [Cypher wrapper pattern](#44-cypher-wrapper-pattern)
5. [Redis](#5-redis)
   - 5.1 [Connection](#51-connection)
   - 5.2 [Key patterns](#52-key-patterns)
   - 5.3 [Streams and consumer groups](#53-streams-and-consumer-groups)
6. [Key SQL Queries](#6-key-sql-queries)

---

## 1. Overview

| Store | Type | Purpose | Managed by |
|-------|------|---------|-----------|
| PostgreSQL `documents` | RDBMS table | Full document metadata and content | `store/vector.py` |
| PostgreSQL `chunks` | RDBMS table | Embedded text chunks (pgvector HNSW + tsvector GIN) | `store/vector.py` |
| PostgreSQL `kg_entity_index` | RDBMS table | AGE entity shadow table for fast BM25+cosine entity search | `store/entity_index.py` |
| PostgreSQL `semantic_cache` | RDBMS table | L3 JWE-encrypted answer cache (pgvector cosine lookup) | `retrieval/semantic_cache.py` |
| PostgreSQL `audit_events` | RDBMS table | Append-only compliance audit log | `api/middleware.py` |
| PostgreSQL `conversations` | RDBMS table | Tier 2 episodic memory — conversation threads | `memory/conversation_store.py` |
| PostgreSQL `messages` | RDBMS table | Tier 2 episodic memory — individual turns (tsvector GIN) | `memory/conversation_store.py` |
| PostgreSQL `user_memories` | RDBMS table | Tier 3 semantic user memory (tsvector GIN + pgvector HNSW) | `memory/mem0_store.py` |
| PostgreSQL `system_prompts` | RDBMS table | Tier 5 procedural memory — versioned system prompt store | `api/app.py` |
| PostgreSQL `gold_samples` | RDBMS table | Evaluation gold dataset | `evaluation/datasets.py` |
| PostgreSQL `eval_runs` | RDBMS table | Evaluation run metadata and regression report | `evaluation/runner.py` |
| PostgreSQL `eval_results` | RDBMS table | Per-sample evaluation results with confidence fields | `evaluation/runner.py` |
| PostgreSQL `user_feedback` | RDBMS table | Explicit thumbs up/down feedback | `api/routes/feedback.py` |
| PostgreSQL `implicit_signals` | RDBMS table | Behavioural signals (reformulation, copy, escalation) | `api/routes/feedback.py` |
| PostgreSQL `token_usage` | RDBMS table | Per-LLM-call token counts (financial records, 7yr retention) | `agent/cost_guard.py` |
| PostgreSQL `tenants` | RDBMS table | Tenant registry | `billing/provisioner.py` |
| PostgreSQL `tenant_quotas` | RDBMS table | Per-tenant rate limits and LLM budget | `api/quota.py` |
| PostgreSQL `billing_events` | RDBMS table | LLM call billing events for Stripe metering | `billing/metering.py` |
| PostgreSQL `scheduled_jobs` | RDBMS table | Periodic ingestion job configuration | `scheduler/job_store.py` |
| Apache AGE graphs | Graph DB | Per-corpus knowledge graphs (openCypher, port 5433) | `store/graph.py` |
| Redis `knowledge:ingest` | Stream | Ingestion job queue | `bus/publisher.py` |
| Redis `knowledge:search` | Stream | Async search batch queue | `bus/publisher.py` |
| Redis `knowledge:eval` | Stream | Evaluation job queue | `bus/publisher.py` |
| Redis `knowledge:events` | Stream | Worker lifecycle events | workers |
| Redis `knowledge:*:dlq` | Stream | Dead-letter queues | `bus/consumer.py` |
| Redis `cache:embed:*` | String | L2 embedding cache (msgpack, 24h TTL) | `store/cache.py` |
| Redis `cache:search:*` | String | L2 search result cache (msgpack, 5min TTL) | `store/cache.py` |
| Redis `cache:doc_fingerprint:*` | String | Document SHA-256 fingerprint for incremental ingest | `store/cache.py` |
| Redis `quota:*` | String | Per-tenant RPM and daily counters | `api/quota.py` |
| Redis `cb:*` | String | Circuit breaker state (CLOSED/OPEN/HALF-OPEN) | `bus/circuit_breaker.py` |
| Redis `job:*` | Hash | Ingest job status hash (status, progress, error) | `ingestion/pipeline.py` |
| Redis `knowledge:logs:recent` | List | LPUSH ring buffer — last 5,000 log entries (24h TTL) | `observability/metrics.py` |

---

## 2. Entity Diagram

```
┌──────────────────────────────────┐
│            tenants               │
├──────────────────────────────────┤
│ id           TEXT        PK      │
│ display_name TEXT        NN      │
│ tier         TEXT        NN      │
│ admin_email  TEXT        NN      │
│ data_region  TEXT        NN      │
└──────────────┬───────────────────┘
               │ 1
               │ has one
               ▼
┌──────────────────────────────────┐
│          tenant_quotas           │
├──────────────────────────────────┤
│ tenant_id  TEXT  PK→tenants.id   │
│ max_qpd    INT   NN              │
│ max_rpm    INT   NN              │
│ llm_enabled BOOL NN             │
└──────────────────────────────────┘

          (tenant_id + corpus_id scope all content)

┌──────────────────────────────────┐
│            documents             │
├──────────────────────────────────┤
│ id         UUID  PK              │
│ title      TEXT  NN              │
│ source     TEXT  NN UNIQUE       │
│ corpus_id  TEXT  NN  ←──────────── per-corpus partition
│ tenant_id  TEXT  NN  ←──────────── RLS enforcement
│ metadata   JSONB NN              │
└──────────────┬───────────────────┘
               │ 1
               │ ON DELETE CASCADE
               │ N
               ▼
┌──────────────────────────────────┐
│              chunks              │
├──────────────────────────────────┤
│ id          UUID  PK             │
│ document_id UUID  FK→documents   │
│ content     TEXT  NN             │
│ content_tsv TSVECTOR (generated) │ ← GIN index (BM25)
│ embedding   vector(768)          │ ← HNSW index (cosine ANN)
│ chunk_index INT   NN             │
│ token_count INT                  │
│ corpus_id   TEXT  NN             │
│ tenant_id   TEXT  NN             │
│ metadata    JSONB NN             │
└──────────────────────────────────┘

┌──────────────────────────────────┐     ┌──────────────────────────────┐
│          conversations           │     │        user_memories         │
├──────────────────────────────────┤     ├──────────────────────────────┤
│ id          UUID  PK             │     │ id         UUID  PK          │
│ session_id  TEXT  UNIQUE         │     │ user_id    TEXT  NN (hashed) │
│ tenant_id   TEXT  NN             │     │ tenant_id  TEXT  NN          │
│ user_id     TEXT  NN (hashed)    │     │ content    TEXT  NN          │
│ corpus_ids  TEXT[]               │     │ content_tsv TSVECTOR (gen.)  │ ← GIN
│ summary     TEXT  (auto at 20t)  │     │ embedding  vector(768)       │ ← HNSW
│ turn_count  INT                  │     │ last_retrieved_at TIMESTAMPTZ│
└──────────────┬───────────────────┘     └──────────────────────────────┘
               │ ON DELETE CASCADE
               ▼
┌──────────────────────────────────┐
│             messages             │
├──────────────────────────────────┤
│ id              UUID  PK         │
│ conversation_id UUID  FK         │
│ role            TEXT  NN         │ ← 'user' | 'assistant'
│ content         TEXT  NN         │
│ content_tsv     TSVECTOR (gen.)  │ ← GIN index (conversation search)
│ citations       JSONB            │
│ pipeline_status TEXT             │
│ cost_usd        FLOAT            │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│           eval_runs              │
├──────────────────────────────────┤
│ id           UUID  PK            │
│ corpus_id    TEXT  NN            │
│ status       TEXT  NN            │
│ report_json  JSONB               │ ← regression diff
└──────────────┬───────────────────┘
               │ ON DELETE CASCADE
               ▼
┌──────────────────────────────────┐
│           eval_results           │
├──────────────────────────────────┤
│ id           UUID  PK            │
│ run_id       UUID  FK            │
│ sample_id    UUID  FK            │
│ hit_rate     FLOAT               │
│ mrr          FLOAT               │
│ faithfulness FLOAT               │
│ pipeline_status TEXT             │ ← answered|abstained_*
│ confidence   FLOAT               │
└──────────────────────────────────┘
```

---

## 3. Main PostgreSQL Database

### 3.1 Connection

| Parameter | Value |
|-----------|-------|
| Driver | `asyncpg` |
| DSN env var | `DATABASE_URL` |
| DSN format | `postgresql://user:pass@host:5432/dbname` |
| Extensions | `uuid-ossp`, `vector` (pgvector) |
| Schema | `public` (default) |
| RLS | Enabled on `documents`, `chunks`, `audit_events` |
| Pool init | `register_vector(conn)` on every new connection |

Set `app.tenant_id` before every query:
```sql
SET LOCAL app.tenant_id = 'acme-corp';
```
PostgreSQL RLS then enforces tenant isolation automatically.

---

### 3.2 Table: `documents`

Primary record for each ingested document.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `title` | `TEXT` | NOT NULL | — | Document title (extracted or filename stem) |
| `source` | `TEXT` | NOT NULL | — | File path or URL — must be unique per corpus |
| `content` | `TEXT` | NULL | — | Raw document content (stored for re-chunking) |
| `corpus_id` | `TEXT` | NOT NULL | — | Corpus namespace (e.g. `acme:hr-policies`) |
| `tenant_id` | `TEXT` | NOT NULL | — | Tenant namespace — enforced by RLS |
| `metadata` | `JSONB` | NOT NULL | `'{}'` | File hash, ingestion date, YAML frontmatter, word count |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Ingestion timestamp |

**Upsert behaviour**: `INSERT … ON CONFLICT (source) DO UPDATE` — re-ingesting a changed document overwrites in place.

---

### 3.3 Table: `chunks`

One row per embedded chunk. The primary search table.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `document_id` | `UUID` | NOT NULL | — | FK → `documents.id` ON DELETE CASCADE |
| `content` | `TEXT` | NOT NULL | — | Contextualized chunk text (heading hierarchy prepended) |
| `content_tsv` | `TSVECTOR` | GENERATED | — | `to_tsvector('english', content)` — BM25 search |
| `embedding` | `vector(768)` | NULL | — | nomic-embed-text embedding — cosine ANN search |
| `chunk_index` | `INTEGER` | NOT NULL | — | Position within the document (0-based) |
| `token_count` | `INTEGER` | NULL | — | Token count via HF tokenizer |
| `corpus_id` | `TEXT` | NOT NULL | — | Denormalised from document for fast corpus-scoped search |
| `tenant_id` | `TEXT` | NOT NULL | — | Enforced by RLS |
| `metadata` | `JSONB` | NOT NULL | `'{}'` | chunk_method, has_context, graph_extraction_failed, … |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |

**Search**: hybrid BM25 (`content_tsv @@ websearch_to_tsquery`) + cosine ANN (`embedding <=> $vec`) fused with RRF (k=60).

---

### 3.4 Table: `kg_entity_index`

Shadow table that mirrors Apache AGE vertices into the main PostgreSQL DB for fast hybrid search. AGE has no GIN or HNSW index support — every CONTAINS scan in AGE is O(n). This table provides sub-millisecond entity search.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `age_uuid` | `TEXT` | NOT NULL | — | PK — mirrors `uuid` property on the AGE vertex |
| `name` | `TEXT` | NOT NULL | — | Entity name |
| `name_tsv` | `TSVECTOR` | GENERATED | — | `to_tsvector('english', name)` — BM25 name search |
| `embedding` | `vector(768)` | NULL | — | Embedded entity name — cosine ANN search |
| `label` | `TEXT` | NOT NULL | — | Vertex label / entity type (from ontology) |
| `corpus_id` | `TEXT` | NOT NULL | — | Corpus namespace |
| `tenant_id` | `TEXT` | NOT NULL | — | Tenant namespace |
| `document_id` | `TEXT` | NOT NULL | `''` | Source document |

**Write path**: `entity_index.upsert()` is called after each vertex upsert in AGE. Embedding is generated by the same embedder used for chunk search.

---

### 3.5 Table: `semantic_cache`

L3 semantic cache. Stores JWE-encrypted LLM answers indexed by query embedding for cosine similarity lookup.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `corpus_ids` | `TEXT[]` | NOT NULL | — | Corpora the answer was generated for |
| `tenant_id` | `TEXT` | NOT NULL | — | Tenant namespace |
| `query_text` | `TEXT` | NOT NULL | — | Original query (stored for debugging) |
| `query_emb` | `vector(768)` | NOT NULL | — | Query embedding — HNSW cosine lookup |
| `answer_jwe` | `TEXT` | NOT NULL | — | JWE-encrypted `RAGResponse` JSON blob |
| `hit_count` | `INTEGER` | NOT NULL | `0` | Number of times this cache entry was served |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |
| `expires_at` | `TIMESTAMPTZ` | NOT NULL | — | TTL (default: 60 minutes from creation) |

**Lookup**: `SELECT … WHERE corpus_ids = $1 AND expires_at > NOW() ORDER BY query_emb <=> $2 LIMIT 1` — returns cache hit if cosine similarity ≥ `semantic_cache_threshold` (default 0.95).

**Pruning**: When row count exceeds `semantic_cache_max_rows` (default 10,000), delete the oldest 10%.

---

### 3.6 Table: `audit_events`

Append-only compliance audit log. Never `UPDATE` or `DELETE`.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `ts` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Event timestamp |
| `user_id` | `TEXT` | NOT NULL | — | `SHA-256(sub + tenant_salt)` — never plaintext |
| `tenant_id` | `TEXT` | NOT NULL | — | Tenant namespace |
| `action` | `TEXT` | NOT NULL | — | `'search'`, `'ingest'`, `'delete'`, `'cache_invalidate'`, … |
| `corpus_id` | `TEXT` | NULL | — | Corpus involved (NULL for tenant-level actions) |
| `query_text` | `TEXT` | NULL | — | `SHA-256(query)` — never plaintext |
| `request_id` | `UUID` | NOT NULL | — | Correlation ID — links to logs and Langfuse trace |
| `ip_address` | `INET` | NULL | — | Client IP |
| `response_ms` | `INTEGER` | NULL | — | Request latency |

**Retention**: 2 years (pruned by nightly job). Right-to-erasure replaces `user_id` with `SHA-256("ERASED" + tenant_salt)` — the row is preserved for compliance.

---

### 3.7 Table: `conversations`

Tier 2 episodic memory — one row per conversation thread.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `session_id` | `TEXT` | NOT NULL | — | UNIQUE — `crypto.randomUUID()` from frontend |
| `tenant_id` | `TEXT` | NOT NULL | — | — |
| `user_id` | `TEXT` | NOT NULL | — | `SHA-256(sub + tenant_salt)` |
| `corpus_ids` | `TEXT[]` | NOT NULL | — | Corpora this conversation covers |
| `title` | `TEXT` | NULL | — | First 60 chars of first user message (set on first turn) |
| `summary` | `TEXT` | NULL | — | Auto-generated by nano model when `turn_count > 20` |
| `turn_count` | `INTEGER` | NOT NULL | `0` | Incremented after every assistant turn |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |
| `last_turn_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Updated on every turn |
| `expires_at` | `TIMESTAMPTZ` | NULL | — | NULL = never; set by tenant retention policy |
| `deleted_at` | `TIMESTAMPTZ` | NULL | — | Soft delete; hard delete after 7-day grace |

**Active window**: `load_active_window(session_id)` returns last 8 messages (or `[summary_message] + last 8` when `summary IS NOT NULL`).

---

### 3.8 Table: `messages`

Tier 2 episodic memory — one row per turn.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `conversation_id` | `UUID` | NOT NULL | — | FK → `conversations.id` ON DELETE CASCADE |
| `role` | `TEXT` | NOT NULL | — | `'user'` or `'assistant'` CHECK constraint |
| `content` | `TEXT` | NOT NULL | — | Message text |
| `content_tsv` | `TSVECTOR` | GENERATED | — | `to_tsvector('english', content)` — GIN for conversation search |
| `citations` | `JSONB` | NULL | — | `list[Citation]` serialized (assistant turns only) |
| `pipeline_status` | `TEXT` | NULL | — | `'answered'` / `'abstained_retrieval'` / … (assistant only) |
| `confidence` | `FLOAT` | NULL | — | Judge confidence (assistant only) |
| `model_tier` | `TEXT` | NULL | — | `'nano'` / `'small'` / `'large'` (assistant only) |
| `prompt_tokens` | `INTEGER` | NULL | — | — |
| `completion_tokens` | `INTEGER` | NULL | — | — |
| `cost_usd` | `FLOAT` | NULL | — | — |
| `cache_hit` | `TEXT` | NULL | — | `'l2'` / `'l3'` / NULL |
| `request_id` | `UUID` | NULL | — | Links to logs and Langfuse trace |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |

**Note**: No `embedding` column — messages are scoped to user+conversation; BM25 alone is sufficient for conversation search.

---

### 3.9 Table: `user_memories`

Tier 3 semantic user memory. Hybrid tsvector + pgvector search — same RRF pattern as `kg_entity_index`.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `user_id` | `TEXT` | NOT NULL | — | `SHA-256(sub + tenant_salt)` |
| `tenant_id` | `TEXT` | NOT NULL | — | — |
| `content` | `TEXT` | NOT NULL | — | Extracted fact (e.g. "User is a senior engineer at a fintech company") |
| `content_tsv` | `TSVECTOR` | GENERATED | — | `to_tsvector('english', content)` — BM25 leg of hybrid search |
| `embedding` | `vector(768)` | NULL | — | Cosine ANN leg of hybrid search |
| `source_message_id` | `UUID` | NULL | — | Message that triggered extraction (for audit) |
| `last_retrieved_at` | `TIMESTAMPTZ` | NULL | — | Updated on every search hit — used for LRU eviction |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Updated when Mem0 resolves a contradiction in place |

**Capacity limit**: hard cap 200 memories per user. When exceeded, delete where `last_retrieved_at < NOW() - 60 days AND created_at < NOW() - 90 days LIMIT 20`.

---

### 3.10 Table: `system_prompts`

Tier 5 procedural memory — versioned system prompt store.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK |
| `name` | `TEXT` | NOT NULL | — | Prompt identifier (e.g. `'rag_agent_v3'`) |
| `content` | `TEXT` | NOT NULL | — | Full system prompt text |
| `version` | `INTEGER` | NOT NULL | `1` | Version number — incremented on every change |
| `active` | `BOOLEAN` | NOT NULL | `FALSE` | Only one version per name may be `active = TRUE` |
| `corpus_id` | `TEXT` | NULL | — | NULL = global; set for corpus-specific overrides |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |
| `created_by` | `TEXT` | NOT NULL | `'system'` | Admin user who set it |

---

### 3.11 Table: `gold_samples`

Evaluation gold dataset. Version-controlled as JSONL in `evaluation/data/` and mirrored here.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Stable ID (never change after creation) |
| `corpus_id` | `TEXT` | NOT NULL | — | Corpus this sample belongs to |
| `query` | `TEXT` | NOT NULL | — | Natural-language query exactly as a user would type it |
| `relevant_doc_sources` | `TEXT[]` | NOT NULL | — | Filename stems of relevant documents (substring match) |
| `ground_truth_answer` | `TEXT` | NULL | — | Known correct answer (optional — needed for correctness metrics) |
| `difficulty` | `TEXT` | NOT NULL | `'medium'` | `'easy'` / `'medium'` / `'hard'` CHECK constraint |
| `tags` | `TEXT[]` | NOT NULL | `'{}'` | `'factual'` / `'multi-hop'` / `'aggregation'` / `'temporal'` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | — |

---

### 3.12 Table: `eval_runs`

One row per triggered evaluation run.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Surrogate PK — returned to caller as `run_id` |
| `corpus_id` | `TEXT` | NOT NULL | — | Corpus evaluated |
| `git_commit` | `TEXT` | NOT NULL | — | Git SHA at eval time — reproducibility anchor |
| `model_tier` | `TEXT` | NOT NULL | — | `'small'` / `'large'` |
| `search_type` | `TEXT` | NOT NULL | — | `'hybrid'` / `'semantic'` / `'text'` |
| `k` | `INTEGER` | NOT NULL | `5` | Top-K for retrieval metrics |
| `started_at` | `TIMESTAMPTZ` | NOT NULL | — | — |
| `completed_at` | `TIMESTAMPTZ` | NULL | — | NULL while running |
| `status` | `TEXT` | NOT NULL | `'queued'` | `'queued'` / `'running'` / `'completed'` / `'failed'` |
| `sample_count` | `INTEGER` | NOT NULL | `0` | Number of samples processed |
| `baseline_run_id` | `UUID` | NULL | — | FK → `eval_runs.id` — run to compare against for regression |
| `report_json` | `JSONB` | NULL | — | Per-metric deltas and regression flags written by `reporter.py` |

---

### 3.13 Table: `eval_results`

Per-sample results. Join with `eval_runs` for context.

Key columns (see `migrations/004_evaluation.sql` for full DDL):

| Group | Columns |
|-------|---------|
| Retrieval | `hit_rate`, `mrr`, `ndcg`, `precision_at_k`, `recall_at_k` |
| Generation | `faithfulness`, `answer_relevance` |
| Correctness | `bleu_4`, `rouge_1_f`, `rouge_2_f`, `rouge_l_f`, `meteor`, `bert_score_f`, `semantic_similarity` |
| Performance | `retrieval_ms`, `llm_first_token_ms`, `generation_ms`, `total_ms`, `prompt_tokens`, `completion_tokens`, `estimated_cost_usd`, `cache_tier_hit` |
| Confidence | `mean_confidence`, `min_confidence`, `low_confidence_flag` |
| Pipeline | `pipeline_status`, `abstention_layer`, `retrieval_aggregate_confidence`, `citation_trustworthy`, `judge_verdict`, `judge_confidence`, `false_abstention` |

---

### 3.14–3.20 Other Tables

See `migrations/005_feedback.sql` through `migrations/007_scheduler.sql` for full DDL of:
`user_feedback`, `implicit_signals`, `token_usage`, `tenants`, `tenant_quotas`, `billing_events`, `scheduled_jobs`.

---

### 3.21 Indexes

| Index | Table | Column(s) | Type | Purpose |
|-------|-------|-----------|------|---------|
| `chunks_content_tsv_gin` | `chunks` | `content_tsv` | GIN | BM25 full-text search |
| `chunks_embedding_hnsw` | `chunks` | `embedding` | HNSW (m=16, ef=64) | Cosine ANN search |
| `chunks_corpus_id_idx` | `chunks` | `corpus_id` | B-tree | Corpus-scoped queries |
| `kg_entity_tsv_gin` | `kg_entity_index` | `name_tsv` | GIN | BM25 entity name search |
| `kg_entity_embedding_hnsw` | `kg_entity_index` | `embedding` | HNSW | Cosine ANN entity search |
| `semantic_cache_embedding_hnsw` | `semantic_cache` | `query_emb` | HNSW | L3 cache lookup |
| `messages_content_tsv_gin` | `messages` | `content_tsv` | GIN | Conversation full-text search |
| `user_memories_tsv_gin` | `user_memories` | `content_tsv` | GIN | BM25 memory search |
| `user_memories_embedding_hnsw` | `user_memories` | `embedding` | HNSW | Cosine ANN memory search |
| `conversations_user_ts` | `conversations` | `(user_id, last_turn_at DESC)` | B-tree | List conversations for user |
| `scheduled_jobs_next_run_idx` | `scheduled_jobs` | `next_run_at` WHERE `is_active` | Partial B-tree | Efficient due-job polling |

**HNSW rebuild**: after deleting > 20% of a corpus, run `REINDEX INDEX CONCURRENTLY chunks_embedding_hnsw` to prevent index degradation.

---

### 3.22 Row-Level Security

RLS is enabled on `documents`, `chunks`, and `audit_events`. The API sets the tenant context before every query:

```sql
SET LOCAL app.tenant_id = 'acme-corp';
-- PostgreSQL enforces: tenant_id = current_setting('app.tenant_id', true)
```

Even if application code forgets a `WHERE tenant_id = $1` clause, the policy blocks cross-tenant reads. The second argument `true` to `current_setting` makes it return NULL (not raise) when the setting is not set — this allows superuser connections (migrations, admin tools) to bypass RLS gracefully.

---

## 4. Apache AGE Graph Database

### 4.1 Connection

| Parameter | Value |
|-----------|-------|
| Driver | `asyncpg` |
| DSN env var | `AGE_DATABASE_URL` |
| Default port | `5433` (separate container from main PostgreSQL) |
| Image | `apache/age:latest` |

Every connection requires two setup statements before any Cypher. Registered as asyncpg `pool init` callback and re-applied on every `pool.acquire()` (AGE state is reset by `RESET ALL` on connection return):

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
```

### 4.2 Graph naming

Each corpus gets its own AGE graph:

```
{age_graph_prefix}_{tenant_id}_{corpus_id}
e.g. kg_acme_corp_hr_policies
```

Created at first ingest: `SELECT create_graph('kg_acme_corp_hr_policies')`. Already-existing graphs are silently skipped.

### 4.3 Vertex and edge structure

Vertices are created per entity type from the user's ontology template. Each vertex carries these properties regardless of type:

| Property | Type | Description |
|----------|------|-------------|
| `uuid` | TEXT | Stable ID, `COALESCE(existing, new_uuid4)` — preserved on re-ingest |
| `nx_id` | TEXT | NetworkX node ID from docling-graph `PipelineContext` |
| `name` | TEXT | Entity name or best identifier |
| `label` | TEXT | Vertex label (entity type from ontology, e.g. `Person`, `Policy`) |
| `corpus_id` | TEXT | Corpus namespace — all queries filter by this |
| `document_id` | TEXT | Source document UUID |

Edges carry `document_id`, `corpus_id`, and the relationship type (e.g. `HAS_MEMBER`, `APPLIES_TO`) from the ontology template's `edge(label="...")` annotations.

### 4.4 Cypher wrapper pattern

All Cypher statements are wrapped in the AGE SQL function:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (n:Person {corpus_id: "acme:hr"})
    RETURN n.name, n.uuid
    LIMIT 20
$$) AS (name agtype, uuid agtype);
```

`agtype` columns are returned as strings by asyncpg. Strip surrounding `"` before use:
```python
value = row["name"].strip('"')  # '"Acme Corp"' → 'Acme Corp'
```

**Important**: `CypherExporter` from docling-graph generates Neo4j-compatible raw Cypher `CREATE` statements that are **not compatible** with the AGE SQL wrapper syntax. The ingestion pipeline iterates `PipelineContext.knowledge_graph` (a NetworkX DiGraph) directly and calls `age_store._upsert_vertex()` and `age_store._add_edge()` — it does not use `CypherExporter`.

---

## 5. Redis

### 5.1 Connection

| Parameter | Value |
|-----------|-------|
| Client | `redis.asyncio` |
| DSN env var | `REDIS_URL` |
| Default | `redis://localhost:6379` |
| Max connections | `REDIS_MAX_CONNECTIONS` (default 20) |
| Serialization | `msgpack` (binary, faster than JSON for vectors) |

### 5.2 Key patterns

| Pattern | Data structure | TTL | Purpose |
|---------|---------------|-----|---------|
| `cache:embed:{sha256(text)}` | String (msgpack) | 24h | L2 embedding dedup — avoids round-trips to Ollama |
| `cache:search:{sha256(query+corpus+filters)}` | String (msgpack) | 5min | L2 search result cache — identical query short-circuit |
| `cache:doc_fingerprint:{sha256(file_content)}` | String `"1"` | 7d | Skip re-ingesting unchanged files |
| `cache:health:{service}` | String (JSON) | 30s | Avoid DB health checks on every probe |
| `quota:{tenant_id}:queries:{YYYY-MM-DD}` | String (counter) | 25h | Daily query count |
| `quota:{tenant_id}:rpm:{minute_bucket}` | String (counter) | 2min | Sliding RPM window |
| `quota:{tenant_id}:cost_usd:{YYYY-MM}` | String (float) | monthly | Monthly LLM spend |
| `cb:{service}:state` | String | — | Circuit breaker state: `CLOSED`/`OPEN`/`HALF-OPEN` |
| `cb:{service}:failures` | String (counter) | 60s | Failure count in current window |
| `cb:{service}:opened_at` | String (timestamp) | — | When circuit opened (for probe timer) |
| `job:{job_id}` | Hash | 48h | Ingest job status: status, progress, error, corpus_id |
| `worker:{id}:heartbeat` | String | 30s | Worker liveness (set every 10s) |
| `knowledge:logs:recent` | List (LPUSH + LTRIM 0 4999) | 24h | Log ring buffer for `/v1/logs` endpoint |
| `rt:{jti}` | String (user_id) | 7d | Refresh token server-side store |

### 5.3 Streams and consumer groups

| Stream | Consumer group | Published by | Consumed by |
|--------|---------------|-------------|-------------|
| `knowledge:ingest` | `ingest-workers` | `bus/publisher.py` | `ingestion/worker.py` (N replicas) |
| `knowledge:search` | `retrieval-workers` | `bus/publisher.py` | `retrieval/worker.py` (M replicas) |
| `knowledge:eval` | `eval-workers` | `bus/publisher.py` | `evaluation/runner.py` |
| `knowledge:events` | — (pub/sub style) | workers | API SSE streams, monitoring |
| `knowledge:ingest:dlq` | — | `bus/consumer.py` | human review |
| `knowledge:search:dlq` | — | `bus/consumer.py` | human review |

**Retry policy**: 3 attempts, exponential backoff (5s → 10s → 20s ± 15% jitter). After 3 failures, job moves to DLQ and an alert email is sent to `rohan.vazirani@gmail.com`.

---

## 6. Key SQL Queries

### Hybrid chunk search (RRF k=60) for a corpus
```sql
WITH
text_ranked AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', $1)) DESC
    ) AS rn
    FROM chunks
    WHERE content_tsv @@ websearch_to_tsquery('english', $1)
      AND corpus_id = $2 AND tenant_id = $3
),
vec_ranked AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY embedding <=> $4::vector ASC
    ) AS rn
    FROM chunks
    WHERE embedding IS NOT NULL
      AND corpus_id = $2 AND tenant_id = $3
    LIMIT 60
),
rrf AS (
    SELECT COALESCE(t.id, v.id) AS id,
           COALESCE(1.0/(60+t.rn), 0) + COALESCE(1.0/(60+v.rn), 0) AS score
    FROM text_ranked t
    FULL OUTER JOIN vec_ranked v ON t.id = v.id
)
SELECT c.id, c.content, c.metadata, r.score
FROM rrf r JOIN chunks c ON c.id = r.id
ORDER BY r.score DESC LIMIT $5;
```

### L3 semantic cache lookup
```sql
SELECT id, answer_jwe, 1 - (query_emb <=> $1) AS sim
FROM semantic_cache
WHERE corpus_ids = $2
  AND tenant_id = $3
  AND expires_at > NOW()
ORDER BY query_emb <=> $1
LIMIT 1;
-- Return cache hit if sim >= semantic_cache_threshold (default 0.95)
```

### User memories hybrid search
```sql
WITH
text_ranked AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', $1)) DESC
    ) AS rn
    FROM user_memories
    WHERE content_tsv @@ websearch_to_tsquery('english', $1)
      AND user_id = $3 AND tenant_id = $4
),
vec_ranked AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY embedding <=> $2::vector ASC
    ) AS rn
    FROM user_memories
    WHERE embedding IS NOT NULL
      AND user_id = $3 AND tenant_id = $4
    LIMIT 20
),
rrf AS (
    SELECT COALESCE(t.id, v.id) AS id,
           COALESCE(1.0/(60+t.rn), 0) + COALESCE(1.0/(60+v.rn), 0) AS score
    FROM text_ranked t FULL OUTER JOIN vec_ranked v ON t.id = v.id
)
SELECT m.content, r.score
FROM rrf r JOIN user_memories m ON m.id = r.id
ORDER BY r.score DESC LIMIT $5;
```

### Retrieve due scheduled jobs
```sql
SELECT * FROM scheduled_jobs
WHERE next_run_at <= NOW()
  AND is_active = TRUE
  AND tenant_id = $1
ORDER BY next_run_at ASC;
```

### Active conversations for a user (newest first)
```sql
SELECT id, session_id, title, summary, turn_count, last_turn_at
FROM conversations
WHERE user_id = $1 AND tenant_id = $2
  AND deleted_at IS NULL
ORDER BY last_turn_at DESC
LIMIT $3;
```

### Evaluation regression check
```sql
SELECT
    er.hit_rate      - br.hit_rate      AS hit_rate_delta,
    er.mrr           - br.mrr           AS mrr_delta,
    er.faithfulness  - br.faithfulness  AS faithfulness_delta,
    er.total_ms      - br.total_ms      AS latency_delta_ms
FROM eval_runs cur
JOIN eval_runs base ON base.id = cur.baseline_run_id
JOIN LATERAL (
    SELECT AVG(hit_rate) AS hit_rate, AVG(mrr) AS mrr,
           AVG(faithfulness) AS faithfulness, AVG(total_ms) AS total_ms
    FROM eval_results WHERE run_id = cur.id
) er ON TRUE
JOIN LATERAL (
    SELECT AVG(hit_rate) AS hit_rate, AVG(mrr) AS mrr,
           AVG(faithfulness) AS faithfulness, AVG(total_ms) AS total_ms
    FROM eval_results WHERE run_id = base.id
) br ON TRUE
WHERE cur.id = $1;
```

### Tenant LLM budget utilisation
```sql
SELECT
    t.id AS tenant_id,
    t.display_name,
    COALESCE(SUM(be.cost_usd), 0) AS spent_usd,
    tq.llm_budget_usd_per_month AS budget_usd,
    ROUND(COALESCE(SUM(be.cost_usd), 0) / NULLIF(tq.llm_budget_usd_per_month, 0) * 100, 1) AS pct_used
FROM tenants t
JOIN tenant_quotas tq ON tq.tenant_id = t.id
LEFT JOIN billing_events be
    ON be.tenant_id = t.id
    AND DATE_TRUNC('month', be.timestamp) = DATE_TRUNC('month', NOW())
WHERE t.deleted_at IS NULL
GROUP BY t.id, t.display_name, tq.llm_budget_usd_per_month
ORDER BY pct_used DESC NULLS LAST;
```
