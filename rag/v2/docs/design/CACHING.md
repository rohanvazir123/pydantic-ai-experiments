# RAG v2 — Caching

## Table of Contents

- [Caching Architecture](#caching-architecture)
  - [L1 — In-Process LRU (per worker)](#l1--in-process-lru-per-worker)
  - [L2 — Redis Distributed Cache](#l2--redis-distributed-cache)
  - [L3 — Semantic Query Cache (pgvector)](#l3--semantic-query-cache-pgvector)
  - [Cache Observability](#cache-observability)
- [Caching Reference](#caching-reference)
  - [Layer summary](#layer-summary)
  - [L1 — In-Process Embedding Cache](#l1--in-process-embedding-cache)
  - [L2 — Redis Cache](#l2--redis-cache)
    - [Embedding cache](#embedding-cache)
    - [Search result cache](#search-result-cache)
    - [Document fingerprint cache](#document-fingerprint-cache)
    - [Health check cache](#health-check-cache)
  - [L3 — Semantic Query Cache (pgvector)](#l3--semantic-query-cache-pgvector-1)
  - [Invalidation decision guide](#invalidation-decision-guide)
  - [Cache hit rate targets](#cache-hit-rate-targets)

---

### Caching Architecture

Three independent cache layers. Each layer has a distinct TTL and eviction strategy.
See [Caching Reference](#caching-reference) below for the complete TTL table, invalidation triggers, and decision guide.

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

### Caching Reference

Complete cache layer guide: what is cached, TTLs, invalidation triggers, and decision rules.

#### Layer summary

| Layer | Where | Scope | Max entries | What is cached |
|-------|-------|-------|-------------|----------------|
| **L1 — in-process** | Worker RAM (Python dict) | Per worker process | 1,000 (FIFO eviction) | Embeddings for repeated texts |
| **L2 — Redis** | Redis strings (msgpack) | Shared across all workers and API pods | Unlimited (TTL-bounded) | Embeddings, search results, doc fingerprints, health checks |
| **L3 — pgvector** | PostgreSQL `semantic_cache` | Shared, per tenant+corpus | 10,000 rows (pruned) | Full JWE-encrypted RAGResponse for near-duplicate queries |

#### L1 — In-Process Embedding Cache

**Implementation:** bounded dict in `knowledge/ingestion/embedder.py`; FIFO eviction at 1,000 entries.

**Key:** `text` (the raw string to embed)  
**Value:** `list[float]` — the embedding vector  
**TTL:** process lifetime (no expiry — evicted by FIFO when full)  
**Scope:** single worker process; not shared across pods

**Hit condition:** `text in self._cache`  
**Miss action:** call `AsyncOpenAI.embeddings.create()` → cache result → return  
**Eviction:** when `len(cache) >= 1000`, delete the oldest inserted key (FIFO)

**Why FIFO not LRU:** embedding cache entries are large (768 floats ≈ 6KB each). A proper LRU requires ordered access tracking. FIFO is sufficient because repeated texts during a batch ingest session are sequential — by the time a text would be evicted by FIFO, it's genuinely stale. True LRU adds overhead with no measurable benefit for this workload.

**Invalidation:** none — process restart clears it. Safe because L2 Redis is the durable layer.

---

#### L2 — Redis Cache

All L2 keys use `msgpack` serialisation (faster than JSON for binary float arrays). All keys are namespaced by type prefix.

##### Embedding cache

| Field | Value |
|-------|-------|
| Key | `cache:embed:{sha256(text)}` |
| Value | msgpack-encoded `list[float]` |
| TTL | **24 hours** |
| Set | After every L1 miss that calls the embedding API |
| Hit | Return deserialised vector; skip API call |
| Invalidation | Never explicitly invalidated — TTL expiry only |
| Rationale | 24h matches typical ingest batch duration. Vectors for unchanged text never change — safe to cache indefinitely within the TTL. Cross-worker benefit: pod A ingests a doc, pod B can reuse the embedding. |

##### Search result cache

| Field | Value |
|-------|-------|
| Key | `cache:search:{sha256(query + sorted_corpus_ids + filters)}` |
| Value | msgpack-encoded `list[dict]` (serialised SearchResult list) |
| TTL | **5 minutes** |
| Set | After every retrieval pipeline run that reaches L3 miss, using `SET NX` (won't overwrite concurrent write) |
| Hit | Deserialise and return; skip retrieval, reranking, and LLM call |
| Invalidation triggers | • `POST /v1/corpus/{id}/cache/invalidate` — admin endpoint scans and deletes all `cache:search:*` keys (corpus-scoped, not full flush) • New document ingested to the corpus — `POST_INGEST` hook calls `cache.invalidate_corpus()` |
| Rationale | 5min balances staleness risk vs cost saving. A corpus being actively ingested benefits from invalidation on every job completion. |

##### Document fingerprint cache

| Field | Value |
|-------|-------|
| Key | `cache:doc_fingerprint:{sha256(file_content_bytes)}` |
| Value | `"1"` (existence flag) |
| TTL | **7 days** |
| Set | After a document is successfully ingested |
| Hit | Skip the document entirely in incremental mode — no DB read, no Docling, no chunking |
| Invalidation triggers | • `cache.delete_fingerprint(sha256)` called when a document is deleted • `make purge-corpus` runs `purge.py` which scans and deletes all fingerprint keys |
| Rationale | 7d covers typical re-ingest cycles. Content-hash keyed (not filename): renaming a file does not bypass the cache; changing a file's content creates a new cache miss. |

##### Health check cache

| Field | Value |
|-------|-------|
| Key | `cache:health:{service_name}` |
| Value | JSON health status |
| TTL | **30 seconds** |
| Hit | Return cached status; skip DB/Redis/Ollama probe |
| Invalidation | TTL-only |
| Rationale | Health checks fire on every liveness probe. 30s prevents cascading DB queries from Kubernetes health checkers. |

---

#### L3 — Semantic Query Cache (pgvector)

**Why L3 exists:** two queries asking the same thing in different words (paraphrases) produce the same answer. The L2 search cache only hits on byte-identical query strings. L3 catches near-duplicates by cosine similarity of the query embedding.

| Field | Value |
|-------|-------|
| Table | `semantic_cache` |
| Key | query embedding vector (HNSW indexed for cosine ANN lookup) |
| Value | JWE-encrypted `RAGResponse` JSON blob (`answer_jwe`) |
| TTL | **60 minutes** (configurable: `semantic_cache_ttl_minutes`) |
| Similarity threshold | **0.95 cosine similarity** (configurable: `semantic_cache_threshold`) |
| Max rows | **10,000** (configurable: `semantic_cache_max_rows`) |
| Scope | Per `tenant_id` + `corpus_ids` array |

**Hit flow:**
```
1. Embed incoming query (L1/L2 hit likely)
2. SELECT ... ORDER BY query_emb <=> $vec LIMIT 1
3. if sim >= 0.95 AND expires_at > NOW():
     decrypt JWE answer → increment hit_count → return (no LLM call)
4. else: miss → full pipeline → INSERT into semantic_cache
```

**Why JWE-encrypted:** cached answers may contain corpus-specific information. Storing plaintext allows any process with DB access to read answers without going through the auth/RBAC layer. JWE ensures the answer is opaque in storage.

**Pruning:** when `COUNT(*) > semantic_cache_max_rows`, delete the oldest 10% by `created_at`. Triggered async after every `store()` call — never blocks the response path.

**Invalidation triggers:**
- `POST /v1/corpus/{id}/cache/invalidate` — deletes all rows where `corpus_ids @> ARRAY['corpus_id']`
- New document ingested — `POST_INGEST` hook triggers L2 + L3 invalidation for the corpus
- Row TTL expiry — `expires_at < NOW()` rows are ignored at lookup; a nightly job prunes them

**Threshold guidance:** 0.95 is strict by design — a wrong cached answer is worse than a cache miss. Tune down to 0.92 per corpus once you have measured confidence distributions in the eval system.

---

#### Invalidation decision guide

Use this when deciding what to invalidate after a data change:

```
Document added/changed/deleted in corpus X
  └── Invalidate L2 search cache:  cache.invalidate_corpus(corpus_id, tenant_id)
      (scans cache:search:* and deletes — O(n) but infrequent)
  └── Invalidate L3 semantic cache: DELETE FROM semantic_cache WHERE corpus_ids @> ARRAY[$1]
  └── Set doc fingerprint:          cache.set_fingerprint(sha256(new_content))  ← for new content
  └── Delete doc fingerprint:       cache.delete_fingerprint(sha256(old_content)) ← for deleted

Model / embedding model changed
  └── Flush ALL L2 embed keys:  SCAN + DELETE cache:embed:*  (make purge-corpus)
  └── Flush ALL L3 entries:     TRUNCATE semantic_cache
  └── Flush ALL L2 search keys: SCAN + DELETE cache:search:*
  (Embedding vectors are model-specific — wrong model → wrong similarity scores)

Corpus deleted (tenant offboarding)
  └── DELETE FROM semantic_cache WHERE tenant_id = $1
  └── SCAN + DELETE all quota:tenant:*, cache:search:* for tenant
  └── Invalidation is handled by billing/provisioner.py delete_tenant()

Corpus cache invalidate endpoint (admin action)
  └── POST /v1/corpus/{id}/cache/invalidate
      calls cache.invalidate_corpus() → L2 search flush + L3 DELETE
```

#### Cache hit rate targets

| Layer | Target hit rate | Alert threshold |
|-------|----------------|-----------------|
| L1 embedding | ≥ 60% during batch ingest | — (process-local, no monitoring) |
| L2 embedding | ≥ 30% across workers | — (L1 usually absorbs most) |
| L2 search | ≥ 10% | `l2_cache_hit_rate < 5%` |
| L3 semantic | ≥ 25% | `l3_cache_hit_rate < 15%` |

A cache hit rate below alert threshold usually means: corpus was recently invalidated (expected), new traffic pattern (expected), or threshold too strict (tune down).

---

