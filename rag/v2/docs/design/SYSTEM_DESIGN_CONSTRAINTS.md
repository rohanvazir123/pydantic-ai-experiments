# RAG v2 — System Design Constraints

## Table of Contents

- [Load Model](#load-model)
- [Retrieval — SLA](#retrieval--sla)
- [Retrieval — Token Budget](#retrieval--token-budget)
- [Ingestion — SLA](#ingestion--sla)
- [Ingestion — Token Budget](#ingestion--token-budget)
- [Scalability — Corpus Size](#scalability--corpus-size)
  - [Why This Is a System-Wide Number, Not Per-Tenant](#why-this-is-a-system-wide-number-not-per-tenant)
  - [Storage & Memory by Scale](#storage--memory-by-scale)
  - [Ingestion Throughput by Scale](#ingestion-throughput-by-scale)
  - [Retrieval Latency by Scale](#retrieval-latency-by-scale)
- [Total System Cost Summary](#total-system-cost-summary-10-k-dau-standard-config)
- [Budget Controls & Cost Circuit Breakers](#budget-controls--cost-circuit-breakers)

---

Two load-bearing workloads with fundamentally different SLA profiles: **retrieval** (interactive, latency-sensitive, user-blocking) and **ingestion** (batch, throughput-sensitive, async and non-user-blocking). The constraints below are derived from the 10 K DAU target and drive every capacity and cost decision in the architecture.


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

#### Scalability — Corpus Size

Everything above (Load Model, SLAs, token budgets) is derived from the 10 K DAU / 100–500 docs-per-day *query and ingestion rate*. This section covers the orthogonal axis: what changes as the **total number of chunks already stored** grows, independent of query rate.

##### Why This Is a System-Wide Number, Not Per-Tenant

`chunks_embedding_hnsw` (`schema/001_initial_schema.sql`) is **one HNSW graph shared by every tenant and corpus** — there is no per-tenant or per-corpus index. Isolation is enforced at query time by Row-Level Security (`SET LOCAL app.tenant_id`, `schema/002_corpus_tenant.sql`) plus a `WHERE corpus_id = $2` predicate in `semantic_search()` (`knowledge/store/vector.py`), not by the index structure itself. Two consequences:

- The chunk-count tiers below are the **sum across every tenant and corpus in one Postgres instance**, not one customer's corpus. A platform with 1,000 tenants at 10K chunks each is already at the 10M row.
- HNSW graph traversal is not filter-aware — it finds nearest neighbours in the *whole* graph, then the `corpus_id`/RLS predicate discards non-matching rows. A small corpus living inside a large shared table can under-recall if too few of the true top-K neighbours happen to belong to that corpus. `OVERFETCH_FACTOR = 3` (`knowledge/store/vector.py`) — fetching `k × 3` candidates before RRF/rerank — is the current mitigation. As system-wide row count grows relative to any one corpus, re-validate recall per corpus with `tests/retrieval` (Hit Rate/MRR/NDCG) before assuming `OVERFETCH_FACTOR = 3` is still enough; raising it or the `hnsw.ef_search` runtime parameter are the levers, at a latency cost.
- True isolation (a dedicated index or dedicated Postgres instance per large tenant) is not implemented today. It's the escape hatch to reach for if one tenant's corpus alone approaches the 1M–10M rows below — declarative table partitioning by `tenant_id` with a per-partition HNSW index, or moving that tenant to its own instance.

##### Storage & Memory by Scale

Planning estimates, not measured benchmarks — validate with a load test before sizing production hardware. Baseline: `nomic-embed-text` (768-dim, 3,072 B/vector raw), HNSW at the schema default (`m=16, ef_construction=64`), ~65 chunks per 10-page document (`docs/design/SYSTEM_DESIGN_CONSTRAINTS.md` Ingestion Token Budget baseline).

| Chunks (system-wide) | ~Documents | Data + HNSW + GIN storage (~10 KB/chunk) | RAM to keep vectors + HNSW graph resident (~8 KB/chunk) | Postgres sizing (rule of thumb) |
|---|---|---|---|---|
| 10K | ~150 | ~100 MB | ~80 MB | Any managed instance, 4 GB RAM — dev / single small tenant |
| 100K | ~1,500 | ~1 GB | ~800 MB | 8 GB RAM (e.g. `db.t4g.medium`) |
| 1M | ~15,000 | ~10 GB | ~8 GB | 16 GB RAM (e.g. `db.r6g.large`) + read replica for retrieval-worker traffic |
| 10M | ~150,000 | ~100 GB | ~80 GB | 64 GB+ RAM (e.g. `db.r6g.2xlarge`), multiple read replicas, or partition per-tenant (see above) |

The RAM column is the one that matters for latency: pgvector HNSW search is fast only while the graph stays in `shared_buffers`/OS page cache. Once it spills to disk, each hop in the graph traversal becomes a random I/O, and search latency degrades sharply — this is a step-function failure, not a gradual one. Provision RAM ahead of the storage column, not behind it.

`m` and `ef_construction` are fixed at `16` / `64` across every HNSW index in the schema (`001`, `002`, `003`, `008`) regardless of table size. That default is adequate through roughly the 1M-row tier; beyond it, recall typically needs `ef_construction` raised (e.g. to 128) and query-time `hnsw.ef_search` raised for the 10M tier — at the cost of slower index builds and slightly higher per-query latency. Changing these is a recall/latency trade-off to validate against `tests/retrieval` metrics, not a settings change to make speculatively.

**HNSW rebuild cost also grows with scale.** `DATASTORE.md` already calls for `REINDEX INDEX CONCURRENTLY chunks_embedding_hnsw` after any operation deletes > 20% of a corpus. At the 10M-chunk tier, a full rebuild can run for hours; schedule it during low-traffic windows and expect degraded `pgvector_search` latency (the retrieval circuit breaker, `docs/RAGV2_DESIGN.md` §Circuit Breakers) for the duration.

##### Ingestion Throughput by Scale

The documented embedding baseline is 65 chunks in < 5 s P95 on one GPU worker (Ingestion — SLA, above) — roughly 13 chunks/s sustained, but the end-to-end pipeline (Docling parse + chunk + embed + optional graph extraction) is throughput-capped at the documented **100–500 docs/day** per worker, i.e. ~6,500–32,500 chunks/day. That rate is constant regardless of how many chunks already exist — corpus size doesn't slow down ingesting new documents. It does determine how long a cold bulk load takes:

| Target chunks | Time to reach it at 500 docs/day (32,500 chunks/day) sustained |
|---|---|
| 10K | < 1 day |
| 100K | ~3 days |
| 1M | ~31 days |
| 10M | ~308 days — steady-state throughput alone is not viable |

A one-time bulk backfill toward the 1M–10M tier needs horizontal scale-out of `ingest-worker`, not just time: scale toward its documented max (2–20 pods, HPA on `knowledge:ingest` stream depth — `docs/design/DEPLOYMENT.md` Scaling Rules) with multiple GPU-backed embedding workers running in parallel. Reaching 10M chunks in, say, a week requires roughly 40–50× the single-worker throughput above — plan the GPU fleet for the backfill window, then scale back down to the steady-state floor once caught up.

##### Retrieval Latency by Scale

Because HNSW search is `O(log n)`, the documented P95 hybrid-retrieval budget (< 120 ms, Retrieval — SLA above) should hold in principle across all four tiers — algorithmic complexity is not the bottleneck at these sizes. The two things that *do* degrade with scale are the RAM-residency cliff (Storage & Memory, above) and filtered-recall on a shared index disproportionately affecting small corpora inside a large system-wide table (Why This Is a System-Wide Number, above). Re-validate P95 retrieval latency and per-corpus recall empirically at each tier via `tests/retrieval` — this table is a planning starting point, not a guarantee.

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

