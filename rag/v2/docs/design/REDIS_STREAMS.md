# RAG v2 — Redis Streams & Async Workers

## Table of Contents

- [Active Streams & Consumer Groups](#active-streams--consumer-groups)
- [Worker Lifecycle](#worker-lifecycle)
- [Job Status API](#job-status-api)
- [Why Search Is on the Sync Path](#why-search-is-on-the-sync-path)
- [Why This Doesn't Become a Concurrency Bottleneck](#why-this-doesnt-become-a-concurrency-bottleneck)

---

## Active Streams & Consumer Groups

**Message bus** uses Redis Streams (`XADD` / `XREADGROUP`) rather than plain pub/sub — streams give persistent delivery, consumer groups, and dead-letter via `XPENDING`.

```
Streams (active):
  knowledge:ingest          # ingestion jobs         ← LIVE
  knowledge:eval            # eval run jobs          ← LIVE
  knowledge:events          # worker heartbeats, job completions  ← LIVE

Consumer groups:
  ingest-workers            # N replicas, each XREADGROUP from knowledge:ingest

Dead-letter:
  knowledge:ingest:dlq      # jobs that failed MAX_RETRIES times

Not yet wired (scaffolding only — publisher + worker exist but no route calls them):
  knowledge:search          # reserved for bulk/background search batches
  retrieval-workers         # consumer group defined but inactive
```

---

## Worker Lifecycle

Implemented in `knowledge/bus/consumer.py`:

1. `XREADGROUP GROUP <group> <worker_id> COUNT 1 BLOCK 5000 STREAMS <stream> >`
2. Deserialize message → `IngestJob` | `EvalJob`
3. Execute pipeline
4. `XACK` on success; increment retry counter + re-enqueue on transient failure
5. After `MAX_RETRIES` → move to DLQ, emit alert event
6. Heartbeat: `SET worker:<id>:heartbeat <ts> EX 30` every 10 s

---

## Job Status API

- `GET /api/v2/ingest/{job_id}/status` → polls `HGETALL job:{job_id}` (hash: status, progress, error, corpus_id)
- `GET /api/v2/ingest/{job_id}/stream` → SSE subscription to `knowledge:events` filtered by job_id

---

## Why Search Is on the Sync Path

`POST /api/v2/search` runs the retriever directly in the API process, not through a Redis Stream worker. This is an intentional architectural decision:

| Dimension | Ingestion (async) | Interactive search (sync) |
|-----------|-------------------|--------------------------|
| User-blocking? | No — fire-and-forget | Yes — user waits for results |
| Latency budget | Minutes (Docling + embed + graph) | < 600ms P95 |
| Compute profile | CPU-bound (Docling), GPU-bound (embed/LLM) | I/O-bound (DB queries + rerank) |
| Fan-out? | No — 1 doc → 1 pipeline | No — 1 query → 1 retrieval |
| Worker overhead | Worthwhile for heavy CPU work | ~1–5s pickup delay exceeds the entire SLA |

The retriever (`knowledge/retrieval/retriever.py`) runs entirely on asyncio I/O — pgvector queries, tsvector queries, optional AGE Cypher, CrossEncoder rerank via `asyncio.to_thread`. There is no reason to offload this to a worker; the API process handles it directly as an async coroutine within the request lifecycle.

The `knowledge:search` stream and `retrieval-workers` group exist for **bulk/background search batches only** — evaluation runs, pre-warming the semantic cache, batch re-ranking jobs. Interactive chat and search never touch that stream.

---

## Why This Doesn't Become a Concurrency Bottleneck

A common question: if many chat sessions run in parallel, doesn't the API process become a bottleneck handling all retrieval synchronously?

No — and here's why:

The retriever is pure async I/O. Every operation it performs (`asyncpg` queries, Redis lookups, CrossEncoder rerank via `asyncio.to_thread`) releases the event loop while waiting. This means a single Uvicorn worker can interleave hundreds of concurrent retrievals with zero blocking:

```
Session A: sends pgvector query → awaits result (event loop free)
Session B: sends tsvector query → awaits result (event loop free)
Session C: sends Redis lookup   → awaits result (event loop free)
Session A: result back → CrossEncoder → asyncio.to_thread (thread pool, non-blocking)
Session B: result back → RRF fusion → continues
...
```

The concurrency model is: Gunicorn launches multiple Uvicorn worker *processes* (typically `2 × CPU cores + 1`), and each worker runs a single-threaded asyncio event loop that handles many concurrent coroutines. The limiting resources are:

| Resource | Limit | How it's managed |
|----------|-------|-----------------|
| asyncpg connection pool | 10–20 connections per worker | `min_size`/`max_size` in pool config; requests queue if pool is exhausted |
| CrossEncoder thread pool | `asyncio` default thread pool (min 5 threads) | CPU-bound rerank runs in thread; other coroutines proceed in parallel |
| Redis connections | Pool per worker | Same pattern as asyncpg |
| Gunicorn workers | `2 × cores + 1` processes | OS-level parallelism for CPU saturation |

At the target load (5 req/s peak, ~10 in-flight), a single Uvicorn worker handles this comfortably. The asyncpg pool depth (10 connections) is the practical concurrency ceiling per worker process — if more than 10 queries are simultaneously in-flight, the 11th queues behind the pool. At 5 req/s this is never reached in practice; each retrieval completes in ~120ms P95, so average in-flight pool occupancy is `5 × 0.12 = 0.6 connections`.

**When would you need workers for retrieval?** Only if the retriever were CPU-bound (e.g. running a local embedding model synchronously) — that would block the event loop and starve other coroutines. Since embeddings go to Ollama over HTTP (async) and reranking uses `asyncio.to_thread`, neither blocks. The sync-path design holds up to the load model without change.
