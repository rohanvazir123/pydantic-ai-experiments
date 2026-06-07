# knowledge/bus/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Redis Streams](#redis-streams)
- [Reliability Guarantees](#reliability-guarantees)

---

## What This Is

The async message bus built on Redis Streams. Decouples the API (which publishes jobs) from the workers (which consume and process them). Provides persistent delivery, consumer groups, dead-letter queuing, exponential backoff, and circuit breakers.

---

## Files

| File | Purpose |
|------|---------|
| `publisher.py` | `XADD` helpers: `publish_ingest_job()`, `publish_eval_job()` |
| `consumer.py` | Base `consume_loop()`: `XREADGROUP` → handler → ack/retry/DLQ |
| `circuit_breaker.py` | `CircuitBreaker`: CLOSED/OPEN/HALF-OPEN state machine in Redis |
| `backoff.py` | `exponential_backoff()`: base=5s, max=125s, 15% jitter |
| `schemas.py` | `IngestJob`, `SearchRequest`, `WorkerEvent`, `EvalJob` Pydantic models |

---

## Redis Streams

| Stream | Consumer group | Purpose |
|--------|---------------|---------|
| `knowledge:ingest` | `ingest-workers` | Ingestion jobs |
| `knowledge:search` | `retrieval-workers` | Async search batches |
| `knowledge:eval` | `eval-workers` | Offline evaluation runs |
| `knowledge:events` | Stream (polled, no consumer group) | Worker heartbeats; API SSE endpoints filter by job_id |
| `knowledge:ingest:dlq` | — | Dead-letter: jobs that failed MAX_RETRIES times |

---

## Reliability Guarantees

- **At-least-once delivery** via `XREADGROUP` + explicit `XACK` on success
- **Retry with backoff**: up to 3 attempts (5s → 10s → 20s ± jitter) on transient errors
- **Dead-letter on permanent failure**: job moves to DLQ, alert email sent, never silently lost
- **Circuit breaker per external service**: opens after 5 failures in 60s; state shared in Redis across all API pods
- **Idempotent jobs**: SHA-256 fingerprint cache prevents re-ingesting unchanged documents
