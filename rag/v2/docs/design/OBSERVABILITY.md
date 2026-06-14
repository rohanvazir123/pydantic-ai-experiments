# RAG v2 — Observability

## Table of Contents

- [Where to Find Logs](#where-to-find-logs)
  - [UI Log Viewer](#ui-log-viewer)
  - [Docker Logs](#docker-logs)
  - [Redis Ring Buffer](#redis-ring-buffer)
- [Structured Logging](#structured-logging)
- [Log Levels](#log-levels)
- [Prometheus Metrics](#prometheus-metrics)
- [Langfuse LLM Traces](#langfuse-llm-traces)
- [Grafana Dashboards](#grafana-dashboards)
- [Alerts](#alerts)

---

## Where to Find Logs

### UI Log Viewer

The fastest way during development. Navigate to **`/logs`** in the frontend UI (`http://localhost:3000/logs`).

- Toggle individual log levels on/off with the level chips (DEBUG · INFO · WARNING · ERROR · CRITICAL)
- Filter by `service`, `request_id`, or `job_id`
- Click any row to expand the full JSON transcript for that entry
- Set auto-refresh to 5s / 15s / 30s for live tailing
- Limit up to 2,000 entries

Reads from the Redis ring buffer `knowledge:logs:recent` (last 5,000 entries, 24h TTL) via `GET /api/v2/logs`.

> The UI log viewer shows in-memory recent logs only. For historical logs beyond the 24h buffer, use Docker logs.

---

### Docker Logs

Full log output from the API and worker containers:

```bash
# All services — live tail
docker compose logs -f

# API only
docker compose logs -f api

# Ingest worker only
docker compose logs -f ingest-worker

# Last 500 lines from API
docker compose logs --tail=500 api

# Filter for errors
docker compose logs api 2>&1 | grep '"level":"ERROR"'
```

Logs are emitted as structured JSON (`structlog`) to stdout. Every line is a valid JSON object.

---

### Redis Ring Buffer

Query the raw ring buffer directly with `redis-cli`:

```bash
# Connect to Redis
docker compose exec redis redis-cli

# See last 10 log entries (newest first)
LRANGE knowledge:logs:recent 0 9

# Count entries in buffer
LLEN knowledge:logs:recent

# Clear the buffer
DEL knowledge:logs:recent
```

Or from Python:

```python
import asyncio, json, redis.asyncio as aioredis

async def tail_logs(n: int = 20) -> None:
    r = aioredis.from_url("redis://localhost:6379")
    entries = await r.lrange("knowledge:logs:recent", 0, n - 1)
    for raw in entries:
        log = json.loads(raw)
        print(f"[{log.get('level','?')}] {log.get('timestamp','')} {log.get('message','')}")
    await r.aclose()

asyncio.run(tail_logs())
```

---

## Structured Logging

All log output uses `structlog` with JSON rendering. Every log entry is a JSON object on stdout.

Key fields present on every entry:

| Field | Description |
|-------|-------------|
| `level` | `DEBUG` · `INFO` · `WARNING` · `ERROR` · `CRITICAL` |
| `timestamp` | ISO 8601 UTC |
| `message` | Human-readable log message |
| `service` | `api` · `ingest-worker` · `retrieval-worker` |
| `request_id` | UUID — correlates all logs for one HTTP request |
| `tenant_id` | Tenant scope (from JWT) |
| `user_id` | SHA-256 prefix of the user sub (never raw PII) |

Additional fields appear per context:

| Field | When present |
|-------|-------------|
| `route` | HTTP route path |
| `status` | HTTP response status code |
| `latency_ms` | Request wall-clock time |
| `corpus_id` | Retrieval / ingestion context |
| `job_id` | Ingestion worker context |
| `stage` | Pipeline stage name |
| `duration_ms` | Stage-level latency |
| `pipeline_status` | `answered` · `abstained_retrieval` · etc. |
| `chunk_count` | Chunks ingested |

---

## Log Levels

| Level | When used | Default on? |
|-------|-----------|-------------|
| `DEBUG` | Fine-grained trace: SQL queries, cache lookups, embedding calls | Off in prod |
| `INFO` | Normal operation: request handled, job completed, cache hit | On |
| `WARNING` | Recoverable issues: retry triggered, degraded mode, low confidence | On |
| `ERROR` | Failures: DB error, LLM timeout, DLQ promotion | On |
| `CRITICAL` | System-wide breach: budget exceeded, circuit open | On |

Set the minimum log level via environment variable:

```bash
# .env
LOG_LEVEL=DEBUG    # show everything (dev)
LOG_LEVEL=INFO     # default
LOG_LEVEL=WARNING  # production minimum
```

The UI log viewer toggles levels client-side — no API restart needed. To suppress DEBUG logs from the ring buffer entirely, set `LOG_LEVEL=INFO` in the API container.

---

## Prometheus Metrics

Exposed at `GET /metrics` (text/plain, Prometheus format). Scraped by Grafana.

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `rag_requests_total{route, status}` | Counter | HTTP requests by route and status code |
| `rag_request_latency_seconds{route}` | Histogram | P50/P95/P99 per route |
| `rag_pipeline_status_total{status}` | Counter | `answered` / `abstained_*` counts |
| `rag_cache_hits_total{layer}` | Counter | L1/L2/L3 cache hits |
| `rag_cache_misses_total{layer}` | Counter | L1/L2/L3 cache misses |
| `rag_ingest_chunks_total{corpus_id}` | Counter | Chunks ingested per corpus |
| `rag_dlq_depth{stream}` | Gauge | Dead-letter queue depth |
| `rag_worker_heartbeat_age_seconds` | Gauge | Seconds since last worker heartbeat |
| `cost_budget_utilization{tenant_id}` | Gauge | 0.0–1.0 monthly budget used |
| `cost_circuit_breaker_triggered_total` | Counter | Budget circuit breaker fires |

Start the observability stack:

```bash
cd rag/v2
docker compose --profile observability up -d
# Grafana:    http://localhost:3001  (admin / admin)
# Prometheus: http://localhost:9090
```

---

## Langfuse LLM Traces

Every LLM call (agent, judge, model router, content policy) is traced via Langfuse using the `@observe` decorator in `knowledge/observability/langfuse.py`.

Each trace captures:
- Input prompt + output completion
- Token counts (prompt / completion)
- Latency per LLM call
- Model tier used
- `request_id` for correlation back to API logs

Access Langfuse at `http://localhost:3002` when running the observability profile. The trace URL is included in `RAGResponse.trace_url` so the UI DebugPanel can link directly to the trace.

---

## Grafana Dashboards

Pre-built dashboards in `infra/grafana/dashboards/`:

| Dashboard | What it shows |
|-----------|--------------|
| `api-overview.json` | Request rate, error rate, P95 latency per route |
| `pipeline.json` | Answered vs abstained breakdown, judge gate pass rate |
| `cache.json` | L1/L2/L3 hit rates, semantic cache similarity distribution |
| `ingestion.json` | Chunks/s, DLQ depth, worker heartbeat age |
| `cost.json` | Per-tenant budget utilization, cost circuit breaker fires |

---

## Alerts

SMTP alerts fire on:

| Condition | Trigger |
|-----------|---------|
| `chat_latency_p95 > 3s` sustained 5 min | PagerDuty + email |
| `search_latency_p99 > 1.5s` | Email |
| `streaming_ttft_p95 > 1000ms` | Email |
| `l3_cache_hit_rate < 15%` | Email |
| `dlq_depth > 0` sustained | Immediate email |
| `cost_circuit_breaker_triggered` | Immediate email |
| `worker_heartbeat_age > 60s` | Email (worker dead) |

Alert destination configured in `.env`:

```bash
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=alerts@yourdomain.com
ALERT_SMTP_PASSWORD=...
ALERT_TO_EMAIL=oncall@yourdomain.com
```
