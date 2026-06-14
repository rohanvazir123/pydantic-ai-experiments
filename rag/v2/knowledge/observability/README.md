# knowledge/observability/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Log Storage](#log-storage)

---

## What This Is

Prometheus metrics, Langfuse LLM tracing, structured logging, and SMTP alert sending. All of these are opt-in — everything defaults to disabled so the system starts cleanly without external services.

---

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Prometheus counters/histograms; `RedisLogProcessor` structlog processor (mirrors logs to Redis ring buffer for `/v1/logs` endpoint) |
| `langfuse.py` | Langfuse `@observe` decorator; `langfuse_context.update_current_observation()` for token counts from Pydantic AI `result.usage()` |
| `alerts.py` | `send_alert(severity, code, detail)`: async SMTP send via `aiosmtplib`; fallback to `logs/alerts.jsonl` when SMTP unreachable |

---

## Log Storage

Logs go to **stdout** (Docker captures per container). For local debugging:

```bash
docker compose logs -f api
docker compose logs -f ingest-worker
# Filter for errors only:
docker compose logs -f 2>&1 | grep '"level":"ERROR"'
```

The `RedisLogProcessor` also mirrors every structlog entry to a Redis ring buffer (`knowledge:logs:recent`, last 5,000 entries, 24h TTL). This powers the `GET /v1/logs` admin endpoint — no file I/O required.

LLM traces go to Langfuse (self-hosted via `docker compose --profile observability up`). Set `LANGFUSE_ENABLED=true` to activate. Token counts are read from Pydantic AI's built-in `result.usage()` — no manual token counting.
