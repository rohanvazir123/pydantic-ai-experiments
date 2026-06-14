"""Prometheus metrics and Redis log ring buffer processor.

Metrics are exported via GET /metrics (Prometheus text format).
The RedisLogProcessor mirrors every structlog entry to a capped Redis list
so the GET /v1/logs endpoint can serve recent logs on-demand.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
# Lazily imported so tests don't require prometheus_client installed

def _counter(name: str, doc: str, labels: list[str]) -> Any:
    try:
        from prometheus_client import Counter
        return Counter(name, doc, labels)
    except ImportError:
        return None

def _histogram(name: str, doc: str, labels: list[str], buckets: list[float] | None = None) -> Any:
    try:
        from prometheus_client import Histogram
        if buckets is not None:
            return Histogram(name, doc, labels, buckets=buckets)
        return Histogram(name, doc, labels)
    except ImportError:
        return None

def _gauge(name: str, doc: str, labels: list[str] | None = None) -> Any:
    try:
        from prometheus_client import Gauge
        return Gauge(name, doc, labels or [])
    except ImportError:
        return None


# Cache layer counters
cache_l1_hits       = _counter("cache_l1_hits_total",    "L1 embedding cache hits",    ["operation"])
cache_l2_hits       = _counter("cache_l2_hits_total",    "L2 Redis cache hits",         ["layer"])
cache_l2_misses     = _counter("cache_l2_misses_total",  "L2 Redis cache misses",       ["layer"])
cache_l3_hits       = _counter("cache_l3_hits_total",    "L3 semantic cache hits",      [])
cache_l3_misses     = _counter("cache_l3_misses_total",  "L3 semantic cache misses",    [])

# Request latency
request_latency     = _histogram(
    "request_latency_seconds", "End-to-end request latency", ["route", "status"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Retrieval stage latency
retrieval_latency   = _histogram(
    "retrieval_latency_seconds", "Retrieval stage latency", ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0],
)

# Token and cost counters
llm_tokens          = _counter("llm_tokens_total",      "LLM token usage",       ["tier", "model", "type"])
llm_cost_usd        = _counter("llm_cost_usd_total",    "LLM cost in USD",       ["tier", "model"])

# Model tier selection
model_tier_selected = _counter("model_tier_selected_total", "Model tier routing", ["tier"])

# Pipeline status (answered / abstained_*)
pipeline_status     = _counter("pipeline_status_total", "Pipeline outcome",       ["status"])

# Circuit breaker
cb_state            = _gauge("circuit_breaker_state",   "Circuit breaker state (0=CLOSED,1=OPEN,2=HALF-OPEN)", ["service"])

# DLQ depth
dlq_depth           = _gauge("dlq_depth",               "Dead letter queue depth", ["stream"])


def inc_cache_l2_hit(layer: str) -> None:
    if cache_l2_hits:
        cache_l2_hits.labels(layer=layer).inc()

def inc_cache_l2_miss(layer: str) -> None:
    if cache_l2_misses:
        cache_l2_misses.labels(layer=layer).inc()

def inc_cache_l3_hit() -> None:
    if cache_l3_hits:
        cache_l3_hits.inc()

def inc_cache_l3_miss() -> None:
    if cache_l3_misses:
        cache_l3_misses.inc()

def observe_request(route: str, status: int, latency_s: float) -> None:
    if request_latency:
        request_latency.labels(route=route, status=str(status)).observe(latency_s)

def observe_retrieval(stage: str, latency_s: float) -> None:
    if retrieval_latency:
        retrieval_latency.labels(stage=stage).observe(latency_s)

def inc_tokens(tier: str, model: str, prompt: int, completion: int) -> None:
    if llm_tokens:
        llm_tokens.labels(tier=tier, model=model, type="prompt").inc(prompt)
        llm_tokens.labels(tier=tier, model=model, type="completion").inc(completion)

def inc_cost(tier: str, model: str, cost: float) -> None:
    if llm_cost_usd:
        llm_cost_usd.labels(tier=tier, model=model).inc(cost)

def inc_pipeline_status(status: str) -> None:
    if pipeline_status:
        pipeline_status.labels(status=status).inc()


# ── Redis log ring buffer ─────────────────────────────────────────────────────

_LOG_KEY   = "knowledge:logs:recent"
_MAX_LINES = 5_000
_TTL_S     = 86_400  # 24h


class RedisLogHandler(logging.Handler):
    """stdlib logging.Handler that mirrors every log record to the Redis ring buffer.

    Attaches to the root logger so all logging.getLogger() calls in the app
    are captured, regardless of whether structlog is used.
    """

    def __init__(self, redis: Any) -> None:
        super().__init__()
        self._redis = redis

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            event_dict: dict[str, Any] = {
                "level":     record.levelname,
                "timestamp": ts,
                "message":   record.getMessage(),
                "service":   record.name,
            }
            if record.exc_info:
                event_dict["exc_info"] = self.formatException(record.exc_info)
            entry = json.dumps(event_dict, default=str)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._write(entry))
            except RuntimeError:
                pass  # no event loop — skip (e.g. during sync startup code)
        except Exception:
            pass  # never let logging break the request

    async def _write(self, entry: str) -> None:
        try:
            await self._redis.lpush(_LOG_KEY, entry)
            await self._redis.ltrim(_LOG_KEY, 0, _MAX_LINES - 1)
            await self._redis.expire(_LOG_KEY, _TTL_S)
        except Exception:
            pass


def configure_structlog(redis: Any) -> None:
    """Attach a Redis log handler to the root logger.

    All logging.getLogger() calls (uvicorn, fastapi, knowledge.*) will be
    captured and written to the Redis ring buffer at knowledge:logs:recent,
    enabling GET /api/v2/logs to serve them on-demand.

    Safe to call multiple times — duplicate handlers are skipped.
    """
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RedisLogHandler):
            return  # already installed
    handler = RedisLogHandler(redis)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # Ensure root logger passes INFO and above to our handler
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
