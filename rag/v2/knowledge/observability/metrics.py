"""Prometheus metrics and Redis log ring buffer processor.

Metrics are exported via GET /metrics (Prometheus text format).
The RedisLogProcessor mirrors every structlog entry to a capped Redis list
so the GET /v1/logs endpoint can serve recent logs on-demand.
"""

import json
import logging
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


# ── Redis log ring buffer processor ──────────────────────────────────────────

class RedisLogProcessor:
    """structlog processor that mirrors each log entry to a Redis ring buffer.

    Enables the GET /v1/logs endpoint to serve recent logs on-demand
    without reading Docker stdout.
    """

    LOG_KEY   = "knowledge:logs:recent"
    MAX_LINES = 5_000
    TTL_S     = 86_400   # 24h

    def __init__(self, redis: Any) -> None:   # redis.asyncio.Redis
        self._redis = redis

    async def process(self, event_dict: dict[str, Any]) -> dict[str, Any]:
        try:
            entry = json.dumps(event_dict, default=str)
            await self._redis.lpush(self.LOG_KEY, entry)
            await self._redis.ltrim(self.LOG_KEY, 0, self.MAX_LINES - 1)
            await self._redis.expire(self.LOG_KEY, self.TTL_S)
        except Exception:
            pass   # never let log processing break the request
        return event_dict
