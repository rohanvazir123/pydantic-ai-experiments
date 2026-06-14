"""Per-tenant rate limiting and budget enforcement.

Redis counters are the fast-path guard; PostgreSQL token_usage is the audit trail.
Quota headers are added to every response even when limits are not exceeded.

Key patterns:
  daily counter:  quota:{tenant_id}:queries:{YYYY-MM-DD}   INCR / expire 25h
  RPM counter:    quota:{tenant_id}:rpm:{minute_bucket}     INCR / expire 2min
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class QuotaHeaders:
    """Headers to add to every API response."""
    rate_limit:         int
    rate_remaining:     int
    rate_reset:         int        # Unix timestamp when the minute window resets
    daily_limit:        int
    daily_used:         int


class QuotaExceeded(Exception):
    def __init__(self, code: str, limit: int, retry_after_s: int | None = None) -> None:
        self.code          = code
        self.limit         = limit
        self.retry_after_s = retry_after_s
        super().__init__(code)


async def enforce_quota(
    tenant_id:     str,
    redis:         aioredis.Redis,
    max_per_day:   int,
    max_per_minute: int,
    request_type:  str = "search",
    llm_enabled:   bool = True,
) -> QuotaHeaders:
    """Check and increment quota counters. Raises QuotaExceeded on breach.

    Called at PRE_VALIDATE — before any DB or LLM work.

    Args:
        tenant_id:      tenant identifier.
        redis:          shared Redis client.
        max_per_day:    daily query limit (0 = unlimited).
        max_per_minute: RPM limit (0 = unlimited).
        request_type:   "chat" | "search" | "ingest".
        llm_enabled:    if False, block chat requests (free tier).
    """
    today        = datetime.now(UTC).strftime("%Y-%m-%d")
    minute_key   = f"quota:{tenant_id}:rpm:{int(time.time() // 60)}"
    daily_key    = f"quota:{tenant_id}:queries:{today}"
    next_minute  = int((time.time() // 60 + 1) * 60)

    # Free tier: block LLM calls
    if not llm_enabled and request_type == "chat":
        raise QuotaExceeded("LLM_NOT_ENABLED_ON_FREE_TIER", limit=0)

    # Increment both counters atomically
    pipe = redis.pipeline()
    pipe.incr(daily_key)
    pipe.expire(daily_key, 90_000)    # 25h buffer
    pipe.incr(minute_key)
    pipe.expire(minute_key, 120)      # 2min sliding window
    daily_count, _, rpm_count, _ = await pipe.execute()

    daily_count = int(daily_count)
    rpm_count   = int(rpm_count)

    if max_per_day > 0 and daily_count > max_per_day:
        raise QuotaExceeded("DAILY_QUOTA_EXCEEDED", limit=max_per_day)

    if max_per_minute > 0 and rpm_count > max_per_minute:
        raise QuotaExceeded(
            "RATE_LIMIT_EXCEEDED", limit=max_per_minute, retry_after_s=60
        )

    return QuotaHeaders(
        rate_limit=max_per_minute,
        rate_remaining=max(0, max_per_minute - rpm_count),
        rate_reset=next_minute,
        daily_limit=max_per_day,
        daily_used=daily_count,
    )
