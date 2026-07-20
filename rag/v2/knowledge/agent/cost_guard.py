# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Cost circuit breaker — fires at PRE_LLM hook before every agent.run() call.

Two levels:
  TenantBudgetExceeded — tenant's monthly LLM budget exhausted → 402
  SystemBudgetExceeded — system-wide daily cap breached → 503

Budget is tracked in Redis (fast-path guard). PostgreSQL token_usage table
is the source of truth for billing — Redis is the enforcement gate.

0.0 for system_daily_cost_limit_usd means disabled (no cap).
0.0 for llm_budget_usd_per_month means unlimited (enterprise prepaid).
"""

import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


class TenantBudgetExceeded(Exception):
    def __init__(self, tenant_id: str, spent: float, limit: float) -> None:
        self.tenant_id = tenant_id
        self.spent     = spent
        self.limit     = limit
        super().__init__(
            f"Tenant '{tenant_id}' LLM budget exhausted: ${spent:.4f} / ${limit:.2f}"
        )


class SystemBudgetExceeded(Exception):
    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"System daily LLM budget exhausted: ${spent:.4f} / ${limit:.2f}")


async def check_cost_circuit_breaker(
    tenant_id: str,
    redis: aioredis.Redis,
    tenant_limit: float,
    settings: Settings | None = None,
) -> None:
    """Raise TenantBudgetExceeded or SystemBudgetExceeded if any budget is exhausted.

    Called at PRE_LLM hook — before every agent.run() / agent.run_stream() call.
    Zero cost is incurred on cache hits (this function is not called on cache hits).

    Args:
        tenant_id:    tenant identifier.
        redis:        shared Redis client.
        tenant_limit: monthly LLM budget in USD (0.0 = unlimited).
        settings:     settings instance (uses load_settings() if None).
    """
    _settings = settings or load_settings()
    month = datetime.now(UTC).strftime("%Y-%m")

    # ── Tenant budget check ───────────────────────────────────────────────────
    if tenant_limit > 0:
        monthly_key  = f"quota:{tenant_id}:cost_usd:{month}"
        raw          = await redis.get(monthly_key)
        monthly_cost = float(raw) if raw else 0.0

        if monthly_cost >= tenant_limit:
            logger.warning(
                "Tenant '%s' budget exhausted: $%.4f / $%.2f",
                tenant_id, monthly_cost, tenant_limit,
            )
            raise TenantBudgetExceeded(tenant_id, monthly_cost, tenant_limit)

    # ── System-wide daily cap ─────────────────────────────────────────────────
    system_limit = _settings.system_daily_cost_limit_usd
    if system_limit > 0:
        daily_key   = "system:cost_usd:daily"
        raw         = await redis.get(daily_key)
        daily_spent = float(raw) if raw else 0.0

        if daily_spent >= system_limit:
            logger.error(
                "System daily budget exhausted: $%.4f / $%.2f",
                daily_spent, system_limit,
            )
            raise SystemBudgetExceeded(daily_spent, system_limit)


async def record_cost(
    tenant_id: str,
    redis: aioredis.Redis,
    cost_usd: float,
) -> None:
    """Increment both tenant monthly and system daily cost counters.

    Called after every successful LLM call. Non-blocking best-effort
    (failures are logged but not raised).
    """
    if cost_usd <= 0:
        return
    month = datetime.now(UTC).strftime("%Y-%m")
    try:
        pipe = redis.pipeline()
        pipe.incrbyfloat(f"quota:{tenant_id}:cost_usd:{month}", cost_usd)
        pipe.incrbyfloat("system:cost_usd:daily", cost_usd)
        await pipe.execute()
    except Exception as exc:
        logger.warning("Failed to record cost: %s", exc)
