"""Built-in placeholder hooks registered at app startup.

All are no-ops until their real implementations are added. They are
registered here so the hook points are always populated — preventing
key-not-found errors — and so the hook names appear in introspection.

Real implementations are added by:
  - audit_log_hook  → knowledge/api/middleware.py (Phase 9)
  - pii_redact_hook → knowledge/hooks/ (future)
  - metrics_hook    → knowledge/observability/metrics.py (Phase 11)
"""

import logging

from knowledge.hooks.context import HookContext
from knowledge.hooks.registry import HookPoint, registry

logger = logging.getLogger(__name__)


async def audit_log_hook(ctx: HookContext) -> HookContext:
    """Placeholder — real impl emits to audit_events table."""
    return ctx


async def pii_redact_hook(ctx: HookContext) -> HookContext:
    """Placeholder — real impl scrubs PII from retrieved chunks."""
    return ctx


async def response_filter_hook(ctx: HookContext) -> HookContext:
    """Placeholder — real impl applies output filtering."""
    return ctx


async def metrics_hook(ctx: HookContext) -> HookContext:
    """Placeholder — real impl increments Prometheus counters."""
    return ctx


def register_builtin_hooks() -> None:
    """Register all built-in placeholder hooks. Called once at app startup."""
    registry.register(HookPoint.POST_LLM,   audit_log_hook,      priority=100, name="audit_log")
    registry.register(HookPoint.POST_RETRIEVE, pii_redact_hook,  priority=100, name="pii_redact")
    registry.register(HookPoint.POST_LLM,   response_filter_hook, priority=200, name="response_filter")
    for point in HookPoint:
        registry.register(point, metrics_hook, priority=999, name=f"metrics_{point.value}")
    logger.info("Built-in placeholder hooks registered")
