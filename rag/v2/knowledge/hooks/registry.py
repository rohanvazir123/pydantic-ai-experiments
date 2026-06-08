"""Hook registry — lifecycle points and async callable chain.

Usage:
    from knowledge.hooks.registry import registry, HookPoint

    @registry.hook(HookPoint.POST_RETRIEVE, priority=10)
    async def my_hook(ctx: HookContext) -> HookContext:
        ctx.metadata["my_key"] = "value"
        return ctx

    # In pipeline:
    ctx = await registry.fire(HookPoint.POST_RETRIEVE, ctx)

Hooks run in ascending priority order (lower number = runs first).
A hook raising HookAbort immediately stops the chain and the exception
propagates to the pipeline — the pipeline returns the abort response.
"""

import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum

from knowledge.hooks.context import HookAbort, HookContext

logger = logging.getLogger(__name__)

Hook = Callable[[HookContext], Awaitable[HookContext]]


class HookPoint(StrEnum):
    PRE_VALIDATE       = "pre_validate"
    POST_VALIDATE      = "post_validate"
    PRE_ROUTE          = "pre_route"
    POST_ROUTE         = "post_route"
    PRE_RETRIEVE       = "pre_retrieve"
    POST_RETRIEVE      = "post_retrieve"
    PRE_LLM            = "pre_llm"
    POST_LLM           = "post_llm"
    PRE_INGEST         = "pre_ingest"
    POST_INGEST        = "post_ingest"
    ON_CACHE_HIT       = "on_cache_hit"
    ON_VALIDATION_FAIL = "on_validation_fail"
    ON_ERROR           = "on_error"


class HookRegistry:
    """Ordered async hook chain with priority-based execution."""

    def __init__(self) -> None:
        # {HookPoint: [(priority, name, fn), ...]} sorted by priority ascending
        self._hooks: dict[HookPoint, list[tuple[int, str, Hook]]] = {
            point: [] for point in HookPoint
        }

    def register(
        self,
        point: HookPoint,
        fn: Hook,
        priority: int = 0,
        name: str | None = None,
    ) -> None:
        """Register a hook at the given lifecycle point.

        Lower priority number runs first.
        """
        hook_name = name or fn.__name__
        self._hooks[point].append((priority, hook_name, fn))
        self._hooks[point].sort(key=lambda t: t[0])
        logger.debug("Registered hook '%s' at %s (priority=%d)", hook_name, point, priority)

    def hook(self, point: HookPoint, priority: int = 0, name: str | None = None):
        """Decorator variant of register()."""
        def decorator(fn: Hook) -> Hook:
            self.register(point, fn, priority=priority, name=name)
            return fn
        return decorator

    async def fire(self, point: HookPoint, ctx: HookContext) -> HookContext:
        """Run all hooks at `point` in priority order.

        Returns the (potentially mutated) context.
        Raises HookAbort if any hook aborts the pipeline.
        Any other exception from a hook is logged and re-raised.
        """
        for priority, name, fn in self._hooks[point]:
            try:
                ctx = await fn(ctx)
            except HookAbort:
                logger.info("HookAbort from '%s' at %s", name, point)
                raise
            except Exception as exc:
                logger.error("Hook '%s' at %s raised: %s", name, point, exc)
                raise
        return ctx

    def registered(self, point: HookPoint) -> list[str]:
        """Return names of hooks registered at a point (for introspection/tests)."""
        return [name for _, name, _ in self._hooks[point]]

    def clear(self, point: HookPoint | None = None) -> None:
        """Remove all hooks at a point, or all hooks if point is None (for tests)."""
        if point is None:
            for p in HookPoint:
                self._hooks[p].clear()
        else:
            self._hooks[point].clear()


# Module-level singleton — imported by all pipeline components
registry = HookRegistry()
