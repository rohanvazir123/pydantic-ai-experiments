# knowledge/hooks/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Hook Points](#hook-points)
- [Writing a Hook](#writing-a-hook)

---

## What This Is

A lifecycle hook system that lets custom async callables intercept the request pipeline at named points — without touching core pipeline logic. Built-in hooks are registered as no-ops at startup; real implementations are added incrementally.

---

## Files

| File | Purpose |
|------|---------|
| `registry.py` | `HookRegistry`, `HookPoint` enum, `Hook` type alias; `fire()` runs hooks in priority order |
| `context.py` | `HookContext` dataclass: carries full request state through the hook chain |
| `builtins.py` | Placeholder hooks registered at startup: `audit_log_hook`, `pii_redact_hook`, `response_filter_hook`, `metrics_hook` |

---

## Hook Points

```
PRE_VALIDATE    → before validation pipeline
POST_VALIDATE   → after validation passes
PRE_ROUTE       → before model router
POST_ROUTE      → after routing decision
PRE_RETRIEVE    → before retrieval (user memories injected here)
POST_RETRIEVE   → after retrieval, before LLM
PRE_LLM         → before LLM call (cost guard fires here)
POST_LLM        → after LLM response (audit log fires here)
PRE_INGEST      → before document ingestion
POST_INGEST     → after ingestion completes
ON_CACHE_HIT    → any cache layer hit
ON_VALIDATION_FAIL → query rejected or pipeline abstained
ON_ERROR        → unhandled exception in pipeline
```

---

## Writing a Hook

```python
from knowledge.hooks.context import HookContext
from knowledge.hooks.registry import HookPoint, registry

async def my_hook(ctx: HookContext) -> HookContext:
    # Read ctx.query, ctx.user_id, ctx.retrieved_chunks, etc.
    # Mutate ctx to pass data to the next hook or pipeline stage
    ctx.metadata["my_key"] = "my_value"
    return ctx

# Register at app startup (lower priority number = runs first)
registry.register(HookPoint.POST_RETRIEVE, my_hook, priority=10)
```

A hook can raise `HookAbort` to short-circuit the pipeline and return a custom response to the client.
