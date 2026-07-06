# Test Report — OrderFlow

## Table of Contents

- [Summary](#summary)
- [Environment](#environment)
- [Results by suite](#results-by-suite)
- [What is verified](#what-is-verified)
- [How to reproduce](#how-to-reproduce)

## Summary

| Gate | Result |
|------|--------|
| **pytest** | ✅ **38 passed** in ~1.7s |
| **ruff** (lint) | ✅ All checks passed |
| **mypy** (types, `app/`) | ✅ Success: no issues in 18 source files |

All tests are included and passing — none were dropped. No external Temporal
server or Postgres instance is required to run the suite (see below).

## Environment

- Python **3.13.14** (uv-managed)
- `temporalio` **1.30.0** · `fastapi` **0.139.0** · `pydantic` **2.13.4**
- `pytest` 9.1.1 · `pytest-asyncio` 1.4.0 (auto mode)
- Workflow tests run against Temporal's **in-process time-skipping test server**
  (started per test); the 2-day SLA timer is fast-forwarded to milliseconds.

## Results by suite

```
tests/test_activities.py ...            [  3 ]  activities (direct call, in-memory repo)
tests/test_api.py .............         [ 13 ]  all FastAPI routes + SSE stream
tests/test_domain.py .........          [  9 ]  pure domain logic
tests/test_e2e.py .                     [  1 ]  full HTTP -> Temporal -> worker -> repo -> HTTP
tests/test_integration.py ..            [  2 ]  real TemporalWorkflowStarter wiring
tests/test_store.py .....               [  5 ]  in-memory repository CRUD
tests/test_workflow.py .....            [  5 ]  workflow: confirm / reject / HIL approve+reject+SLA
======================== 38 passed in ~1.7s ========================
```

Slowest: the full-stack HTTP e2e (~0.22s) and Temporal test-server startups (~0.11s each).

## What is verified

- **Domain rules** — line-total math, order validation (empty item, non-positive
  quantity, negative price), high-value boundary at exactly $1,000.
- **Persistence** — create/get/list/set-status; missing-key error.
- **Activities** — `load_order` (incl. non-retryable not-found), `mark_status`.
- **Workflow** (real Temporal engine):
  - low-value order → `confirmed`
  - invalid order → `rejected`
  - high-value + **approval signal** → `confirmed`
  - high-value + **rejection signal** → `rejected`
  - high-value + **no decision → SLA timer fires** → `rejected`
- **Wiring** — `TemporalWorkflowStarter.start_order` / `.approve` drive the workflow.
- **Full stack** — an order POSTed over HTTP is processed by the workflow/worker and
  reads back as `confirmed` over HTTP.
- **API** — create/list/get/approval routes, 202/404/422 status codes, the
  server-rendered web pages + form redirect, and the **SSE** status stream
  (`text/event-stream` framing + 404 for a missing order).

## How to reproduce

```bash
cd basics/fullstack
uv sync
uv run pytest -v          # 36 passed
uv run ruff check app tests
uv run mypy app
# or all three:
make check
```
