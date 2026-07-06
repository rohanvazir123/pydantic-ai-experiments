# OrderFlow — a full-stack order-processing app on Temporal

A small but complete full-stack application demonstrating **durable orchestration
with [Temporal](https://temporal.io)**: orders are submitted over HTTP, a Temporal
workflow validates each one, routes **high-value orders through a human-approval
gate** (signal + SLA timer — the human-in-the-loop pattern), and confirms or rejects
them. Built on the stack from
[`../.system_design_practice/impl_template.md`](../.system_design_practice/impl_template.md)
(FastAPI + Postgres + Pydantic + Jinja2), with **Temporal swapped in** for the
orchestration layer.

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Quick start](#quick-start)
- [API surface](#api-surface)
- [The Temporal workflow (HIL)](#the-temporal-workflow-hil)
- [Testing](#testing)
- [Test report](#test-report)
- [Environment notes](#environment-notes)

## What it does

- Submit an order (item, quantity, unit price) via a web form or JSON API.
- A Temporal workflow processes it durably:
  - **valid + low value (≤ $1,000)** → auto-confirmed.
  - **valid + high value (> $1,000)** → suspends for **manager approval**; a
    2-day SLA timer auto-rejects if no decision arrives.
  - **invalid** → rejected.
- Poll status over HTTP or view it in the server-rendered UI.

## Architecture

```
Browser / API client
        │  HTTP (JSON + HTML form)          ▲  SSE (live status: text/event-stream)
        ▼                                   │
┌───────────────────────┐     start_workflow / signal      ┌──────────────────┐
│  FastAPI (app/api)     │ ───────────────────────────────▶ │  Temporal server │
│  - JSON API + Jinja2   │        (Temporal task queue =     └────────┬─────────┘
│  - SSE live updates    │         durable intake queue)              │ task
│  - WorkflowStarter     │                                            ▼
└───────────┬───────────┘                          ┌───────────────────────────┐
            │ create/read                           │  Worker (app/temporal)    │
            ▼                                        │  - OrderWorkflow          │
     ┌──────────────┐   load_order / mark_status    │  - OrderActivities (I/O)  │
     │  Postgres    │ ◀──────────────────────────── │                           │
     │ (repository) │                                └───────────────────────────┘
     └──────────────┘
```

The **API and worker are decoupled processes** connected by Temporal's task queue —
no separate message broker needed. The SSE endpoint reads order status from the store
and pushes changes to the browser live.

The **workflow** is pure orchestration (deterministic, no I/O). All side effects
(DB reads/writes) happen in **activities**. The API talks to Temporal through a
`WorkflowStarter` abstraction, and to the DB through an `OrderRepository` interface —
both are swapped for lightweight fakes/in-memory impls in tests.

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.13 (asyncio) |
| API | **FastAPI**; dev = uvicorn, **prod = gunicorn + uvicorn workers** |
| Frontend | Jinja2 server-rendered templates + **SSE** (live status, no polling) |
| Queuing / decoupling | **Temporal task queue** — API enqueues (`start_workflow`), worker consumes. No Redis Streams (would be redundant with Temporal; see note). |
| Orchestration | **Temporal** (`temporalio`) — workflow, activities, signals, SLA timer |
| Database | PostgreSQL 16 via **asyncpg** (SQL-first), behind a repository interface |
| Models / config | Pydantic + pydantic-settings |
| Tests | pytest + pytest-asyncio, Temporal **time-skipping test server**, httpx ASGI |
| Quality | ruff + mypy |
| Packaging | uv |

> **Why no Redis Streams?** The `impl_template.md` default stack uses Redis Streams
> as its durable queue *because it has no Temporal*. Here **Temporal owns durable
> queuing, decoupling, retries, and timers**, so a Redis Stream in front of it would
> duplicate that. Live status is pushed to browsers via **SSE** instead — the one gap
> Temporal doesn't cover (pushing to clients). If outbound fan-out to *external*
> subscribers were needed later, a Redis Stream event bus is the place it would slot in.

## Project layout

```
fullstack/
├── app/
│   ├── domain.py          # pure business logic (validation, pricing, high-value rule)
│   ├── models.py          # Pydantic DTOs
│   ├── config.py          # pydantic-settings
│   ├── main.py            # entrypoint: `python -m app.main api|worker`
│   ├── store/             # OrderRepository: in-memory (tests) + Postgres (prod)
│   ├── temporal/          # activities, workflow, client/starter, worker
│   ├── api/               # FastAPI app factory + SSE + asgi.py (prod entrypoint)
│   └── web/templates/     # Jinja2 pages (order page uses EventSource for live status)
├── tests/                 # domain / store / activities / workflow / api / integration / e2e
├── docker-compose.yml     # Postgres + Temporal dev server (UI on :8233)
├── Makefile               # install / test / lint / type / up / run-*
└── pyproject.toml
```

## Quick start

> Full step-by-step walkthrough (with the approval flow and troubleshooting):
> **[RUNNING.md](RUNNING.md)**.

```bash
cd basics/fullstack
uv sync                       # install deps into .venv (Python 3.13)

# Run the tests (no external services needed — uses Temporal's in-process test server)
uv run pytest                 # or: make test

# --- run the real app ---
cp .env.example .env
make up                       # start Postgres + Temporal dev server (Docker)
make run-worker               # terminal 1: Temporal worker
make run-api                  # terminal 2: FastAPI on http://localhost:8000

# open the UI
open http://localhost:8000    # submit orders
open http://localhost:8233    # Temporal Web UI (inspect workflows)
```

## API surface

| Method & path | Purpose |
|---------------|---------|
| `POST /api/orders` | Create an order → `202` `{order_id, status}`; starts the workflow |
| `GET  /api/orders` | List orders |
| `GET  /api/orders/{id}` | Get one order (`404` if missing) |
| `GET  /api/orders/{id}/events` | **SSE** stream (`text/event-stream`) — live status changes |
| `POST /api/orders/{id}/approval` | Manager decision `{decision: approved\|rejected}` → signals the workflow |
| `GET  /health` | Liveness |
| `GET  /` · `POST /orders` · `GET /orders/{id}` | Web UI (form + status pages) |

## The Temporal workflow (HIL)

`OrderWorkflow` (in `app/temporal/workflow.py`):

1. mark `processing` → `load_order` (activity) → validate (pure domain).
2. **high value?** → mark `awaiting_approval`, then
   `await workflow.wait_condition(decision is set, timeout=2 days)`:
   - **approval signal** arrives → confirm or reject per the decision;
   - **SLA timer** fires first → auto-reject.
3. otherwise → mark `confirmed`.

While suspended on `wait_condition`, the workflow holds no thread and survives worker
restarts — Temporal persists its state. The 2-day timer is durable. (This is the same
human-in-the-loop pattern documented in the LoanApproval design's Deep Dive 4.)

## Testing

Test-driven; every layer is covered **without any external service** — Temporal's
in-process **time-skipping test server** runs the real workflow engine (and
fast-forwards the 2-day SLA timer to milliseconds), and the store/starter are
in-memory/fake behind their interfaces.

| Suite | Covers | Needs |
|-------|--------|-------|
| `test_domain.py` | pricing, validation, high-value rule | nothing |
| `test_store.py` | in-memory repository CRUD | nothing |
| `test_activities.py` | activities (direct call, in-memory repo) | nothing |
| `test_workflow.py` | confirm / reject / **HIL approve, reject, SLA timeout** | Temporal test server (in-process) |
| `test_integration.py` | real `TemporalWorkflowStarter` wiring | Temporal test server |
| `test_e2e.py` | full HTTP → FastAPI → Temporal → worker → repo → HTTP | Temporal test server |
| `test_api.py` | all routes via `TestClient` + fake starter; **SSE** via httpx ASGI | nothing |

```bash
make check      # ruff + mypy + pytest
```

## Test report

See [TEST_REPORT.md](TEST_REPORT.md). Latest run:

```
38 passed in ~1.7s   (ruff: clean · mypy: clean)
```

All tests are **included and green** — nothing was dropped.

## Environment notes

- **Python 3.13, not 3.12.** The template targets 3.12; a 3.12 interpreter download
  failed here (flaky network), so the project pins 3.13 (already present). `temporalio`
  fully supports 3.13. `requires-python` allows `>=3.12,<3.14`.
- **No external Temporal/Postgres needed for tests** — the suite uses Temporal's
  time-skipping test server and the in-memory repository. Docker Compose is only for
  running the *real* app.
