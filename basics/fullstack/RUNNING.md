# Running OrderFlow

Step-by-step guide to run the tests and the full application.

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Install dependencies](#1-install-dependencies)
- [2. Run the tests (no services needed)](#2-run-the-tests-no-services-needed)
- [3. Start backing services (Postgres + Temporal)](#3-start-backing-services-postgres--temporal)
- [4. Start the worker](#4-start-the-worker)
- [5. Start the API](#5-start-the-api)
- [6. Try it out](#6-try-it-out)
- [7. Stop everything](#7-stop-everything)
- [Ports](#ports)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python package/toolchain manager)
- **Docker** + Docker Compose (for Postgres and the Temporal dev server)
- Free local ports: `8000`, `5432`, `7233`, `8233` (see [Ports](#ports))

> The Python interpreter is managed by `uv` (pinned to 3.13 in `.python-version`) —
> you do **not** need a system Python 3.13.

All commands run from `basics/fullstack/`.

## 1. Install dependencies

```bash
cd basics/fullstack
uv sync            # creates .venv and installs everything
```

## 2. Run the tests (no services needed)

The suite is self-contained — it uses Temporal's in-process test server and an
in-memory store, so **no Docker/Postgres/Temporal is required**:

```bash
uv run pytest          # 36 passed
# or the full gate:
make check             # ruff + mypy + pytest
```

To run the real app instead, continue below.

## 3. Start backing services (Postgres + Temporal)

```bash
cp .env.example .env   # default values already point at the compose services
make up                # docker compose up -d  (postgres + temporal dev server)
```

First run pulls the `postgres:16` and `temporalio/temporal` images (needs network).
Give them a few seconds; check they're healthy:

```bash
docker compose ps
```

The database schema is created automatically on first connect (the repository runs
`CREATE TABLE IF NOT EXISTS`), so there's no separate migration step.

## 4. Start the worker

The worker hosts the Temporal workflow + activities. **Orders will not progress
unless the worker is running.** In one terminal:

```bash
make run-worker        # python -m app.main worker
```

Leave it running.

## 5. Start the API

In a second terminal:

```bash
make run-api           # python -m app.main api  -> http://localhost:8000  (dev, single process)
```

For production, run under **gunicorn with uvicorn workers** instead:

```bash
make run-api-prod
# = uv run gunicorn app.api.asgi:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
```

`app.api.asgi:app` connects Postgres + Temporal on startup (FastAPI lifespan).

## 6. Try it out

### Web UI

- App: <http://localhost:8000> — submit orders. The order detail page updates its
  **status live via SSE** (Server-Sent Events) — no manual refresh.
- Temporal Web UI: <http://localhost:8233> — inspect running/closed workflows,
  their event history, and signals.

### Live status stream (SSE)

```bash
curl -N http://localhost:8000/api/orders/<id>/events
# event: status
# data: {"order_id":"<id>","status":"processing"}
# event: status
# data: {"order_id":"<id>","status":"confirmed"}
```

### API (curl)

Low-value order — auto-confirms:

```bash
# create ($10 total -> below the $1,000 threshold)
curl -s -X POST http://localhost:8000/api/orders \
  -H 'content-type: application/json' \
  -d '{"item":"Widget","quantity":2,"unit_price_cents":500}'
# -> {"order_id":"<id>","status":"pending"}

# poll — becomes "confirmed" once the worker processes it
curl -s http://localhost:8000/api/orders/<id>
```

High-value order — requires approval (human-in-the-loop):

```bash
# create ($2,000 total -> above threshold)
curl -s -X POST http://localhost:8000/api/orders \
  -H 'content-type: application/json' \
  -d '{"item":"Server","quantity":1,"unit_price_cents":200000}'
# -> {"order_id":"<id>","status":"pending"}

# it moves to "awaiting_approval" and the workflow suspends (2-day SLA timer)
curl -s http://localhost:8000/api/orders/<id>        # status: awaiting_approval

# approve (or "rejected") -> signals the workflow -> "confirmed"
curl -s -X POST http://localhost:8000/api/orders/<id>/approval \
  -H 'content-type: application/json' \
  -d '{"decision":"approved"}'

curl -s http://localhost:8000/api/orders/<id>        # status: confirmed
```

List all orders: `curl -s http://localhost:8000/api/orders`
Health: `curl -s http://localhost:8000/health`

## 7. Stop everything

- Stop the API and worker with `Ctrl-C` in their terminals.
- Stop the services:

```bash
make down              # docker compose down
```

## Ports

| Port | Service |
|------|---------|
| 8000 | FastAPI app |
| 5432 | PostgreSQL |
| 7233 | Temporal gRPC (client/worker connect here) |
| 8233 | Temporal Web UI |

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Orders stay `pending`/`processing` forever | The **worker isn't running** (step 4), or it can't reach Temporal. Start it and check its logs. |
| `connection refused` on startup | Services not up yet — run `make up` and wait for `docker compose ps` to show them healthy. |
| High-value order never leaves `awaiting_approval` | Expected — it's waiting for your approval call (step 6). It auto-rejects only after the 2-day SLA timer. |
| Image pull fails on `make up` | Network issue reaching Docker Hub; retry once services/registry are reachable. |
| Port already in use | Another process holds 8000/5432/7233/8233 — stop it or change `API_PORT` in `.env` / the compose port mappings. |
| Want to change the approval threshold | Edit `HIGH_VALUE_THRESHOLD_CENTS` in `app/domain.py`. |

> Alternative to Docker for Temporal: if you have the Temporal CLI installed, run
> `temporal server start-dev` (serves gRPC on `7233`, UI on `8233`) instead of the
> compose `temporal` service. You'd still need a Postgres for the app.
