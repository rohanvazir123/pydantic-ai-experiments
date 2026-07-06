# app/api/

The FastAPI surface — JSON API + server-rendered web pages.

## Table of Contents

- [Purpose](#purpose)
- [Files](#files)
- [Routes](#routes)

## Purpose

Accepts orders and approval decisions over HTTP, persists via the repository, and
starts/signals workflows through the `WorkflowStarter` abstraction (faked in tests).

## Files

| File | Role |
|------|------|
| `app.py` | `create_app(repo, starter, *, lifespan=None)` factory — routes read deps from `app.state`. |
| `deps.py` | `WorkflowStarter` Protocol (`start_order` / `approve`). |
| `asgi.py` | Production entrypoint (`app.api.asgi:app`) — lifespan connects Postgres + Temporal; served by gunicorn/uvicorn. |

## Routes

- JSON: `POST /api/orders`, `GET /api/orders`, `GET /api/orders/{id}`,
  `POST /api/orders/{id}/approval`, `GET /health`.
- **SSE:** `GET /api/orders/{id}/events` — `text/event-stream`, pushes live status
  changes (used by the order page's `EventSource`); no client polling.
- Web (Jinja2): `GET /` (form + list), `POST /orders` (form → redirect),
  `GET /orders/{id}` (status page).

Tests inject `repo` + a fake `starter`; production builds them in `asgi.py`'s
lifespan. Both paths store them on `app.state`, which the routes read at request time.
