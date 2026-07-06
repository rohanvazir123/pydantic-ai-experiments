# app/store/

Persistence layer — the `OrderRepository` interface and its implementations.

## Table of Contents

- [Purpose](#purpose)
- [Files](#files)

## Purpose

Decouples the rest of the app from any concrete database. The API and the Temporal
activities depend only on the `OrderRepository` Protocol, so production injects the
**SQLModel** store and the broader test suite injects the in-memory double — no code
changes either way.

## Files

| File | Role |
|------|------|
| `base.py` | `OrderRepository` Protocol (`create` / `get` / `set_status` / `list`). |
| `sqlmodel_repo.py` | `SqlModelOrderRepository` (async SQLAlchemy/SQLModel) — the production store; runs on Postgres (`postgresql+asyncpg://…`) and on SQLite for its own tests. Includes `create_engine` + `init_db`. |
| `memory.py` | `InMemoryOrderRepository` — fast test double for the rest of the suite; no DB. |

The `Order` SQLModel `table=True` model lives in `app/models.py` (one class = table + API schema).
