# app/

The OrderFlow application package.

## Table of Contents

- [Purpose](#purpose)
- [Modules](#modules)
- [Layering rule](#layering-rule)

## Purpose

Houses all application code: pure domain logic, DTOs, config, the persistence
layer, the Temporal orchestration layer, and the FastAPI surface.

## Modules

| Path | Responsibility |
|------|----------------|
| `domain.py` | Pure business logic — validation, pricing, high-value rule, enums. No I/O. |
| `models.py` | Pydantic DTOs shared across API, store, and Temporal payloads. |
| `config.py` | `pydantic-settings` configuration (env / `.env`). |
| `main.py` | Entrypoint — `python -m app.main api|worker`. |
| `store/` | `OrderRepository` interface + in-memory and Postgres implementations. |
| `temporal/` | Activities, workflow, client/starter, worker. |
| `api/` | FastAPI app factory + the `WorkflowStarter` protocol. |
| `web/` | Jinja2 templates. |

## Layering rule

`domain` imports nothing framework-specific and does no I/O — it is safe to call
from inside a Temporal workflow. All side effects live in `store` (DB) and are only
invoked from `temporal` activities or the `api` layer.
