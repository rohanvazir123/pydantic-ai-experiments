# app/store/

Persistence layer — the `OrderRepository` interface and its implementations.

## Table of Contents

- [Purpose](#purpose)
- [Files](#files)

## Purpose

Decouples the rest of the app from any concrete database. The API and the Temporal
activities depend only on the `OrderRepository` Protocol, so tests inject the
in-memory store and production injects Postgres with no code changes.

## Files

| File | Role |
|------|------|
| `base.py` | `OrderRepository` Protocol (`create` / `get` / `set_status` / `list`). |
| `memory.py` | `InMemoryOrderRepository` — used by the whole test suite; no DB. |
| `postgres.py` | `PostgresOrderRepository` (asyncpg, SQL-first) — the production store, incl. schema DDL. |
