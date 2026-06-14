# tests/api/

## Table of Contents

- [What This Is](#what-this-is)
- [Requirements](#requirements)

---

## What This Is

HTTP surface tests. Verifies every route returns the correct status code and response envelope, SSE endpoints stream valid events, JWT auth is enforced, rate limiting returns 429, and error codes match the spec in `RAGV2_DESIGN.md §Error Handling`.

---

## Requirements

Full stack running. Tests use `httpx.AsyncClient(app=app)` — no real HTTP server needed, but PostgreSQL + Redis must be available for the app to start.
