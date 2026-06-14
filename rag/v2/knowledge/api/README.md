# knowledge/api/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Middleware Stack](#middleware-stack)
- [Auth](#auth)

---

## What This Is

The FastAPI application layer. Exposes all REST and SSE endpoints, enforces JWT auth and RBAC, applies rate limiting, and emits audit events. The app is served by Gunicorn + UvicornWorker in production.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI factory: lifespan (connect stores, start scheduler), middleware stack |
| `auth.py` | `require_jwt` dependency, JWKS fetch+cache, JWE encrypt/decrypt helpers |
| `middleware.py` | CorrelationID header, structured request log, background audit event emit |
| `quota.py` | `enforce_quota()`: Redis counter checks for RPM and daily limits |
| `timeout.py` | `TimeoutBudget` dataclass: per-stage sub-deadlines for the request pipeline |
| `schemas.py` | Pydantic request/response models (`ChatRequest`, `RAGResponse`, `ErrorDetail`, …) |
| `routes/` | One file per API group — see `routes/README.md` |

---

## Middleware Stack

Applied in this order (outermost first):

1. **CorrelationID** — sets `X-Request-ID`; injects into `contextvars` for log correlation
2. **StructuredLog** — emits one JSON log line per request with latency, user_id, corpus_id
3. **AuditEmitter** — background task: `INSERT INTO audit_events` after every authenticated request
4. **CORS** — configurable allowed origins
5. **RateLimiter** — `slowapi` per JWT `sub`; returns `429` with `Retry-After`

---

## Auth

All routes except `/health` and `/metrics` require a JWT. The `require_jwt` FastAPI dependency:

1. Decodes the `Authorization: Bearer <token>` header using RS256 + cached JWKS
2. Extracts `sub` (user_id), `roles`, and `tenant_id` from claims
3. Injects as `TokenClaims` into the route handler

RBAC is checked per-corpus in `check_corpus_access()` — called by any route that targets a specific corpus.
