# knowledge/api/routes/

## Table of Contents

- [What This Is](#what-this-is)
- [Route Files](#route-files)

---

## What This Is

One Python file per API group. Each file is an `APIRouter` included by `app.py`. Routes are thin — they validate input, delegate to the appropriate service layer, and format the response. No business logic lives here.

---

## Route Files

| File | Routes | Auth |
|------|--------|------|
| `auth.py` | `POST /v1/auth/token`, `POST /v1/auth/refresh` | none |
| `chat.py` | `POST /v1/chat`, `POST /v1/chat/stream` (SSE) | reader |
| `search.py` | `POST /v1/search` | reader |
| `ingest.py` | `POST /v1/ingest`, `GET /v1/ingest/{id}/status`, `GET /v1/ingest/{id}/stream` | writer |
| `corpus.py` | `GET /v1/corpus`, `POST /v1/corpus/{id}/cache/invalidate`, `GET/POST/DELETE /v1/corpus/{id}/ontology` | reader / admin |
| `scheduler.py` | `GET/POST/PATCH/DELETE /v1/scheduler/jobs`, `POST /v1/scheduler/jobs/{id}/run-now` | writer |
| `evaluate.py` | `POST /v1/evaluate/run`, `GET /v1/evaluate/run/{id}`, `GET /v1/evaluate/compare` | admin |
| `feedback.py` | `POST /v1/feedback`, `POST /v1/signals` | reader / service |
| `memory.py` | `GET/DELETE /v1/conversations`, `GET/POST/DELETE /v1/memories` | reader |
| `logs.py` | `GET /v1/logs` | admin |
| `health.py` | `GET /health`, `GET /metrics` | none / service |
