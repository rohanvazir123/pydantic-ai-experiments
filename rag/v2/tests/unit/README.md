# tests/unit/

## Table of Contents

- [What This Is](#what-this-is)
- [Rules](#rules)
- [Files](#files)

---

## What This Is

Pure unit tests. No external services, no network, no database. These run in < 30 seconds and pass on any machine with Python + dev dependencies installed.

---

## Rules

- No `asyncpg`, no `redis`, no `httpx` calls — mock everything at the boundary
- Use `fakeredis` for Redis-dependent tests (circuit breaker, quota)
- Use `mock.patch.dict(os.environ, ..., clear=True)` for settings tests
- Every test class is labelled `@pytest.mark.unit`

---

## Files

| File | Tests |
|------|-------|
| `test_settings.py` | Settings loading, defaults, corpus config parsing, masking, constraints, singleton |
| `test_backoff.py` | `exponential_backoff()`: schedule, jitter bounds, max cap _(not yet implemented)_ |
| `test_circuit_breaker.py` | State transitions CLOSED→OPEN→HALF-OPEN→CLOSED _(not yet implemented)_ |
| `test_fusion.py` | RRF math, confidence sigmoid, confidence filter _(not yet implemented)_ |
| `test_validation.py` | V1–V5 validation chain rejection paths _(not yet implemented)_ |
| `test_quota.py` | Daily quota, RPM limit, free-tier LLM block _(not yet implemented)_ |
| `test_scheduler.py` | `compute_next_run_at`, `get_due_jobs` logic _(not yet implemented)_ |
| `test_chunker.py` | `DoclingHybridChunker` with mock `DoclingDocument` _(not yet implemented)_ |
| `test_auth.py` | JWT decode, RBAC, expired token _(not yet implemented)_ |
