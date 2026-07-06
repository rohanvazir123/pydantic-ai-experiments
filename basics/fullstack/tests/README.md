# tests/

The test suite (pytest + pytest-asyncio). See [../TEST_REPORT.md](../TEST_REPORT.md)
for the latest run.

## Table of Contents

- [Approach](#approach)
- [Files](#files)
- [Running](#running)

## Approach

Test-driven and self-contained: **no external Temporal server or Postgres** is
required. Workflow tests run against Temporal's in-process **time-skipping test
server** (which also fast-forwards the 2-day SLA timer), and the store/starter are
in-memory/fake behind their interfaces. Shared fixtures live in `conftest.py`
(`repo`, `temporal_env`).

## Files

| File | Covers |
|------|--------|
| `test_domain.py` | Pure domain logic. |
| `test_store.py` | In-memory repository CRUD. |
| `test_sqlmodel_repo.py` | Production SQLModel repo against in-memory SQLite. |
| `test_activities.py` | Activities called directly with an in-memory repo. |
| `test_workflow.py` | Workflow: confirm, reject, and all three HIL outcomes. |
| `test_integration.py` | Real `TemporalWorkflowStarter` wiring. |
| `test_e2e.py` | Full HTTP → FastAPI → Temporal → worker → repo → HTTP. |
| `test_api.py` | All routes via `TestClient` + a fake starter. |

## Running

```bash
uv run pytest          # 36 passed
make check             # ruff + mypy + pytest
```
