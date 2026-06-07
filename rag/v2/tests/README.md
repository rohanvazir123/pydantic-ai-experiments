# tests/

## Table of Contents

- [What This Is](#what-this-is)
- [Categories](#categories)
- [Running Tests](#running-tests)
- [Test Gates](#test-gates)

---

## What This Is

The full pytest test suite. Each sub-folder maps to a test category with different external dependency requirements. See `TESTS.md` for the complete test plan including file maps, acceptance criteria, and CI integration spec.

---

## Categories

| Folder | What it tests | Requires | CI |
|--------|--------------|----------|----|
| `unit/` | Pure logic, no I/O — settings, metrics math, backoff, fusion | None | ✅ always |
| `integration/` | End-to-end layer tests against real services | PostgreSQL + Redis | ✅ with services |
| `retrieval/` | IR quality metrics against gold datasets | PostgreSQL + Ollama + ingested data | ✅ with services |
| `ingestion/` | Docling pipeline, incremental ingest, graph extraction | PostgreSQL + Redis + Ollama | ✅ with services |
| `agent/` | Confidence-aware pipeline, streaming, judge | Full stack | ✅ with services |
| `api/` | HTTP surface: status codes, SSE, error envelopes | Full stack | ✅ with services |
| `load/` | Locust load scenarios — find breaking point, sustained peak | Staging | Manual only |
| `chaos/` | Fault injection — kill Redis/Ollama/PostgreSQL; verify degraded mode | Staging | Manual only |

---

## Running Tests

```bash
cd rag/v2

# Unit tests only (no services, < 30s)
make test-unit

# All tests (requires Docker Compose services)
make test

# Single file
python -m pytest tests/unit/test_settings.py -v

# Category with verbose output
python -m pytest tests/retrieval/ -v --log-cli-level=INFO
```

---

## Test Gates

Each implementation phase has a test gate that must pass before the next phase starts. See `TODO_implementation.md §Phase Test Gates` for the full table with runnable commands.
