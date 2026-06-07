# tests/integration/

## Table of Contents

- [What This Is](#what-this-is)
- [Requirements](#requirements)
- [Files](#files)

---

## What This Is

End-to-end layer tests that hit real services. Each test brings up its fixtures, performs operations, and asserts database or cache state. No mocking of infrastructure.

---

## Requirements

```bash
docker compose up -d postgres age redis
# For agent tests, also:
docker compose up -d ollama && make pull-models
```

Tests auto-skip when services are unreachable (via `pytest_asyncio.fixture` with `try/except ConnectionFailure`).

---

## Files

| File | Tests |
|------|-------|
| `test_vector_store.py` | Upsert → search → corpus isolation → RLS tenant check → delete _(not yet implemented)_ |
| `test_cache.py` | Redis L2 cache hit/miss; corpus invalidation _(not yet implemented)_ |
| `test_semantic_cache.py` | L3 store/lookup/prune/JWE round-trip _(not yet implemented)_ |
| `test_ingestion_pipeline.py` | File → Docling → chunks → DB; incremental skip; graph disabled path _(not yet implemented)_ |
| `test_retrieval_pipeline.py` | Query → confidence populated → cache hit on repeat → corpus isolation _(not yet implemented)_ |
| `test_agent.py` | Confidence-aware pipeline answered + abstained paths; streaming _(not yet implemented)_ |
| `test_api.py` | All HTTP routes: status codes, SSE events, error envelopes _(not yet implemented)_ |
| `test_memory.py` | Two-turn conversation; auto-summarization at turn 21; memory extraction/injection _(not yet implemented)_ |
