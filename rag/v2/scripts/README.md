# scripts/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [seed.py](#seedpy)

---

## What This Is

Utility scripts for bootstrapping and maintaining the dev environment. Run from the `rag/v2/` directory via `make <target>` or directly with `uv run python scripts/<file>.py`.

---

## Files

| File | Makefile target | Purpose |
|------|----------------|---------|
| `seed.py` | `make seed` | Connectivity checks → corpus config → sample doc ingestion → verification |

---

## seed.py

Sets up a fresh dev environment end-to-end.

```bash
cd rag/v2
make seed          # full seed (idempotent)
make dev-reset     # wipe all data and re-seed (asks for confirmation)
```

**What it does:**
1. Checks that PostgreSQL, Redis, and Ollama are reachable
2. Writes default corpus config to `.env` if `CORPUS_CONFIGS_JSON` is empty
3. Ingests `../../rag/documents/` (NeuralFlow AI sample docs) into corpus `default:neuralflow`
4. Verifies chunk count > 0

**What is already seeded by migrations** (runs automatically via `make migrate`):
- Default tenant (`id='default'`, tier=free)
- Default tenant quotas (500 req/day, 10 RPM, LLM disabled)
- Default system prompt (`rag_agent_v1`)

**Idempotent:** safe to run twice — incremental mode skips unchanged files, and the corpus config is only written if currently empty.
