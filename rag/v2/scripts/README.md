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
| `purge_default_corpus.sql` | (called by purge.py) | SQL to delete documents/chunks/entity index/semantic cache for default corpus |
| `purge.py` | `make purge-corpus` | Full purge + Redis fingerprint clear + AGE graph drop + forced re-ingestion |

---

## purge-corpus

Wipes all ingested content for `default:neuralflow` and forces a full re-ingestion.

```bash
make purge-corpus
```

**What it purges (3 layers):**

| Layer | What | How |
|-------|------|-----|
| PostgreSQL | documents, chunks (cascade), kg_entity_index, semantic_cache | `purge_default_corpus.sql` via psql |
| Redis | `cache:doc_fingerprint:*` + `cache:search:*` | Scan + DELETE |
| Apache AGE | `kg_default_neuralflow` graph | `drop_graph()` via asyncpg |

**What is preserved:**

| Table | Why |
|-------|-----|
| `audit_events` | Compliance — append-only, never purged |
| `token_usage`, `billing_events` | Financial records — 7yr retention |
| `conversations`, `messages`, `user_memories` | User memory — not content |
| `tenants`, `tenant_quotas`, `system_prompts` | Config + procedural memory |
| `gold_samples`, `eval_runs`, `eval_results` | Evaluation history |
| `scheduled_jobs` | Scheduler config |

**Deduplication guarantee:** After purge, the next `make seed` uses `mode=full` which bypasses the incremental SHA-256 check and processes all files unconditionally.

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
