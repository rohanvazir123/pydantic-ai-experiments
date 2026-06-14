# schema/

> **Why "schema" not "migrations"?** The term *migrations* comes from ORMs (Django, Rails, Alembic) where each file represents an incremental *change* to an existing schema. Here, every file uses `IF NOT EXISTS` and `ON CONFLICT DO NOTHING` — you can run them against a blank database and get a complete, working schema in one shot. That's a *schema definition*, not an incremental migration. The folder is named accordingly.

## Table of Contents

- [What Schema Files Are](#what-migrations-are)
- [How They Work](#how-they-work)
- [Running Schema Setup](#running-migrations)
- [Schema Files](#migration-files)
- [Adding a New Schema File](#adding-a-new-migration)
- [Which Database](#which-database)

---

## What Migrations Are

Migrations are plain SQL files that build and evolve the PostgreSQL database schema. Each file is a set of `CREATE TABLE`, `CREATE INDEX`, `ALTER TABLE`, and similar statements that define what tables and indexes exist. Running them in order takes a blank database and produces a fully schema'd, ready-to-use database.

They are **not** application code — no Python, no Pydantic, no FastAPI. Just SQL that a database administrator or automated deploy script can read and audit directly.

---

## How They Work

- Files are numbered `001`, `002`, … and **must be run in order**. Each migration can depend on tables and columns created by earlier ones (e.g., `003` creates a table that references `documents` from `001`).
- Every statement uses `IF NOT EXISTS` or `ON CONFLICT DO NOTHING` so migrations are **idempotent** — safe to re-run if a migration was partially applied or if you want to verify the schema is up to date.
- Migrations are **additive only**. They add columns, tables, indexes, and policies. They never drop or rename things. Destructive changes go in a new numbered file, never in an existing one.
- There is no migration runner framework (no Alembic, no Flyway). The `Makefile` runs each `*.sql` file in glob order using `psql`. Simple and auditable.

---

## Running Migrations

**All at once (recommended):**
```bash
cd rag/v2
make migrate
# equivalent to:
for f in schema/*.sql; do psql "$DATABASE_URL" -f "$f"; done
```

**Single file (for debugging):**
```bash
psql "$DATABASE_URL" -f schema/003_semantic_cache.sql
```

**Verify tables were created:**
```bash
psql "$DATABASE_URL" -c "\dt"
```

`DATABASE_URL` must be set in your environment or `.env` file. See `.env.example` for the format.

---

## Migration Files

| File | What it creates | Key tables / objects |
|------|----------------|----------------------|
| `001_initial_schema.sql` | Core document store | `documents`, `chunks` (HNSW + GIN), `audit_events` |
| `002_corpus_tenant.sql` | Multi-tenant isolation | `corpus_id` + `tenant_id` columns on all tables; Row-Level Security policies; `kg_entity_index` (tsvector + pgvector) |
| `003_semantic_cache.sql` | L3 semantic cache | `semantic_cache` (HNSW on `query_emb`, JWE-encrypted answers) |
| `004_evaluation.sql` | Offline evaluation | `gold_samples`, `eval_runs` (with `report_json`), `eval_results` (retrieval metrics + confidence + pipeline status) |
| `005_feedback.sql` | Online signals | `user_feedback`, `implicit_signals`, `token_usage` |
| `006_billing.sql` | SaaS billing | `tenants`, `tenant_quotas`, `billing_events`; seeds free-tier default tenant |
| `007_scheduler.sql` | Periodic ingest | `scheduled_jobs` (cron expression + source config + corpus target) |
| `008_memory.sql` | Memory system | `conversations` + `messages` (tsvector GIN), `user_memories` (tsvector + pgvector HNSW), `system_prompts` (versioned) |

### Index strategy

Every searchable text column has a `tsvector GENERATED ALWAYS AS` column with a GIN index for BM25 full-text search. Every embedding column has an HNSW index for approximate nearest-neighbour cosine search. Hybrid search (RRF k=60) combines both — see `basics/rag/memory/MEMORY_DESIGN.md` for the query pattern.

### Row-Level Security (RLS)

`002` enables RLS on `documents`, `chunks`, and `audit_events`. Before every query the API sets:
```sql
SET LOCAL app.tenant_id = 'acme-corp';
```
PostgreSQL then enforces the `tenant_isolation` policy automatically — rows from other tenants are invisible even if a bug in the application forgets to add a `WHERE tenant_id = $1` clause.

---

## Adding a New Migration

1. Create `schema/009_your_feature.sql` (next number in sequence).
2. Use `IF NOT EXISTS` on all `CREATE` statements.
3. Use `ON CONFLICT DO NOTHING` on all `INSERT` seed data.
4. Test it on a blank database: `psql "$DATABASE_URL" -f schema/009_your_feature.sql`.
5. Run the full suite to verify idempotency: `make migrate && make migrate` (second run should produce no errors).
6. Add a row to the table in this README.

---

## Which Database

Migrations run against the **main PostgreSQL database** (`DATABASE_URL`, default port 5432). This is the `pgvector/pgvector:pg16` container — it handles document chunks, semantic cache, evaluation, billing, and memory.

The **Apache AGE database** (`AGE_DATABASE_URL`, port 5433) has its own graph schema managed by `AgeGraphStore.initialize()` at runtime (it creates graphs dynamically per corpus). There are no SQL migration files for AGE.
