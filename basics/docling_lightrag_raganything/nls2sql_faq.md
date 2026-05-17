# NL-to-SQL FAQ

---

## Schema Discovery

**Q: Why store schema chunks in pgvector + tsvector instead of just passing the full schema in every prompt?**

A full schema for 100 tables can easily exceed the LLM's context window and increases cost per query. By embedding schema chunks and retrieving only the top-K relevant tables at query time you keep the prompt small and focused. tsvector provides a keyword fallback for exact column or table name matches that vector similarity might miss.

---

**Q: How does schema retrieval scale to 100+ tables?**

The schema cache holds one chunk per table (columns + descriptions + sample values). At query time, the normalized NL query is embedded and an ANN search returns the top-50 candidate chunks, which are then re-ranked to top-K. Only those K tables are injected into the prompt. A warehouse with 1 000+ tables would need a second-stage retrieval or domain routing layer on top.

---

**Q: How often does the Schema Discovery Service run?**

It is event-driven (schema-change hook) with a periodic fallback (e.g. nightly). Any ALTER TABLE, new table creation, or column rename should trigger a re-index of the affected chunks. Stale schema is the most common cause of hallucinated column names.

---

**Q: What metadata is stored alongside each schema embedding?**

Each chunk stores a `db:schema:table:column` path string as metadata. This allows ANN results to be filtered by database or schema before being passed to the prompt — critical for multi-tenant isolation.

---

## Prompt Engineering

**Q: Why the `<thinking>` and `<query>` output format?**

Structured output forces the model to separate reasoning from the SQL statement, which improves SQL quality (the model "thinks before it writes") and makes parsing deterministic — the validation pipeline can extract the SQL block with a simple tag split rather than regex heuristics. The `<thinking>` block is also useful for debugging wrong queries.

---

**Q: What goes into the RBAC context in the prompt?**

Two things: (1) an explicit list of permitted tables and columns for the requesting tenant/role, and (2) mandatory WHERE filters (e.g. `region = 'North America'`) that must appear in the generated SQL. These are injected as hard requirements, not suggestions, so the LLM cannot generate queries that cross tenant boundaries.

---

**Q: How are business acronyms handled?**

The prompt normalization stage should resolve known acronyms to their expanded form before schema retrieval (e.g. "MCR" → "Monthly Conversion Rate"). A static acronym dictionary per tenant/domain is the simplest approach; a retrieval-augmented glossary works for large or changing vocabularies.

---

**Q: Why resolve dates to YYYY-MM-DD in normalization?**

LLMs are inconsistent about date formatting across requests — "last quarter" might produce `Q3 2024`, `'2024-07-01'`, or a relative expression depending on the model run. Normalizing to ISO format in the prompt guardrail eliminates ambiguity and ensures the generated SQL uses a format the database can parse.

---

## SQL Generation

**Q: Why generate N candidates instead of one?**

A single LLM sample can fail schema validation or produce a suboptimal join order. Generating N candidates with a confidence score means the validation pipeline can fall back to the next-best candidate without re-prompting the LLM, saving a full round-trip.

**Q: Where are the ranked candidates stored?**

In memory for the duration of the request. If the top candidate fails validation, the pipeline pops the next candidate from the ordered list. There is no need to persist them to Redis unless the repair loop needs to survive a process restart, which is an edge case.

---

**Q: Why Qwen-2.5 as the generation model?**

Qwen-2.5 performs competitively on SQL benchmarks (Spider, BIRD) while running locally — no API cost, no data leaving the environment. For a multi-tenant analytical system this matters because queries may touch sensitive commercial data. Swap to a larger model (Qwen-2.5-72B or GPT-4o) if accuracy requirements increase.

---

## SQL Validation

**Q: Why use sqlglot for validation instead of running a dry-run EXPLAIN?**

`EXPLAIN` requires a live database connection and leaks query patterns to the query planner. sqlglot validates syntax and schema entirely in-process, with no network round-trip. It also provides structured AST inspection for detecting DDL nodes (DROP, DELETE, etc.) more reliably than regex.

---

**Q: What is the repair loop and when does it trigger?**

When a generated SQL fails a recoverable check (schema validation error, disallowed column), the validation pipeline re-prompts the LLM with the original prompt + the failing SQL + the normalized error message. The LLM uses this context to correct the query. Unrecoverable failures (policy violations, hard DDL blocks) are not retried — they surface directly as user-facing errors.

---

**Q: How many repair attempts are made before giving up?**

A fixed maximum (typically 3). Each attempt costs one LLM round-trip (~2–4s). Three attempts still fits within the 10s SLA for most cases. On exhaustion the system returns a structured error and optionally falls back to the next-ranked candidate from the generation step.

---

**Q: What counts as a "hard" vs "recoverable" error?**

| Error | Type | Reason |
|-------|------|--------|
| Disallowed keyword (INSERT, DROP) | Hard | The LLM intentionally generated a write; re-prompting is likely to reproduce it |
| RBAC policy violation | Hard | The query accesses a forbidden column; re-prompting without changing permissions won't help |
| Unknown column name | Recoverable | The LLM hallucinated a column; the error message names it so the LLM can correct |
| Syntax error | Recoverable | Malformed SQL the LLM can fix given the error |
| Schema type mismatch | Recoverable | Wrong data type in a comparison; correctable with the schema in context |

---

## Execution

**Q: Why cursor-based pagination instead of OFFSET/LIMIT?**

OFFSET forces the database to scan and discard the first N rows on every page, which degrades linearly as the page number grows. A cursor (keyset pagination using an ordered unique column) fetches only the next page's rows regardless of depth — much faster for large analytical result sets.

---

**Q: How is multi-tenancy enforced at the execution layer?**

Each tenant gets a dedicated connection pool bound to a read-only database user scoped to their schema or database. This means a misconfigured query cannot cross tenant data even if the RBAC prompt injection fails — the database user simply has no SELECT permission on other tenants' tables.

---

**Q: What happens when a query times out?**

The execution pipeline issues a cancellation signal to the database connection (PostgreSQL: `pg_cancel_backend`; DuckDB: `conn.interrupt()`). The timeout error is returned to the caller as a structured response. The query is not retried automatically — a timeout typically indicates the query needs optimization, not a transient failure.

---

## Caching

**Q: When should the NL→Results cache be used vs the SQL→Results cache?**

Use **NL→Results** for recurring business questions that always produce the same SQL and the same data (e.g. a fixed dashboard metric). Use **SQL→Results** when different NL phrasings generate the same SQL — this catches semantically equivalent questions without re-executing the query. Both require data-freshness consideration: invalidate on ETL completion or use a short TTL.

---

**Q: How is the schema cache invalidated?**

The Schema Discovery Service writes new embeddings for changed tables and deletes stale chunks identified by their `db:schema:table` metadata path. There is no global cache flush — only affected chunks are replaced, so the cache remains warm for unchanged tables.
