# FAQ - NL-to-SQL System

Code references: line numbers point to files under `nl2sql/` in this repo.

---

## NLP-to-SQL System

<a id="q210"></a>
**Q210. Walk me through the end-to-end flow of the NLP-to-SQL system.**

`ConversationManager.run_query(nl_query)` is called:

1. NL query is normalized (lowercase + whitespace-collapsed) and checked against the NL cache — exact normalized match returns the cached `QueryResult` immediately.
2. Prompt is built: schema text + last 3 successful conversation turns (Q/SQL/result preview) + the new question.
3. `agent.run_sync(prompt)` calls GPT-4o (or Claude), which returns SQL.
4. `strip_sql_fences()` cleans any markdown wrapping.
5. SQL is MD5-hashed and checked against the SQL hash cache — if the same SQL was already generated for a different question, the cached result is returned.
6. `conn.execute(sql)` runs the SQL in DuckDB. Column names come from `cursor.description`.
7. On success: `QueryResult` (with `columns`, `rows`, `attempts`) is stored in both caches and history, then returned.
8. On failure: the error is fed back to the LLM in a correction prompt and step 3 retries (up to `max_retries`, default 3). If all retries fail, `QueryResult(error=...)` is returned.

---

<a id="q211"></a>
**Q211. How does schema discovery work?**

`UnifiedDataSource.generate_schema()` introspects all sources at startup:

**GCS Parquet:** Lists blob virtual prefixes with `delimiter="/"` — each subfolder becomes a table name. Creates a DuckDB `VIEW` over `parquet_scan('gs://...')`. Then `DESCRIBE view_name` gives column names and types.

**PostgreSQL:** After attaching via DuckDB's postgres extension, queries `{alias}.information_schema.tables` and `{alias}.information_schema.columns` through DuckDB's catalog.

Everything is serialized into a single schema string prepended to every LLM prompt. Schema is captured once at startup — changes require a restart.

---

<a id="q212"></a>
**Q212. Why DuckDB over Spark, Trino, pg_parquet, or duckdb_fdw?**

| Option | Problem |
|---|---|
| **Spark / Trino** | Cluster-based, heavy infrastructure. Overkill for single-analyst workloads. Adds 10–30s startup latency. |
| **pg_parquet** | PostgreSQL reads Parquet; limited SQL, no GCS HMAC auth, PostgreSQL is the bottleneck. |
| **duckdb_fdw** | Wrong direction — PostgreSQL queries DuckDB via FDW. Complex Windows setup, server-side changes required. |
| **DuckDB postgres_scanner** | DuckDB ATTACHes PostgreSQL as a catalog and JOINs it with GCS Parquets in a single in-process query. Zero server-side changes. Zero extra infrastructure. |

---

<a id="q213"></a>
**Q213. How do cross-source JOINs work?**

100% inside DuckDB's in-memory engine. GCS Parquets are read lazily via `httpfs` (predicate pushdown where possible). PostgreSQL tables are scanned via `postgres_scanner` (full table scan — no index pushdown). DuckDB handles the JOIN, aggregation, and projection internally. The user writes plain DuckDB SQL; the naming convention (`bare name` vs `alias.main.table`) tells DuckDB which catalog to use.

---

<a id="q214"></a>
**Q214. What are the limitations of this architecture?**

| Limitation | Detail |
|---|---|
| **PostgreSQL full scans** | `postgres_scanner` reads entire PG tables; no index pushdown. Large PG tables (>10M rows) are slow. |
| **In-memory result sets** | DuckDB defaults to in-memory. Very large results can OOM. No result pagination implemented. |
| **Static schema** | Captured at startup. Table changes require restart. |
| **GCS auth** | HMAC keys only. Service account JSON / Workload Identity not implemented. |
| **No timeout enforcement** | No per-query timeout. |
| **Semantically wrong SQL** | Syntactically valid but logically incorrect SQL returns wrong results silently. |

---

<a id="q215"></a>
**Q215. How is the model prompted to generate correct SQL?**

System prompt enforces DuckDB-specific table naming rules:
- GCS Parquet tables → bare table name (`FROM orders`)
- rag_db tables → `rag.main.<table>`
- local_pg tables → `local_pg.main.<table>`

And mandates plain SQL output (no markdown fences, no explanation, no comments).

The schema string (all tables and columns from all sources) is injected in the user-turn prompt. The last 3 successful conversation turns are included as history context for follow-up questions. Zero-shot — no hardcoded few-shot examples.

---

<a id="q216"></a>
**Q216. How are hallucinated table or column names handled?**

**v1:** No handling. DuckDB throws a `CatalogException`, the `except` block catches it, and `None` is returned.

**v2:** Self-correcting retry loop. The error message is sent back to the LLM:

```
The following SQL you generated failed:
Question: {original_nl_query}
SQL: {bad_sql}
Error: {duckdb_error_message}

Return ONLY the corrected SQL.
```

The model reads `"Table orders_2024 does not exist"` and corrects to `FROM orders`. Up to `max_retries` attempts (default 3).

---

<a id="q217"></a>
**Q217. What happens with semantically valid but semantically wrong SQL?**

Silent failure — the query executes and returns wrong results. No semantic validation layer. The conversation history partially mitigates this: a wrong answer in turn 1 can be corrected in turn 2 if the user notices. Real mitigation would require row count sanity checks, column type validation, or chain-of-thought reasoning before returning SQL.

---

<a id="q218"></a>
**Q218. How is ambiguous natural language handled?**

It isn't — the model guesses. *"Show me recent findings"* typically produces `ORDER BY created_at DESC LIMIT 10` or `WHERE date > NOW() - INTERVAL '7 days'` based on the model's priors. The fix is to inject current date/time and business-specific term definitions into the prompt, or ask a clarifying question before generating SQL.

---

<a id="q219"></a>
**Q219. How does ConversationManager maintain context across follow-ups?**

`history: list[tuple[str, str, QueryResult]]` stores every turn as `(nl_query, sql, result)`. `_history_context(n=3)` serializes the last 3 **successful** turns as:

```
Q: Revenue per customer?
SQL: SELECT c.name, SUM(s.revenue) ...
Result preview: [('Alice', 3000.0), ('Bob', 1400.0)]
```

This block is prepended to every new prompt as "Conversation so far:". Failed turns are recorded in `history` for audit but **excluded** from `_history_context()` so bad SQL examples don't confuse the model.

---

<a id="q220"></a>
**Q220. How is GCS authentication handled in DuckDB?**

HMAC keys (not service account JSON), stored in `.env` as `GCS_HMAC_ID` + `GCS_HMAC_SECRET`. Registered in DuckDB as:

```sql
CREATE OR REPLACE SECRET gcs_secret (
    TYPE gcs,
    KEY_ID  '{GCS_HMAC_ID}',
    SECRET  '{GCS_HMAC_SECRET}'
)
```

DuckDB's `httpfs` extension picks this up for all `gs://` paths automatically.

---

<a id="q221"></a>
**Q221. What did v2 improve over v1?**

| | v1 | v2 |
|---|---|---|
| **Return type** | Raw `Any` (list of tuples or `None` on error) | `QueryResult` with `.columns`, `.rows`, `.success`, `.error`, `.attempts` |
| **SQL errors** | Silent `None`, dead end | Self-correcting retry loop: error fed back to LLM, up to `max_retries` (default 3) |
| **NL cache matching** | Exact string equality | Normalized: lowercase + whitespace-collapsed |
| **Column names** | Anonymous tuples | Populated from `cursor.description` |
| **Provider** | OpenAI only, hardcoded Windows path | `provider="openai"` or `"anthropic"`, caller-supplied env paths |
| **History context** | Includes failed turns | Failed turns excluded from context shown to model |
| **Guardrails** | None | SELECT-only enforcement, result row cap, query timeout |

---

<a id="q222"></a>
**Q222. What guardrails are built into the NLP-to-SQL pipeline?**

Three execution-time guardrails run inside `ConversationManager.run_query()` after each SQL generation step but before DuckDB execution:

| Guardrail | When it fires | Effect |
|---|---|---|
| **SELECT-only** | Generated SQL contains a write/DDL keyword | Treated as an attempt error; self-correcting retry loop can request a new query from the LLM |
| **Result row cap** | Generated SQL has no `LIMIT` clause | `LIMIT N` appended automatically (`max_result_rows=10_000` by default) |
| **Query timeout** | DuckDB query exceeds wall-clock budget | `conn.interrupt()` cancels the running query; error surfaces as a retry-able attempt failure |

All three are applied in sequence on every attempt, so a query that passes the read-only check still gets capped and timed out.

---

<a id="q223"></a>
**Q223. How does the SELECT-only guardrail work?**

```python
_WRITE_PATTERN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def _check_readonly(sql: str) -> str | None:
    m = _WRITE_PATTERN.search(sql)
    if m:
        return f"Only SELECT queries are permitted. Detected keyword: {m.group(0).upper()}"
    return None
```

`run_query()` calls `_check_readonly(sql)` after stripping markdown fences. If it returns an error string, that string is stored as `last_error` and the attempt `continue`s — exactly like a DuckDB execution error. Because the self-correcting retry loop feeds `last_error` back to the LLM in the correction prompt, the model has a chance to rewrite the query as a SELECT. After `max_retries` failures the error is returned in the final `QueryResult`.

Why regex rather than SQL parsing? A full parser (e.g. `sqlglot`) would be more precise, but it is an optional dependency with its own versioning surface. The word-boundary regex catches the vast majority of LLM-generated write attempts and has zero false positives on SELECT/CTE queries.

---

<a id="q224"></a>
**Q224. How does the result row cap work?**

```python
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

def _apply_row_cap(sql: str, limit: int) -> str:
    if not _LIMIT_PATTERN.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql}\nLIMIT {limit}"
    return sql
```

If the LLM omits a `LIMIT` clause, `_apply_row_cap` appends one before the query reaches DuckDB. The cap defaults to `max_result_rows=10_000` (configurable per `ConversationManager`). If the LLM already includes a `LIMIT`, the original limit is left untouched — the guardrail only adds, never overrides.

The trailing semicolon is stripped before appending because `SELECT * FROM t;\nLIMIT 10` is invalid SQL.

---

<a id="q225"></a>
**Q225. How does the query timeout work?**

```python
def _execute_with_timeout(conn, sql, timeout):
    timed_out = threading.Event()

    def _cancel():
        timed_out.set()
        conn.interrupt()

    timer = threading.Timer(timeout, _cancel)
    timer.start()
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in (cursor.description or [])]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        timer.cancel()
```

`threading.Timer` fires `_cancel()` on a background thread after `timeout` seconds (default `query_timeout=30.0`). `conn.interrupt()` is DuckDB's thread-safe cancellation API — it signals the running query to abort, causing `conn.execute()` to raise an exception containing "Interrupted". `run_query()` catches that exception, labels it as a timeout error, and lets the retry loop handle it:

```python
except Exception as exc:
    err_str = str(exc)
    if "Interrupted" in err_str or "interrupted" in err_str:
        last_error = f"Query timed out after {self.query_timeout}s"
    else:
        last_error = err_str
```

`timer.cancel()` in the `finally` block prevents the timer from firing if the query completes normally, avoiding a spurious interrupt on the next query.

---

## System Design — Schema Discovery

**Q: Why store schema chunks in pgvector + tsvector instead of passing the full schema in every prompt?**

A full schema for 100 tables can easily exceed the LLM's context window and increases cost per query. By embedding schema chunks and retrieving only the top-K relevant tables at query time you keep the prompt small and focused. tsvector provides a keyword fallback for exact column or table name matches that vector similarity might miss.

---

**Q: How does schema retrieval scale to 100+ tables?**

The schema cache holds one chunk per table (columns + descriptions + sample values). At query time, the normalized NL query is embedded and an ANN search returns the top-50 candidate chunks, which are then re-ranked to top-K. Only those K tables are injected into the prompt. A warehouse with 1 000+ tables would need a second-stage retrieval or domain routing layer on top.

---

**Q: How often does the Schema Discovery Service run?**

Event-driven (schema-change hook) with a periodic fallback (e.g. nightly). Any ALTER TABLE, new table creation, or column rename should trigger a re-index of the affected chunks. Stale schema is the most common cause of hallucinated column names.

---

**Q: What metadata is stored alongside each schema embedding?**

Each chunk stores a `db:schema:table:column` path string. This allows ANN results to be filtered by database or schema before being passed to the prompt — critical for multi-tenant isolation.

---

## System Design — Prompt Engineering

**Q: Why the `<thinking>` and `<query>` output format?**

Structured output forces the model to separate reasoning from the SQL statement, which improves quality (the model "thinks before it writes") and makes parsing deterministic — the validation pipeline extracts the SQL block with a simple tag split rather than regex heuristics. The `<thinking>` block is useful for debugging wrong queries.

---

**Q: What goes into the RBAC context in the prompt?**

Two things: (1) an explicit list of permitted tables and columns for the requesting tenant/role, and (2) mandatory WHERE filters (e.g. `region = 'North America'`) that must appear in the generated SQL. These are injected as hard requirements so the LLM cannot generate queries that cross tenant boundaries.

---

**Q: How are business acronyms handled?**

The normalization stage resolves known acronyms to their expanded form before schema retrieval (e.g. "MCR" → "Monthly Conversion Rate"). A static acronym dictionary per tenant/domain is the simplest approach; a retrieval-augmented glossary works for large or changing vocabularies.

---

**Q: Why resolve dates to YYYY-MM-DD in normalization?**

LLMs are inconsistent about date formatting across requests — "last quarter" might produce `Q3 2024`, `'2024-07-01'`, or a relative expression. Normalizing to ISO format eliminates ambiguity and ensures the generated SQL uses a format the database can parse.

---

## System Design — SQL Generation

**Q: Why generate N candidates instead of one?**

A single LLM sample can fail schema validation or produce a suboptimal join order. N candidates with confidence scores let the validation pipeline fall back to the next-best candidate without re-prompting the LLM, saving a full round-trip.

---

**Q: Why Qwen-2.5 as the generation model?**

Qwen-2.5 performs competitively on SQL benchmarks (Spider, BIRD) while running locally — no API cost, no data leaving the environment. For a multi-tenant analytical system this matters because queries may touch sensitive commercial data. Swap to a larger model (Qwen-2.5-72B or GPT-4o) if accuracy requirements increase.

---

**Q: Are local LLMs trained to generate SQL from natural language?**

Yes — and SQL is one of the strongest areas for local models. Most popular local LLMs (Llama 3, Qwen 2.5 Coder, Mistral, DeepSeek-Coder) have seen large volumes of SQL during pre-training: Stack Overflow answers, GitHub repositories, Kaggle notebooks, database documentation, and academic datasets. Several standard NL→SQL benchmarks (Spider, WikiSQL, BIRD) have also been used in instruction fine-tuning. Qwen 2.5 Coder in particular is explicitly fine-tuned on code generation tasks including SQL.

**What works well:**
- Simple `SELECT … WHERE … GROUP BY` queries reliably generated from a brief schema description.
- Multi-table joins when the schema context names the join key clearly.
- Common aggregations (`COUNT`, `SUM`, `AVG`, `MAX/MIN`, window functions in larger models).
- Self-correction: models read their own SQL errors and fix them on retry.

**Where they struggle:**
- Dialect-specific syntax — DuckDB's `QUALIFY`, `::` casting, `LIST_AGG`, and `PIVOT` are less common in training data than standard SQL.
- Complex date arithmetic — "last rolling 12 weeks" or fiscal-quarter logic is often wrong on the first attempt.
- Schema hallucination — when schema context is missing or ambiguous, models invent plausible-sounding column names that don't exist.
- Multi-step reasoning — queries requiring a CTE to pre-aggregate before joining tend to degrade in quality as model size decreases.

**Practical implication for this system:** The schema text injected into every prompt is the single most important input. A correct, complete schema description recovers most of the failures listed above. The self-correcting retry loop (up to 3 attempts with the DuckDB error fed back) handles the rest. A 7B model with good schema context outperforms a 70B model with a vague schema description.

---

## System Design — SQL Validation

**Q: Why sqlglot for validation instead of a dry-run EXPLAIN?**

`EXPLAIN` requires a live database connection and leaks query patterns to the query planner. sqlglot validates syntax and schema entirely in-process with no network round-trip. It also provides structured AST inspection for detecting DDL nodes (DROP, DELETE, etc.) more reliably than regex.

---

**Q: What counts as a "hard" vs "recoverable" validation error?**

| Error | Type | Reason |
|-------|------|--------|
| Disallowed keyword (INSERT, DROP) | Hard | Re-prompting is likely to reproduce it |
| RBAC policy violation | Hard | Re-prompting without changing permissions won't help |
| Unknown column name | Recoverable | Error message names the column so the LLM can correct |
| Syntax error | Recoverable | Malformed SQL the LLM can fix |
| Schema type mismatch | Recoverable | Correctable with the schema in context |

---

## System Design — Execution & Caching

**Q: Why cursor-based pagination instead of OFFSET/LIMIT?**

OFFSET forces the database to scan and discard the first N rows on every page, degrading linearly with page depth. A cursor (keyset pagination on an ordered unique column) fetches only the next page's rows regardless of depth.

---

**Q: How is multi-tenancy enforced at the execution layer?**

Each tenant gets a dedicated connection pool bound to a read-only database user scoped to their schema or database. A misconfigured query cannot cross tenant data even if RBAC prompt injection fails — the database user simply has no SELECT permission on other tenants' tables.

---

**Q: When should the NL→Results cache be used vs the SQL→Results cache?**

Use **NL→Results** for recurring business questions that always produce the same SQL and data (fixed dashboard metrics). Use **SQL→Results** when different NL phrasings generate the same SQL — catches semantically equivalent questions without re-executing. Both need data-freshness consideration: invalidate on ETL completion or use a short TTL.

---

**Q: How is the schema cache invalidated?**

The Schema Discovery Service writes new embeddings for changed tables and deletes stale chunks by their `db:schema:table` metadata path. No global cache flush — only affected chunks are replaced, so the cache stays warm for unchanged tables.

---

## Graph Query Generation — NL→Cypher (Apache AGE)

**Q: Does NL→Cypher need its own pipeline or can it reuse the SQL pipeline?**

The five-stage structure (discover → prompt → generate → validate → execute) is the same. What changes are the inputs at each stage: graph schema instead of relational schema, Cypher syntax in the prompt, a write-keyword guard instead of sqlglot, and the AGE execution wrapper instead of a plain SQL connection. It is the same pipeline with a different adapter at each stage, not a separate system.

---

**Q: What inputs does the Cypher generation stage need that SQL generation does not?**

| Input | SQL | Cypher |
|-------|-----|--------|
| Table/column definitions | `information_schema` | — |
| Node labels | — | `ag_catalog.ag_label WHERE kind = 'v'` |
| Edge/relationship types | — | `ag_catalog.ag_label WHERE kind = 'e'` |
| Property keys per label | — | Sampled via `MATCH (n:Label) RETURN keys(n)` |
| Graph name | — | Required for `ag_catalog.cypher('graph_name', …)` |
| AS column list | — | Required; must match RETURN clause exactly |

---

**Q: Why does Cypher need a separate `<columns>` output block from the LLM?**

AGE requires an `AS (col1 agtype, col2 agtype, …)` clause whose column count must exactly match the Cypher `RETURN` clause. If the count is off by one, AGE throws `column definition list has too few/many entries`. Rather than parsing the RETURN clause with regex post-hoc, asking the LLM to output the column list explicitly in a `<columns>` tag is simpler and more reliable.

---

**Q: Why can't we use `$1` parameters in Cypher like we do in SQL?**

AGE does not support parameterised Cypher — all values must be inlined into the query string. This means string values from user input must be escaped before insertion: replace `"` with `\"`. The write-keyword guard and label allowlist (see below) are the main defences against injection, since parameterisation is unavailable.

---

**Q: How are node labels and relationship types validated to prevent injection?**

Validate every label and relationship type against an allowlist before interpolating it into Cypher:

```python
_VALID_LABELS = frozenset({
    "Contract", "Party", "Jurisdiction", "Date",
    "LicenseClause", "TerminationClause", "RestrictionClause",
    "IPClause", "LiabilityClause", "Clause",
})
_VALID_REL_TYPES = frozenset({
    "PARTY_TO", "GOVERNED_BY_LAW", "HAS_LICENSE", "HAS_TERMINATION",
    "HAS_RESTRICTION", "HAS_IP_CLAUSE", "HAS_LIABILITY", "HAS_CLAUSE",
})
```

If the LLM generates an unknown label, the validation stage rejects the query and triggers the repair loop with an error message listing the valid labels.

---

**Q: How is graph schema discovered for AGE — it has no `information_schema`?**

Two queries replace `information_schema` introspection:

```sql
-- Node labels
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
  AND kind = 'v'::"char" AND name NOT LIKE '\_ag\_%' ESCAPE '\';

-- Edge types
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
  AND kind = 'e'::"char" AND name NOT LIKE '\_ag\_%' ESCAPE '\';
```

Because AGE properties are schema-less (agtype/JSONB), property keys must be sampled:

```sql
SELECT * FROM ag_catalog.cypher('legal_graph', $$
    MATCH (n:Party) RETURN keys(n) LIMIT 100
$$) AS (k agtype);
```

Run at discovery time per label; deduplicate and store with the schema chunk.

---

**Q: How often should graph schema be re-discovered?**

On the same schedule as relational schema: event-driven on label creation or bulk ingestion, with a periodic fallback. Unlike relational schema, a new vertex or edge does not change the schema — only a new *label* or new *property key* does. Graph schema is more stable than relational schema in practice.

---

**Q: What does the AGE execution wrapper look like?**

```python
async def run_cypher(conn, graph, cypher, columns_str):
    col_names = [c.strip().split(".")[-1] for c in columns_str.split(",")]
    as_clause = ", ".join(f"{c} agtype" for c in col_names)
    sql = f"SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$) AS ({as_clause})"
    rows = await conn.fetch(sql)
    return [
        {col: (row[i].strip('"') if row[i] not in (None, "null") else None)
         for i, col in enumerate(col_names)}
        for row in rows
    ]
```

Each connection in the pool must have AGE loaded: register `init=_init_age_conn` on `asyncpg.create_pool`, where `_init_age_conn` runs `LOAD 'age'` and `SET search_path = ag_catalog, "$user", public`.

---

**Q: Why do agtype values come back as quoted strings like `'"Acme Corp"'`?**

AGE's `agtype` is a custom PostgreSQL type. asyncpg does not have a codec for it, so it falls back to the text representation, which includes surrounding double-quotes for string values. Strip them with `val.strip('"')`. The special value `"null"` (the string) represents a graph null — convert it to Python `None`.

---

**Q: What Cypher constructs should the LLM be instructed to avoid?**

Beyond write operations (CREATE, MERGE, SET, DELETE, REMOVE, DETACH, DROP), also instruct the model to avoid:

| Construct | Reason |
|-----------|--------|
| `MATCH (e)` without a label | Scans all vertex tables — slow on large graphs |
| Variable-length paths `*1..N` with large N and no LIMIT | Can explode exponentially |
| `CALL` procedures | Not available in AGE's Cypher subset |
| `null` property values in MERGE | AGE rejects null in property maps |
| `LIKE` for string matching | Use `toLower(n.prop) CONTAINS '...'` instead |

---

**Q: Are local LLMs trained to generate Apache AGE Cypher from natural language?**

Poorly, and unreliably — this is one of the hardest NL→query tasks for local models. Two compounding problems make it significantly harder than NL→SQL:

**Problem 1 — Training data scarcity**

openCypher (Neo4j's dialect) has moderate representation in pre-training data: Neo4j documentation, GitHub repos, and some Stack Overflow answers. Apache AGE has almost none — it is a PostgreSQL extension with a small user base, and the AGE-specific query format is essentially invisible in public training corpora. Models have no exposure to:
- The `ag_catalog.cypher('graph', $$ … $$) AS (col agtype, …)` execution wrapper
- The `agtype` column type and why it requires the `AS` clause
- The prohibition on `$1` parameterisation inside Cypher strings
- AGE-specific `OPTIONAL MATCH` requirements (e.g. `WHERE property IS NULL` patterns that work in Neo4j fail in AGE)

**Problem 2 — Non-standard output format required**

The `<cypher>` + `<columns>` dual-block output format used by this system is entirely custom. The `<columns>` block must list every value in the RETURN clause, comma-separated, with the correct count — a mismatch by even one entry causes AGE to throw `column definition list has too few/many entries`. No model has seen this format in training; it relies entirely on prompt engineering and often fails on the first attempt.

**What happens in practice with a capable model (Qwen 2.5 Coder 14B+, Llama 3.1 70B, frontier APIs):**
- Syntactically plausible openCypher MATCH queries: generated reliably.
- Correct node labels and relationship types when schema context is explicit: works most of the time.
- `LIMIT` clause: frequently omitted without explicit instruction.
- `toLower(n.prop) CONTAINS '...'` vs `LIKE`: models default to `LIKE` which AGE does not support.
- `<columns>` count matching RETURN clause: fails ~20–30% of the time — the most common runtime error.
- Multi-hop paths requiring two or more relationship hops: degrade significantly in quality on smaller models.

**Why this system avoids the LLM for Cypher generation:**

The rule-based pipeline (`IntentParser` + `QUERY_CAPABILITIES` + builder functions) covers the 23 most common legal KG intents with zero LLM calls, zero hallucination risk, and deterministic output. The LLM is only proposed as a fallback (Gap 7 in `nl2sql/docs/SYSTEM_DESIGN.md`) for the catch-all case — questions that don't match any known intent pattern — where the rule-based system would return a useless generic result anyway. In that fallback scenario, an imperfect LLM-generated Cypher (with validation) is strictly better than the catch-all `list_contracts` template.

**Bottom line:** Use a rule-based pipeline as the primary path for well-defined intents. Reserve LLM-based Cypher generation for the long tail of free-form questions, with strict post-generation validation (write-keyword guard + RETURN↔columns parity) before execution.
