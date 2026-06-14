# NL-to-Query — System Design

Natural language to SQL (relational) and Cypher (knowledge graph) over multi-tenant PostgreSQL, DuckDB, and Apache AGE.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [LLM Model](#2-llm-model)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Caching Strategy](#4-caching-strategy)
5. [Schema Discovery Service](#5-schema-discovery-service)
6. [Prompt Generation Pipeline](#6-prompt-generation-pipeline)
7. [SQL Generation Pipeline](#7-sql-generation-pipeline)
8. [SQL Validation Pipeline](#8-sql-validation-pipeline)
9. [SQL Executor Pipeline](#9-sql-executor-pipeline)
10. [SQL Best Practices](#10-sql-best-practices-prompt-guardrails)
11. [Graph Query Generation — NL→Cypher (Apache AGE)](#11-graph-query-generation--nlcypher-apache-age)
12. [Sample LLM Prompts](#12-sample-llm-prompts)
13. [Implementation Gaps & What Should Be Built](#13-implementation-gaps--what-should-be-built)

---

## Implementation Status

Legend: ✅ Built · 🔶 Partially built · ❌ Not built · 📋 Planned (design only)

| Section | Feature | Status | Notes |
|---------|---------|--------|-------|
| §3 | Pipeline overview | 🔶 | Single-prompt path built; N-candidate scoring not built |
| §4 | NL cache (in-memory LRU) | ✅ | `OrderedDict`, keyed on normalized NL string |
| §4 | SQL hash cache (in-memory LRU) | ✅ | `OrderedDict`, keyed on MD5 of SQL string |
| §4 | Semantic NL→SQL cache (pgvector) | ❌ | Design target; not implemented |
| §4 | NL→Results cache | ❌ | Design target; not implemented |
| §5 | Schema discovery — startup scan | ✅ | `UnifiedDataSource` reads `information_schema` at init |
| §5 | Schema Discovery Service — background / event-driven | ❌ | Design target; not implemented |
| §5 | Schema cache with pgvector embedding | ❌ | Design target; not implemented |
| §6 | Prompt normalization (whitespace, case) | ✅ | `_normalize_nl()` in `ConversationManager` |
| §6 | Date normalization | ❌ | Not implemented |
| §6 | RBAC constraints in prompt | ❌ | Design target; not implemented |
| §6 | `<thinking>` + `<query>` output format | ✅ | `_parse_tagged_output()` extracts both; falls back to raw SQL when local model ignores tags |
| §6 | Conversation history context | ✅ | Last 3 successful turns injected into user prompt |
| §7 | N-candidate generation + confidence scoring | ❌ | Design target; not implemented |
| §7 | Single-generation + self-correcting retry (≤3) | ✅ | `ConversationManager.run_query()` |
| §8 | Static guardrail — regex read-only check | ✅ | `_check_readonly()` blocks DDL/DML keywords |
| §8 | Static guardrail — row cap | ✅ | `_apply_row_cap()` appends `LIMIT` if absent |
| §8 | Static guardrail — query timeout | ✅ | `threading.Timer` + `conn.interrupt()` |
| §8 | Schema validation — SQLGlot AST | 🔶 | `_validate_sql()` syntax-checks all SQL; column validation requires `schema_dict` from caller. Full catalog-qualified name validation (`rag.main.documents`) deferred — SQLGlot doesn't handle DuckDB multi-catalog naming cleanly |
| §8 | RBAC policy check | ❌ | Design target; not implemented |
| §8 | Repair loop (re-prompt with error) | ✅ | `_build_correction_prompt()`, up to 3 attempts |
| §9 | SQL execution (DuckDB + PostgreSQL scanner) | ✅ | `_execute_with_timeout()` |
| §9 | Cursor-based pagination | ❌ | Row cap (`LIMIT`) only; no cursor |
| §9 | Output adapters (CSV / charts / images) | ❌ | Not implemented |
| §9 | Structured observability logs | ✅ | `_emit_event()` emits JSON per `run_query()` call: `latency_ms`, `cache_tier`, `attempts`, `error` |
| §9 | DB Index Updater Service | ❌ | Design target; not implemented |
| §10 | SQL best practices in system prompt | ✅ | Enforced via `_SYSTEM_PROMPT` |
| §11 | NL→Cypher — rule-based (`IntentParser`) | ✅ | `kg/legal/retrieval/nl2cypher.py` |
| §11 | NL→Cypher — LLM-based (free-form) | ❌ | Design target; not implemented |
| — | Agent orchestration — single-prompt v1 | ✅ | `nl2sql/nlp_sql_postgres_v2.py` |
| — | Agent orchestration — tool-calling v2 | ✅ | `nl2sql/sql_discovery.py` |
| — | Query router (SQL vs Cypher target selection) | ❌ | Deferred — medium effort; design in Gap 6 |
| — | LLM NL→Cypher fallback (free-form) | ❌ | Deferred — prompts designed in §11/Gap 7; medium effort |
| — | Session persistence (`HistoryStore`) | ✅ | asyncpg-backed `conversation_history` table |

---

## 1. Requirements

### High Level

- Multi-tenant database
- Natural language to SQL generation and execution
- PostgreSQL and DuckDB as initial targets
- Analytical / OLAP workloads

**Example queries:**
- "How many users bought Product X from Region Y?"
- "Total sales for the last quarter"
- "Sales dipped year on year for Q4 for Product X in Region Y — is it related to low inventory, shipment delays, or price increases?"
- Queries can include business acronyms (e.g. "MCR Q4 sales")

### Low Level

| Constraint | Value |
|------------|-------|
| Target latency | 5s – 10s end-to-end |
| Max tables per DB | ~100 (average ~8 queried per request) |
| Query timeouts | Required on all queries |
| Pagination | Always paginated (cursor-based or offset + limit) |

---

## 2. LLM Model

**Current decision:** `qwen-2.5-coder:7b`

> TODO: evaluate other candidates

---

## 3. Pipeline Overview

```
User NL query
      │
      ▼
┌─────────────────────────────────┐
│  Prompt Generation Pipeline     │
│  1. Normalize query             │
│  2. Retrieve schema context     │──→ Schema Cache (pgvector + tsvector)
│  3. Inject RBAC + guardrails    │
│  4. Assemble structured prompt  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SQL Generation Pipeline        │
│  Qwen-2.5 → N candidates        │
│  <thinking> + <query> format    │
│  Score candidates (1–10)        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SQL Validation Pipeline        │
│  1. Static guardrails           │
│  2. sqlglot syntax + schema     │
│  3. RBAC policy check           │
│  └── fail → LLM repair loop     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SQL Execution Pipeline         │
│  Route → connection pool        │
│  Timeout + cursor pagination    │
│  Output: CSV / grid / chart     │
└─────────────────────────────────┘

Background:
  Schema Discovery Service  ──→  Schema Cache
  DB Index Updater          ──→  query frequency tracking
```

Each pipeline can have one or more stages. The system is composed of:

1. **Schema Discovery Service** — background service, event-driven
2. **Prompt Generation Pipeline** — normalize query, assemble context
3. **SQL Generation Pipeline** — generate N candidate SQL queries
4. **SQL Validation Pipeline** — static guards, schema checks, RBAC
5. **SQL Execution Pipeline** — route, execute, paginate, return results
6. **DB Index Updater Service** — update indexes for frequent queries

> **What's built (2026-05-17):** The single-prompt path is fully implemented in `nl2sql/nlp_sql_postgres_v2.py` — schema is pre-built at startup, one SQL generation per attempt, self-correcting retry loop (up to 3), three regex-based guardrails, and two in-memory LRU caches. N-candidate generation, confidence scoring, RBAC, SQLGlot validation, pgvector caching, and the DB Index Updater are design targets only.

---

## 4. Caching Strategy

| Cache | Key | Populated by | Use case |
|-------|-----|--------------|----------|
| Schema cache | NL query embedding → top-K tables/columns | Schema Discovery Service | Avoid full schema scan per query |
| NL → SQL | Normalized NL query | SQL Generation Pipeline | Skip generation for repeated questions |
| NL → Results | Normalized NL query | SQL Execution Pipeline | Recurring analytics / fixed dashboards |
| SQL → Results | SQL string hash | SQL Execution Pipeline | Same SQL reached via different NL phrasings |

All caches stored in pgvector (semantic ANN) + tsvector (keyword fallback).

> **What's built:** Two in-memory `OrderedDict` LRU caches per `ConversationManager` instance (configurable `cache_size`, default 20):
> - **NL cache** — key: `" ".join(nl.lower().split())` (normalized whitespace + case). Hit: return cached `QueryResult`, zero LLM calls.
> - **SQL hash cache** — key: MD5 of the SQL string after guardrails. Hit: return cached rows, skip DuckDB execution.
>
> Both caches are warmed from `HistoryStore` on session resume. pgvector-based semantic caching and the NL→Results cache are not implemented.

---

## 5. Schema Discovery Service

Runs in the background — triggered periodically or on schema-change events.

### Process

1. Scan tables and generate schema using SQLAlchemy
2. Emit JSON schema chunks (one per table/group)
3. Embed each chunk with pgvector and store in `embedding` column
4. Generate `tsvector` and store in `content_tsv` column
5. Store metadata path `<db_name>:<schema_name>:<table_name>:<column_name>` for ANN lookups

### Schema Chunk Format

```json
{
  "database_name": "Ariel_Inc_Products",
  "schema_name": "Products_schema",
  "tables": [
    {
      "table_name": "Products",
      "columns": [
        {
          "column_name": "PRODUCT_ID",
          "data_type": "KEY",
          "description": "Unique identifier for each Product",
          "sample_values": [1, 2, 3]
        },
        {
          "column_name": "PRODUCT_CATEGORY",
          "data_type": "INT",
          "description": "Product category key",
          "sample_values": [10, 20]
        }
      ]
    },
    {
      "table_name": "Orders",
      "columns": [
        {
          "column_name": "ORDER_ID",
          "data_type": "KEY",
          "description": "Unique identifier for each order",
          "sample_values": [15, 25, 35]
        },
        {
          "column_name": "PRODUCT",
          "data_type": "INT",
          "description": "Product key for the order",
          "sample_values": [1, 2]
        }
      ]
    }
  ]
}
```

### Schema Retrieval

- Use NL query embedding to do ANN search → top-50 candidate schema chunks
- **Open question:** Follow up ANN search with a reranker? If the generated SQL is bad, fall back to the next-best set of columns?
- **Scalability question:** In a large warehouse with thousands of tables, how does this scale?

> **What's built:** `UnifiedDataSource` runs a full `information_schema` scan at startup and serialises the entire schema into a plain-text string (`schema_text`). That string is embedded verbatim into every prompt — no embedding, no ANN search, no per-query retrieval. Works well for small schemas (< ~30 tables); will hit context-window limits on large warehouses. The background Schema Discovery Service and pgvector-based retrieval are not implemented.

---

## 6. Prompt Generation Pipeline

### Stage 1 — Normalization

- Remove leading/trailing whitespace and extraneous characters (emojis, etc.), retain case
- Resolve natural-language dates into `YYYY-MM-DD` format

Example prompt instructions injected downstream:

```
Use table and column names exactly as provided in the schema context. Do not change their case.
When you use dates in SQL, always format them as 'YYYY-MM-DD'.
```

### Stage 2 — Context Assembly

Assemble the prompt from the following parts:

| Part | Description |
|------|-------------|
| **System role** | What the model is, what it must and must not do |
| **Schema context** | Retrieved from NL→schema cache or schema vector DB |
| **RBAC constraints** | Injected as hard SQL requirements, e.g. `"Filter by region = North America."` |
| **Static guardrails** | Read-only enforcement, PII column restrictions, complexity limits |

**Static guardrail examples:**
- Do not access PII columns (e.g. `email`, `phone_number`) unless explicitly permitted
- Never generate `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`
- Always include `LIMIT`
- Max 5 levels of nested subqueries
- Max query size: 1,000 characters

### Stage 3 — Output Format

The model is instructed to respond strictly in this format:

```
<thinking>
[Explain your reasoning briefly. Clarify how you interpreted the user question,
which tables you chose, how you applied filters, and how you handled any dates.]
</thinking>
<query>
[Write a single valid SQL SELECT statement here. No backticks, no explanation, no comments.]
</query>
Do NOT include anything outside these tags.
Do NOT include natural language outside <thinking> and <query>.
```

> **Open questions:**
> - Should we enforce the `<thinking>` / `<query>` output format?
> - What do we do with the reasoning steps — feed them into downstream pipeline stages?

> **What's built:** The `<thinking>`+`<query>` format is **not implemented**. The actual system prompt instructs the model to return raw SQL with no tags:
>
> ```
> You are an expert SQL assistant working with DuckDB.
>
> Table naming rules (IMPORTANT — always follow these):
> - GCS Parquet tables  -> bare table name,          e.g.  FROM orders
> - rag_db tables       -> rag.main.<table>,          e.g.  FROM rag.main.documents
> - local_pg tables     -> local_pg.main.<table>,     e.g.  FROM local_pg.main.baby_names
>
> Return ONLY plain SQL. No Markdown fences, no explanation, no comments.
> ```
>
> The user turn (first attempt) is: `Schema:\n{schema_text}\n\nConversation so far:\n{history}\n\nQuestion: {nl}`.  
> On retry: `Schema:\n{schema_text}\n\nThe following SQL you generated failed:\nQuestion: …\nSQL: …\nError: …\n\nReturn ONLY the corrected SQL.`  
> RBAC constraints and static guardrails are **not injected into the prompt** — they are enforced post-generation via code (`_check_readonly()`, `_apply_row_cap()`).

### Stage 4 — Cache Update

Write the resolved NL → schema mapping to the NL→schema cache.

---

## 7. SQL Generation Pipeline

1. Check the SQL→Results cache — on hit, skip all subsequent steps
2. Feed the structured prompt from the Prompt Generation Pipeline to the model with **sampling enabled** to generate **N candidate queries**
3. Each candidate has the format: `<thinking>...</thinking> <query>SELECT ...</query>`
4. Rank candidates by attaching a confidence score (1–10) to each

> **Open questions:**
> - How is the confidence scoring implemented?
> - Where are the ranked candidates stored — in memory, or persisted to Redis so that if the top-ranked query fails validation, the next-best can be used as fallback?

> **What's built:** N-candidate generation and confidence scoring are **not implemented**. The actual flow is:
> 1. Check SQL hash cache (MD5 of SQL) — hit → return cached rows
> 2. Call `agent.run(prompt)` — single generation, temperature default
> 3. Strip Markdown fences (`strip_sql_fences()`)
> 4. Run guardrails → execute via DuckDB
> 5. On exception: build correction prompt → retry (up to `max_retries=3`)
>
> See `ConversationManager.run_query()` in `nl2sql/nlp_sql_postgres_v2.py`.

---

## 8. SQL Validation Pipeline

Runs after SQL Generation. On failure for recoverable errors, triggers an LLM repair loop with the generation pipeline.

### Check 1 — Static Guardrails

Reject any query containing:

- DDL/DML keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, etc.
- Multiple statements separated by `;`
- Suspicious constructs: `--` comments, `/* */` blocks, `xp_` stored procedures

Enforce complexity limits:
- Max query length (tokens / characters)
- Max depth of nested subqueries

**Failure response:**

```json
{
  "error_type": "disallowed_keyword",
  "details": "Query contains UPDATE, only SELECT allowed."
}
```

### Check 2 — Schema Validation (SQLGlot) 📋 Not built

SQLGlot validates syntax, schema conformance, and read-only safety.

**Syntax check:**

```python
import sqlglot

def is_syntax_valid(llm_sql: str) -> bool:
    try:
        # duckdb dialect required for QUALIFY, :: casting, etc.
        sqlglot.parse_one(llm_sql, read="duckdb")
        return True
    except sqlglot.errors.ParseError:
        return False
```

**Schema conformance check:**

```python
from sqlglot.optimizer import optimize

schema = {"sales": {"date": "DATE", "amount": "DOUBLE", "region": "TEXT"}}

def validate_against_schema(llm_sql: str, schema: dict) -> bool:
    try:
        # qualify_columns resolves and validates all column references
        optimize(sqlglot.parse_one(llm_sql, read="duckdb"), schema=schema)
        return True
    except Exception:
        return False
```

**Read-only AST check:**

```python
from sqlglot import exp

def is_read_only(llm_sql: str) -> bool:
    expression = sqlglot.parse_one(llm_sql, read="duckdb")
    if not isinstance(expression, (exp.Select, exp.Union)):
        return False
    forbidden = (exp.Drop, exp.Delete, exp.Insert, exp.Update, exp.Alter)
    if any(expression.find(node) for node in forbidden):
        return False
    return True
```

**Failure response structure:**

> TODO: finalise error schema

```json
{
  "error_type": "syntax_error | schema_failure | safety_failure",
  "details": "Column 'product_sku' not found in table 'Products'."
}
```

### Check 3 — RBAC Policy 📋 Not built

```json
{
  "error_type": "policy_violation",
  "details": "Access to column 'email' is not permitted for this role."
}
```

### Repair Loop ✅ Built

For recoverable errors, re-invoke the SQL Generation Pipeline with:

1. The original prompt
2. The failing SQL
3. The normalized error message

If auto-repair fails after **N attempts**, return a graceful error to the user.

For hard errors (non-recoverable), fail the NL query immediately.

---

## 9. SQL Executor Pipeline

### Router

Determine which database to connect to based on tenant / query context.

### Connection Pooling

- Pool per tenant / database
- Read-only credentials only
- Configurable max connections

### Execution

| Concern | Approach |
|---------|----------|
| Query timeout | Configurable per query |
| Cancellation | Cancellation points at execution boundaries |
| Retryable errors | Detect and retry transient failures |
| Pagination | Cursor-based or offset + limit |

### Output Adapters

Results can be returned as:
- CSV
- Grid / table
- Charts
- Images

### Observability

- Scan and emit observability logs for every execution

### Index Feedback

- Track frequently executed queries and feed patterns to the DB Index Updater Service

---

## 10. SQL Best Practices (prompt guardrails)

Enforced via system prompt instructions injected at prompt assembly time.

| Rule | Reason |
|------|--------|
| `SELECT` specific columns — never `SELECT *` | Reduces I/O and memory |
| Apply `WHERE` before `JOIN` / `GROUP BY` | Narrows dataset early |
| `UNION ALL` instead of `UNION` | No de-duplication overhead |
| `EXISTS` instead of `IN` for subqueries | Stops on first match |
| Avoid `LIKE '%value'` | Prevents index use (forces full scan) |
| Covering indexes on `WHERE` + `JOIN` + `ORDER BY` columns | Engine skips table reads |

**Latency targets:** <100ms for simple lookups · <500ms for aggregations · <10s system SLA

---

## 11. Graph Query Generation — NL→Cypher (Apache AGE)

The same five-stage pipeline (discover → prompt → generate → validate → execute) applies to knowledge graph queries. The inputs, prompt, validation rules, and execution wrapper all differ from the SQL path.

### How it fits

```
User NL query
      │
      ├──[target = relational]──→ SQL pipeline (sections 5–9)
      │
      └──[target = graph]──→ Cypher pipeline (this section)
                                    │
                          ┌─────────▼──────────┐
                          │  Graph Schema       │
                          │  Discovery          │
                          │  ag_catalog tables  │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Prompt Assembly    │
                          │  labels + rel types │
                          │  + sampled props    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Cypher Generation  │
                          │  Qwen-2.5           │
                          │  MATCH…RETURN + AS  │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Cypher Validation  │
                          │  write-keyword guard│
                          │  RETURN↔AS parity   │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  AGE Execution      │
                          │  ag_catalog.cypher()│
                          │  agtype stripping   │
                          └────────────────────┘
```

---

### Graph Schema Discovery (inputs to the prompt)

AGE stores graph metadata in `ag_catalog` tables. Properties are schema-less (agtype/JSONB), so schema context must be built from two sources:

**1. Structural metadata — from catalog tables**

```sql
-- Node labels
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
  AND kind = 'v'::"char"
  AND name NOT LIKE '\_ag\_%' ESCAPE '\';

-- Edge / relationship types
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
  AND kind = 'e'::"char"
  AND name NOT LIKE '\_ag\_%' ESCAPE '\';
```

**2. Property keys — sampled from live data (no fixed schema)**

```sql
-- Sample property keys for a label
SELECT * FROM ag_catalog.cypher('legal_graph', $$
    MATCH (n:Party) RETURN keys(n) LIMIT 100
$$) AS (k agtype);
```

Run this for each label at discovery time; deduplicate and store as part of the schema chunk.

**Schema chunk format for a graph:**

```json
{
  "graph": "legal_graph",
  "node_labels": [
    {
      "label": "Party",
      "property_keys": ["uuid", "name", "document_id", "label"]
    },
    {
      "label": "Contract",
      "property_keys": ["uuid", "name", "document_id", "label"]
    }
  ],
  "edge_types": [
    "PARTY_TO", "GOVERNED_BY_LAW", "HAS_LICENSE",
    "HAS_TERMINATION", "HAS_RESTRICTION", "HAS_IP_CLAUSE",
    "HAS_LIABILITY", "HAS_CLAUSE"
  ]
}
```

---

### Prompt differences vs SQL

| Dimension | SQL prompt | Cypher prompt |
|-----------|-----------|---------------|
| Schema context | Table names + column names + types | Node labels + property keys + edge types |
| Query syntax | `SELECT … FROM … WHERE … GROUP BY` | `MATCH (n:Label)-[:REL]->(m) WHERE … RETURN …` |
| Output format | Raw SQL | Cypher body **and** a matching `AS` column list |
| Null handling | `IS NULL` / `COALESCE` | `OPTIONAL MATCH`; AGE rejects null property values in MERGE |
| Case sensitivity | Column names case-insensitive (usually) | Label names and property keys are case-sensitive |
| Joins | Explicit JOIN … ON | Implicit via pattern: `(a)-[:REL]->(b)` |

**System prompt additions for Cypher:**

```
You are a Cypher query generator for Apache AGE running on PostgreSQL.

Rules:
- Only generate read-only MATCH … RETURN queries. Never use CREATE, MERGE, SET,
  DELETE, REMOVE, DETACH, DROP, or CALL.
- Use node labels and property keys exactly as listed in the schema context.
  Labels are case-sensitive: Party ≠ party.
- Always include a LIMIT clause.
- Do not reference properties that are not in the provided property_keys list.
- For string comparisons use toLower(n.name) CONTAINS '...' (not LIKE).

Output format — respond with exactly two blocks:
<cypher>
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
WHERE toLower(p.name) CONTAINS 'acme'
RETURN p.name, c.name
LIMIT 20
</cypher>
<columns>p.name, c.name</columns>
```

The `<columns>` block lists every value in the RETURN clause, comma-separated. This is used to build the required `AS (col agtype, …)` clause for AGE execution.

---

### Cypher Validation

**1. Write-keyword guard**

```python
_CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH|DROP|CALL)\b",
    re.IGNORECASE,
)

def is_cypher_readonly(cypher: str) -> bool:
    return not _CYPHER_WRITE.search(cypher)
```

**2. RETURN ↔ AS column parity**

The number of names in `<columns>` must equal the number of items in the Cypher RETURN clause. Mismatch causes AGE to throw `column definition list has too few/many entries`.

```python
def build_as_clause(columns_str: str) -> str:
    cols = [c.strip().split(".")[-1] for c in columns_str.split(",")]
    return ", ".join(f"{c} agtype" for c in cols)
```

**3. No parameterised values**

AGE does not support `$1` parameters inside Cypher. All values are inlined. String values must be escaped: `value.replace('"', '\\"')`.

---

### AGE Execution

**Connection requirements (per connection in the pool):**

```python
async def _init_age_conn(conn: asyncpg.Connection) -> None:
    await conn.execute("LOAD 'age'")
    await conn.execute("SET search_path = ag_catalog, \"$user\", public")
```

Register this as `init=` on `asyncpg.create_pool` so every connection is AGE-ready.

**Execution wrapper:**

```python
async def run_cypher(
    conn: asyncpg.Connection,
    graph: str,
    cypher: str,
    columns_str: str,
) -> list[dict]:
    as_clause = build_as_clause(columns_str)
    sql = (
        f"SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$)"
        f" AS ({as_clause})"
    )
    rows = await conn.fetch(sql)
    col_names = [c.strip().split(".")[-1] for c in columns_str.split(",")]
    return [
        {col: row[i].strip('"') if row[i] not in (None, "null") else None
         for i, col in enumerate(col_names)}
        for row in rows
    ]
```

**agtype stripping:** asyncpg returns agtype values as quoted strings (`'"Acme Corp"'`). Strip surrounding quotes and convert `"null"` to `None` before returning results.

---

### SQL vs Cypher pipeline comparison

| | SQL (sections 5–9) | Cypher (section 11) |
|--|-------------------|---------------------|
| Schema source | `information_schema.tables/columns` | `ag_catalog.ag_label` + sampled `keys(n)` |
| Schema type | Fixed columns + data types | Labels + property key lists (schema-less) |
| Generation output | `<thinking>` + `<query>` | `<cypher>` + `<columns>` |
| Validation | sqlglot AST + DDL/DML guard | Regex write-keyword guard + RETURN↔AS parity |
| Parameterisation | `$1` placeholders (safe) | Inline string escaping (AGE limitation) |
| Execution | `conn.execute(sql)` | `ag_catalog.cypher('graph', $$ … $$) AS (…)` |
| Connection init | Standard asyncpg pool | `LOAD 'age'` + `SET search_path` on every connection |
| Null handling | NULL / COALESCE | `"null"` string → Python `None` (agtype quirk) |

---

## 12. Sample LLM Prompts

Concrete prompts as they exist in the current implementation. For the planned `<thinking>`+`<query>` format and Cypher prompt design see sections 6 and 11 respectively.

---

### NL→SQL — System Prompt

Sent once per session as the `system` role. Source: `_SYSTEM_PROMPT` in `nl2sql/nlp_sql_postgres_v2.py`.

```
You are an expert SQL assistant working with DuckDB.

Table naming rules (IMPORTANT — always follow these):
- GCS Parquet tables  -> bare table name,          e.g.  FROM orders
- rag_db tables       -> rag.main.<table>,          e.g.  FROM rag.main.documents
- local_pg tables     -> local_pg.main.<table>,     e.g.  FROM local_pg.main.baby_names

Return ONLY plain SQL. No Markdown fences, no explanation, no comments.
```

---

### NL→SQL — First-Attempt User Prompt

Built by `_build_prompt(nl_query)`. History block omitted on the first turn of a session.

```
Schema:
=== GCS Parquet tables (use bare name) ===
Table: orders
  - order_id (VARCHAR)
  - order_date (DATE)
  - customer_id (VARCHAR)
  - amount (DOUBLE)

=== rag_db tables (prefix: rag.main.<table>) ===
Table: rag.main.documents
  - id (UUID)
  - title (TEXT)
  - source (TEXT)
  - created_at (TIMESTAMP WITH TIME ZONE)

Conversation so far:
Q: How many orders are there?
SQL: SELECT COUNT(*) FROM orders LIMIT 10000
Result preview: [(42891,)]

Question: How many orders were placed in Q4?
```

Only the last 3 **successful** turns are included — failed SQL is excluded.

**Expected model output:**

```sql
SELECT COUNT(*) FROM orders
WHERE order_date >= '2024-10-01' AND order_date < '2025-01-01'
LIMIT 10000
```

---

### NL→SQL — Self-Correction Prompt

Built by `_build_correction_prompt(nl_query, bad_sql, error)` on attempt 2 and 3. Error is truncated to 400 characters.

```
Schema:
=== GCS Parquet tables (use bare name) ===
Table: orders
  - order_id (VARCHAR)
  - order_date (DATE)
  - customer_id (VARCHAR)
  - amount (DOUBLE)

The following SQL you generated failed:
Question: How many orders were placed in Q4?
SQL: SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2024-10-01' AND '2024-31-12'
Error: Conversion Error: date field value out of range: "2024-31-12"

Return ONLY the corrected SQL.
```

**Expected model output:**

```sql
SELECT COUNT(*) FROM orders
WHERE order_date >= '2024-10-01' AND order_date < '2025-01-01'
LIMIT 10000
```

---

### NL→SQL — Planned `<thinking>`+`<query>` Format (Gap 2, not yet built)

Target format once reasoning extraction is implemented (see Gap 2 in section 13):

```
<thinking>
The user asks for Q4 orders. Q4 is October–December.
I'll use order_date >= '2024-10-01' AND < '2025-01-01' to avoid
timezone-edge issues with BETWEEN on DATE columns.
The orders table is a GCS Parquet table so I use the bare name.
</thinking>
<query>
SELECT COUNT(*) FROM orders
WHERE order_date >= '2024-10-01' AND order_date < '2025-01-01'
</query>
```

---

### NL→Cypher — System Prompt

For the LLM-based Cypher generation path (Apache AGE). Source: section 11 of this document + `nl2sql/docs/ARCHITECTURE.md`.

> **Note:** The current implementation uses a rule-based `IntentParser` — no LLM is called. This prompt applies to the planned LLM fallback (Gap 7).

```
You are a Cypher query generator for Apache AGE running on PostgreSQL.

Rules:
- Only generate read-only MATCH … RETURN queries. Never use CREATE, MERGE, SET,
  DELETE, REMOVE, DETACH, DROP, or CALL.
- Use node labels and property keys exactly as listed in the schema context.
  Labels are case-sensitive: Party ≠ party.
- Always include a LIMIT clause.
- Do not reference properties that are not in the provided property_keys list.
- For string comparisons use toLower(n.name) CONTAINS '...' (not LIKE).

Output format — respond with exactly two blocks:
<cypher>
MATCH (n:Label)-[:REL]->(m)
WHERE ...
RETURN n.prop, m.prop
LIMIT 20
</cypher>
<columns>n.prop, m.prop</columns>
```

The `<columns>` block must list every value in the RETURN clause, comma-separated.
It is used to build the required `AS (col agtype, …)` clause for `ag_catalog.cypher()`.

---

### NL→Cypher — User Prompt

Graph schema context is discovered from `ag_catalog` at query time (see section 11).

```
Graph: legal_graph

Node labels:
  Party:        [uuid, name, document_id, label]
  Contract:     [uuid, name, document_id, label]
  Clause:       [uuid, text, clause_type, document_id]
  Risk:         [uuid, description, risk_type, severity, document_id]
  Jurisdiction: [uuid, name, document_id]

Relationship types:
  PARTY_TO, GOVERNED_BY_LAW, HAS_LICENSE, HAS_TERMINATION,
  HAS_RESTRICTION, HAS_IP_CLAUSE, HAS_LIABILITY, HAS_CLAUSE,
  INCREASES_RISK_FOR, CAUSES

Question: Which contracts does Acme Corp appear in as a party?
```

**Expected model output:**

```
<cypher>
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
WHERE toLower(p.name) CONTAINS 'acme corp'
RETURN p.name, c.name
LIMIT 20
</cypher>
<columns>p.name, c.name</columns>
```

---

## 13. Implementation Gaps & What Should Be Built

This section documents gaps between the design and the current implementation. Each item includes a rationale for why it matters and a rough sense of effort. **The goal here is to capture intent, not to prescribe timelines.** Build order should be driven by pain felt in practice.

---

### Gap 1 — SQLGlot Schema Validation

**What's missing:** After SQL generation, there is no validation that the generated table and column names actually exist in the schema. The system only checks for DDL/DML keywords (regex). A hallucinated column name (`orders.customer_nme` instead of `orders.customer_name`) will only fail at execution time, burning a DuckDB round-trip and consuming one retry attempt.

**Why it matters:** Catches the most common LLM failure mode — hallucinated identifiers — before any I/O happens. Gives the repair loop a precise, actionable error message ("Column 'customer_nme' not found in table 'orders'") rather than a cryptic DuckDB exception.

**What to build:**
```python
import sqlglot
from sqlglot.optimizer import optimize

def validate_against_schema(sql: str, schema: dict) -> tuple[bool, str]:
    try:
        optimize(sqlglot.parse_one(sql, read="duckdb"), schema=schema)
        return True, ""
    except Exception as e:
        return False, str(e)
```
The `schema` dict is already available from `UnifiedDataSource.schema_text` — it just needs to be kept in structured form alongside the serialised string.

**Effort:** Small. Schema dict is already built; SQLGlot is already a natural dependency.

---

### Gap 2 — `<thinking>` Reasoning Extraction

**What's missing:** The model returns raw SQL. There is no reasoning trace exposed to the user or downstream systems. The `<thinking>` / `<query>` output format is specified in the design but not enforced.

**Why it matters:**
- Users who get a wrong answer have no visibility into why the model made the choices it did.
- The reasoning trace is the most useful input to the repair loop — it tells you *what the model thought it was doing*, not just what SQL it produced.
- Exposing reasoning in the UI ("I interpreted 'last quarter' as 2024-10-01 to 2024-12-31") dramatically increases user trust and debuggability.

**What to build:** Change `_SYSTEM_PROMPT` to enforce the tagged format. Add a `strip_thinking()` parser that extracts the `<query>` block and optionally surfaces `<thinking>` to the caller. `QueryResult` gets an optional `reasoning: str | None` field.

**Effort:** Small — prompt change + string parser. Risk: some models (especially smaller Ollama models) may not reliably honour structured output tags; may need to evaluate.

---

### Gap 3 — Semantic NL Cache (pgvector)

**What's missing:** The NL cache is keyed on exact normalized string match. "How many orders last month?" and "Count of orders in the previous month?" will both miss and trigger separate LLM calls even though they're semantically identical.

**Why it matters:** Recurring analytics questions are rarely phrased identically. In a multi-user environment, the same underlying question arrives with surface variation constantly. A semantic cache with a similarity threshold would collapse these into a single cached result.

**What to build:** On cache miss, embed the normalized NL query and do an ANN lookup in a pgvector table. If cosine similarity > threshold (e.g. 0.92), return the cached result. On execution success, write the NL embedding + `QueryResult` to the table.

**Effort:** Medium. Requires:
- A pgvector table for NL→SQL cache entries
- Embedding call per query (adds ~50–200ms latency on cold path)
- Similarity threshold tuning (too low → wrong cache hits; too high → no benefit)

**Dependency:** Embedding provider must be available at query time (Ollama or OpenAI-compatible).

---

### Gap 4 — N-Candidate Generation + Confidence Scoring

**What's missing:** The system generates one SQL candidate per attempt. The design calls for N candidates with confidence scores, using the highest-scoring valid candidate.

**Why it matters:** SQL generation is non-deterministic. For ambiguous questions, a single sample at temperature=0 locks in whatever the model's mode answer is. Generating 3–5 candidates with temperature > 0 and picking the one that parses + validates + scores highest significantly improves accuracy on hard queries (joins across multiple tables, complex aggregations, date arithmetic).

**What to build:**
1. Call `agent.run()` N times with `temperature > 0` (or use a single call with a prompt asking for N variants).
2. Run each candidate through guardrails + SQLGlot validation.
3. Score valid candidates: a simple heuristic scorer works (prefers candidates with fewer subqueries, present in schema, no `SELECT *`).
4. Execute the top-scoring candidate; fall back to next-best if it fails.

**Effort:** Medium. The retry loop already exists — N-candidate extends it. Main challenge is scoring heuristic design and whether to call the LLM N times (cost) or prompt for N at once (reliability).

---

### Gap 5 — RBAC Policy Check

**What's missing:** No access control. Any user can query any table and column.

**Why it matters:** Critical for multi-tenant deployments. Without RBAC, a user in tenant A can trivially query tenant B's data by crafting a natural-language question that references the right table.

**What to build:** A policy engine that takes (user_role, sql_ast) and checks:
- Table-level access: is this table in the user's allowed set?
- Column-level access: are any PII columns (e.g. `email`, `phone`) referenced by a role that doesn't have clearance?
- Row-level security: inject a mandatory `WHERE tenant_id = {tenant}` predicate if it's missing.

The SQLGlot AST (Gap 1) is the natural input — parse the SQL, walk the AST for table/column references, check against a policy config.

**Effort:** Medium–large. Policy config design is the hardest part; the AST walking is straightforward once SQLGlot is in place.

---

### Gap 6 — Query Router (SQL vs Cypher Target Selection)

**What's missing:** The two pipelines (SQL via DuckDB, Cypher via AGE) are called separately by the caller. There is no unified entry point that inspects the question and routes it to the right backend.

**Why it matters:** From a user's perspective, "Who are the parties to the Acme agreement?" and "How many documents were ingested last week?" are both natural-language questions — the user shouldn't need to know which backend to target.

**What to build:** A lightweight router that classifies the question as `relational` or `graph` before dispatching. Options:
- **Rule-based:** regex patterns on entity/relationship vocabulary (fast, brittle)
- **LLM-based:** single cheap classification call ("Is this a graph traversal question or a tabular aggregation?") before the main pipeline (adds ~200ms latency)
- **Confidence-based:** try both pipelines, return whichever produces a non-empty result (slow, expensive)

A rule-based router using the existing `IntentParser` vocabulary is the natural starting point — if the question matches any KG intent, route to Cypher; otherwise route to SQL.

**Effort:** Small for rule-based. Medium for LLM-based.

---

### Gap 7 — LLM-Based NL→Cypher (Free-Form Graph Queries)

**What's missing:** The current NL→Cypher uses a rule-based `IntentParser` (regex → fixed Cypher templates). Questions that don't match a known intent pattern fall through to `list_contracts` — a generic catch-all. Free-form graph queries ("Find all contracts where the governing law is California and the indemnification clause mentions IP") cannot be handled.

**Why it matters:** The rule-based system covers the top-N most common intents reliably and fast. But it has a hard ceiling — any query requiring multi-hop traversal or combining two or more intent patterns requires a new hand-written template. An LLM-based fallback removes that ceiling.

**What to build:** When `IntentParser` returns `list_contracts` (the catch-all), instead of executing the generic template, fall back to an LLM-based Cypher generator. The system prompt and output format (`<cypher>` + `<columns>`) are fully specified in section 11 of this document. The graph schema context (labels, property keys, edge types) already comes from the AGE catalog queries.

**Effort:** Small for the fallback path (prompts are designed). Effort is in testing — verifying the LLM doesn't generate write Cypher or hallucinate labels.

---

### Gap 8 — Cursor-Based Pagination

**What's missing:** Large result sets are truncated by a hard `LIMIT` row cap (`_apply_row_cap()`, default 10,000 rows). There is no way to page through results beyond that cap.

**Why it matters:** Analytical queries over large corpora legitimately return millions of rows (e.g. "Export all orders from 2023"). A row cap is a blunt instrument — it silently truncates results, which can produce misleading aggregations if the user re-aggregates client-side.

**What to build:** Cursor-based pagination for DuckDB result sets. The `execute()` call already returns a cursor; `fetchmany(page_size)` yields pages. The API response includes a `next_cursor` token. The `QueryResult` dataclass gains `has_more: bool` and `cursor: str | None`.

**Effort:** Small for the DuckDB side. Medium for the API/UI surface.

---

### Gap 9 — Structured Observability

**What's missing:** No structured logging, metrics, or tracing per query execution. There is no way to answer: "What fraction of queries hit the cache?", "What is p95 generation latency?", "Which questions fail most often?"

**Why it matters:** Without observability, tuning the system is guesswork. Cache hit rate tells you whether to invest in semantic caching. Failure patterns tell you which prompt guardrails to tighten. Latency breakdowns tell you where the bottleneck is (embedding vs generation vs execution).

**What to build:** Emit a structured event at the end of every `run_query()` call:
```python
{
  "session_id": "...",
  "nl_query": "...",
  "sql": "...",
  "cached": true,
  "attempts": 1,
  "latency_ms": 312,
  "error": null,
  "cache_tier": "nl" | "sql" | "miss"
}
```
Send to Langfuse (already used in the RAG system), a local log file, or a metrics sink. The `HistoryStore` table already captures most of this — adding `latency_ms` and `cache_tier` columns is the main change.

**Effort:** Small. The data is already available at the end of `run_query()`; it just isn't emitted anywhere structured.

---

### Priority order (suggested)

| Priority | Gap | Why first |
|----------|-----|-----------|
| 1 | Gap 2 — `<thinking>` reasoning | Highest user-facing value, lowest risk, pure prompt change |
| 2 | Gap 1 — SQLGlot validation | Improves repair loop quality; small effort |
| 3 | Gap 9 — Observability | Needed to make data-driven decisions on everything else |
| 4 | Gap 7 — LLM-based NL→Cypher fallback | Prompts already designed; removes hard ceiling on KG queries |
| 5 | Gap 6 — Query router | Enables unified entry point; rule-based version is fast |
| 6 | Gap 3 — Semantic NL cache | Medium effort; highest payoff in multi-user environments |
| 7 | Gap 4 — N-candidate generation | Accuracy improvement; evaluate after observability shows failure rate |
| 8 | Gap 8 — Cursor pagination | Needed for production data volumes |
| 9 | Gap 5 — RBAC | Required before any multi-tenant deployment |
