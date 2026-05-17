# NL-to-SQL System Design

Natural language to SQL over multi-tenant PostgreSQL and DuckDB databases.

---

## Requirements

### Functional
- Accept natural language queries (including business acronyms like "MCR Q4 sales")
- Generate and execute SQL against PostgreSQL or DuckDB
- Support analytical queries: aggregations, time-series, regional breakdowns, inventory correlations
- Multi-tenant: isolate schema context and access permissions per tenant

### Non-Functional
- Latency: 5–10s end-to-end
- DB scale: up to 100 tables, ~8 columns avg
- All queries: paginated (cursor-based), timeout-enforced
- Read-only: no DDL/DML permitted

---

## Architecture Overview

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

---

## Caching

| Cache | Key | Populated by | Use case |
|-------|-----|--------------|----------|
| Schema cache | NL query embedding → top-K tables/columns | Schema Discovery Service | Avoid full schema scan on every query |
| NL → SQL | Normalized NL query | SQL Generation Pipeline | Skip generation for repeated questions |
| NL → Results | Normalized NL query | SQL Execution Pipeline | Recurring analytics / dashboards |
| SQL → Results | SQL string hash | SQL Execution Pipeline | Same SQL generated for different questions |

All caches stored in pgvector (semantic) + tsvector (keyword) for hybrid retrieval.

---

## 1. Schema Discovery Service

Runs periodically or on schema-change events.

**Output:** JSON schema chunks per table:

```json
{
  "database_name": "Ariel_Inc_Products",
  "schema_name": "Products_schema",
  "tables": [
    {
      "table_name": "Products",
      "columns": [
        { "column_name": "PRODUCT_ID", "data_type": "KEY", "description": "Unique product identifier", "sample_values": [1, 2, 3] },
        { "column_name": "PRODUCT_CATEGORY", "data_type": "INT", "description": "Category key", "sample_values": [10, 20] }
      ]
    }
  ]
}
```

**Storage per chunk:**
- `embedding vector` — pgvector ANN search (top-50 candidate retrieval)
- `content_tsv tsvector` — keyword fallback
- `metadata text` — `db:schema:table:column` path for filtering

---

## 2. Prompt Generation Pipeline

### Normalization
- Strip whitespace, emojis, control characters; preserve case
- Resolve relative dates → `YYYY-MM-DD`

### Schema Context Retrieval
- Embed normalized query → ANN search against schema cache → top-K tables/columns
- Falls back to keyword search on cache miss

### Prompt Assembly

```
System role:     What the model is, hard rules (no INSERT/UPDATE/DELETE/DDL,
                 max 5 nested subqueries, max 1 000 token query)
Schema context:  Top-K tables + column descriptions from schema cache
RBAC context:    Permitted tables/columns for this tenant and role
                 Mandatory filters: e.g. "Filter by region = North America"
Static guards:   No PII columns (email, phone_number) unless explicitly permitted
Output format:   <thinking>...</thinking> <query>...</query>
```

### Output Format

```xml
<thinking>
Brief reasoning: which tables, how filters were applied, date interpretation.
</thinking>
<query>
SELECT ... single valid SQL SELECT statement, no backticks, no comments
</query>
```

Updates NL→schema cache after retrieval.

---

## 3. SQL Generation Pipeline

- Feeds the structured prompt to **Qwen-2.5** with sampling enabled
- Generates **N candidates** (each: `<thinking>` + `<query>` block)
- Scores each candidate 1–10 on confidence
- Passes top-ranked candidate to validation; retains others for fallback

---

## 4. SQL Validation Pipeline

Three sequential checks. On failure, returns a structured error to the generation pipeline for LLM-assisted repair (up to N attempts before returning a user-facing error).

### Static Guardrails
Reject if query contains DDL/DML keywords, multiple `;`-separated statements, `--` or `/* */` comments, or exceeds complexity limits (query length, nested subquery depth).

```json
{ "error_type": "disallowed_keyword", "details": "Query contains UPDATE, only SELECT allowed." }
```

### Schema Validation (sqlglot)
```python
import sqlglot
from sqlglot.optimizer import optimize

def validate(sql: str, schema: dict) -> bool:
    tree = sqlglot.parse_one(sql, read="duckdb")
    optimize(tree, schema=schema)   # raises if columns don't exist in schema
    return isinstance(tree, (sqlglot.exp.Select, sqlglot.exp.Union))
```

### RBAC Policy Check
```json
{ "error_type": "policy_violation", "details": "Access to column 'email' is not permitted for this role." }
```

### LLM Repair Loop
On recoverable failure, re-prompt the LLM with: original prompt + failing SQL + normalized error. Retry up to N times; on exhaustion, surface a graceful error to the user.

---

## 5. SQL Execution Pipeline

### Routing
Determine target database (PostgreSQL or DuckDB) from tenant config.

### Connection Pool
- Pool per tenant/database
- Read-only credentials only
- Configurable max connections

### Execution
- Hard query timeout with cancellation
- Retryable error handling (transient connection issues)
- Cursor-based pagination (not OFFSET/LIMIT) for large result sets

### Output Adapters
CSV · tabular grid · charts · images

### Post-Execution
- Emit observability logs (query, latency, rows, tenant)
- Update query frequency index for the DB Index Updater Service

---

## 6. Schema Discovery & Index Updater Services

| Service | Trigger | Action |
|---------|---------|--------|
| Schema Discovery | Periodic timer or DB schema-change event | Regenerate schema chunks, re-embed, update schema cache |
| DB Index Updater | Query execution logs | Track most-frequent query patterns; recommend or create indexes |

---

## SQL Best Practices (enforced via prompt guardrails)

- `SELECT` specific columns — never `SELECT *`
- Apply `WHERE` before `JOIN` / `GROUP BY`
- `UNION ALL` instead of `UNION` (no de-duplication overhead)
- `EXISTS` instead of `IN` for subqueries
- Avoid `LIKE '%value'` (disables index use)
- Prefer covering indexes on `WHERE` + `JOIN` + `ORDER BY` columns

**Latency targets:** <100ms for simple lookups · <500ms for aggregations · <10s system SLA
