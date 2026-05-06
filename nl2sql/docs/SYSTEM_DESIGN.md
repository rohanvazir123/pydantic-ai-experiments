# NL-to-SQL — System Design

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

Each pipeline can have one or more stages. The system is composed of:

1. **Schema Discovery Service** — background service, event-driven
2. **Prompt Generation Pipeline** — normalize query, assemble context
3. **SQL Generation Pipeline** — generate N candidate SQL queries
4. **SQL Validation Pipeline** — static guards, schema checks, RBAC
5. **SQL Execution Pipeline** — route, execute, paginate, return results
6. **DB Index Updater Service** — update indexes for frequent queries

---

## 4. Caching Strategy

### Schema Cache
- Updated by the Schema Discovery Service via schema embeddings
- At query time: embed the NL query and run ANN search against the vector DB to retrieve the most relevant schema chunks

### NL Query → SQL Cache
- Given a user's NL query, reuse a previously generated SQL statement
- Keyed on normalized NL query

### NL Query → Results Cache
- For fully deterministic dashboards and recurring analytics questions
- Bypasses SQL generation and execution entirely

### SQL Query → Results Cache
- If the same SQL is executed frequently and data freshness requirements allow
- Keyed on SQL hash

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

### Check 2 — Schema Validation (SQLGlot)

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

### Check 3 — RBAC Policy

```json
{
  "error_type": "policy_violation",
  "details": "Access to column 'email' is not permitted for this role."
}
```

### Repair Loop

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
