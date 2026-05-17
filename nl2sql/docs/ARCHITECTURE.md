# NL-to-SQL — Architecture

## Table of Contents

1. [Detailed Architecture Diagram](#detailed-architecture-diagram)
2. [Overview](#overview)
3. [Stack](#stack)
4. [Components](#components)
5. [Data Flow — NL Query](#data-flow--nl-query)
6. [Sample LLM Prompts — NL→SQL](#sample-llm-prompts--nlsql)
7. [Sample LLM Prompts — NL→Cypher](#sample-llm-prompts--nlcypher)
8. [Agent Orchestration: Single Prompt vs Tool Calling](#agent-orchestration-single-prompt-vs-tool-calling)
9. [DuckDB ↔ PostgreSQL Bridge](#duckdb--postgresql-bridge)
10. [Guardrails](#guardrails)
11. [Caching](#caching)
12. [Key Configuration](#key-configuration-env)
13. [API Endpoints](#api-endpoints)
14. [Running](#running)

---

## Detailed Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           NL-to-SQL System                                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌─────────────────────────┐    ┌──────────────────────────────────────────┐   ║
║   │       Entry Points      │    │              Data Sources                │   ║
║   │                         │    │                                          │   ║
║   │  ┌───────────────────┐  │    │  ┌──────────────────────────────────┐   │   ║
║   │  │  Streamlit UI     │  │    │  │       UnifiedDataSource           │   │   ║
║   │  │  (chat interface) │  │    │  │                                  │   │   ║
║   │  │  _build_manager() │  │    │  │  ┌────────────┐  ┌───────────┐  │   │   ║
║   │  │  [st.cache_resrc] │  │    │  │  │ GCS Parquet│  │PostgreSQL │  │   │   ║
║   │  └────────┬──────────┘  │    │  │  │  (gcsfs +  │  │  DB(s)    │  │   │   ║
║   │           │             │    │  │  │            │  │           │  │   │   ║
║   │  ┌───────────────────┐  │    │  │  │  datasets) │  │rag_db     │  │   │   ║
║   │  │  FastAPI REST     │  │    │  │  │            │  │local_pg   │  │   │   ║
║   │  │  POST /v1/query   │  │    │  │  │ lazy reads │  │(pgvector) │  │   │   ║
║   │  │  _get_manager()   │  │    │  │  └─────┬──────┘  └─────┬─────┘  │   │   ║
║   │  │  [module singleton│  │    │  │        │                │        │   │   ║
║   │  └────────┬──────────┘  │    │  │        ▼                ▼        │   │   ║
║   └───────────┼─────────────┘    │  │  ┌─────────────────────────────┐ │   │   ║
║               │                  │  │  │     DuckDB  (:memory:)      │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  conn.register('orders',    │ │   │   ║
║               │                  │  │  │    pyarrow.dataset) ← lazy  │ │   │   ║
║               │                  │  │  │  (predicate pushdown: only  │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │   needed row groups read)   │ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  INSTALL postgres; LOAD pg  │ │   │   ║
║               │                  │  │  │  ATTACH '...' AS rag_db     │ │   │   ║
║               │                  │  │  │    (TYPE postgres, READ_ONLY)│ │   │   ║
║               │                  │  │  │  ATTACH '...' AS local_pg   │ │   │   ║
║               │                  │  │  │    (TYPE postgres, READ_ONLY)│ │   │   ║
║               │                  │  │  │                             │ │   │   ║
║               │                  │  │  │  Schema introspection:      │ │   │   ║
║               │                  │  │  │  DESCRIBE <view>            │ │   │   ║
║               │                  │  │  │  information_schema.tables  │ │   │   ║
║               │                  │  │  │  information_schema.columns  │ │   │   ║
║               │                  │  │  └──────────────┬──────────────┘ │   │   ║
║               │                  │  │                 │ schema_text     │   │   ║
║               │                  │  └─────────────────┼────────────────┘   │   ║
║               │                  │                    │                     │   ║
║               │                  └────────────────────┼─────────────────────┘   ║
║               │                                       │                         ║
║               ▼                                       ▼                         ║
║   ┌───────────────────────────────────────────────────────────────────────────┐ ║
║   │                        ConversationManager                                │ ║
║   │                                                                           │ ║
║   │   nl_query                                                                │ ║
║   │      │                                                                    │ ║
║   │      ▼                                                                    │ ║
║   │   _normalize_nl()  ──►  NL Cache (LRU, size=20)                          │ ║
║   │   "how many docs?" ◄──  OrderedDict[normalized_nl → QueryResult]         │ ║
║   │      │ miss                                                               │ ║
║   │      ▼                                                                    │ ║
║   │   ┌─────────────────────────────────────────────────────────────────┐    │ ║
║   │   │               Retry Loop  (attempt 1 .. max_retries=3)          │    │ ║
║   │   │                                                                  │    │ ║
║   │   │  attempt == 1                   attempt > 1                      │    │ ║
║   │   │  _build_prompt()               _build_correction_prompt()        │    │ ║
║   │   │  ┌────────────────────┐        ┌────────────────────────────┐   │    │ ║
║   │   │  │ Schema:\n{schema}  │        │ Schema:\n{schema}          │   │    │ ║
║   │   │  │ History (last 3    │        │ Failed SQL: {bad_sql}      │   │    │ ║
║   │   │  │ successful turns)  │        │ Error: {error[:400]}       │   │    │ ║
║   │   │  │ Question: {nl}     │        │ Return ONLY corrected SQL. │   │    │ ║
║   │   │  └────────┬───────────┘        └───────────┬────────────────┘   │    │ ║
║   │   │           │                                │                     │    │ ║
║   │   │           └──────────────┬─────────────────┘                     │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │              ┌─────────────────────┐                             │    │ ║
║   │   │              │   Pydantic AI Agent  │                             │    │ ║
║   │   │              │   agent.run(prompt)  │                             │    │ ║
║   │   │              │                      │                             │    │ ║
║   │   │              │  OpenAI / Anthropic  │                             │    │ ║
║   │   │              │  or Ollama (local)   │                             │    │ ║
║   │   │              │  system_prompt:      │                             │    │ ║
║   │   │              │  "Return ONLY plain  │                             │    │ ║
║   │   │              │   SQL. No fences."   │                             │    │ ║
║   │   │              └──────────┬──────────┘                             │    │ ║
║   │   │                         │ raw SQL string                         │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │              strip_sql_fences()   (remove ```sql ... ```)        │    │ ║
║   │   │                         │                                        │    │ ║
║   │   │                         ▼                                        │    │ ║
║   │   │  ┌──────────────────────────────────────────────────────────┐   │    │ ║
║   │   │  │                   Guardrails                              │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  G1: _check_readonly()                                    │   │    │ ║
║   │   │  │      regex: DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|...  │   │    │ ║
║   │   │  │      → error string if matched → retry with error         │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  G2: _apply_row_cap()                                     │   │    │ ║
║   │   │  │      if no LIMIT clause → append LIMIT {max_result_rows}  │   │    │ ║
║   │   │  │      safe_sql = sql + "\nLIMIT 10000"                     │   │    │ ║
║   │   │  └──────────────────────────────┬────────────────────────────┘   │    │ ║
║   │   │                                 │ safe_sql                       │    │ ║
║   │   │                                 ▼                                │    │ ║
║   │   │              _hash(safe_sql) ──► SQL Cache (LRU, size=20)        │    │ ║
║   │   │                             ◄── OrderedDict[md5 → QueryResult]   │    │ ║
║   │   │                                 │ miss                           │    │ ║
║   │   │                                 ▼                                │    │ ║
║   │   │  ┌──────────────────────────────────────────────────────────┐   │    │ ║
║   │   │  │  G3: _execute_with_timeout(conn, safe_sql, timeout=30s)  │   │    │ ║
║   │   │  │                                                           │   │    │ ║
║   │   │  │  threading.Timer(30, _cancel)                             │   │    │ ║
║   │   │  │      _cancel: timed_out.set() + conn.interrupt()          │   │    │ ║
║   │   │  │  conn.execute(safe_sql)   [DuckDB]                        │   │    │ ║
║   │   │  │      → pulls data from PostgreSQL via postgres scanner     │   │    │ ║
║   │   │  │      → or reads GCS Parquet via httpfs                    │   │    │ ║
║   │   │  │  cursor.description → columns                             │   │    │ ║
║   │   │  │  cursor.fetchall()  → rows                                │   │    │ ║
║   │   │  └──────────────────────────────┬────────────────────────────┘   │    │ ║
║   │   │                                 │                                │    │ ║
║   │   │              ┌──────────────────┴───────────────────┐           │    │ ║
║   │   │              │ Success                   Exception   │           │    │ ║
║   │   │              ▼                           ▼           │           │    │ ║
║   │   │     populate SQL cache           last_error = str(exc)          │    │ ║
║   │   │     populate NL cache            (or TimeoutError)              │    │ ║
║   │   │     → break loop                 → next attempt                 │    │ ║
║   │   └─────────────────────────────────────────────────────────────────┘    │ ║
║   │                                                                           │ ║
║   │      ▼  (after loop)                                                      │ ║
║   │   append turn to self.history                                             │ ║
║   │   if history_store: await history_store.save(session_id, ...)             │ ║
║   │      │                                                                    │ ║
║   │      ▼                                                                    │ ║
║   │   QueryResult(nl_query, sql, columns, rows, error, cached, attempts)      │ ║
║   └───────────────────────────────────────────────────────────────────────────┘ ║
║               │                                                                  ║
║               ▼                                                                  ║
║   ┌───────────────────────────────────────────────────────────────────────────┐ ║
║   │                   HistoryStore  (optional)                                │ ║
║   │                                                                           │ ║
║   │   asyncpg pool → PostgreSQL table: conversation_history                   │ ║
║   │   ┌────────────────────────────────────────────────────────────────────┐  │ ║
║   │   │ id │ session_id │ nl_query │ sql │ columns │ rows │ error │ ts     │  │ ║
║   │   └────────────────────────────────────────────────────────────────────┘  │ ║
║   │   save(): INSERT on every successful/failed turn                           │ ║
║   │   load(): SELECT ORDER BY ts → warm caches on session resume              │ ║
║   │   sessions(): list all session_ids                                         │ ║
║   └───────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### Table naming by source

```
Query: "How many orders were placed in Q4?"

                      LLM sees schema text:
          ┌───────────────────────────────────────────┐
          │ === GCS Parquet tables (use bare name) === │
          │ Table: orders                              │
          │   - order_id (VARCHAR)                     │
          │   - order_date (DATE)                      │
          │                                            │
          │ === rag tables (prefix: rag.main.<table>) ││
          │ Table: rag.main.documents                  │
          │   - id (UUID)                              │
          │   - title (TEXT)                           │
          │                                            │
          │ === local_pg tables (prefix:              ││
          │        local_pg.main.<table>)              │
          │ Table: local_pg.main.baby_names            │
          │   - name (VARCHAR)                         │
          └───────────────────────────────────────────┘
                           │
                           ▼
          LLM generates: SELECT COUNT(*) FROM orders
                         WHERE order_date >= '2024-10-01'
                           │
                           ▼ DuckDB resolves:
                    orders → GCS Parquet view
                    rag.main.documents → postgres scanner
                    local_pg.main.* → postgres scanner
```

### Cache hit paths

```
NL Cache hit (same question, different casing/whitespace):
  "How many docs?" → normalize → "how many docs?"
  "HOW MANY DOCS?" → normalize → "how many docs?"  → cache hit, 0 LLM calls

SQL Cache hit (different question → identical SQL):
  "Count all docs" → LLM → SELECT COUNT(*) FROM rag.main.documents LIMIT 10000
  "Total documents?" → LLM → SELECT COUNT(*) FROM rag.main.documents LIMIT 10000
                                       → md5 match → cache hit, rows returned from memory

History context (last 3 successful turns fed into next prompt):
  Only successful turns included — failed SQL is excluded to avoid confusing the LLM.
  On session resume: HistoryStore.load() replays DB rows into self.history
                     then caches are warmed from those turns.
```

## Overview

Natural-language to SQL over the RAG PostgreSQL database.
A Pydantic AI agent translates user questions into SQL SELECT statements,
executes them via DuckDB's PostgreSQL scanner, and returns results.
A self-correcting retry loop feeds SQL errors back to the LLM for up to 3 attempts.

## Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (local) or any OpenAI-compatible API |
| SQL execution | DuckDB in-memory (attaches PostgreSQL via postgres scanner) |
| Agent framework | Pydantic AI |
| History persistence | asyncpg PostgreSQL (optional) |
| UI | Streamlit (`apps/nl2sql/streamlit_app.py`) |
| REST API | FastAPI (`apps/nl2sql/api.py`) |

## Components

```
apps/nl2sql/
├── streamlit_app.py   — Chat UI with SQL display and result tables
└── api.py             — FastAPI: /health, /v1/query, /v1/history, /v1/schema

nlp2sql/
├── nlp_sql_postgres_v2.py   — Core logic
│   ├── ConversationManager  — NL → SQL → execute → retry loop
│   ├── HistoryStore         — asyncpg-backed session persistence
│   ├── QueryResult          — Structured result (columns, rows, error, cached, attempts)
│   ├── UnifiedDataSource    — Multi-source: PostgreSQL + GCS Parquet via DuckDB
│   ├── _check_readonly()    — Guardrail: block non-SELECT statements
│   ├── _apply_row_cap()     — Guardrail: enforce LIMIT
│   └── _execute_with_timeout() — Guardrail: interrupt long queries
└── README.md
```

## Data Flow — NL Query

```
User question ("How many documents are stored?")
    │
    ▼
ConversationManager.run_query(nl_query)
    │
    ├── NL cache hit? → return cached QueryResult
    │
    ├── [loop: attempt 1..max_retries]
    │       │
    │       ├── Build prompt: schema_text + conversation history + question
    │       │   (attempt > 1: include failed SQL + error for self-correction)
    │       │
    │       ├── Pydantic AI agent.run(prompt) → raw SQL string
    │       │
    │       ├── Guardrail 1: _check_readonly() — block DROP/DELETE/INSERT/UPDATE/…
    │       │
    │       ├── Guardrail 2: _apply_row_cap() — append LIMIT if missing
    │       │
    │       ├── SQL hash cache hit? → return cached result
    │       │
    │       └── _execute_with_timeout(conn, sql, timeout)
    │               └── DuckDB → PostgreSQL (via postgres scanner attachment)
    │
    └── Return QueryResult(sql, columns, rows, cached, attempts, error)
```

## Sample LLM Prompts — NL→SQL

The agent is a Pydantic AI `Agent` with a fixed system prompt and a dynamic user turn built per query.
Source: `nl2sql/nlp_sql_postgres_v2.py` — `_SYSTEM_PROMPT`, `_build_prompt()`, `_build_correction_prompt()`.

### System prompt (sent once per session)

```
You are an expert SQL assistant working with DuckDB.

Table naming rules (IMPORTANT — always follow these):
- GCS Parquet tables  -> bare table name,          e.g.  FROM orders
- rag_db tables       -> rag.main.<table>,          e.g.  FROM rag.main.documents
- local_pg tables     -> local_pg.main.<table>,     e.g.  FROM local_pg.main.baby_names

Return ONLY plain SQL. No Markdown fences, no explanation, no comments.
```

### First-attempt user prompt (`_build_prompt`)

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

The "Conversation so far" block is omitted on the very first turn of a session.
Only the last 3 **successful** turns are included — failed SQL is excluded to avoid confusing the model.

**Expected LLM response:**

```sql
SELECT COUNT(*) FROM orders
WHERE order_date >= '2024-10-01' AND order_date < '2025-01-01'
LIMIT 10000
```

### Self-correction prompt (`_build_correction_prompt`)

Sent on attempt 2 and 3 when the previous SQL raised an exception during execution.

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

The error string is truncated to 400 characters before being embedded.

**Expected LLM response:**

```sql
SELECT COUNT(*) FROM orders
WHERE order_date >= '2024-10-01' AND order_date < '2025-01-01'
LIMIT 10000
```

---

## Sample LLM Prompts — NL→Cypher

> **Current implementation note:** `kg/legal/retrieval/nl2cypher.py` uses a **rule-based** pipeline — `IntentParser` (regex patterns) maps the question to an intent, and `QUERY_CAPABILITIES[intent](params)` builds the Cypher string directly. **No LLM is involved.**
>
> The prompts below document the LLM-based Cypher generation described in `nl2sql/docs/SYSTEM_DESIGN.md` section 11, which applies when the system is extended to handle free-form graph queries that cannot be covered by fixed intent patterns.

### System prompt

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
It is used to build the required `AS (col agtype, …)` clause for AGE execution.

### User prompt (with graph schema context)

Graph schema context is discovered at query time from `ag_catalog` tables and sampled property keys.

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
  HAS_RESTRICTION, HAS_IP_CLAUSE, HAS_LIABILITY, HAS_CLAUSE, CAUSES_RISK

Question: Which contracts does Acme Corp appear in as a party?
```

**Expected LLM response:**

```
<cypher>
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
WHERE toLower(p.name) CONTAINS 'acme corp'
RETURN p.name, c.name
LIMIT 20
</cypher>
<columns>p.name, c.name</columns>
```

### Prompt vs SQL differences

| Dimension | SQL prompt | Cypher prompt |
|-----------|-----------|---------------|
| Schema context | Table names + column names + data types | Node labels + property key lists + edge types |
| Query syntax | `SELECT … FROM … WHERE … GROUP BY` | `MATCH (n:Label)-[:REL]->(m) WHERE … RETURN …` |
| Output format | Raw SQL string | `<cypher>` block **and** a matching `<columns>` block |
| Null handling | `IS NULL` / `COALESCE` | `OPTIONAL MATCH`; AGE rejects null in MERGE |
| Case sensitivity | Column names case-insensitive (DuckDB) | Label names and property keys are case-sensitive |
| Joins | Explicit `JOIN … ON` | Implicit via path pattern `(a)-[:REL]->(b)` |
| Parameterisation | `$1` placeholders (safe) | Inline string escaping (AGE limitation) |

---

## Agent Orchestration: Single Prompt vs Tool Calling

Two implementations exist in this repo. They make a fundamentally different choice about when and how the LLM accesses schema information.

### v1 — Single-prompt (stateless) · `nl2sql/nlp_sql_postgres_v2.py`

Schema is discovered **before** the LLM is invoked. The `UnifiedDataSource` inspects `information_schema` at startup and serialises the entire schema into a plain-text block. That block is embedded directly in the user message on every call.

```
┌─────────────────────────────────────────────────────────┐
│  User question                                          │
└──────────────────────────┬──────────────────────────────┘
                           │
                 schema already in memory
                           │
                           ▼
           ┌───────────────────────────────┐
           │     Pydantic AI agent.run()   │
           │                               │
           │  [system]  _SYSTEM_PROMPT     │   ← fixed, sent once
           │  [user]    Schema:\n{schema}  │
           │            History:\n{…}      │   ← last 3 good turns
           │            Question: {nl}     │
           └───────────────┬───────────────┘
                           │
                           │  single LLM generation
                           ▼
                      raw SQL string
                           │
                  guardrails + execution
                           │
                  success? → done
                  error?   → _build_correction_prompt()
                             → agent.run() again (≤ 3 attempts)
```

**Turn sequence (happy path, 1 attempt):**

| # | Role | Content |
|---|------|---------|
| 1 | system | `_SYSTEM_PROMPT` (table naming rules + "Return ONLY plain SQL") |
| 2 | user | `Schema:\n{schema_text}\n\nConversation so far:\n{history}\n\nQuestion: {nl}` |
| 3 | assistant | `SELECT COUNT(*) FROM orders WHERE … LIMIT 10000` |

**Turn sequence (self-correction, attempt 2):**

| # | Role | Content |
|---|------|---------|
| 1 | system | `_SYSTEM_PROMPT` |
| 2 | user | `Schema:\n{schema_text}\n\nThe following SQL you generated failed:\nQuestion: …\nSQL: …\nError: …\n\nReturn ONLY the corrected SQL.` |
| 3 | assistant | corrected SQL |

**Characteristics:**
- LLM sees the full schema on every call — context window cost scales with schema size.
- No round-trips for schema discovery; single generation per attempt is fast.
- Works well when the schema is small enough to fit in one prompt.
- Retry loop handles transient SQL errors without user intervention.

---

### v2 — Tool-calling (agentic) · `nl2sql/sql_discovery.py`

The agent has no schema injected upfront. It uses **Pydantic AI tool calls** to discover the schema at runtime — deciding which tables to inspect before writing SQL. The result is a structured `SQLResponse` Pydantic model, not a raw string.

```
┌─────────────────────────────────────────────────────────┐
│  User question                                          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │     Pydantic AI agent.run()   │
           │     system_prompt:            │
           │     "Discover the schema      │
           │      first."                  │
           └───────────────┬───────────────┘
                           │ LLM decides to call tools
                           ▼
              tool: list_tables("postgres")
                → ["documents", "chunks", …]   ← live DB call
                           │
              tool: describe_table("postgres", "documents")
                → "documents: id (UUID), title (TEXT), …" ← live DB call
                           │
                           ▼ LLM now has enough schema context
           ┌───────────────────────────────┐
           │  LLM generates final response │
           │  SQLResponse(                 │
           │    database_type="postgres",  │
           │    sql="SELECT COUNT(*) …",   │
           │    explanation="Used the      │
           │      documents table because…"│
           │  )                            │
           └───────────────────────────────┘
                           │
              execute on postgres or DuckDB
              based on database_type field
```

**Turn sequence:**

| # | Role | Content |
|---|------|---------|
| 1 | system | `"You are a data expert with access to Postgres and DuckDB. Discover the schema first."` |
| 2 | user | `"How many documents are in the RAG database?"` |
| 3 | assistant (tool call) | `list_tables(db_type="postgres")` |
| 4 | tool result | `["documents", "chunks", "conversation_history"]` |
| 5 | assistant (tool call) | `describe_table(db_type="postgres", table_name="documents")` |
| 6 | tool result | `"Postgres table documents: id (UUID), title (TEXT), source (TEXT), …"` |
| 7 | assistant (final) | `SQLResponse(database_type="postgres", sql="SELECT COUNT(*) FROM documents", explanation="…")` |

**Characteristics:**
- Schema is discovered lazily — the LLM decides which tables to inspect, so irrelevant tables are never read.
- Handles large or unknown schemas where pre-building the full schema text is impractical.
- Multiple round-trips (one per tool call) add latency — each tool call is a separate LLM generation.
- Returns a structured `SQLResponse` with an `explanation` field, not a raw SQL string — easier to display reasoning to the user.
- No built-in retry loop; error handling is the caller's responsibility.

---

### Comparison

| | v1: Single-prompt | v2: Tool-calling |
|--|-------------------|------------------|
| **File** | `nl2sql/nlp_sql_postgres_v2.py` | `nl2sql/sql_discovery.py` |
| **Schema delivery** | Pre-built text injected into every user message | LLM calls `list_tables` / `describe_table` at runtime |
| **LLM round-trips** | 1 (+ retry on error) | 2–4 (schema discovery) + 1 (SQL generation) |
| **Result type** | Raw SQL string → stripped by `strip_sql_fences()` | `SQLResponse` Pydantic model (typed, validated) |
| **Retry / self-correction** | Built-in: `_build_correction_prompt()` up to 3 attempts | Not built-in |
| **Context window cost** | Full schema text on every call | Only inspected tables |
| **Best for** | Known, bounded schema; latency-sensitive paths | Large/unknown schema; when reasoning trace matters |

---

## DuckDB ↔ PostgreSQL Bridge

```
duckdb.connect(":memory:")
    └── ATTACH 'postgresql://...' AS rag_db (TYPE postgres, READ_ONLY)
            └── exposes tables as: rag_db.main.<table>

Schema introspection:
    SELECT table_name FROM rag_db.information_schema.tables
    SELECT column_name, data_type FROM rag_db.information_schema.columns
```

DuckDB executes the LLM-generated SQL locally, pulling data from PostgreSQL
via the postgres scanner. This avoids running arbitrary SQL directly on PostgreSQL
and gives DuckDB's optimizer control over the query plan.

## Guardrails

| Guardrail | Implementation | Behavior |
|---|---|---|
| Read-only enforcement | `_check_readonly()` regex | Blocks DROP/DELETE/INSERT/UPDATE/TRUNCATE/ALTER/CREATE/GRANT/REVOKE |
| Result row cap | `_apply_row_cap()` | Appends `LIMIT {max_result_rows}` if none present |
| Query timeout | `threading.Timer` + `conn.interrupt()` | Kills query after `query_timeout` seconds |
| Self-correcting retry | `ConversationManager.run_query()` loop | Re-prompts LLM with error message on failure |

## Caching

Two LRU caches per `ConversationManager`:

- **NL cache** — keyed by `" ".join(nl.lower().split())` (normalized whitespace + case)
- **SQL hash cache** — keyed by MD5 of the SQL string (catches same SQL from different phrasing)

Cache size is configurable (`cache_size` param, default 20).
History-warmed: prior session turns pre-populate both caches on resume.

## Key Configuration (`.env`)

```
DATABASE_URL=postgresql://...          # RAG PostgreSQL (attached as rag_db)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB and LLM connectivity |
| POST | `/v1/query` | NL → SQL → execute → return results |
| GET | `/v1/history` | Recent conversation history |
| GET | `/v1/schema` | Database schema text used for SQL generation |

## Running

```bash
# UI
streamlit run apps/nl2sql/streamlit_app.py

# API
uvicorn apps.nl2sql.api:app --port 8001 --reload

# Example query
curl -X POST http://localhost:8001/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many documents are stored?"}'
```
