# Apache AGE Deep Dive — Property Graphs in PostgreSQL

Apache AGE (A Graph Extension) adds native openCypher graph queries to
PostgreSQL. This tutorial covers everything needed to use it effectively in
this project.

---

## Table of Contents

1. [How AGE Works](#how-age-works)
2. [Setup & Connection](#setup--connection)
3. [Core Concepts](#core-concepts)
4. [Cypher Basics](#cypher-basics)
5. [AGE + asyncpg Patterns](#age--asyncpg-patterns)
6. [Vertex Labels vs Flat Entity Type](#vertex-labels-vs-flat-entity-type)
7. [Common Queries for Legal Contracts](#common-queries-for-legal-contracts)
8. [Injection Safety](#injection-safety)
9. [Debugging & Introspection](#debugging--introspection)
10. [Performance](#performance)
11. [Gotchas](#gotchas)

---

## How AGE Works

AGE wraps Cypher inside a PostgreSQL function call:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
    RETURN p.name, c.name
$$) AS (party_name agtype, contract_name agtype);
```

Key points:

- The graph is stored in the same PostgreSQL database as your relational tables.
- `agtype` is AGE's internal type. asyncpg receives it as a string like `"Acme Corp"`.
- Every connection must load AGE and set the search path before issuing Cypher:

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
```

This is why we register an `init` callback on the asyncpg pool (see
`rag/knowledge_graph/age_graph_store.py`).

---

## Setup & Connection

### Docker Compose (this project)

```yaml
# docker-compose.yml
services:
  rag_age:
    image: apache/age:latest
    ports:
      - "5433:5432"
    environment:
      POSTGRES_USER: age_user
      POSTGRES_PASSWORD: age_pass
      POSTGRES_DB: legal_graph
```

```bash
docker-compose up -d rag_age
```

### Create the Graph

```sql
-- Connect to legal_graph DB, then:
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT create_graph('legal_graph');
```

### asyncpg Pool Initialization

```python
import asyncpg

async def _init_age_conn(conn: asyncpg.Connection) -> None:
    await conn.execute("LOAD 'age'")
    await conn.execute("SET search_path = ag_catalog, \"$user\", public")

pool = await asyncpg.create_pool(
    dsn="postgresql://age_user:age_pass@localhost:5433/legal_graph",
    init=_init_age_conn,
)
```

The `init` callback runs once per connection as it is created from the pool —
so every connection is always AGE-ready without needing to call LOAD in
every query.

---

## Core Concepts

### Vertices (Nodes)

A vertex has:
- A **label** — like a class or table name (`:Party`, `:Contract`)
- **Properties** — key-value pairs stored as agtype (JSON-like)

```cypher
-- Create a vertex
CREATE (:Party {uuid: "e1", name: "Acme Corp", document_id: "doc1"})

-- MERGE — create only if not exists (idempotent)
MERGE (:Party {uuid: "e1", name: "Acme Corp"})
```

### Edges (Relationships)

An edge has:
- A **relationship type** — uppercase by convention (`:PARTY_TO`, `:INDEMNIFIES`)
- **Properties** — optional
- A direction — `(a)-[:REL]->(b)`

```cypher
-- Create an edge
MATCH (p:Party {uuid: "e1"}), (c:Contract {uuid: "e2"})
CREATE (p)-[:PARTY_TO]->(c)

-- MERGE edge (idempotent)
MATCH (p:Party {uuid: "e1"}), (c:Contract {uuid: "e2"})
MERGE (p)-[:PARTY_TO]->(c)
```

### Properties

All property values in AGE must be literal strings, numbers, or booleans.
AGE does **not** support null values in Cypher property maps.
Use `COALESCE` to substitute a default:

```cypher
MERGE (e:Party {
    uuid: "e1",
    name: COALESCE("Acme Corp", "unknown"),
    document_id: "doc1"
})
```

---

## Cypher Basics

### MATCH — Find vertices/edges

```cypher
-- All parties
MATCH (p:Party) RETURN p.name LIMIT 20

-- Parties connected to a specific contract
MATCH (p:Party)-[:PARTY_TO]->(c:Contract {uuid: "abc"})
RETURN p.name

-- All contracts with their governing jurisdictions
MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
RETURN c.name, j.name
```

### Pattern Matching

```cypher
-- Two-hop: party → contract → jurisdiction
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS 'delaware'
RETURN p.name, c.name, j.name

-- Variable-length path (1 to 3 hops)
MATCH path = (p:Party)-[:PARTY_TO|SIGNED_BY*1..3]->(c:Contract)
RETURN p.name, length(path)
```

### Filtering

```cypher
-- Case-insensitive name search
MATCH (e) WHERE toLower(e.name) CONTAINS 'amazon'
RETURN e.name, e.label

-- Filter by label property (cross-type queries)
MATCH (e) WHERE e.label = 'Party'
RETURN e.name LIMIT 10
```

### Aggregation

```cypher
-- Count entities per label
MATCH (e) RETURN e.label, count(*) AS cnt ORDER BY cnt DESC

-- Most common governing jurisdictions
MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
RETURN j.name, count(c) AS contract_count
ORDER BY contract_count DESC LIMIT 10

-- Parties in the most contracts
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
RETURN p.name, count(c) AS num_contracts
ORDER BY num_contracts DESC LIMIT 20
```

### EXISTS subquery

```cypher
-- Contracts that have a termination clause but no indemnity clause
MATCH (c:Contract)
WHERE EXISTS { MATCH (c)-[:HAS_TERMINATION]->(:TerminationClause) }
AND NOT EXISTS { MATCH (c)-[:INDEMNIFIES]->() }
RETURN c.name
```

### OPTIONAL MATCH

```cypher
-- All contracts, with jurisdiction if it exists
MATCH (c:Contract)
OPTIONAL MATCH (c)-[:GOVERNED_BY]->(j:Jurisdiction)
RETURN c.name, j.name
```

---

## AGE + asyncpg Patterns

### Basic Query Wrapper

```python
async def run_cypher(conn: asyncpg.Connection, graph: str, cypher: str) -> list[dict]:
    """Execute a read-only Cypher MATCH query and return rows as dicts."""
    # Extract RETURN clause to infer column names
    import re
    ret = re.search(r'\bRETURN\b(.*?)(?:\bORDER\b|\bLIMIT\b|\bSKIP\b|$)',
                    cypher, re.IGNORECASE | re.DOTALL)
    cols = []
    if ret:
        for col in ret.group(1).split(','):
            col = col.strip()
            alias = re.search(r'\bAS\s+(\w+)$', col, re.IGNORECASE)
            cols.append(alias.group(1) if alias else col.split('.')[-1].strip())
    
    # Build AS clause for ag_catalog.cypher()
    as_clause = ", ".join(f"{c} agtype" for c in cols) if cols else "result agtype"
    
    sql = f"SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$) AS ({as_clause})"
    rows = await conn.fetch(sql)
    
    # agtype values come back as quoted strings: '"Acme Corp"' → 'Acme Corp'
    result = []
    for row in rows:
        d = {}
        for i, col in enumerate(cols):
            val = row[i]
            if isinstance(val, str):
                val = val.strip('"')
                # Handle agtype null
                if val in ('null', 'NULL'):
                    val = None
            d[col] = val
        result.append(d)
    return result
```

### Upsert Vertex (MERGE + SET)

```python
async def upsert_vertex(conn, graph, label, props):
    """
    MERGE on uuid — update all other properties.
    AGE does not support $params in Cypher; values must be inlined.
    """
    name_esc = props["name"].replace('"', '\\"')
    cypher = (
        f'MERGE (e:{label} {{uuid: "{props["uuid"]}"}}) '
        f'SET e.name = "{name_esc}", '
        f'    e.document_id = "{props["document_id"]}", '
        f'    e.label = "{label}" '
        f'RETURN e.uuid'
    )
    sql = f"SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$) AS (uuid agtype)"
    await conn.execute(sql)
```

### Add Edge

```python
async def add_edge(conn, graph, src_uuid, rel_type, tgt_uuid):
    """
    MATCH source and target by uuid (label-agnostic), then MERGE the edge.
    Using (e) without a label avoids having to know the vertex type.
    """
    cypher = (
        f'MATCH (s {{uuid: "{src_uuid}"}}), (t {{uuid: "{tgt_uuid}"}}) '
        f'MERGE (s)-[:{rel_type}]->(t)'
    )
    sql = f"SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$) AS (r agtype)"
    await conn.execute(sql)
```

### Fetch Entity with Relationships

```python
async def get_entity_context(conn, graph, uuid):
    cypher = f"""
        MATCH (e {{uuid: "{uuid}"}})
        OPTIONAL MATCH (e)-[r]->(n)
        RETURN e.name, e.label, type(r) AS rel, n.name AS neighbour
        LIMIT 20
    """
    sql = f"""
        SELECT * FROM ag_catalog.cypher('{graph}', $$ {cypher} $$)
        AS (name agtype, label agtype, rel agtype, neighbour agtype)
    """
    rows = await conn.fetch(sql)
    return [dict(row) for row in rows]
```

---

## Vertex Labels vs Flat Entity Type

This project uses **distinct vertex labels** (`:Party`, `:Contract`, `:Clause`)
rather than a flat `:Entity {entity_type: "Party"}` model.

### Why distinct labels?

```cypher
-- Flat model — must filter on property
MATCH (e:Entity {entity_type: 'Party'})-[:PARTY_TO]->(c:Entity {entity_type: 'Contract'})

-- Distinct labels — idiomatic, faster, index-able
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
```

Benefits:
1. Cleaner Cypher — semantically correct, easier to read
2. AGE can create label-specific indexes
3. `MATCH (p:Party)` scans only the Party vertex table, not all vertices

Trade-off: `MATCH (e)` (no label) scans all vertex tables — always use the
`e.label` property for cross-type queries to aid filtering.

### Label safety

Never interpolate user-supplied label names directly into Cypher. Always
validate against an allowlist:

```python
_VALID_LABELS = frozenset({
    "Contract", "Section", "Clause", "Party", "Jurisdiction",
    "EffectiveDate", "ExpirationDate", "RenewalTerm", "LiabilityClause",
    "IndemnityClause", "TerminationClause", "GoverningLawClause",
})

def safe_label(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", raw)
    if cleaned not in _VALID_LABELS:
        return "Clause"  # safe fallback
    return cleaned
```

---

## Common Queries for Legal Contracts

### Find all parties to a contract

```cypher
MATCH (p:Party)-[:PARTY_TO]->(c:Contract {uuid: $contract_id})
RETURN p.name, p.uuid
LIMIT 20
```

### Contracts governed by a jurisdiction

```cypher
MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS 'delaware'
RETURN c.name, c.uuid
LIMIT 50
```

### Indemnification chains

```cypher
MATCH (a:Party)-[:INDEMNIFIES]->(b:Party)
RETURN a.name AS indemnifier, b.name AS indemnified
LIMIT 30
```

### Multi-hop: party → contracts → jurisdictions

```cypher
MATCH (p:Party {uuid: "party-uuid-here"})
      -[:PARTY_TO]->(c:Contract)
      -[:GOVERNED_BY]->(j:Jurisdiction)
RETURN p.name, c.name, j.name
```

### Contracts with termination clause but no liability cap

```cypher
MATCH (c:Contract)-[:HAS_TERMINATION]->(:TerminationClause)
WHERE NOT EXISTS {
    MATCH (c)-[:LIMITS_LIABILITY]->(:LiabilityClause)
}
RETURN c.name, c.uuid
```

### Entity count per contract (complexity score)

```cypher
MATCH (e)-[:PARTY_TO|HAS_TERMINATION|HAS_LICENSE|GOVERNED_BY|INDEMNIFIES*0..1]->(c:Contract)
RETURN c.name, count(DISTINCT e) AS entity_count
ORDER BY entity_count DESC
LIMIT 20
```

### Cross-licensing: same party as both licensor and licensee

```cypher
MATCH (p:Party)-[:PARTY_TO]->(c1:Contract)
MATCH (p)-[:PARTY_TO]->(c2:Contract)
WHERE c1.uuid <> c2.uuid
RETURN p.name, c1.name, c2.name
LIMIT 10
```

### Graph statistics

```cypher
MATCH (e) RETURN e.label AS label, count(*) AS count
ORDER BY count DESC
```

---

## Injection Safety

AGE does not support parameterised Cypher — values are always inlined. This
makes injection prevention critical.

### Rules

1. **Validate labels** against `_VALID_LABELS` allowlist
2. **Validate relationship types** against `_VALID_REL_TYPES` allowlist  
3. **Escape string values**: replace `"` → `\"` in any user-supplied text
4. **Block destructive keywords** in query execution:

```python
_BLOCKED = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|REMOVE|CALL)\b",
    re.IGNORECASE,
)

def validate_read_only(cypher: str) -> None:
    if _BLOCKED.search(cypher):
        raise ValueError(f"Destructive Cypher not allowed: {cypher[:80]}")
```

### Safe value escaping

```python
def esc(value: str) -> str:
    """Escape a string value for inline Cypher."""
    return value.replace("\\", "\\\\").replace('"', '\\"')

# Usage:
name = esc(user_input)
cypher = f'MATCH (e {{name: "{name}"}}) RETURN e.uuid'
```

---

## Debugging & Introspection

### List all graphs

```sql
SELECT * FROM ag_catalog.ag_graph;
```

### List all vertex labels in a graph

```sql
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
AND kind = 'v';  -- 'v' = vertex, 'e' = edge
```

### Count vertices per label

```cypher
MATCH (e) RETURN e.label, count(*) ORDER BY count(*) DESC
```

### List all edge types

```sql
SELECT name FROM ag_catalog.ag_label
WHERE graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph')
AND kind = 'e';
```

### Check AGE version

```sql
SELECT * FROM pg_extension WHERE extname = 'age';
```

### Explain a Cypher query

AGE Cypher queries run through PostgreSQL's query planner. To see the plan:

```sql
EXPLAIN
SELECT * FROM ag_catalog.cypher('legal_graph', $$
    MATCH (p:Party)-[:PARTY_TO]->(c:Contract)
    RETURN p.name, c.name
    LIMIT 100
$$) AS (party agtype, contract agtype);
```

### Reset the graph (DANGER — drops all data)

```sql
SELECT drop_graph('legal_graph', true);   -- true = cascade
SELECT create_graph('legal_graph');
```

---

## Performance

### What's fast

- Label-specific `MATCH (p:Party)` — scans only the Party vertex table
- Simple property lookups with exact values (`{uuid: "..."}`) — O(1) with index
- Short traversals (1-2 hops) with LIMIT

### What's slow

- `MATCH (e)` without label — full scan of all vertex tables
- Variable-length paths `*1..10` without LIMIT — can explode
- Aggregation over all vertices without filtering first

### Index strategy

AGE creates an index automatically on the `id` column (internal vertex ID).
For property-based lookups (e.g., by `uuid`), create a PostgreSQL index on
the underlying vertex table:

```sql
-- Find the vertex table for the Party label
SELECT relation FROM ag_catalog.ag_label
WHERE name = 'Party'
AND graph = (SELECT oid FROM ag_catalog.ag_graph WHERE name = 'legal_graph');

-- Create index on uuid property (stored in the 'properties' agtype column)
-- Note: AGE 1.5+ supports expression indexes on agtype properties
CREATE INDEX party_uuid_idx ON legal_graph."Party"
    ((properties->>'uuid'));
```

### Connection pool size

AGE queries are PostgreSQL queries — pool size recommendations are the same:
- CPU-bound (complex Cypher): pool = num CPUs × 2
- I/O-bound (simple lookups): pool = 10-20

In this project the default pool size is 10 (set in `AgeGraphStore`).

---

## Gotchas

### 1. agtype strings come back quoted

asyncpg returns agtype as a string with surrounding quotes:
- `"Acme Corp"` → strip to `Acme Corp`
- `42::numeric` → strip to `42`
- `null` → Python `None`

```python
def strip_agtype(val: str | None) -> str | None:
    if val is None:
        return None
    val = val.strip()
    if val in ('null', 'NULL'):
        return None
    return val.strip('"')
```

### 2. LOAD 'age' must run on every connection

If you forget this (or the pool doesn't have the `init` callback), you'll see:

```
ERROR: function ag_catalog.cypher(unknown, unknown) does not exist
```

Fix: register `init=_init_age_conn` on the pool, not just on individual connections.

### 3. Property values cannot be NULL in MERGE

```cypher
-- WRONG: AGE may reject null property values
MERGE (e:Party {uuid: null})

-- RIGHT: use COALESCE or skip null properties
MERGE (e:Party {uuid: "e1"})
SET e.alias = COALESCE("Corp", "")
```

### 4. AS clause column count must match RETURN clause

If your RETURN has 3 columns but the AS clause declares 2, AGE throws:

```
ERROR: column definition list has too few entries
```

Always keep AS clause in sync with RETURN.

### 5. Integer property values

AGE stores all properties as agtype. When you RETURN an integer property,
asyncpg gets back `"42"` (quoted string), not `42` (int). Cast explicitly:

```cypher
RETURN toInteger(e.chunk_index) AS chunk_index
```

### 6. Relationship type names are case-sensitive

`[:party_to]` and `[:PARTY_TO]` are different edge types. Always use
UPPERCASE for relationship types in this project (see `_VALID_REL_TYPES`).

### 7. graph_name must match exactly

The graph name (`'legal_graph'`) must match the name used in `create_graph()`.
A typo gives:

```
ERROR: graph "legel_graph" does not exist
```

### 8. asyncpg fetch vs execute for INSERT/MERGE

- Use `conn.execute(sql)` for MERGE/CREATE — returns no rows
- Use `conn.fetch(sql)` for MATCH/RETURN — returns rows
- Using `conn.fetch` on a CREATE gives no error but returns empty

---

## Running Examples

```bash
# Connect to the AGE container
docker exec -it rag_age psql -U age_user -d legal_graph

# Inside psql:
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT create_graph('legal_graph');  -- first time only

-- After running the extraction pipeline:
SELECT * FROM ag_catalog.cypher('legal_graph', $$
    MATCH (e) RETURN e.label, count(*) ORDER BY count(*) DESC
$$) AS (label agtype, cnt agtype);
```

See `rag/knowledge_graph/age_graph_store.py` for the full production
implementation used in this project.
