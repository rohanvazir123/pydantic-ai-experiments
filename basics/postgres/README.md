# PostgreSQL with asyncpg — Reference & FAQ

Domain used throughout: **e-commerce** (users, products, orders, order_items, reviews).

---

## Table of Contents

1. [asyncpg vs psycopg3](#asyncpg-vs-psycopg3)
2. [Pool Sizing Guidelines](#pool-sizing-guidelines)
3. [fetch / fetchrow / fetchval / execute: Decision Table](#fetch-methods-decision-table)
4. [Transaction Isolation Levels](#transaction-isolation-levels)
5. [Cursor vs fetch(): Memory Tradeoffs](#cursor-vs-fetch-memory-tradeoffs)
6. [COPY vs executemany vs INSERT: Throughput](#copy-vs-executemany-vs-insert)
7. [Full-Text Search: tsvector vs LIKE vs pg_trgm](#full-text-search)
8. [JSONB vs Separate Columns](#jsonb-vs-separate-columns)
9. [Window Functions: Common Patterns](#window-functions-common-patterns)
10. [Recursive CTEs](#recursive-ctes)
11. [LISTEN/NOTIFY vs Polling vs Message Queues](#listennotify-vs-polling-vs-message-queues)
12. [Advisory Locks](#advisory-locks)
13. [SKIP LOCKED: Job Queue Pattern](#skip-locked-job-queue-pattern)
14. [Common Gotchas](#common-gotchas)
15. [Index Types](#index-types)

---

## asyncpg vs psycopg3

### Q: What is the fundamental difference?

**asyncpg** speaks the PostgreSQL binary wire protocol natively and is async-only. It never implements DB-API 2.0, so there is no cursor object — all methods are on the connection directly (`conn.fetch`, `conn.execute`, etc.).

**psycopg3** (the `psycopg` package, not `psycopg2`) is a full DB-API 2.0 driver with both sync and async modes. It defaults to the text protocol but can use binary for specific types.

### Q: Which is faster?

asyncpg is generally 2–5x faster than psycopg3 for pure throughput due to binary protocol, no DB-API overhead, and a C extension that avoids the GIL for encoding/decoding. For most web applications, the difference is irrelevant compared to query time. asyncpg matters most for high-throughput services or workloads that move large volumes of data (COPY, bulk fetch).

### Q: When should I choose each?

| Situation | Choose |
|---|---|
| Pure async service (FastAPI, Starlette, asyncio) | asyncpg |
| Need sync support (Django ORM, Flask, scripts) | psycopg3 |
| DB-API 2.0 compatibility required | psycopg3 |
| SQLAlchemy 2.x async | psycopg3 (native) or asyncpg (via asyncpg dialect) |
| Maximum raw throughput, COPY operations | asyncpg |
| Familiarity / team prefers cursor pattern | psycopg3 |

### Q: How do parameterized queries differ?

```python
# asyncpg — PostgreSQL-native positional placeholders
await conn.fetch("SELECT * FROM users WHERE id = $1 AND tier = $2", 1, "gold")

# psycopg3 — DB-API %s style
cur.execute("SELECT * FROM users WHERE id = %s AND tier = %s", (1, "gold"))
# or named
cur.execute("SELECT * FROM users WHERE id = %(uid)s", {"uid": 1})
```

### Q: How do return types differ?

asyncpg returns `asyncpg.Record` objects (subscriptable by name or index, but NOT a dict). psycopg3 returns tuples by default; use `row_factory=psycopg.rows.dict_row` for dicts.

```python
# asyncpg
row = await conn.fetchrow("SELECT id, email FROM users WHERE id = $1", 1)
row["email"]   # ok
dict(row)      # explicit conversion to dict
row.get("email")  # NOT available — Record has no .get()

# psycopg3 with dict_row
with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
    row = conn.execute("SELECT id, email FROM users WHERE id = %s", (1,)).fetchone()
    row["email"]  # ok, it's a real dict
```

### Q: Feature comparison summary

| Feature | asyncpg | psycopg3 |
|---|---|---|
| Protocol | Binary (faster) | Text by default |
| Return type | asyncpg.Record | tuple (or dict w/ row_factory) |
| Sync support | No | Yes (and async) |
| Placeholders | $1, $2, ... | %s or %(name)s |
| Prepared statements | Explicit conn.prepare() | Automatic server-side |
| COPY API | copy_records_to_table() | cursor.copy("COPY ...") |
| Type registration | set_type_codec() | register_adapter() |
| Pool | asyncpg.create_pool() | psycopg_pool.AsyncConnectionPool() |
| DB-API 2.0 compat | No | Yes |

---

## Pool Sizing Guidelines

### Q: How many connections should I configure?

PostgreSQL performs best when the number of **active** connections is at most `2 x CPU cores` on the database server. Each connection is a separate OS process; too many causes context-switching overhead and memory pressure.

**Rule of thumb for `max_size`:**

| Scenario | Formula |
|---|---|
| CPU-bound queries (complex analytics) | `db_cpu_cores x 2` |
| IO-bound queries (simple CRUD, index lookups) | `db_cpu_cores x 4` to `x 8` |
| Behind PgBouncer (transaction pooling) | Set asyncpg `max_size` to match PgBouncer's pool size; keep DB connections low |

**Example:** 4-core database server running CRUD queries -> `max_size=16` per app instance. If you run 4 app instances, that is 64 connections total — likely too many. Add PgBouncer.

### Q: What do min_size and max_inactive_connection_lifetime do?

- `min_size`: connections created eagerly at pool startup and kept alive always. Set to the typical steady-state concurrency (e.g., 2–5).
- `max_inactive_connection_lifetime`: a connection idle longer than this (seconds) is closed and replaced. Prevents stale connections after a DB restart or firewall timeout.
- `max_queries`: recycle a connection after N queries. Prevents long-running connection state leaks.

### Q: How does PgBouncer interact with asyncpg?

PgBouncer sits between the app and PostgreSQL. In **transaction pooling** mode (most common):
- Each `pool.acquire()` call may get a different underlying PostgreSQL connection.
- Prepared statements (`conn.prepare()`) break because they are connection-local. Disable with `statement_cache_size=0` on asyncpg.
- Session-level advisory locks and `SET LOCAL` are lost after each transaction.

```python
pool = await asyncpg.create_pool(
    dsn,
    statement_cache_size=0,  # required for PgBouncer transaction mode
)
```

---

## Fetch Methods Decision Table

| Method | Returns | Use when |
|---|---|---|
| `execute()` | Status string e.g. `"UPDATE 5"` | DDL, UPDATE, DELETE, INSERT where you don't need the row back |
| `fetch()` | `list[Record]` (empty list if none) | Multiple rows: search results, reports, bulk processing |
| `fetchrow()` | `Record | None` | Exactly one row by primary key or unique constraint |
| `fetchval()` | scalar `| None` | `COUNT(*)`, `EXISTS(...)`, a single column from a single row |
| `executemany()` | `None` | Same DML with many parameter tuples; better than N `execute()` calls |

```python
# execute — returns status string, not data
status = await conn.execute("DELETE FROM orders WHERE status = $1", "cancelled")
# status == "DELETE 12"

# fetch — list of Records
orders = await conn.fetch("SELECT id, total FROM orders WHERE user_id = $1", 42)

# fetchrow — one Record or None
user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", "alice@example.com")

# fetchval — bare scalar
count = await conn.fetchval("SELECT count(*) FROM orders WHERE status = $1", "pending")

# executemany — bulk DML
await conn.executemany(
    "INSERT INTO products (sku, name, price) VALUES ($1, $2, $3)",
    [("A1", "Widget", 9.99), ("B2", "Gadget", 49.99)]
)
```

---

## Transaction Isolation Levels

### Q: What are the four isolation levels and what anomalies do they prevent?

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Serialization Anomaly |
|---|---|---|---|---|
| READ UNCOMMITTED | Prevented in PG* | Possible | Possible | Possible |
| READ COMMITTED (default) | Prevented | Possible | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Prevented in PG* | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented | Prevented |

*PostgreSQL implements READ UNCOMMITTED as READ COMMITTED. Its REPEATABLE READ also prevents phantoms, which is stricter than the SQL standard requires.

### Q: When should I use each?

- **READ COMMITTED (default):** CRUD operations, most web requests. Each statement sees data committed before that statement started.
- **REPEATABLE READ:** Reports or analytics that read the same table multiple times and need a consistent snapshot. Example: generating an invoice.
- **SERIALIZABLE:** Financial transfers, inventory deductions, any operation where two concurrent transactions must produce the same result as if they ran sequentially. Be prepared for serialization failure errors and retry.

```python
# SERIALIZABLE with retry loop
for attempt in range(5):
    try:
        async with conn.transaction(isolation="serializable"):
            stock = await conn.fetchval(
                "SELECT stock_qty FROM products WHERE id = $1 FOR UPDATE", pid
            )
            if stock > 0:
                await conn.execute(
                    "UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1", pid
                )
        break  # success
    except asyncpg.SerializationError:
        if attempt == 4:
            raise
        await asyncio.sleep(0.05 * (2 ** attempt))  # exponential backoff
```

---

## Cursor vs fetch(): Memory Tradeoffs

### Q: When should I use a server-side cursor instead of fetch()?

`fetch()` loads **all** matching rows into Python memory at once. For a query returning 10 million rows, that can be several gigabytes.

A server-side cursor fetches rows in batches (`prefetch`), keeping only N rows in memory at a time. Use it when:

- The result set is too large to fit in memory.
- You are streaming data to a file, queue, or downstream service.
- You want to start processing rows before the full query completes.

```python
# fetch() — all rows in RAM
orders = await conn.fetch("SELECT * FROM orders")           # could be huge

# cursor — streamed, prefetch 500 rows per round-trip
async with conn.transaction():
    async for row in conn.cursor("SELECT * FROM orders", prefetch=500):
        process(row)
```

### Q: What is the prefetch parameter?

`prefetch` controls how many rows asyncpg requests from PostgreSQL in each network round-trip. Larger values reduce round-trips (faster total throughput) but use more memory. The default is 50. For large result sets, 200–1000 is typical.

### Q: Does the cursor require a transaction?

Yes. PostgreSQL requires a cursor to be inside a transaction block. asyncpg enforces this.

---

## COPY vs executemany vs INSERT

### Q: Which method should I use for bulk inserts?

| Method | Relative Throughput | Best For |
|---|---|---|
| N x `execute()` (loop) | 1x (baseline, slowest) | 1–10 rows, simple scripts |
| `executemany()` | 5–20x | Hundreds of rows; single statement, many arg tuples |
| `copy_records_to_table()` | 50–200x | Thousands+ rows from Python lists |
| `copy_to_table()` (from file/stream) | 100–300x | CSV/TSV files, ETL pipelines |

The COPY protocol bypasses SQL parsing and trigger overhead (unless triggers are defined). It is the fastest path for bulk data loading.

```python
# executemany — good for ~hundreds of rows
await conn.executemany(
    "INSERT INTO users (email, name) VALUES ($1, $2)",
    [("a@example.com", "Alice"), ("b@example.com", "Bob")]
)

# copy_records_to_table — best for thousands of rows from Python
await conn.copy_records_to_table(
    "users",
    records=[("a@example.com", "Alice"), ("b@example.com", "Bob")],
    columns=["email", "name"],
)
```

### Q: Does COPY support conflict handling (upsert)?

No. COPY always inserts. For upsert with bulk data, load into a temporary table with COPY, then `INSERT INTO target SELECT ... FROM tmp ON CONFLICT DO UPDATE`.

---

## Full-Text Search

### Q: tsvector vs LIKE vs pg_trgm — when to use which?

| Method | Indexable | Stemming | Ranking | Best For |
|---|---|---|---|---|
| `LIKE '%term%'` | No (seq scan) | No | No | Simple prefix/suffix match on small tables |
| `pg_trgm` (`%` operator) | GIN/GiST | No | Similarity score | Fuzzy match, typos, autocomplete |
| `tsvector / tsquery` | GIN | Yes (English, etc.) | `ts_rank` | Natural language search, multi-word, phrase |

Use `tsvector` when you need language-aware keyword search with ranking. Use `pg_trgm` when you need fuzzy matching or autocomplete (handles misspellings). Use `LIKE` only for exact prefix searches on short indexed columns.

### Q: What is the difference between to_tsquery, websearch_to_tsquery, and phraseto_tsquery?

- `to_tsquery('english', 'widget & great')` — requires explicit `&`/`|`/`!` operators; parse error on natural language input.
- `websearch_to_tsquery('english', 'great widget -broken')` — Google-style; spaces mean AND, `-` means NOT, quotes mean phrase. Best for user-facing search boxes.
- `phraseto_tsquery('english', 'great widget')` — enforces that terms appear in this exact order and adjacent.

```python
# User types in a search box — use websearch_to_tsquery
rows = await conn.fetch(
    "SELECT id, name FROM products WHERE search_vec @@ websearch_to_tsquery('english', $1)",
    user_input
)
```

### Q: How do I add a full-text search column efficiently?

Use a `GENERATED ALWAYS AS ... STORED` column so PostgreSQL maintains the tsvector automatically:

```sql
ALTER TABLE products ADD COLUMN search_vec TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(name,'') || ' ' || coalesce(description,''))
    ) STORED;

CREATE INDEX idx_products_search ON products USING GIN (search_vec);
```

---

## JSONB vs Separate Columns

### Q: When should I store data as JSONB instead of normalized columns?

| Prefer JSONB | Prefer Columns |
|---|---|
| Schema varies per row (product attributes differ by category) | Schema is stable and well-known |
| Sparse attributes (most rows have NULL for most fields) | Most fields have values for most rows |
| Read the whole blob at once; rarely filter on individual keys | Frequently filter, sort, or join on individual fields |
| Prototyping — evolve schema without migrations | Production — need query planner statistics, foreign keys, constraints |

### Q: How do I index JSONB efficiently?

- **GIN on the whole column** (`CREATE INDEX ON products USING GIN (tags)`) — supports `@>`, `?`, `?|`, `?&`. Best for containment and existence queries.
- **Expression index on a specific path** (`CREATE INDEX ON users ((metadata ->> 'tier'))`) — supports `=` and `<` on a single extracted field; much smaller than a full GIN index.

```sql
-- GIN for containment
CREATE INDEX idx_products_tags ON products USING GIN (tags);
-- Expression index for a known key
CREATE INDEX idx_users_tier ON users ((metadata ->> 'tier'));
```

### Q: What are the query operator gotchas?

- `->>`  returns **text**, not typed. Cast explicitly: `(metadata ->> 'price')::numeric`.
- `@>` only works if both sides are JSONB: `WHERE metadata @> '{"tier": "gold"}'::jsonb`.
- `jsonb_set` replaces a key's value; it does not merge nested objects.

---

## Window Functions: Common Patterns

### Running total

```sql
SUM(total) OVER (PARTITION BY user_id ORDER BY created_at
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
```

### Rank within group (with and without gap handling)

```sql
RANK()       OVER (PARTITION BY category ORDER BY price DESC) AS rank_with_gaps
DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rank_no_gaps
```

### Moving average (last 3 rows)

```sql
AVG(total) OVER (PARTITION BY user_id ORDER BY created_at
                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS avg_3
```

### Previous and next row values

```sql
LAG(total,  1, 0) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_total
LEAD(total, 1, 0) OVER (PARTITION BY user_id ORDER BY created_at) AS next_total
```

### Gaps and islands (finding consecutive sequences)

```sql
-- Assign a group number to consecutive rows with the same status
ROW_NUMBER() OVER (ORDER BY created_at)
- ROW_NUMBER() OVER (PARTITION BY status ORDER BY created_at) AS grp
```

### Percent rank and distribution buckets

```sql
PERCENT_RANK() OVER (PARTITION BY category ORDER BY price) AS pct_rank
NTILE(10)      OVER (ORDER BY price)                       AS decile
```

### Named windows (avoid repeating PARTITION BY)

```sql
SELECT ..., SUM(total) OVER w, AVG(total) OVER w
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY created_at)
```

---

## Recursive CTEs

### Q: When should I use a recursive CTE vs application-side recursion?

Use a recursive CTE when:
- The hierarchy lives entirely in PostgreSQL and you need to traverse it in SQL (category trees, org charts, bill of materials, graph traversal).
- You want to avoid N+1 queries fetching level by level in application code.
- The depth is bounded and the table is not enormous.

Use application-side recursion when:
- The hierarchy is already in memory.
- You need complex logic between levels that is awkward in SQL.
- The tree is very deep (recursive CTEs have a default cycle limit).

```sql
WITH RECURSIVE tree AS (
    -- Anchor: root nodes
    SELECT id, parent_id, name, 0 AS depth
    FROM categories WHERE parent_id IS NULL

    UNION ALL

    -- Recursive: children
    SELECT c.id, c.parent_id, c.name, t.depth + 1
    FROM categories c
    JOIN tree t ON t.id = c.parent_id
)
SELECT * FROM tree ORDER BY depth, name;
```

### Q: How do I detect cycles in a recursive CTE?

Track visited IDs in an array and stop when the current ID is already in it:

```sql
WITH RECURSIVE tree AS (
    SELECT id, parent_id, ARRAY[id] AS visited, false AS is_cycle
    FROM categories WHERE parent_id IS NULL

    UNION ALL

    SELECT c.id, c.parent_id, t.visited || c.id, c.id = ANY(t.visited)
    FROM categories c
    JOIN tree t ON t.id = c.parent_id
    WHERE NOT t.is_cycle
)
SELECT * FROM tree WHERE NOT is_cycle;
```

---

## LISTEN/NOTIFY vs Polling vs Message Queues

### Q: When should I use LISTEN/NOTIFY?

LISTEN/NOTIFY is ideal when:
- You need **low-latency** notification that something changed (cache invalidation, real-time dashboard).
- The payload is small (max 8000 bytes).
- Missed notifications are acceptable (no persistence — if the listener is disconnected, it misses events).
- You want to avoid a separate infrastructure dependency.

### Q: What are its limitations?

- **No persistence:** if no listener is connected when NOTIFY fires, the message is lost.
- **No delivery guarantee:** at-most-once.
- **8000-byte payload limit.**
- **Requires a dedicated long-lived connection** — cannot share a pool connection that might be returned to the pool mid-listen.

### Q: When should I use polling instead?

Polling (e.g., `SELECT * FROM jobs WHERE status = 'pending' LIMIT 1`) is simpler and works when:
- Latency of a few seconds is acceptable.
- You want persistence (jobs stay in the table until processed).
- Correctness matters more than immediate notification.

Combine with `SKIP LOCKED` to make polling safe for concurrent workers.

### Q: When should I use a real message queue (Redis, RabbitMQ, Kafka)?

- You need guaranteed delivery and persistence across restarts.
- You need fan-out to multiple consumer groups.
- Payload volumes are high.
- You need ordered processing at scale.
- You want dead-letter queues, retry policies, or consumer group semantics.

LISTEN/NOTIFY is a lightweight convenience feature, not a replacement for a proper message broker.

---

## Advisory Locks

### Q: Session vs transaction scope — what is the difference?

| Scope | Acquired by | Released by | Survives COMMIT? |
|---|---|---|---|
| Session | `pg_advisory_lock($1)` | `pg_advisory_unlock($1)` or connection close | Yes |
| Transaction | `pg_advisory_xact_lock($1)` | Automatic at COMMIT or ROLLBACK | No |

Use **session scope** when the lock must span multiple transactions (e.g., a cron job that holds a lock for its entire run). Use **transaction scope** when you want the lock to be held only during a single transaction — safer because PostgreSQL releases it automatically.

### Q: What is the two-argument form?

`pg_advisory_lock(class_id, object_id)` lets you namespace locks. Use the class to represent a lock type (e.g., a table) and the object to represent a specific row ID.

```sql
-- Lock "product inventory" (class 100) for product id 42
SELECT pg_advisory_xact_lock(100, 42);
```

### Q: How do I implement a distributed singleton task?

```python
async def run_as_singleton(conn, task_id: int):
    acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", task_id)
    if not acquired:
        return  # another instance is running
    try:
        await do_the_work()
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", task_id)
```

---

## SKIP LOCKED: Job Queue Pattern

### Q: How does SKIP LOCKED work?

`FOR UPDATE SKIP LOCKED` attempts to lock matching rows but **skips** any row already locked by another transaction instead of blocking. Multiple workers can run concurrently — each claims a different job.

```sql
-- Worker claims one job atomically
SELECT id, payload
FROM job_queue
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

### Q: What is the full claim-process-complete pattern?

```python
async with conn.transaction():
    job = await conn.fetchrow(
        """
        SELECT id, payload FROM job_queue
        WHERE status = 'pending'
        ORDER BY created_at LIMIT 1
        FOR UPDATE SKIP LOCKED
        """
    )
    if not job:
        return  # nothing to do
    await conn.execute(
        "UPDATE job_queue SET status = 'processing' WHERE id = $1", job["id"]
    )
# Transaction commits — row is now 'processing'

try:
    await process(job["payload"])
    await conn.execute(
        "UPDATE job_queue SET status = 'done' WHERE id = $1", job["id"]
    )
except Exception:
    await conn.execute(
        "UPDATE job_queue SET status = 'failed' WHERE id = $1", job["id"]
    )
```

### Q: SKIP LOCKED vs advisory locks — which should I use for a job queue?

- **SKIP LOCKED** is simpler: the row IS the lock. Works well when the job table is the source of truth.
- **Advisory locks** work when you cannot modify the table or when the lock must outlive the row (e.g., after the row is deleted).
- Both are better than application-level mutexes or polling without locking.

---

## Common Gotchas

### Q: asyncpg uses $1 not ? — why does my query fail?

asyncpg uses PostgreSQL's native protocol which uses `$1`, `$2`, ... positional placeholders. The `?` placeholder belongs to DB-API 2.0 (psycopg, sqlite3). Always use `$N` with asyncpg.

### Q: asyncpg.Record is not a dict

`asyncpg.Record` supports `row["col"]` and `row[0]` but does NOT support `.get()` or `json.dumps()` directly. Convert with `dict(row)` when you need a real dict.

```python
row = await conn.fetchrow("SELECT id, email FROM users WHERE id = $1", 1)
d = dict(row)           # convert to dict
```

### Q: Pool exhaustion — all workers are waiting

Symptoms: requests hang or time out waiting for a connection. Causes:
- A long-running transaction holds a connection from the pool.
- `max_size` is too small for the concurrency.
- A slow query blocks progress.

Fix: always release connections quickly; avoid holding a connection across user-facing I/O; size the pool correctly. Add `command_timeout` to kill runaway queries.

### Q: Idle connections are closing unexpectedly

If your cloud provider (RDS, Cloud SQL, Neon) has an idle connection timeout, connections older than that are killed at the network level, but asyncpg still thinks they are alive. Symptoms: `ConnectionDoesNotExistError` on the next query.

Fix: set `max_inactive_connection_lifetime` shorter than the provider's timeout (e.g., 300 seconds), or use `keepalives_idle` in the DSN.

### Q: NUMERIC columns come back as Python Decimal, not float

asyncpg decodes PostgreSQL `NUMERIC`/`DECIMAL` as Python `Decimal` to preserve exact precision. If you want a float, cast in SQL (`SELECT price::float8`) or in Python (`float(row["price"])`). Never store money as float.

### Q: TIMESTAMPTZ vs TIMESTAMP — which should I use?

Always use `TIMESTAMPTZ` (timestamp with time zone). PostgreSQL stores it as UTC internally and converts to the session's `TimeZone` setting on output. `TIMESTAMP` (no timezone) is a naive datetime — it stores whatever you give it with no conversion. Using `TIMESTAMP` leads to subtle timezone bugs in production.

asyncpg returns `TIMESTAMPTZ` as a timezone-aware Python `datetime` (UTC). `TIMESTAMP` comes back as a naive `datetime`.

```python
row = await conn.fetchrow("SELECT created_at FROM orders WHERE id = $1", 1)
row["created_at"]  # datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
```

### Q: Why does fetchval return None instead of a value?

`fetchval` returns `None` when the query returns **no rows** (not the same as a row with a NULL value). Always check for None on queries that may match nothing.

---

## Index Types

### Q: Which PostgreSQL index type should I use?

| Type | Operator support | Best for |
|---|---|---|
| **B-tree** (default) | `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `LIKE 'prefix%'`, `IS NULL` | Most columns; range queries; the default choice |
| **Hash** | `=` only | Equality-only lookups on large tables; slightly smaller than B-tree for this case |
| **GIN** | `@>`, `<@`, `?`, `?|`, `?&`, `@@` (full-text), `&&` (arrays) | JSONB columns, array columns, full-text search tsvector |
| **GiST** | Geometric, range types, full-text (`@@`), nearest-neighbor | Geospatial (PostGIS), range overlaps, pgvector ANN search |
| **BRIN** | Range min/max per block | Very large append-only tables with natural physical ordering (e.g., time-series, log tables) |
| **SP-GiST** | Geometric, range, text prefix | Space-partitioned structures; good for non-overlapping data like IP ranges |

### Q: When should I use a GIN index?

GIN (Generalized Inverted Index) is the right choice for:
- `tsvector` columns with full-text search operators (`@@`).
- `JSONB` columns with containment (`@>`) or existence (`?`) operators.
- Array columns with overlap (`&&`) or containment (`@>`) operators.

GIN indexes are large and slow to build/update but very fast to query. Use `fastupdate=on` (the default) to buffer writes; run `VACUUM` regularly.

### Q: When should I use a BRIN index?

BRIN (Block Range INdex) stores only the minimum and maximum value per range of table blocks. It is tiny (a few KB even for a billion-row table) but only useful when the column's physical order on disk matches the query range — most commonly:
- Append-only timestamp columns in log or event tables.
- Auto-increment IDs in append-only tables.

For random-order data, BRIN gives no benefit over a seq scan.

### Q: When do partial indexes help?

A partial index covers only rows matching a WHERE condition. It is smaller, faster to scan, and cheaper to maintain than a full index.

```sql
-- Index only pending jobs (the hot path)
CREATE INDEX idx_pending_jobs ON job_queue (created_at) WHERE status = 'pending';

-- Unique constraint only for non-deleted records
CREATE UNIQUE INDEX idx_active_email ON users (email) WHERE deleted_at IS NULL;
```

### Q: When do expression indexes help?

Index the result of an expression so queries using that expression can use the index:

```sql
-- Queries on lower(email) can use this index
CREATE INDEX idx_users_email_lower ON users (lower(email));
-- Now this is an index scan, not a seq scan:
-- SELECT * FROM users WHERE lower(email) = lower($1)
```
