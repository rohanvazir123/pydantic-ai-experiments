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
16. [PARTITION BY: Window Function Frame Clauses](#partition-by-window-function-frame-clauses)
17. [Table Partitioning](#table-partitioning)
18. [TOAST: Automatic Column-Level Storage](#toast-automatic-column-level-storage)
19. [JSONB: Advanced Patterns](#jsonb-advanced-patterns)
20. [BYTEA: Binary Data Storage](#bytea-binary-data-storage)
21. [MVCC and Pessimistic Locking](#mvcc-and-pessimistic-locking)
22. [Change Stream (Trigger + LISTEN/NOTIFY)](#change-stream-trigger--listennotify)
23. [Read-After-Write Consistency](#read-after-write-consistency)

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

## Change Stream (Trigger + LISTEN/NOTIFY)

PostgreSQL has no native change stream API, but the same pattern is achievable by combining
a PL/pgSQL trigger with `NOTIFY` and `LISTEN`.

### How it works

1. A `AFTER INSERT OR UPDATE OR DELETE` trigger fires on every row change.
2. The trigger function builds a JSON payload from `OLD`/`NEW` and calls `pg_notify(channel, payload)`.
3. Any client that has run `LISTEN channel` on a dedicated connection receives the event asynchronously.

```sql
CREATE OR REPLACE FUNCTION orders_change_notify()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE payload JSONB;
BEGIN
    IF TG_OP = 'DELETE' THEN
        payload := jsonb_build_object('op', 'DELETE', 'id', OLD.id, 'status', OLD.status);
    ELSIF TG_OP = 'INSERT' THEN
        payload := jsonb_build_object('op', 'INSERT', 'id', NEW.id, 'status', NEW.status);
    ELSE
        payload := jsonb_build_object('op', 'UPDATE', 'id', NEW.id,
                                      'old_status', OLD.status, 'new_status', NEW.status);
    END IF;
    PERFORM pg_notify('order_changes', payload::TEXT);
    RETURN NEW;
END;
$$;

CREATE TRIGGER orders_change_trigger
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION orders_change_notify();
```

### asyncpg listener

```python
def on_change(conn, pid, channel, payload):
    event = json.loads(payload)
    print(event["op"], event["id"])

listener_conn = await asyncpg.connect(dsn)
await listener_conn.add_listener("order_changes", on_change)
# ... run your writes on another connection ...
await asyncio.sleep(0.1)   # let the event loop deliver notifications
await listener_conn.remove_listener("order_changes", on_change)
await listener_conn.close()
```

`LISTEN` requires a **dedicated connection** that stays open — do not use a pooled connection for it, as `asyncpg.Pool` may recycle or reuse it.

### LISTEN/NOTIFY vs WAL-based CDC

| Feature | LISTEN/NOTIFY | WAL / Debezium |
|---|---|---|
| **Durability** | None — fire & forget | Durable (replication slot) |
| **Resume / replay** | No | Yes (LSN position) |
| **Payload size** | 8 000 bytes | Unlimited |
| **Schema changes** | Manual in trigger | Automatic |
| **Setup complexity** | Low (SQL trigger) | High (connector infra) |
| **Replica set needed** | No | No (logical replication on any PG) |
| **Best for** | Cache busting, simple hooks | Full CDC, Kafka, audit pipelines |

### Compared to MongoDB change streams

| | MongoDB | PostgreSQL |
|---|---|---|
| **Mechanism** | Oplog-backed cursor | Trigger + pg_notify |
| **Resume token** | Yes (`_id` of each event) | No |
| **Scope** | Collection / DB / cluster | Per-table trigger |
| **Durability** | At-least-once (oplog) | None |
| **Payload limit** | 16 MB (BSON doc limit) | 8 000 bytes |
| **Infrastructure** | Replica set required | Any PG instance |

For durable, replayable CDC from PostgreSQL use [Debezium](https://debezium.io/) with the `pgoutput` plugin (built into PG 10+) — no extra extensions needed.

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

---

## PARTITION BY: Window Function Frame Clauses

`PARTITION BY` inside a window function divides rows into independent groups; the function
resets and recalculates for each partition. It is fundamentally different from `GROUP BY`:
`GROUP BY` collapses rows into one summary row per group; `PARTITION BY` preserves every
input row while making group-level metrics available alongside row-level data.

### `$` vs `$$` — field reference vs variable (PostgreSQL analogy)

In window function syntax:
- `PARTITION BY col` — partition boundary, resets the window per distinct value of `col`
- `ORDER BY col` — sort order within each partition (required for ranking/running functions)
- Frame clause — controls which rows within the partition are visible to the function

### Frame clause units

| Unit | Counts | Tie handling |
|---|---|---|
| `ROWS` | Physical row positions | Each row is distinct regardless of ORDER BY value |
| `RANGE` | Rows sharing the same ORDER BY value | Peer rows are all included in each other's frame |
| `GROUPS` | Distinct peer groups (PG 11+) | Like RANGE but counts groups, not rows |

```sql
-- ROWS: each row is a distinct position — running total ticks up per row
SUM(total) OVER (
    PARTITION BY user_id ORDER BY created_at
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)

-- RANGE: on a day with 3 orders, all three show the same running total
-- (the entire day's amount) because they are ORDER BY peers
SUM(total) OVER (
    PARTITION BY user_id ORDER BY created_at::date
    RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)

-- GROUPS: "1 PRECEDING" means the previous peer group, not the previous row
SUM(total) OVER (
    PARTITION BY user_id ORDER BY created_at::date
    GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW   -- current day + previous day
)
```

### Frame boundary keywords

```
UNBOUNDED PRECEDING  — start of the partition
<n> PRECEDING        — n rows/range-units/groups before current row
CURRENT ROW          — the current row (or its peer group for RANGE/GROUPS)
<n> FOLLOWING        — n rows/range-units/groups after current row
UNBOUNDED FOLLOWING  — end of the partition
```

### Common patterns

**Centred moving average (1 preceding + current + 1 following):**
```sql
AVG(total) OVER (
    PARTITION BY user_id ORDER BY created_at
    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
) AS centred_3row_avg
```

**Multi-column partition (rank within user × status):**
```sql
RANK() OVER (
    PARTITION BY user_id, status   -- independent window per (user, status) pair
    ORDER BY total DESC
) AS rank_within_user_status
```

**Share of group total (no ORDER BY needed — full partition frame):**
```sql
total / SUM(total) OVER (PARTITION BY user_id) AS share_of_user_spend
```
No `ORDER BY` means the frame is the entire partition by default — correct for
proportions and averages, wrong for running totals (which need `ORDER BY` to be meaningful).

**Named window — define once, reference many times:**
```sql
SELECT
    ROW_NUMBER() OVER w          AS row_num,
    RANK()       OVER w          AS rank,
    SUM(total)   OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY created_at)
```
A named window can be refined in the `OVER()` clause with an additional frame clause,
but cannot change the `PARTITION BY` or `ORDER BY`.

### `LAST_VALUE` gotcha

`LAST_VALUE` without an explicit frame only sees up to the current row (default frame is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`). To get the true last value in the
partition you must extend the frame:

```sql
-- WRONG — returns current row's total, not the last one
LAST_VALUE(total) OVER (PARTITION BY user_id ORDER BY created_at)

-- CORRECT — extend frame to end of partition
LAST_VALUE(total) OVER (
    PARTITION BY user_id ORDER BY created_at
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
```

---

## Table Partitioning

PostgreSQL declarative table partitioning (PG 10+) splits a logical parent table into
physical child tables called partitions. The planner uses **partition pruning** to skip
partitions whose ranges cannot satisfy the query filter — a major performance tool for
large tables.

### Three strategies

| Strategy | Syntax | Best for |
|---|---|---|
| `RANGE` | `FOR VALUES FROM (start) TO (end)` | Dates, timestamps, sequential IDs |
| `LIST` | `FOR VALUES IN (val1, val2, ...)` | Enumerations: region, status, tenant |
| `HASH` | `FOR VALUES WITH (MODULUS n, REMAINDER r)` | Even distribution, no natural range |

### RANGE partitioning by date

```sql
-- Parent: holds schema and partition key, no rows
CREATE TABLE order_log (
    id         BIGSERIAL,
    user_id    INT          NOT NULL,
    total      NUMERIC      NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (id, created_at)   -- partition key must be part of every unique key
) PARTITION BY RANGE (created_at);

-- Children: FOR VALUES FROM (inclusive) TO (exclusive)
CREATE TABLE order_log_2023 PARTITION OF order_log
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE order_log_2024 PARTITION OF order_log
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- DEFAULT catches anything outside all explicit ranges
-- Without this, inserting out-of-range raises an error
CREATE TABLE order_log_default PARTITION OF order_log DEFAULT;

-- Index on parent propagates to all partitions automatically (PG 11+)
CREATE INDEX ON order_log (user_id, created_at);
```

**Partition pruning** — queries with a filter on the partition key skip irrelevant partitions:
```sql
-- Scans ONLY order_log_2024; other partitions skipped
SELECT * FROM order_log
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';

-- Verify with EXPLAIN — look for Append node listing only the relevant child
EXPLAIN SELECT * FROM order_log WHERE created_at >= '2024-01-01';
```

**Which partition holds each row:**
```sql
SELECT tableoid::regclass AS partition, id, created_at
FROM order_log ORDER BY created_at;
```

### LIST partitioning by region

```sql
CREATE TABLE regional_orders (
    id     BIGSERIAL,
    region TEXT    NOT NULL,
    amount NUMERIC NOT NULL,
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

CREATE TABLE regional_orders_eu PARTITION OF regional_orders
    FOR VALUES IN ('DE', 'FR', 'NL', 'ES', 'IT');

CREATE TABLE regional_orders_us PARTITION OF regional_orders
    FOR VALUES IN ('US', 'CA', 'MX');
```

### HASH partitioning for even distribution

```sql
CREATE TABLE events (
    id      BIGSERIAL,
    user_id INT  NOT NULL,
    event   TEXT NOT NULL,
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

-- 3 partitions — MODULUS = total, REMAINDER = this partition's slot
CREATE TABLE events_0 PARTITION OF events FOR VALUES WITH (MODULUS 3, REMAINDER 0);
CREATE TABLE events_1 PARTITION OF events FOR VALUES WITH (MODULUS 3, REMAINDER 1);
CREATE TABLE events_2 PARTITION OF events FOR VALUES WITH (MODULUS 3, REMAINDER 2);
```

### Attach / detach for zero-downtime archiving

```sql
-- Detach makes the partition a standalone table instantly — no data copy, no lock on other partitions
ALTER TABLE order_log DETACH PARTITION order_log_2023;
-- PG 14+: non-blocking
ALTER TABLE order_log DETACH PARTITION order_log_2023 CONCURRENTLY;

-- Re-attach (child must satisfy the range constraint)
ALTER TABLE order_log ATTACH PARTITION order_log_2023
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
```

Detach + dump + drop is the standard pattern for archiving old time partitions without
a slow `DELETE` that bloats WAL and stalls vacuum.

### Constraints and gotchas

| Constraint | Detail |
|---|---|
| Partition key in every unique key | `PRIMARY KEY (id, created_at)` — not just `(id)` |
| No FK pointing TO a partitioned table | Supported from PG 16 with restrictions; avoid in earlier versions |
| Indexes created per partition | `CREATE INDEX ON parent` fans out automatically (PG 11+) |
| `DEFAULT` partition blocks adding new partitions | New explicit partition must not overlap with existing `DEFAULT` rows — check first |
| `VACUUM` / `ANALYZE` per partition | Run on children for fine-grained control; running on parent covers all |
| Sub-partitioning | A partition can itself be partitioned — e.g. RANGE by year → LIST by region |

---

## TOAST: Automatic Column-Level Storage

TOAST (The Oversized Attribute Storage Technique) is PostgreSQL's transparent mechanism
for storing values larger than ~2 KB out-of-line, in a separate TOAST table. You never
interact with it directly — it is completely automatic.

### How it works

PostgreSQL's page size is 8 KB. A row must fit on a single page. When a column value
would push the row over the page limit, PostgreSQL applies one or more TOAST strategies:

```
Column value > ~2 KB (TOAST_TUPLE_THRESHOLD)
    ↓
1. Compress in-line (pglz or lz4)           ← try this first
   If compressed value still > page limit:
2. Move value to TOAST table (out-of-line)  ← transparent pointer left in main row
   Main row stores a 4-byte pointer only.
```

The TOAST table (`pg_toast.<oid>`) is a separate heap with its own storage, vacuum,
and indexes. Reads transparently fetch and decompress the value.

### TOAST storage strategies (per column)

Set with `ALTER TABLE ... ALTER COLUMN ... SET STORAGE ...`:

| Strategy | Compresses? | Moves out-of-line? | Best for |
|---|---|---|---|
| `PLAIN` | No | No | Small values that must never be TOASTed (e.g. INT, BOOL) |
| `MAIN` | Yes | Only as last resort | Columns you want to keep in the main row if possible |
| `EXTENDED` | Yes | Yes (default for TEXT, JSONB, BYTEA) | Large variable-length columns |
| `EXTERNAL` | No | Yes | Large values where you want fast substring access (no decompress) |

```sql
-- Check current strategy for each column
SELECT attname, attstorage
FROM pg_attribute
WHERE attrelid = 'my_table'::regclass AND attnum > 0;
-- p = PLAIN, m = MAIN, x = EXTENDED, e = EXTERNAL

-- Change to EXTERNAL for a BYTEA column where you need fast slicing
ALTER TABLE documents ALTER COLUMN raw_bytes SET STORAGE EXTERNAL;

-- Use EXTENDED (default) for JSONB and TEXT — compression usually helps
ALTER TABLE events ALTER COLUMN payload SET STORAGE EXTENDED;
```

### What TOAST means for query performance

```sql
-- TOAST columns are fetched only when selected — excluded columns cost nothing
SELECT id, created_at FROM documents;          -- TOAST body column not read at all
SELECT id, created_at, body FROM documents;    -- body fetched + decompressed per row

-- EXTERNAL storage lets PostgreSQL read a substring without decompressing the whole value
-- (useful for very large TEXT/BYTEA when you only need a prefix)
SELECT substring(raw_bytes FROM 1 FOR 100) FROM documents;
-- With EXTENDED: decompresses full value first → slower for large values
-- With EXTERNAL: reads only the needed bytes from TOAST → faster for large values
```

### Contrast with MongoDB's 16 MB document limit

| | PostgreSQL TOAST | MongoDB |
|---|---|---|
| Per-value size limit | ~1 GB (practical) | 16 MB total document |
| Overflow handling | Automatic, transparent | Manual: GridFS, bucketing, or error |
| Compression | pglz or lz4, automatic | None (BSON is uncompressed) |
| Partial reads | Yes (`EXTERNAL` strategy) | No |
| User awareness required | Zero | Must architect around the limit |

### TOAST and JSONB

`JSONB` uses `EXTENDED` storage by default. Large JSONB documents are compressed and
moved out of line automatically. This means:

```sql
-- This never hits a hard size limit the way MongoDB does
-- (practical limit ~255 MB before other constraints apply)
UPDATE events SET payload = $1 WHERE id = $2;   -- 50 MB JSONB: fine

-- But fetching many rows with large JSONB is expensive — select only what you need
SELECT id, payload->>'status' FROM events;      -- reads only the key, but still decompresses full payload
-- Better: store hot scalar fields as real columns, keep JSONB for the variable remainder
```

### asyncpg and TOAST

asyncpg reads TOAST values transparently — there is nothing special to do. The only
performance consideration is avoiding `SELECT *` on tables with large TOAST columns when
you do not need them, since fetching and decompressing is done per row per column selected.

---

## JSONB: Advanced Patterns

### SQL/JSON path queries (PG 12+)

`jsonb_path_query` and `jsonb_path_exists` support a richer path syntax than `->` / `#>`:

```sql
-- Extract city from nested address
SELECT jsonb_path_query(metadata, '$.address.city') FROM users;

-- Filter rows where tags array contains "vip"
SELECT id FROM users
WHERE jsonb_path_exists(metadata, '$.tags[*] ? (@ == "vip")');
```

### Merge, delete, and strip

```sql
-- || shallow merge (right-side wins on duplicate keys)
UPDATE users SET metadata = metadata || '{"tier": "gold", "verified": true}'::jsonb
WHERE id = 1;

-- - delete a key
UPDATE users SET metadata = metadata - 'temp_flag' WHERE id = 1;

-- jsonb_strip_nulls — remove keys whose value is null
SELECT jsonb_strip_nulls('{"a": 1, "b": null, "c": {"d": null, "e": 2}}'::jsonb);
-- → {"a": 1, "c": {"e": 2}}
```

### Expand to rows

```sql
-- jsonb_each_text: top-level keys → (key text, value text) rows
SELECT u.id, kv.key, kv.value
FROM users u, jsonb_each_text(u.metadata) AS kv
WHERE u.id = 1;

-- jsonb_array_elements: JSON array → one row per element
SELECT id, elem->>'name' AS tag_name
FROM users, jsonb_array_elements(metadata->'tags') AS elem
WHERE metadata ? 'tags';
```

### Generated columns for hot scalar fields

Extracting a scalar field on every query decompresses the full JSONB value per row.
Generated columns extract the value once at write time and store it as a real column:

```sql
ALTER TABLE users
    ADD COLUMN tier TEXT GENERATED ALWAYS AS (metadata->>'tier') STORED;

CREATE INDEX ON users (tier);   -- normal B-tree index, no JSONB decompression at query time
-- SELECT * FROM users WHERE tier = 'gold'  → index scan
```

Use this pattern for any JSONB field that appears in `WHERE`, `ORDER BY`, or `JOIN` clauses.

### GIN index strategies

```sql
-- Default (jsonb_ops): supports @>, ?, ?|, ?&, @?
CREATE INDEX ON users USING GIN (metadata);

-- jsonb_path_ops: smaller index, faster for @> and @? only
CREATE INDEX ON users USING GIN (metadata jsonb_path_ops);
```

Use `jsonb_path_ops` when your queries exclusively use containment (`@>`) — the index
is ~30–40% smaller and faster for that operator. Use the default when you also need
key-existence (`?`, `?|`, `?&`).

---

## BYTEA: Binary Data Storage

`BYTEA` stores arbitrary binary data. asyncpg maps it to Python `bytes`. No MongoDB-style
size wall — TOAST handles large values transparently up to ~1 GB per column.

### Basic insert and retrieve

```python
import hashlib

sample_pdf = b"%PDF-1.4 ..."
sha = hashlib.sha256(sample_pdf).digest()   # 32 raw bytes

doc_id = await conn.fetchval(
    "INSERT INTO documents (name, content, sha256) VALUES ($1, $2, $3) RETURNING id",
    "sample.pdf", sample_pdf, sha            # asyncpg accepts bytes directly
)

row = await conn.fetchrow("SELECT content, sha256 FROM documents WHERE id = $1", doc_id)
content: bytes = bytes(row["content"])       # asyncpg returns memoryview — coerce to bytes
```

### Hex encode / decode in SQL

```sql
-- bytes → hex string for display or API response
SELECT encode(sha256, 'hex') AS sha256_hex FROM documents WHERE id = $1;

-- hex string → bytes
SELECT decode('deadbeef', 'hex');
```

### Partial reads and STORAGE EXTERNAL

```sql
-- Read just the first 8 bytes (file magic) without fetching the full value
SELECT substring(content FROM 1 FOR 8) FROM documents WHERE id = $1;
```

With default `EXTENDED` storage, the full column is decompressed before slicing.
With `EXTERNAL`, PostgreSQL reads only the requested bytes from TOAST — much faster
for large blobs when you only need a prefix:

```sql
ALTER TABLE documents ALTER COLUMN content SET STORAGE EXTERNAL;
```

### Length, hashing, deduplication

```sql
-- Size without fetching data
SELECT octet_length(content) FROM documents WHERE id = $1;

-- Server-side hash comparison (PG 11+)
SELECT sha256(content) = sha256 FROM documents WHERE id = $1;

-- Hash index for O(1) deduplication
CREATE INDEX ON documents USING HASH (sha256);
SELECT id FROM documents WHERE sha256 = $1;   -- exists check before insert
```

### BYTEA vs TEXT vs Large Object vs object store

| Approach | Max size | Streaming | SQL ops | Best for |
|---|---|---|---|---|
| `BYTEA` | ~1 GB | No | Yes | Files < ~50 MB, transactional, simple |
| `TEXT` | ~1 GB | No | Yes | Text content |
| Large Object (`lo_*`) | 4 TB | Yes | Limited | Multi-GB files; awkward API |
| External (S3/GCS) + URL column | Unlimited | Yes | No | Everything > 50 MB |

**Rule:** use `BYTEA` for files under ~50 MB where you want transactional guarantees and
simple SQL access. Store only the object store URL for larger files. Large Objects
(`pg_largeobject`) are rarely the right choice — separate vacuuming, not replicated via
logical replication, awkward client API.

---

## MVCC and Pessimistic Locking

### How MVCC works

PostgreSQL uses **Multi-Version Concurrency Control (MVCC)**. Instead of locking rows
for reads, every write creates a new row version with transaction timestamps (`xmin`,
`xmax`). Readers see a snapshot of the database at their transaction start time — they
never block on writers, and writers never block on readers.

```
Row versions in heap:
  (xmin=100, xmax=0,   data="Alice")   ← visible to txn 101+ while xmax=0
  (xmin=102, xmax=0,   data="Alice2")  ← created by txn 102, visible to 102+
  (xmin=100, xmax=102, data="Alice")   ← old version, dead to txns >= 102
```

`VACUUM` reclaims dead row versions. High write rates with slow VACUUM cause table bloat.

**Key MVCC behaviours:**
- `SELECT` never blocks on `INSERT` / `UPDATE` / `DELETE` from other transactions
- `UPDATE` creates a new row version; it does not modify in place
- Two concurrent `UPDATE`s to the same row: the second blocks until the first commits,
  then re-evaluates its predicate against the new version (lost-update protection)
- `REPEATABLE READ` and `SERIALIZABLE` snapshots can cause serialization failures
  (`ERROR: could not serialize access`) — the application must retry

### `SELECT ... FOR UPDATE` — pessimistic row locking

`FOR UPDATE` acquires an exclusive row lock at `SELECT` time, preventing other transactions
from updating or locking the same rows until the current transaction commits or rolls back.
Use it when you need to read-then-modify without a lost-update race.

```python
async with conn.transaction():
    # Lock the row — blocks any concurrent FOR UPDATE or UPDATE on this row
    row = await conn.fetchrow(
        "SELECT id, stock FROM products WHERE id = $1 FOR UPDATE",
        product_id
    )
    if row["stock"] < quantity:
        raise ValueError("insufficient stock")
    await conn.execute(
        "UPDATE products SET stock = stock - $1 WHERE id = $2",
        quantity, product_id
    )
    # Lock released on commit
```

### Lock strength variants

```sql
FOR UPDATE           -- exclusive: blocks all other FOR UPDATE, FOR SHARE, UPDATE, DELETE
FOR NO KEY UPDATE    -- like FOR UPDATE but allows concurrent FOR KEY SHARE
FOR SHARE            -- shared: blocks FOR UPDATE / FOR NO KEY UPDATE, allows other FOR SHARE
FOR KEY SHARE        -- weakest: only blocks FOR UPDATE
```

Use `FOR NO KEY UPDATE` when your update does not touch the primary/foreign key columns —
it allows referencing foreign-key checks to proceed concurrently.

### `SKIP LOCKED` vs `NOWAIT`

```sql
-- NOWAIT: fail immediately if any row is locked (raises LockNotAvailable)
SELECT id FROM jobs WHERE status = 'pending' FOR UPDATE NOWAIT LIMIT 1;

-- SKIP LOCKED: skip locked rows, return only immediately available ones
-- The canonical job queue pattern — no two workers claim the same job
SELECT id FROM jobs WHERE status = 'pending'
ORDER BY created_at
FOR UPDATE SKIP LOCKED LIMIT 1;
```

### Deadlock risk with pessimistic locking

When two transactions each lock a row and then try to lock the row the other holds,
PostgreSQL detects the deadlock and aborts one with:
`ERROR: deadlock detected`

Prevent deadlocks by always acquiring locks in a **consistent order**:

```python
# Always lock product rows in ascending ID order
product_ids = sorted([product_id_a, product_id_b])
rows = await conn.fetch(
    "SELECT id, stock FROM products WHERE id = ANY($1) ORDER BY id FOR UPDATE",
    product_ids
)
```

### MVCC vs pessimistic locking — when to use each

| Pattern | Mechanism | Best for |
|---|---|---|
| Plain `UPDATE` | MVCC lost-update protection | Non-critical updates where last-writer-wins is acceptable |
| Optimistic (version counter) | Application-level CAS | Low contention; retry on conflict is cheap |
| `SELECT FOR UPDATE` | Pessimistic row lock | Read-then-modify where you cannot afford a retry (stock, balance) |
| `SELECT FOR UPDATE SKIP LOCKED` | Pessimistic + skip | Job queues, task dispatching |
| `SERIALIZABLE` isolation | MVCC serialization | Complex multi-row invariants; accept serialization errors + retry |

**Pessimistic locking is a code smell when:**
- The lock is held across an external API call or user interaction (lock duration is unbounded)
- You lock many rows at once without a consistent order (deadlock risk)
- Contention is low — optimistic locking has lower overhead and no deadlock risk

---

## Read-After-Write Consistency

### PostgreSQL guarantees it automatically

Unlike MongoDB, PostgreSQL provides read-after-write consistency out of the box on the primary:

- Every write is synchronously applied before the statement returns.
- `READ COMMITTED` (the default isolation level) means every new statement gets a fresh snapshot that includes all data committed by any session up to that moment.
- The same connection, a different pool connection, or a completely separate process will all see a committed write immediately.

No sessions, no write concerns, no special settings required.

### Within a transaction

Writes you make inside an open transaction are visible **to your own connection** but not to other sessions until `COMMIT`. This is correct READ COMMITTED behaviour — not a consistency gap.

```python
async with conn.transaction():
    new_id = await conn.fetchval("INSERT INTO orders ... RETURNING id", ...)

    # Your connection sees the uncommitted row
    row = await conn.fetchrow("SELECT id FROM orders WHERE id = $1", new_id)
    assert row is not None   # visible to self

    # Another pool connection cannot see it yet
    async with pool.acquire() as other:
        invisible = await other.fetchrow("SELECT id FROM orders WHERE id = $1", new_id)
    assert invisible is None  # invisible to peers — not yet committed
# COMMIT happens here; now visible to everyone
```

### The only exception: read replicas

Streaming replication is **asynchronous by default**. A standby may lag behind the primary by milliseconds to seconds. A read routed to a replica immediately after a primary write may miss that write.

### Mitigations for replica lag

| Option | How | Latency cost |
|---|---|---|
| **Read from primary** | Point asyncpg at the primary DSN for write-then-read flows | None |
| `synchronous_commit = 'remote_apply'` | Primary waits for one replica to replay WAL before ACKing | One replica round-trip |
| `pg_wal_replay_wait(lsn)` (PG 16+) | Capture `pg_current_wal_lsn()` on primary; block replica query until it catches up | Proportional to lag |
| **Sticky routing** | After any write, route that user's reads to the primary for N seconds | Reduces replica offload |

```sql
-- Capture LSN on the primary after writing
SELECT pg_current_wal_lsn();   -- e.g. 0/1A3F000

-- On the replica (PG 16+): block until replayed
SELECT pg_wal_replay_wait('0/1A3F000');
-- Returns once the replica has applied that WAL position
```

### Compared to MongoDB

| | PostgreSQL | MongoDB |
|---|---|---|
| **Primary reads** | Automatic (READ COMMITTED) | Automatic (w=1 default) |
| **Replica reads** | Lag possible; mitigate with `synchronous_commit` or pg_wal_replay_wait | Lag possible; mitigate with causal consistency session + `w=majority` + `rc=majority` |
| **Cross-session guarantee** | Automatic at READ COMMITTED | Requires explicit causal session |
| **Configuration overhead** | None for primary reads | Must set three knobs for replica safety |
