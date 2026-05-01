"""
PostgreSQL Reference with asyncpg — E-Commerce Domain
======================================================
Domain schema: users, products, orders, order_items, reviews

All code is structured as named async functions demonstrating one concept each.
No top-level side-effects; nothing executes on import.

Run individual sections via:
    import asyncio
    asyncio.run(demo_connections())
"""

import sys
import io as _io_utf8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = _io_utf8.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io_utf8.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import asyncpg
from decimal import Decimal
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Schema (run once to set up the demo database)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    sku         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    price       NUMERIC(10, 2) NOT NULL,
    category    TEXT,
    tags        JSONB DEFAULT '[]',
    stock_qty   INTEGER DEFAULT 0,
    search_vec  TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
                ) STORED
);

CREATE INDEX IF NOT EXISTS idx_products_search ON products USING GIN (search_vec);
CREATE INDEX IF NOT EXISTS idx_products_tags   ON products USING GIN (tags);

CREATE TABLE IF NOT EXISTS orders (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending/paid/shipped/delivered/cancelled
    total       NUMERIC(10, 2),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_items (
    id          SERIAL PRIMARY KEY,
    order_id    INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id  INTEGER REFERENCES products(id),
    qty         INTEGER NOT NULL,
    unit_price  NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id          SERIAL PRIMARY KEY,
    product_id  INTEGER REFERENCES products(id) ON DELETE CASCADE,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
    body        TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (product_id, user_id)
);

-- Job queue table used in SKIP LOCKED examples
CREATE TABLE IF NOT EXISTS job_queue (
    id          SERIAL PRIMARY KEY,
    payload     JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Category hierarchy for recursive CTE example
CREATE TABLE IF NOT EXISTS categories (
    id          SERIAL PRIMARY KEY,
    parent_id   INTEGER REFERENCES categories(id),
    name        TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# 1. Connection & Pool Setup
# ---------------------------------------------------------------------------

async def demo_connections(dsn: str = "postgresql://rag_user:rag_pass@localhost:5434/ecommerce") -> None:
    """
    asyncpg.create_pool() with all key parameters.
    pool.acquire() gives a connection as a context manager.
    Always call pool.close() when done.
    """

    async def _init_connection(conn: asyncpg.Connection) -> None:
        """
        Called once per new connection in the pool.
        Perfect for SET commands, registering codecs, or loading extensions.
        """
        await conn.execute("SET application_name = 'ecommerce_app'")
        # Register pgvector here if using it:
        # await register_vector(conn)

    pool = await asyncpg.create_pool(
        dsn,
        min_size=2,                            # connections kept alive always
        max_size=10,                           # hard cap on concurrent connections
        max_queries=50_000,                    # recycle a connection after this many queries
        max_inactive_connection_lifetime=300,  # drop idle connections after 5 min
        command_timeout=30,                    # seconds before a query is cancelled
        init=_init_connection,                 # called once per new connection
    )

    # Acquire a connection explicitly
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT version()")
        print(version)

    # Or pass the pool directly — asyncpg acquires internally
    rows = await pool.fetch("SELECT id, name FROM users LIMIT 5")
    for row in rows:
        print(dict(row))  # asyncpg Record is not a dict; convert with dict()

    # Always close the pool when the application shuts down
    await pool.close()


async def demo_single_connection(dsn: str = "postgresql://rag_user:rag_pass@localhost:5434/ecommerce") -> None:
    """
    asyncpg.connect() for scripts that need exactly one connection.
    Prefer create_pool() in long-running services.
    """
    conn = await asyncpg.connect(dsn)
    try:
        uid = await conn.fetchval("SELECT id FROM users LIMIT 1")
        print(uid)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 2. Fetch Methods
# ---------------------------------------------------------------------------

async def demo_fetch_methods(conn: asyncpg.Connection) -> None:
    """
    execute()     — DDL/DML, returns a status string like "INSERT 0 1"
    fetch()       — all matching rows as list[Record]
    fetchrow()    — first matching row as Record | None
    fetchval()    — first column of first row as a scalar | None
    executemany() — run same statement with many argument tuples
    """

    # execute(): DDL or DML where you don't need rows back
    status = await conn.execute("UPDATE orders SET status = $1 WHERE status = $2", "pending", "new")
    print(status)   # e.g. "UPDATE 42"

    # fetch(): list of all matching rows — use for search results, reports
    orders = await conn.fetch(
        "SELECT id, user_id, status, total FROM orders WHERE status = $1",
        "pending"
    )
    for order in orders:
        # Access by column name (asyncpg.Record supports attribute-style and dict-style)
        print(order["id"], order["total"])

    # fetchrow(): exactly one row — use for "get by primary key"
    user = await conn.fetchrow(
        "SELECT id, email, name FROM users WHERE id = $1",
        1
    )
    if user:
        print(user["email"])        # column access by name
        print(user[1])              # column access by index

    # fetchval(): single scalar — ideal for COUNT, EXISTS, MAX, a single column
    count = await conn.fetchval("SELECT count(*) FROM orders WHERE status = $1", "pending")
    print(count)  # int, not a Record

    exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)",
        "alice@example.com"
    )
    print(exists)  # bool

    # executemany(): same statement with multiple argument tuples (one round-trip)
    new_products = [
        ("SKU-001", "Widget A", "A great widget", Decimal("9.99"), "widgets"),
        ("SKU-002", "Widget B", "An even better widget", Decimal("14.99"), "widgets"),
        ("SKU-003", "Gadget X", "The gadget you need", Decimal("49.99"), "gadgets"),
    ]
    await conn.executemany(
        "INSERT INTO products (sku, name, description, price, category) VALUES ($1,$2,$3,$4,$5)",
        new_products
    )
    # executemany returns None — use COPY for highest throughput (see section 9)


# ---------------------------------------------------------------------------
# 3. Parameterized Queries
# ---------------------------------------------------------------------------

async def demo_parameterized(conn: asyncpg.Connection) -> None:
    """
    asyncpg uses $1, $2, … placeholders (PostgreSQL native protocol).
    Pass None for SQL NULL. asyncpg coerces Python types automatically.
    """

    # $1, $2 positional placeholders — NEVER use string formatting
    row = await conn.fetchrow(
        "SELECT id FROM users WHERE email = $1 AND name = $2",
        "alice@example.com", "Alice"
    )

    # Pass None for NULL
    await conn.execute(
        "INSERT INTO reviews (product_id, user_id, rating, body) VALUES ($1, $2, $3, $4)",
        1, 1, 5, None   # body is NULL
    )

    # Type coercion: Python int -> INTEGER, Decimal -> NUMERIC, datetime -> TIMESTAMPTZ
    await conn.execute(
        "UPDATE products SET price = $1, stock_qty = $2 WHERE id = $3",
        Decimal("19.99"), 100, 1
    )

    # Reuse the same placeholder — PostgreSQL does NOT support named params natively
    # For readability, build helpers or use string interpolation for identifiers only
    table = "orders"  # dynamic table name — must be an identifier, not a value
    rows = await conn.fetch(f"SELECT * FROM {table} WHERE user_id = $1", 1)
    # WARNING: only do this for identifiers you fully control; values must always use $N


# ---------------------------------------------------------------------------
# 4. RETURNING Clause
# ---------------------------------------------------------------------------

async def demo_returning(conn: asyncpg.Connection) -> None:
    """
    RETURNING avoids a second SELECT round-trip after INSERT/UPDATE/DELETE.
    """

    # INSERT … RETURNING — get the auto-generated id back
    new_user = await conn.fetchrow(
        """
        INSERT INTO users (email, name) VALUES ($1, $2)
        RETURNING id, email, name, created_at
        """,
        "bob@example.com", "Bob"
    )
    print(new_user["id"], new_user["created_at"])

    # Just the scalar id — use fetchval for brevity
    new_order_id = await conn.fetchval(
        "INSERT INTO orders (user_id, status) VALUES ($1, $2) RETURNING id",
        new_user["id"], "pending"
    )

    # UPDATE … RETURNING — see what changed in one trip
    updated = await conn.fetchrow(
        "UPDATE orders SET status = $1 WHERE id = $2 RETURNING id, status, total",
        "paid", new_order_id
    )

    # DELETE … RETURNING — audit what was removed
    deleted = await conn.fetchrow(
        "DELETE FROM users WHERE id = $1 RETURNING id, email",
        new_user["id"]
    )

    # Multiple rows returned by RETURNING — use fetch()
    cancelled = await conn.fetch(
        """
        DELETE FROM orders
        WHERE status = 'pending' AND created_at < NOW() - INTERVAL '30 days'
        RETURNING id, user_id
        """
    )
    print(f"Cancelled {len(cancelled)} stale orders")


# ---------------------------------------------------------------------------
# 5. Transactions
# ---------------------------------------------------------------------------

async def demo_transactions(conn: asyncpg.Connection) -> None:
    """
    asyncpg transactions use `async with conn.transaction()`.
    Nested calls create savepoints automatically.
    """

    # Basic transaction — ROLLBACK on any exception, COMMIT on clean exit
    async with conn.transaction():
        order_id = await conn.fetchval(
            "INSERT INTO orders (user_id, status) VALUES ($1, $2) RETURNING id",
            1, "pending"
        )
        await conn.execute(
            "INSERT INTO order_items (order_id, product_id, qty, unit_price) VALUES ($1,$2,$3,$4)",
            order_id, 1, 2, Decimal("9.99")
        )
        # Both inserts committed atomically; any exception here rolls back both

    # Isolation levels
    async with conn.transaction(isolation="serializable"):
        # Prevents phantom reads and serialization anomalies
        # Use for financial transfers, inventory deductions
        stock = await conn.fetchval("SELECT stock_qty FROM products WHERE id = $1 FOR UPDATE", 1)
        if stock and stock > 0:
            await conn.execute("UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1", 1)

    async with conn.transaction(isolation="repeatable_read"):
        # Prevents non-repeatable reads; same row reads are consistent within txn
        pass

    async with conn.transaction(isolation="read_committed"):
        # Default PostgreSQL level; each statement sees committed data at that moment
        pass

    # Read-only transaction — PostgreSQL can route to a standby replica
    async with conn.transaction(readonly=True):
        rows = await conn.fetch("SELECT id, total FROM orders WHERE user_id = $1", 1)

    # Deferrable + serializable — defers constraint checks to commit time
    async with conn.transaction(isolation="serializable", readonly=True, deferrable=True):
        snapshot = await conn.fetch("SELECT id, email FROM users")


# ---------------------------------------------------------------------------
# 6. Savepoints
# ---------------------------------------------------------------------------

async def demo_savepoints(conn: asyncpg.Connection) -> None:
    """
    Nesting conn.transaction() inside an existing transaction creates savepoints.
    Roll back only the inner block without aborting the outer transaction.
    """

    async with conn.transaction():  # outer transaction — BEGIN
        await conn.execute(
            "INSERT INTO users (email, name) VALUES ($1, $2)",
            "outer@example.com", "OuterUser"
        )

        try:
            async with conn.transaction():  # inner — SAVEPOINT sp1
                await conn.execute(
                    "INSERT INTO users (email, name) VALUES ($1, $2)",
                    "inner@example.com", "InnerUser"
                )
                # Simulate a domain error inside the inner block
                raise ValueError("Inner operation failed")
        except ValueError:
            # asyncpg rolled back to sp1 automatically; outer txn still alive
            pass

        # This insert is still committed because outer txn survived
        await conn.execute(
            "INSERT INTO users (email, name) VALUES ($1, $2)",
            "after@example.com", "AfterUser"
        )
    # COMMIT — only outer@example.com and after@example.com are saved


# ---------------------------------------------------------------------------
# 7. Row-Level Locking
# ---------------------------------------------------------------------------

async def demo_row_locking(conn: asyncpg.Connection) -> None:
    """
    SELECT … FOR UPDATE / FOR SHARE / SKIP LOCKED / NOWAIT
    Used to coordinate concurrent writes without application-level locks.
    """

    # FOR UPDATE — exclusive row lock; other transactions block until this txn commits
    async with conn.transaction():
        product = await conn.fetchrow(
            "SELECT id, stock_qty FROM products WHERE id = $1 FOR UPDATE",
            1
        )
        if product and product["stock_qty"] > 0:
            await conn.execute("UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1", 1)

    # FOR NO KEY UPDATE — like FOR UPDATE but allows FK reference changes from other txns
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT id, status FROM orders WHERE id = $1 FOR NO KEY UPDATE",
            1
        )

    # FOR SHARE — allow other FOR SHARE locks, block FOR UPDATE
    # Good for reading a parent row while inserting child rows
    async with conn.transaction():
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE id = $1 FOR SHARE",
            1
        )

    # SKIP LOCKED — job queue consumer: grab one job nobody else is processing
    async with conn.transaction():
        job = await conn.fetchrow(
            """
            SELECT id, payload FROM job_queue
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
        if job:
            await conn.execute(
                "UPDATE job_queue SET status = 'processing' WHERE id = $1",
                job["id"]
            )

    # NOWAIT — raise immediately if the row is locked (no blocking)
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT id FROM orders WHERE id = $1 FOR UPDATE NOWAIT",
                1
            )
    except asyncpg.LockNotAvailableError:
        print("Row is locked by another transaction; try again later")


# ---------------------------------------------------------------------------
# 7b. MVCC internals and pessimistic locking patterns
# ---------------------------------------------------------------------------

async def demo_mvcc(conn: asyncpg.Connection) -> None:
    """
    PostgreSQL MVCC (Multi-Version Concurrency Control):
    - Every write creates a new row version stamped with xmin/xmax transaction IDs.
    - Readers see a consistent snapshot at their transaction start time.
    - Readers never block writers; writers never block readers.
    - VACUUM reclaims dead versions (xmax set, no live txn needs them).

    Pessimistic locking with SELECT … FOR UPDATE serialises concurrent
    read-then-modify operations at the row level without application locks.
    """

    # ── Inspect MVCC system columns ───────────────────────────────────────
    # xmin: transaction that inserted this version
    # xmax: transaction that deleted/updated this version (0 = still live)
    # ctid: physical location (page, offset) — changes on UPDATE
    rows = await conn.fetch(
        """
        SELECT id, xmin, xmax, ctid, status
        FROM   orders
        WHERE  id = 1
        """
    )
    for r in rows:
        print(f"  id={r['id']} xmin={r['xmin']} xmax={r['xmax']} "
              f"ctid={r['ctid']} status={r['status']}")
    # After an UPDATE: xmax on old version is set; a new version appears with
    # a new xmin and the same id but a different ctid.
    # VACUUM removes old versions where xmax is set and no live txn needs them.

    # ── Dead tuple bloat check ────────────────────────────────────────────
    # High n_dead_tup with low autovacuum_count signals vacuum is falling behind.
    bloat = await conn.fetch(
        """
        SELECT relname,
               n_live_tup,
               n_dead_tup,
               round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 1)
                   AS dead_pct,
               last_autovacuum
        FROM   pg_stat_user_tables
        WHERE  relname IN ('orders', 'products', 'users')
        ORDER  BY dead_pct DESC NULLS LAST
        """
    )
    for r in bloat:
        print(f"  {r['relname']}: live={r['n_live_tup']} dead={r['n_dead_tup']} "
              f"dead_pct={r['dead_pct']}% last_vacuum={r['last_autovacuum']}")

    # ── SELECT FOR UPDATE — read-then-modify without lost updates ─────────
    # Without FOR UPDATE: two concurrent transactions both read stock=5,
    # both decide to decrement, both write stock=4 → one sale is lost.
    # With FOR UPDATE: second transaction blocks until first commits, then
    # re-reads the updated value (stock=4) before deciding.
    async with conn.transaction():
        product = await conn.fetchrow(
            # Acquires exclusive row lock — concurrent FOR UPDATE / UPDATE blocks here
            "SELECT id, stock_qty FROM products WHERE id = $1 FOR UPDATE",
            1
        )
        if product and product["stock_qty"] > 0:
            await conn.execute(
                "UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1", 1
            )
        # Lock released on commit (end of `async with conn.transaction()` block)

    # ── FOR UPDATE with multiple rows — always lock in consistent order ────
    # Locking rows in arbitrary order across transactions causes deadlocks.
    # Rule: always ORDER BY the locking query so all callers acquire in the same sequence.
    product_ids = sorted([2, 3])   # sort in application before sending to DB
    async with conn.transaction():
        rows = await conn.fetch(
            """
            SELECT id, stock_qty
            FROM   products
            WHERE  id = ANY($1::int[])
            ORDER  BY id              -- consistent lock order prevents deadlocks
            FOR UPDATE
            """,
            product_ids
        )
        for row in rows:
            await conn.execute(
                "UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1",
                row["id"]
            )

    # ── FOR NO KEY UPDATE — lighter lock, allows FK reference checks ──────
    # Use when your UPDATE does not change the primary key or referenced columns.
    # Allows concurrent transactions doing FK lookups (FOR KEY SHARE) to proceed.
    async with conn.transaction():
        await conn.fetchrow(
            "SELECT id, status FROM orders WHERE id = $1 FOR NO KEY UPDATE", 1
        )
        await conn.execute(
            "UPDATE orders SET status = 'shipped' WHERE id = $1", 1
        )

    # ── Optimistic locking — version counter (low-contention alternative) ──
    # No lock held. Read version → compute new value → update only if version unchanged.
    # If version changed (concurrent writer won), retry.
    async def optimistic_update_price(product_id: int, new_price: float,
                                      max_retries: int = 5) -> bool:
        for attempt in range(max_retries):
            row = await conn.fetchrow(
                "SELECT price, version FROM products WHERE id = $1", product_id
            )
            if row is None:
                return False
            result = await conn.execute(
                # Guard: only update if version still matches what we read
                "UPDATE products SET price = $1, version = version + 1 "
                "WHERE id = $2 AND version = $3",
                new_price, product_id, row["version"]
            )
            # asyncpg returns "UPDATE N" — N=1 means success, N=0 means conflict
            if result == "UPDATE 1":
                return True
            print(f"  version conflict on attempt {attempt + 1}, retrying …")
        return False

    # Requires a `version` column: ALTER TABLE products ADD COLUMN version INT DEFAULT 0;
    # await optimistic_update_price(1, 29.99)

    # ── Snapshot isolation — what each level sees ─────────────────────────
    print("""
    Isolation level   | Dirty read | Non-repeatable read | Phantom read | Serialization anomaly
    ------------------|------------|--------------------|--------------|-----------------------
    READ COMMITTED    | No (PG)    | Yes                 | Yes          | Yes
    REPEATABLE READ   | No         | No                  | No (PG)      | Yes
    SERIALIZABLE      | No         | No                  | No           | No (SSI detects)

    PG never allows dirty reads even at READ COMMITTED.
    PG REPEATABLE READ also prevents phantoms (stronger than SQL standard requires).
    SERIALIZABLE uses SSI (Serializable Snapshot Isolation) — detects conflicts and
    raises serialization errors that the application must retry.

    SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
    SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    """)

    # ── Serialization failure retry pattern ───────────────────────────────
    print("""
    import asyncpg

    async def run_with_retry(conn, operation, max_retries=3):
        for attempt in range(max_retries):
            try:
                async with conn.transaction(isolation='serializable'):
                    return await operation(conn)
            except asyncpg.SerializationFailureError:
                if attempt == max_retries - 1:
                    raise   # surface after max retries
                # back off briefly before retry
                await asyncio.sleep(0.05 * (2 ** attempt))
    """)


# ---------------------------------------------------------------------------
# 8. Server-Side Cursors
# ---------------------------------------------------------------------------

async def demo_cursors(conn: asyncpg.Connection) -> None:
    """
    Server-side cursors stream rows from PostgreSQL without loading everything
    into memory.  MUST be inside a transaction.
    prefetch controls how many rows are fetched per network round-trip.
    """

    # Iterate over all orders without loading all rows into RAM
    async with conn.transaction():
        async for row in conn.cursor(
            "SELECT id, user_id, total FROM orders ORDER BY created_at",
            prefetch=200,   # fetch 200 rows per round-trip (default 50)
        ):
            # Process each row one at a time
            print(row["id"], row["total"])

    # With query parameters
    async with conn.transaction():
        async for row in conn.cursor(
            "SELECT id, email FROM users WHERE created_at > $1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            prefetch=500,
        ):
            print(row["email"])

    # Explicit cursor object — fetch N rows at a time manually
    async with conn.transaction():
        cur = await conn.cursor(
            "SELECT id, total FROM orders WHERE status = $1",
            "pending",
        )
        while True:
            batch = await cur.fetch(100)  # read up to 100 rows
            if not batch:
                break
            for row in batch:
                print(row["id"])


# ---------------------------------------------------------------------------
# 9. Bulk Operations
# ---------------------------------------------------------------------------

async def demo_bulk(conn: asyncpg.Connection) -> None:
    """
    executemany — same statement, many arg tuples
    copy_records_to_table — fastest Python-list → table path
    copy_to_table — load from CSV/file
    copy_from_query — export query results as CSV
    """

    # executemany: one prepared statement, many tuples — better than N INSERTs
    records = [
        ("SKU-A1", "Alpha Widget", "desc a", Decimal("5.00"), "widgets"),
        ("SKU-B2", "Beta Gadget",  "desc b", Decimal("25.00"), "gadgets"),
    ]
    await conn.executemany(
        "INSERT INTO products (sku, name, description, price, category) VALUES ($1,$2,$3,$4,$5)",
        records
    )

    # copy_records_to_table: uses PostgreSQL COPY protocol — much faster than executemany
    # for large datasets (thousands+ rows). No per-row round-trip.
    bulk_users = [
        ("carol@example.com", "Carol"),
        ("dave@example.com",  "Dave"),
        ("eve@example.com",   "Eve"),
    ]
    await conn.copy_records_to_table(
        "users",
        records=bulk_users,
        columns=["email", "name"],  # must match record tuple order
    )

    # copy_to_table: load from a CSV file using COPY FROM STDIN
    csv_data = b"frank@example.com,Frank\ngrace@example.com,Grace\n"
    import io
    await conn.copy_to_table(
        "users",
        source=io.BytesIO(csv_data),
        columns=["email", "name"],
        format="csv",
    )

    # copy_from_query: export a query result as CSV bytes
    # asyncpg requires an `output` sink — BytesIO collects all chunks
    import io as _io
    _buf = _io.BytesIO()
    await conn.copy_from_query(
        "SELECT email, name FROM users ORDER BY id",
        output=_buf,
        format="csv",
        header=True,
    )
    print(_buf.getvalue().decode())  # CSV text


# ---------------------------------------------------------------------------
# 10. UPSERT
# ---------------------------------------------------------------------------

async def demo_upsert(conn: asyncpg.Connection) -> None:
    """
    ON CONFLICT … DO UPDATE / DO NOTHING
    EXCLUDED refers to the row that failed to insert.
    """

    # Basic upsert: update email if the sku already exists
    await conn.execute(
        """
        INSERT INTO products (sku, name, price, category)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (sku) DO UPDATE
            SET name  = EXCLUDED.name,
                price = EXCLUDED.price
        """,
        "SKU-001", "Widget A v2", Decimal("11.99"), "widgets"
    )

    # ON CONFLICT DO NOTHING: silently ignore duplicates
    await conn.execute(
        "INSERT INTO users (email, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        "alice@example.com", "Alice"
    )

    # ON CONFLICT ON CONSTRAINT: target a named constraint explicitly
    await conn.execute(
        """
        INSERT INTO reviews (product_id, user_id, rating, body)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT ON CONSTRAINT reviews_product_id_user_id_key
        DO UPDATE SET rating = EXCLUDED.rating, body = EXCLUDED.body
        """,
        1, 1, 4, "Updated review body"
    )

    # Partial index upsert: only conflict on active SKUs
    # Requires a matching partial unique index:
    #   CREATE UNIQUE INDEX idx_active_sku ON products(sku) WHERE stock_qty > 0;
    await conn.execute(
        """
        INSERT INTO products (sku, name, price, category, stock_qty)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (sku) WHERE stock_qty > 0
        DO UPDATE SET price = EXCLUDED.price
        """,
        "SKU-001", "Widget A", Decimal("9.99"), "widgets", 10
    )

    # UPSERT with RETURNING — get the row whether inserted or updated
    row = await conn.fetchrow(
        """
        INSERT INTO users (email, name) VALUES ($1, $2)
        ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
        RETURNING id, email, name
        """,
        "alice@example.com", "Alice Updated"
    )
    print(row["id"])


# ---------------------------------------------------------------------------
# 11. Full-Text Search
# ---------------------------------------------------------------------------

async def demo_full_text_search(conn: asyncpg.Connection) -> None:
    """
    tsvector + GIN index for keyword search on products.
    The products table has a GENERATED ALWAYS tsvector column (search_vec).
    """

    # Basic full-text search with to_tsquery (terms ANDed with &)
    rows = await conn.fetch(
        """
        SELECT id, name, ts_rank(search_vec, query) AS rank
        FROM products, to_tsquery('english', $1) query
        WHERE search_vec @@ query
        ORDER BY rank DESC
        """,
        "widget & great"
    )

    # websearch_to_tsquery: Google-style input ("widget great" → widget & great)
    rows = await conn.fetch(
        """
        SELECT id, name
        FROM products
        WHERE search_vec @@ websearch_to_tsquery('english', $1)
        ORDER BY ts_rank(search_vec, websearch_to_tsquery('english', $1)) DESC
        """,
        "great widget"
    )

    # phraseto_tsquery: exact phrase search ("great widget" stays in order)
    rows = await conn.fetch(
        """
        SELECT id, name FROM products
        WHERE search_vec @@ phraseto_tsquery('english', $1)
        """,
        "great widget"
    )

    # ts_headline: highlight matching terms in the original text
    rows = await conn.fetch(
        """
        SELECT
            id,
            name,
            ts_headline(
                'english',
                description,
                websearch_to_tsquery('english', $1),
                'StartSel=<b>, StopSel=</b>, MaxWords=20'
            ) AS snippet
        FROM products
        WHERE search_vec @@ websearch_to_tsquery('english', $1)
        """,
        "great widget"
    )

    # Multi-language: index German text with 'german' config
    # On-the-fly tsvector (no generated column) with different language configs
    rows = await conn.fetch(
        """
        SELECT id, name
        FROM products
        WHERE to_tsvector('german', description) @@ to_tsquery('german', $1)
        """,
        "widget"
    )

    # Ranking with cover density (ts_rank_cd penalises distance between terms)
    rows = await conn.fetch(
        """
        SELECT id, name,
               ts_rank_cd(search_vec, q) AS rank
        FROM products, websearch_to_tsquery('english', $1) q
        WHERE search_vec @@ q
        ORDER BY rank DESC
        LIMIT 10
        """,
        "great widget"
    )


# ---------------------------------------------------------------------------
# 12. JSONB
# ---------------------------------------------------------------------------

async def demo_jsonb(conn: asyncpg.Connection) -> None:
    """
    asyncpg automatically decodes JSONB columns to Python dicts/lists.
    GIN index on JSONB enables fast containment and existence queries.
    """

    # @>  containment: rows where metadata contains {"tier": "gold"}
    rows = await conn.fetch(
        "SELECT id, name FROM users WHERE metadata @> $1",
        '{"tier": "gold"}'   # pass as JSON string; asyncpg sends as JSONB
    )

    # <@  contained-in: check if a JSON value is a subset of a column
    rows = await conn.fetch(
        "SELECT id FROM users WHERE $1::jsonb <@ metadata",
        '{"active": true}'
    )

    # ?   key existence
    rows = await conn.fetch("SELECT id FROM users WHERE metadata ? $1", "tier")

    # ?|  any of these keys exist
    rows = await conn.fetch(
        "SELECT id FROM users WHERE metadata ?| $1::text[]",
        ["tier", "plan"]
    )

    # ?&  all of these keys exist
    rows = await conn.fetch(
        "SELECT id FROM users WHERE metadata ?& $1::text[]",
        ["tier", "active"]
    )

    # ->  get JSON object field as JSONB
    rows = await conn.fetch("SELECT metadata -> $1 FROM users", "tier")

    # ->> get JSON object field as text (for filtering/display)
    rows = await conn.fetch(
        "SELECT id FROM users WHERE metadata ->> $1 = $2",
        "tier", "gold"
    )

    # #>  path access as JSONB: metadata #> '{address, city}'
    rows = await conn.fetch(
        "SELECT metadata #> $1 FROM users",
        ["address", "city"]
    )

    # jsonb_set: update a single key inside a JSONB column
    await conn.execute(
        "UPDATE users SET metadata = jsonb_set(metadata, $1, $2) WHERE id = $3",
        ["tier"], '"platinum"', 1
    )

    # jsonb_build_object: construct JSONB in SQL
    rows = await conn.fetch(
        """
        SELECT jsonb_build_object(
            'id',    id,
            'email', email,
            'tier',  metadata ->> 'tier'
        ) AS user_json
        FROM users
        """
    )

    # jsonb_agg: aggregate rows into a JSON array
    rows = await conn.fetch(
        """
        SELECT o.id AS order_id,
               jsonb_agg(jsonb_build_object(
                   'product_id', oi.product_id,
                   'qty',        oi.qty,
                   'unit_price', oi.unit_price
               )) AS items
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.id
        GROUP BY o.id
        """
    )


# ---------------------------------------------------------------------------
# 12b. JSONB — advanced patterns
# ---------------------------------------------------------------------------

async def demo_jsonb_advanced(conn: asyncpg.Connection) -> None:
    """
    Advanced JSONB patterns: path queries, array ops, indexing strategy,
    JSONB vs generated columns, and the TOAST interaction.
    """

    # ── jsonb_path_query (SQL/JSON path — PG 12+) ─────────────────────────
    # SQL/JSON path expressions are more expressive than the #> operator.
    # @  is the current item; . is member access; [*] iterates an array.
    await conn.fetch(
        """
        SELECT id,
               jsonb_path_query(metadata, '$.address.city') AS city
        FROM users
        WHERE jsonb_path_exists(metadata, '$.tags[*] ? (@ == "vip")')
        """
    )

    # ── jsonb_strip_nulls — remove null-valued keys ───────────────────────
    await conn.fetchval(
        "SELECT jsonb_strip_nulls($1::jsonb)",
        '{"a": 1, "b": null, "c": {"d": null, "e": 2}}'
        # → {"a": 1, "c": {"e": 2}}
    )

    # ── || merge operator — shallow merge (right wins on duplicate keys) ──
    await conn.execute(
        """
        UPDATE users
        SET metadata = metadata || $1::jsonb
        WHERE id = $2
        """,
        '{"tier": "gold", "verified": true}', 1
    )

    # ── - delete key operator ─────────────────────────────────────────────
    await conn.execute(
        "UPDATE users SET metadata = metadata - $1 WHERE id = $2",
        "temp_flag", 1
    )

    # ── jsonb_each / jsonb_each_text — expand top-level keys to rows ──────
    await conn.fetch(
        """
        SELECT u.id, kv.key, kv.value
        FROM users u,
             jsonb_each_text(u.metadata) AS kv  -- lateral implicit join
        WHERE u.id = 1
        """
    )

    # ── jsonb_array_elements — expand a JSON array into rows ─────────────
    await conn.fetch(
        """
        SELECT id,
               elem->>'name'  AS tag_name,
               elem->>'weight' AS tag_weight
        FROM users,
             jsonb_array_elements(metadata->'tags') AS elem
        WHERE metadata ? 'tags'
        """
    )

    # ── Generated column from JSONB (PG 12+) ─────────────────────────────
    # Store hot scalar fields as real generated columns so they can be
    # indexed normally — avoids full JSONB decompression on filter.
    # (DDL shown as a comment — cannot run inside a transaction on
    #  an existing table without careful migration.)
    print("""
    -- Generated column: extracted from JSONB, indexed as a plain B-tree
    ALTER TABLE users
        ADD COLUMN tier TEXT GENERATED ALWAYS AS (metadata->>'tier') STORED;

    CREATE INDEX ON users (tier);   -- fast equality / range scan on tier
    -- Now: SELECT * FROM users WHERE tier = 'gold'  → index scan, no JSONB decompression
    """)

    # ── GIN index strategies ──────────────────────────────────────────────
    print("""
    -- Default GIN (jsonb_ops): supports @>, ?, ?|, ?&
    CREATE INDEX ON users USING GIN (metadata);

    -- jsonb_path_ops: smaller, faster for @> only (no key-existence queries)
    CREATE INDEX ON users USING GIN (metadata jsonb_path_ops);

    -- Rule: use jsonb_path_ops when you only ever use @> (containment).
    --       use jsonb_ops (default) when you also need ?, ?|, ?&.
    """)


# ---------------------------------------------------------------------------
# 12c. BYTEA — binary data storage
# ---------------------------------------------------------------------------

async def demo_bytea(conn: asyncpg.Connection) -> None:
    """
    BYTEA stores arbitrary binary data.  asyncpg maps it to Python bytes.
    No size limit per column (TOAST handles large values automatically —
    up to ~1 GB practical; no MongoDB-style 16 MB wall).

    Common uses: file contents, images, cryptographic hashes, encrypted fields,
    serialised binary formats (protobuf, msgpack, Parquet row groups).
    """

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            name        TEXT        NOT NULL,
            content     BYTEA,                     -- raw file bytes
            sha256      BYTEA,                     -- 32-byte hash
            thumbnail   BYTEA,                     -- small image preview
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )

    # ── Insert raw bytes from Python ──────────────────────────────────────
    # asyncpg accepts Python bytes directly for BYTEA parameters.
    sample_pdf = b"%PDF-1.4 ... (truncated binary content) ..."
    import hashlib
    sha = hashlib.sha256(sample_pdf).digest()  # 32 raw bytes, not hex string

    doc_id = await conn.fetchval(
        "INSERT INTO documents (name, content, sha256) VALUES ($1, $2, $3) RETURNING id",
        "sample.pdf", sample_pdf, sha
    )

    # ── Retrieve as Python bytes ──────────────────────────────────────────
    row = await conn.fetchrow(
        "SELECT name, content, sha256 FROM documents WHERE id = $1", doc_id
    )
    assert isinstance(row["content"], (bytes, memoryview))
    content: bytes = bytes(row["content"])   # memoryview → bytes if needed

    # ── Hex encode / decode in SQL ────────────────────────────────────────
    # encode(col, 'hex') → text;  decode(hex_text, 'hex') → bytea
    await conn.fetch(
        "SELECT encode(sha256, 'hex') AS sha256_hex FROM documents WHERE id = $1",
        doc_id
    )

    # ── Partial reads — substring without fetching the full BYTEA ─────────
    # With STORAGE EXTERNAL the server reads only the requested bytes from
    # the TOAST table (no decompression of the full value).
    # With STORAGE EXTENDED (default) the full value is decompressed first.
    await conn.fetchval(
        "SELECT substring(content FROM 1 FOR 8) FROM documents WHERE id = $1",
        doc_id
    )   # returns bytes — useful for reading magic bytes / file headers

    # ── Store EXTERNAL for large blobs you will substring ─────────────────
    print("""
    -- Switch content to EXTERNAL: no compression, but partial reads are fast
    ALTER TABLE documents ALTER COLUMN content SET STORAGE EXTERNAL;

    -- Keep thumbnail as EXTENDED (small enough that compression helps)
    ALTER TABLE documents ALTER COLUMN thumbnail SET STORAGE EXTENDED;
    """)

    # ── Length without fetching the data ─────────────────────────────────
    await conn.fetchval(
        "SELECT octet_length(content) FROM documents WHERE id = $1", doc_id
    )

    # ── Hashing and equality in SQL ───────────────────────────────────────
    # Use sha256(content) (PG 11+) to compute hash server-side
    await conn.fetchval(
        "SELECT sha256(content) = sha256 FROM documents WHERE id = $1", doc_id
    )   # returns True if stored hash matches current content

    # ── Deduplication with hash index ─────────────────────────────────────
    print("""
    -- Hash index on sha256 for O(1) deduplication lookups
    CREATE INDEX ON documents USING HASH (sha256);

    -- Deduplicate before insert:
    SELECT id FROM documents WHERE sha256 = $1
    """)

    # ── BYTEA vs TEXT vs large object ────────────────────────────────────
    print("""
    | Approach             | Max size | Streaming | SQL ops | Best for             |
    |----------------------|----------|-----------|---------|----------------------|
    | BYTEA                | ~1 GB    | No        | Yes     | Files < ~100 MB      |
    | TEXT                 | ~1 GB    | No        | Yes     | Text content         |
    | Large Object (lo_*)  | 4 TB     | Yes       | Limited | Multi-GB files       |
    | External (S3/GCS)    | Unlimited| Yes       | No      | Everything > 100 MB  |

    Rule: store files < ~50 MB as BYTEA (simple, transactional, no extra infra).
    Use an object store (S3, GCS) and store only the URL/key for larger files.
    Large Objects (pg_largeobject) are rarely the right choice — awkward API,
    separate vacuuming, not replicated via logical replication.
    """)

    await conn.execute("DROP TABLE IF EXISTS documents CASCADE")


# ---------------------------------------------------------------------------
# 13. Window Functions
# ---------------------------------------------------------------------------

async def demo_window_functions(conn: asyncpg.Connection) -> None:
    """
    Window functions compute values across a set of rows related to the current row
    without collapsing them into groups like GROUP BY.
    """

    rows = await conn.fetch(
        """
        SELECT
            id,
            user_id,
            total,
            created_at,

            -- Ranking per user by total descending
            ROW_NUMBER()  OVER w_user AS row_num,
            RANK()        OVER w_user AS rank,       -- gaps after ties
            DENSE_RANK()  OVER w_user AS dense_rank, -- no gaps
            NTILE(4)      OVER w_user AS quartile,   -- split into 4 buckets

            -- Access other rows without a self-join
            LAG(total,  1) OVER w_user AS prev_order_total,
            LEAD(total, 1) OVER w_user AS next_order_total,

            -- First and last in the window
            FIRST_VALUE(total) OVER w_user AS first_order_total,
            LAST_VALUE(total)  OVER (
                PARTITION BY user_id ORDER BY created_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_order_total,

            -- Running total per user
            SUM(total) OVER (
                PARTITION BY user_id ORDER BY created_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_total,

            -- 3-order moving average per user
            AVG(total) OVER (
                PARTITION BY user_id ORDER BY created_at
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) AS moving_avg_3

        FROM orders
        -- Named window: reuse the same PARTITION/ORDER definition
        WINDOW w_user AS (PARTITION BY user_id ORDER BY created_at)
        ORDER BY user_id, created_at
        """
    )

    # Percentile of a product's price within its category
    rows = await conn.fetch(
        """
        SELECT
            id,
            name,
            category,
            price,
            PERCENT_RANK() OVER (PARTITION BY category ORDER BY price) AS price_percentile
        FROM products
        """
    )


# ---------------------------------------------------------------------------
# 13b. PARTITION BY — frame clauses and multi-column partitions
# ---------------------------------------------------------------------------

async def demo_partition_by(conn: asyncpg.Connection) -> None:
    """
    Deep dive into PARTITION BY inside window functions.

    PARTITION BY divides rows into independent groups (partitions); the window
    function resets and recalculates for each partition.  It is NOT the same as
    GROUP BY — every input row is still present in the output.

    Frame clause controls which rows within the partition are visible to the
    function.  Three units:
        ROWS   — counts physical rows (position-based, ignores ties)
        RANGE  — counts rows with the same ORDER BY value as the current row
        GROUPS — counts peer groups (PostgreSQL 11+)

    Syntax:
        OVER (
            PARTITION BY <cols>          -- reset boundary
            ORDER BY <cols>              -- sort within partition
            ROWS|RANGE|GROUPS
                BETWEEN <start> AND <end>  -- frame boundary
            EXCLUDE <option>             -- rows to skip (PG 14+)
        )

    Frame boundary keywords:
        UNBOUNDED PRECEDING  — start of partition
        <n> PRECEDING        — n rows/range-units before current
        CURRENT ROW          — the current row (or peer group)
        <n> FOLLOWING        — n rows/range-units after current
        UNBOUNDED FOLLOWING  — end of partition
    """

    # ── ROWS vs RANGE: the tie-handling difference ────────────────────────
    # Two orders placed on the same day by the same user have the same
    # ORDER BY value.  ROWS treats them as distinct positions; RANGE
    # includes both in each other's frame (peer rows).
    await conn.fetch(
        """
        SELECT
            id,
            user_id,
            total,
            created_at::date AS order_date,

            -- ROWS: running sum counts physical rows — each row is distinct
            SUM(total) OVER (
                PARTITION BY user_id
                ORDER BY created_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_total_rows,

            -- RANGE: running sum includes all rows with the same created_at
            -- value (peers).  On a day with 3 orders, all three see the
            -- same running total (the day's combined amount).
            SUM(total) OVER (
                PARTITION BY user_id
                ORDER BY created_at::date    -- date-level peers
                RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_total_range

        FROM orders
        ORDER BY user_id, created_at
        """
    )

    # ── Moving window with FOLLOWING ─────────────────────────────────────
    # 3-row centred average: 1 preceding + current + 1 following.
    # ROWS is almost always the right choice here; RANGE with FOLLOWING
    # can pull in an unpredictable number of peer rows.
    await conn.fetch(
        """
        SELECT
            id,
            user_id,
            total,
            AVG(total) OVER (
                PARTITION BY user_id
                ORDER BY created_at
                ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING   -- centred 3-row window
            ) AS centred_avg,
            AVG(total) OVER (
                PARTITION BY user_id
                ORDER BY created_at
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW   -- trailing 3-row window
            ) AS trailing_avg
        FROM orders
        ORDER BY user_id, created_at
        """
    )

    # ── Multi-column PARTITION BY ─────────────────────────────────────────
    # Partition on two columns: each (user_id, status) combination is an
    # independent partition.  Useful for segmented ranking or per-status
    # running totals.
    await conn.fetch(
        """
        SELECT
            id,
            user_id,
            status,
            total,
            -- Rank each order within (user × status) by total descending
            RANK() OVER (
                PARTITION BY user_id, status
                ORDER BY total DESC
            ) AS rank_within_user_status,
            -- Running total per (user × status)
            SUM(total) OVER (
                PARTITION BY user_id, status
                ORDER BY created_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_total_by_status
        FROM orders
        ORDER BY user_id, status, created_at
        """
    )

    # ── GROUPS unit (PostgreSQL 11+) ──────────────────────────────────────
    # Like RANGE but counts distinct peer groups rather than rows.
    # "2 PRECEDING" means the 2 peer groups before the current group,
    # not the 2 rows.
    await conn.fetch(
        """
        SELECT
            id,
            user_id,
            total,
            created_at::date AS order_date,
            SUM(total) OVER (
                PARTITION BY user_id
                ORDER BY created_at::date
                GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW  -- current day + previous day
            ) AS two_day_group_sum
        FROM orders
        ORDER BY user_id, created_at
        """
    )

    # ── Named window reuse ────────────────────────────────────────────────
    # Define PARTITION BY / ORDER BY once in a WINDOW clause; reference by
    # name.  Each OVER() can still add its own frame clause on top.
    await conn.fetch(
        """
        SELECT
            id,
            user_id,
            total,
            ROW_NUMBER()  OVER w                                          AS row_num,
            RANK()        OVER w                                          AS rank,
            SUM(total)    OVER (w ROWS BETWEEN UNBOUNDED PRECEDING
                                           AND CURRENT ROW)              AS running_total,
            LAG(total, 1) OVER w                                          AS prev_total,
            LEAD(total,1) OVER w                                          AS next_total
        FROM orders
        WINDOW w AS (PARTITION BY user_id ORDER BY created_at)
        ORDER BY user_id, created_at
        """
    )

    # ── PARTITION BY vs GROUP BY — key difference ─────────────────────────
    # GROUP BY collapses rows; PARTITION BY preserves them.
    # Use GROUP BY when you want one summary row per group.
    # Use PARTITION BY when you want per-row values alongside the group metric.
    await conn.fetch(
        """
        SELECT
            user_id,
            total,
            -- What fraction of this user's total spend is this single order?
            total / SUM(total) OVER (PARTITION BY user_id) AS share_of_user_spend,
            -- How does this order compare to the user's average?
            total - AVG(total) OVER (PARTITION BY user_id) AS delta_from_user_avg,
            -- Global percentile (no PARTITION BY = one partition = all rows)
            PERCENT_RANK() OVER (ORDER BY total)           AS global_percentile
        FROM orders
        ORDER BY user_id, total DESC
        """
    )


# ---------------------------------------------------------------------------
# 13c. Table partitioning — PARTITION OF ... FOR VALUES FROM ... TO ...
# ---------------------------------------------------------------------------

async def demo_table_partitioning(conn: asyncpg.Connection) -> None:
    """
    PostgreSQL declarative table partitioning (PG 10+).

    A partitioned table is a logical parent; data lives in child partition
    tables.  The planner uses partition pruning to skip partitions whose
    ranges cannot match the query filter — a large-table performance tool.

    Three strategies:
        RANGE  — partition by value range (dates, IDs)
                 child: FOR VALUES FROM (start) TO (end)  [start inclusive, end exclusive]
        LIST   — partition by explicit value set (region, status)
                 child: FOR VALUES IN (val1, val2, ...)
        HASH   — partition by hash of key mod divisor (even distribution)
                 child: FOR VALUES WITH (MODULUS n, REMAINDER r)

    Constraints:
        - The partition key must be part of every unique / primary key index.
        - Foreign keys TO a partitioned table are not supported (PG 15-).
        - Each partition can itself be partitioned (sub-partitioning).
        - A DEFAULT partition catches rows that match no other partition.
    """

    # ── Tear down from previous runs ─────────────────────────────────────
    await conn.execute("DROP TABLE IF EXISTS order_log CASCADE")
    await conn.execute("DROP TABLE IF EXISTS order_log_2023 CASCADE")
    await conn.execute("DROP TABLE IF EXISTS order_log_2024 CASCADE")
    await conn.execute("DROP TABLE IF EXISTS order_log_default CASCADE")
    await conn.execute("DROP TABLE IF EXISTS regional_orders CASCADE")
    await conn.execute("DROP TABLE IF EXISTS regional_orders_eu CASCADE")
    await conn.execute("DROP TABLE IF EXISTS regional_orders_us CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sharded_events CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sharded_events_0 CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sharded_events_1 CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sharded_events_2 CASCADE")

    # ── A. RANGE partitioning by date ─────────────────────────────────────
    # Parent table — holds no data itself, just the schema + partition key.
    await conn.execute(
        """
        CREATE TABLE order_log (
            id         BIGSERIAL,
            user_id    INT          NOT NULL,
            total      NUMERIC      NOT NULL,
            created_at TIMESTAMPTZ  NOT NULL,
            PRIMARY KEY (id, created_at)   -- partition key must be in PK
        ) PARTITION BY RANGE (created_at)
        """
    )

    # Child partitions — FOR VALUES FROM (inclusive) TO (exclusive).
    # Rows with created_at in [2023-01-01, 2024-01-01) go to order_log_2023.
    await conn.execute(
        """
        CREATE TABLE order_log_2023
            PARTITION OF order_log
            FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')
        """
    )
    await conn.execute(
        """
        CREATE TABLE order_log_2024
            PARTITION OF order_log
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
        """
    )
    # DEFAULT partition catches anything outside the explicit ranges.
    # Without this, inserting a row outside all ranges raises an error.
    await conn.execute(
        """
        CREATE TABLE order_log_default
            PARTITION OF order_log DEFAULT
        """
    )

    # Indexes are created per-partition; CREATE INDEX on the parent
    # propagates automatically (PG 11+).
    await conn.execute(
        "CREATE INDEX ON order_log (user_id, created_at)"
    )

    # Insert — routed automatically to the correct child partition.
    await conn.executemany(
        "INSERT INTO order_log (user_id, total, created_at) VALUES ($1, $2, $3)",
        [
            (1, 120.00, datetime(2023, 6, 15, 10, 0, 0, tzinfo=timezone.utc)),
            (2,  80.50, datetime(2023, 11, 1, 14, 30, 0, tzinfo=timezone.utc)),
            (1, 200.00, datetime(2024, 3, 10, 9, 0, 0, tzinfo=timezone.utc)),
            (3,  45.00, datetime(2024, 7, 22, 16, 45, 0, tzinfo=timezone.utc)),
        ],
    )

    # Partition pruning — the planner scans ONLY order_log_2024;
    # order_log_2023 and order_log_default are skipped entirely.
    rows = await conn.fetch(
        """
        SELECT id, user_id, total
        FROM   order_log
        WHERE  created_at >= '2024-01-01'
          AND  created_at <  '2025-01-01'
        ORDER  BY created_at
        """
    )
    # Verify pruning with EXPLAIN (run manually):
    # EXPLAIN SELECT ... FROM order_log WHERE created_at >= '2024-01-01' ...
    # Look for "Append" node with only "order_log_2024" listed.

    # Which physical partition does each row live in?
    rows = await conn.fetch(
        """
        SELECT tableoid::regclass AS partition, id, created_at
        FROM   order_log
        ORDER  BY created_at
        """
    )
    for r in rows:
        print(f"  partition={r['partition']}  id={r['id']}  ts={r['created_at']}")

    # ── B. LIST partitioning by region ────────────────────────────────────
    await conn.execute(
        """
        CREATE TABLE regional_orders (
            id      BIGSERIAL,
            region  TEXT        NOT NULL,
            amount  NUMERIC     NOT NULL,
            PRIMARY KEY (id, region)
        ) PARTITION BY LIST (region)
        """
    )
    await conn.execute(
        """
        CREATE TABLE regional_orders_eu
            PARTITION OF regional_orders
            FOR VALUES IN ('DE', 'FR', 'NL', 'ES', 'IT')
        """
    )
    await conn.execute(
        """
        CREATE TABLE regional_orders_us
            PARTITION OF regional_orders
            FOR VALUES IN ('US', 'CA', 'MX')
        """
    )

    await conn.executemany(
        "INSERT INTO regional_orders (region, amount) VALUES ($1, $2)",
        [("DE", 99.0), ("US", 149.0), ("FR", 55.0), ("CA", 210.0)],
    )

    # Query targets EU partition only — US partition not scanned.
    await conn.fetch(
        "SELECT * FROM regional_orders WHERE region IN ('DE', 'FR')"
    )

    # ── C. HASH partitioning for even distribution ─────────────────────────
    # No natural range or list — hash the key across N buckets.
    # MODULUS = total number of partitions; REMAINDER = this partition's slot.
    await conn.execute(
        """
        CREATE TABLE sharded_events (
            id      BIGSERIAL,
            user_id INT  NOT NULL,
            event   TEXT NOT NULL,
            PRIMARY KEY (id, user_id)
        ) PARTITION BY HASH (user_id)
        """
    )
    for remainder in range(3):
        await conn.execute(
            f"""
            CREATE TABLE sharded_events_{remainder}
                PARTITION OF sharded_events
                FOR VALUES WITH (MODULUS 3, REMAINDER {remainder})
            """
        )

    await conn.executemany(
        "INSERT INTO sharded_events (user_id, event) VALUES ($1, $2)",
        [(uid, "login") for uid in range(1, 10)],
    )

    # Each partition holds roughly ⅓ of the rows (hash distribution).
    for remainder in range(3):
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM sharded_events_{remainder}"
        )
        print(f"  sharded_events_{remainder}: {count} rows")

    # ── Attach / detach a partition ────────────────────────────────────────
    # Detach makes the child a standalone table instantly (no data move).
    # Useful for archiving old partitions without a DELETE.
    # PG 14+: DETACH CONCURRENTLY avoids locking the parent.
    print("""
    -- Archive 2023 data: detach without deleting (instant, no data copy)
    ALTER TABLE order_log DETACH PARTITION order_log_2023;
    -- order_log_2023 is now a normal table; dump/restore/move independently.

    -- Re-attach (must satisfy the range constraint):
    ALTER TABLE order_log ATTACH PARTITION order_log_2023
        FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
    """)

    # Cleanup
    await conn.execute("DROP TABLE IF EXISTS order_log CASCADE")
    await conn.execute("DROP TABLE IF EXISTS regional_orders CASCADE")
    await conn.execute("DROP TABLE IF EXISTS sharded_events CASCADE")


# ---------------------------------------------------------------------------
# 14. CTEs
# ---------------------------------------------------------------------------

async def demo_ctes(conn: asyncpg.Connection) -> None:
    """
    CTEs (WITH clauses) improve readability and enable recursive queries.
    """

    # Basic CTE — name a subquery for reuse
    rows = await conn.fetch(
        """
        WITH top_spenders AS (
            SELECT user_id, SUM(total) AS lifetime_value
            FROM orders
            WHERE status = 'delivered'
            GROUP BY user_id
            HAVING SUM(total) > 500
        )
        SELECT u.id, u.email, ts.lifetime_value
        FROM users u
        JOIN top_spenders ts ON ts.user_id = u.id
        ORDER BY ts.lifetime_value DESC
        """
    )

    # MATERIALIZED vs NOT MATERIALIZED
    # MATERIALIZED: force evaluation once and cache (useful when referenced many times)
    # NOT MATERIALIZED: inline the CTE like a view (useful to let planner optimise)
    rows = await conn.fetch(
        """
        WITH recent_orders AS MATERIALIZED (
            SELECT id, user_id, total FROM orders
            WHERE created_at > NOW() - INTERVAL '7 days'
        )
        SELECT user_id, SUM(total) FROM recent_orders GROUP BY user_id
        """
    )

    # Recursive CTE: walk a category tree (parent_id self-reference)
    rows = await conn.fetch(
        """
        WITH RECURSIVE cat_tree AS (
            -- Anchor: start at root categories
            SELECT id, parent_id, name, 0 AS depth, ARRAY[id] AS path
            FROM categories
            WHERE parent_id IS NULL

            UNION ALL

            -- Recursive: join children to the accumulating result
            SELECT c.id, c.parent_id, c.name, ct.depth + 1, ct.path || c.id
            FROM categories c
            JOIN cat_tree ct ON ct.id = c.parent_id
        )
        SELECT id, name, depth, path FROM cat_tree ORDER BY path
        """
    )

    # CTE for UPDATE with side-effects: mark orders paid and return them
    rows = await conn.fetch(
        """
        WITH paid AS (
            UPDATE orders
            SET status = 'paid'
            WHERE status = 'pending' AND total > 0
            RETURNING id, user_id, total
        )
        SELECT u.email, p.total
        FROM paid p
        JOIN users u ON u.id = p.user_id
        """
    )


# ---------------------------------------------------------------------------
# 15. LATERAL Joins
# ---------------------------------------------------------------------------

async def demo_lateral(conn: asyncpg.Connection) -> None:
    """
    LATERAL lets a subquery reference columns from tables to its left.
    Essential for top-N-per-group without window functions.
    """

    # LATERAL subquery: for each user, get their latest 3 orders
    rows = await conn.fetch(
        """
        SELECT u.id, u.email, recent.id AS order_id, recent.total
        FROM users u
        CROSS JOIN LATERAL (
            SELECT id, total
            FROM orders
            WHERE user_id = u.id          -- references the outer u.id
            ORDER BY created_at DESC
            LIMIT 3
        ) recent
        """
    )

    # LEFT JOIN LATERAL: keep users even if they have no orders (like LEFT JOIN)
    rows = await conn.fetch(
        """
        SELECT u.id, u.email, last_order.total
        FROM users u
        LEFT JOIN LATERAL (
            SELECT total
            FROM orders
            WHERE user_id = u.id
            ORDER BY created_at DESC
            LIMIT 1
        ) last_order ON true
        """
    )

    # LATERAL with a function: unnest tags array and join products
    rows = await conn.fetch(
        """
        SELECT p.id, p.name, tag
        FROM products p,
        LATERAL jsonb_array_elements_text(p.tags) AS t(tag)
        WHERE tag = $1
        """,
        "sale"
    )


# ---------------------------------------------------------------------------
# 16. Advisory Locks
# ---------------------------------------------------------------------------

async def demo_advisory_locks(conn: asyncpg.Connection) -> None:
    """
    Advisory locks are application-defined, cooperative locks stored in PostgreSQL.
    Session-scoped: held until released or session ends.
    Transaction-scoped: released automatically at COMMIT/ROLLBACK.
    """

    # Session advisory lock — must be released explicitly
    lock_key = 12345  # arbitrary integer key agreed upon by all app instances
    await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
    try:
        # Critical section: only one instance runs this at a time
        await conn.execute("UPDATE products SET stock_qty = stock_qty - 1 WHERE id = $1", 1)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)

    # Non-blocking try — returns True if lock acquired, False otherwise
    acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", lock_key)
    if acquired:
        try:
            pass  # do the work
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)
    else:
        print("Another instance holds the lock; skipping")

    # Transaction-scoped advisory lock — no explicit release needed
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)
        # Lock released automatically when this transaction ends
        await conn.execute("UPDATE job_queue SET status = 'processing' WHERE id = $1", 1)

    # Two-argument variant: (class_id, object_id) for namespacing
    await conn.execute("SELECT pg_advisory_lock($1, $2)", 100, 42)
    await conn.execute("SELECT pg_advisory_unlock($1, $2)", 100, 42)

    # Distributed task queue pattern using advisory locks
    async def claim_next_job(conn: asyncpg.Connection):
        """Claim the next pending job using an advisory lock keyed by job id."""
        jobs = await conn.fetch(
            "SELECT id FROM job_queue WHERE status = 'pending' ORDER BY created_at LIMIT 10"
        )
        for job in jobs:
            acquired = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", job["id"]
            )
            if acquired:
                await conn.execute(
                    "UPDATE job_queue SET status = 'processing' WHERE id = $1",
                    job["id"]
                )
                return job["id"]
        return None  # no job available


# ---------------------------------------------------------------------------
# 17. LISTEN / NOTIFY
# ---------------------------------------------------------------------------

async def demo_listen_notify(dsn: str = "postgresql://rag_user:rag_pass@localhost:5434/ecommerce") -> None:
    """
    LISTEN/NOTIFY is PostgreSQL's lightweight pub/sub.
    asyncpg callbacks fire in the event loop when a notification arrives.
    Use a dedicated connection (not from the pool) — it must stay open.
    """

    notifications_received: list[dict] = []

    def on_order_event(
        conn: asyncpg.Connection,
        pid: int,           # PID of the notifying backend
        channel: str,       # channel name
        payload: str,       # arbitrary string payload
    ) -> None:
        print(f"[{channel}] from PID {pid}: {payload}")
        notifications_received.append({"pid": pid, "channel": channel, "payload": payload})

    listener_conn = await asyncpg.connect(dsn)
    await listener_conn.add_listener("order_events", on_order_event)

    # Sender: another connection (or the same one) fires NOTIFY
    sender_conn = await asyncpg.connect(dsn)
    await sender_conn.execute("NOTIFY order_events, 'order_placed:42'")
    await sender_conn.close()

    # Give the event loop a moment to deliver the notification
    await asyncio.sleep(0.1)

    await listener_conn.remove_listener("order_events", on_order_event)
    await listener_conn.close()

    # Use cases:
    # - Cache invalidation: notify when a product price changes
    # - Real-time dashboard: notify when an order status changes
    # - Background jobs: trigger a worker when a new job is inserted
    # Limitation: payload max is 8000 bytes; no persistence (missed if not connected)


# ---------------------------------------------------------------------------
# 18. Change Stream (Trigger + LISTEN/NOTIFY)
# ---------------------------------------------------------------------------

async def demo_change_stream(conn: asyncpg.Connection, dsn: str) -> None:
    """
    PostgreSQL change stream equivalent: PL/pgSQL trigger fires pg_notify()
    on INSERT/UPDATE/DELETE; a dedicated asyncpg connection receives the events.

    Compared to MongoDB change streams:
      - No resume token / replay — missed events while disconnected are gone.
        Use a WAL-based CDC tool (Debezium, pglogical) for durability.
      - Payload capped at 8 000 bytes — serialize only key fields, not full rows.
      - No db-level or cluster-level watch — each trigger targets one table.
      - No oplog required — works on any PostgreSQL instance (no replica set).
    """
    import json as _json

    # ── 1. Trigger function ──────────────────────────────────────────────────
    # TG_OP   — INSERT | UPDATE | DELETE
    # TG_TABLE_NAME — name of the table that fired the trigger
    # OLD / NEW — row images (OLD undefined on INSERT; NEW undefined on DELETE)
    await conn.execute("""
        CREATE OR REPLACE FUNCTION orders_change_notify()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
            payload JSONB;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                payload := jsonb_build_object(
                    'op',        'DELETE',
                    'table',     TG_TABLE_NAME,
                    'id',        OLD.id,
                    'status',    OLD.status
                );
            ELSIF (TG_OP = 'INSERT') THEN
                payload := jsonb_build_object(
                    'op',        'INSERT',
                    'table',     TG_TABLE_NAME,
                    'id',        NEW.id,
                    'status',    NEW.status,
                    'total',     NEW.total
                );
            ELSE   -- UPDATE
                payload := jsonb_build_object(
                    'op',         'UPDATE',
                    'table',      TG_TABLE_NAME,
                    'id',         NEW.id,
                    'old_status', OLD.status,
                    'new_status', NEW.status,
                    'total',      NEW.total
                );
            END IF;

            -- pg_notify(channel, payload::text) — payload max 8 000 bytes
            PERFORM pg_notify('order_changes', payload::TEXT);
            RETURN NEW;
        END;
        $$
    """)

    await conn.execute("""
        DROP TRIGGER IF EXISTS orders_change_trigger ON orders;
        CREATE TRIGGER orders_change_trigger
        AFTER INSERT OR UPDATE OR DELETE ON orders
        FOR EACH ROW EXECUTE FUNCTION orders_change_notify()
    """)

    # ── 2. Listener connection ────────────────────────────────────────────────
    # Must be a dedicated connection — LISTEN holds the connection open.
    received: list[dict] = []

    def on_change(
        connection: asyncpg.Connection,
        pid: int,        # backend PID that fired NOTIFY
        channel: str,
        payload: str,
    ) -> None:
        event = _json.loads(payload)
        op = event["op"]
        if op == "INSERT":
            print(f"  INSERT   id={event['id']}  status={event['status']}")
        elif op == "UPDATE":
            print(f"  UPDATE   id={event['id']}  "
                  f"{event['old_status']} -> {event['new_status']}")
        elif op == "DELETE":
            print(f"  DELETE   id={event['id']}")
        received.append(event)

    listener_conn = await asyncpg.connect(dsn)
    await listener_conn.add_listener("order_changes", on_change)

    # ── 3. Fire INSERT / UPDATE / DELETE on orders ───────────────────────────
    new_id = await conn.fetchval(
        "INSERT INTO orders (user_id, status) VALUES ($1, $2) RETURNING id",
        1, "pending",
    )
    await conn.execute(
        "UPDATE orders SET status = $1, total = $2 WHERE id = $3",
        "paid", 99.99, new_id,
    )
    await conn.execute("DELETE FROM orders WHERE id = $1", new_id)

    await asyncio.sleep(0.1)   # let the event loop deliver the three notifications

    await listener_conn.remove_listener("order_changes", on_change)
    await listener_conn.close()

    # ── 4. LISTEN/NOTIFY vs WAL-based CDC ────────────────────────────────────
    print("""
  LISTEN/NOTIFY vs WAL-based CDC (Debezium / pgoutput):

  | Feature              | LISTEN/NOTIFY          | WAL / Debezium             |
  |----------------------|------------------------|----------------------------|
  | Durability           | None — fire & forget   | Durable (replication slot) |
  | Resume / replay      | No                     | Yes (LSN position)         |
  | Payload size         | 8 000 bytes            | Unlimited                  |
  | Schema changes       | Manual in trigger      | Automatic                  |
  | Setup complexity     | Low (SQL trigger)      | High (connector infra)     |
  | Replica set required | No                     | No (logical replication)   |
  | Best for             | Cache busting, hooks   | Full CDC, Kafka, audit     |
    """)

    # ── 5. Teardown ───────────────────────────────────────────────────────────
    await conn.execute("DROP TRIGGER IF EXISTS orders_change_trigger ON orders")
    await conn.execute("DROP FUNCTION IF EXISTS orders_change_notify")


# ---------------------------------------------------------------------------
# 19. Read-After-Write Consistency
# ---------------------------------------------------------------------------

async def demo_read_after_write(conn: asyncpg.Connection, pool: asyncpg.Pool) -> None:
    """
    PostgreSQL guarantees read-after-write consistency automatically — no extra
    settings required.  Every write is synchronously applied before the command
    returns; READ COMMITTED (the default) ensures every new statement sees all
    data committed so far by any session.

    The only exception: read replicas.  Streaming replication is asynchronous
    by default — a replica may lag behind the primary, so a read routed to a
    replica immediately after a primary write may miss that write.
    """

    # ── 1. Same connection — trivially consistent ─────────────────────────────
    # Write and read on the same connection; the read always sees the write.
    rac_id = await conn.fetchval(
        "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING id",
        "rac@example.com", "RAC User",
    )
    user = await conn.fetchrow(
        "SELECT id, email FROM users WHERE id = $1", rac_id
    )
    assert user is not None
    print(f"  [same conn] wrote and read back: {user['email']}")

    # ── 2. Across pool connections — consistent at READ COMMITTED ─────────────
    # asyncpg auto-commits each statement when outside an explicit transaction.
    # Any other connection in the pool immediately sees the committed row.
    async with pool.acquire() as other_conn:
        user2 = await other_conn.fetchrow(
            "SELECT id, email FROM users WHERE id = $1", rac_id
        )
        assert user2 is not None
        print(f"  [other conn] visible to pool peer: {user2['email']}")

    # ── 3. Open transaction — invisible to other sessions until COMMIT ────────
    # Within an open transaction your writes are visible to YOUR OWN queries
    # (READ COMMITTED within-transaction semantics) but not to other sessions.
    async with conn.transaction():
        uncommitted_id = await conn.fetchval(
            "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING id",
            "uncommitted@example.com", "Uncommitted",
        )
        # Your own connection can read the uncommitted row
        own_view = await conn.fetchrow(
            "SELECT id FROM users WHERE id = $1", uncommitted_id
        )
        assert own_view is not None

        # Another pool connection cannot see it yet
        async with pool.acquire() as other_conn:
            invisible = await other_conn.fetchrow(
                "SELECT id FROM users WHERE id = $1", uncommitted_id
            )
        assert invisible is None
        print("  [open txn]  uncommitted write: visible to self, invisible to peers")
    # Transaction commits here; the row is now visible to all connections.

    # ── 4. Read replica caveat and mitigations ────────────────────────────────
    print("""
  Read replica caveat:
    asyncpg connects to a single host.  If reads are routed to a standby,
    streaming replication lag (typically < 100 ms but unbounded) means the
    replica may not yet reflect a just-committed write.

  Mitigations (in order of increasing latency cost):
    a) Route write-then-read to the PRIMARY (simplest; no latency added).
    b) SET synchronous_commit = 'remote_apply';
       The primary waits until one replica has replayed the WAL before ACKing.
       Eliminates replica lag for that write; adds one round-trip of latency.
    c) pg_wal_replay_wait(target_lsn)  (PG 16+)
       Capture pg_current_wal_lsn() after the write on the primary; pass it
       to a replica query — the function blocks until the replica has replayed
       up to that LSN, then returns.
    d) Application-level sticky routing: after any write, direct that user's
       reads to the primary for N seconds.
    """)

    # Cleanup
    await conn.execute(
        "DELETE FROM users WHERE email = ANY($1::text[])",
        ["rac@example.com", "uncommitted@example.com"],
    )


# ---------------------------------------------------------------------------
# 20. Prepared Statements
# ---------------------------------------------------------------------------

async def demo_prepared_statements(conn: asyncpg.Connection) -> None:
    """
    conn.prepare() sends the query to PostgreSQL once for parsing/planning.
    Subsequent executions send only parameter values — faster for hot paths.
    """

    # Prepare once
    stmt = await conn.prepare(
        "SELECT id, email, name FROM users WHERE id = $1"
    )

    # Execute many times with different args — no re-parse overhead
    for user_id in range(1, 100):
        row = await stmt.fetchrow(user_id)
        if row:
            print(row["email"])

    # stmt.fetch() — multiple rows
    rows = await stmt.fetch(1)

    # stmt.fetchval() — single value
    val = await stmt.fetchval(1)

    # stmt.executemany() — bulk with a prepared plan
    insert_stmt = await conn.prepare(
        "INSERT INTO job_queue (payload) VALUES ($1)"
    )
    payloads = [('{"task": "send_email", "user_id": ' + str(i) + '}',) for i in range(1, 11)]
    await insert_stmt.executemany(payloads)

    # Inspect the inferred parameter and return types
    print(stmt.get_parameters())   # list of asyncpg type objects
    print(stmt.get_attributes())   # list of column definitions


# ---------------------------------------------------------------------------
# 20. Type Codecs
# ---------------------------------------------------------------------------

async def demo_type_codecs(dsn: str = "postgresql://rag_user:rag_pass@localhost:5434/ecommerce") -> None:
    """
    conn.set_type_codec() maps a PostgreSQL type to Python encode/decode callables.
    The pgvector pattern shows how to register a custom binary codec.
    """

    import json

    async def _init(conn: asyncpg.Connection) -> None:
        # Example: store/retrieve Python dicts as JSONB without manual json.dumps
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )

        # Register a simple domain type
        await conn.set_type_codec(
            "text",
            encoder=str,
            decoder=str,
            schema="pg_catalog",
            format="text",
        )

    pool = await asyncpg.create_pool(dsn, init=_init)

    # pgvector registration pattern (requires pgvector installed)
    # from pgvector.asyncpg import register_vector
    #
    # async def _init_with_vector(conn):
    #     await register_vector(conn)
    #
    # pool = await asyncpg.create_pool(dsn, init=_init_with_vector)
    #
    # Then you can pass/receive numpy arrays directly:
    # await conn.execute("INSERT INTO embeddings (vec) VALUES ($1)", np.array([0.1, 0.2, 0.3]))
    # row = await conn.fetchrow("SELECT vec FROM embeddings WHERE id = $1", 1)
    # vec: np.ndarray = row["vec"]

    await pool.close()


# ---------------------------------------------------------------------------
# 20. psycopg3 Comparison
# ---------------------------------------------------------------------------

async def demo_psycopg3_comparison() -> None:
    """
    Side-by-side comparison of asyncpg and psycopg3 patterns.
    Requires: pip install psycopg[binary]   (psycopg3, NOT psycopg2)
    """

    # -------------------------------------------------------------------------
    # CONNECTION
    # -------------------------------------------------------------------------
    #
    # asyncpg (async only):
    #   conn = await asyncpg.connect("postgresql://rag_user:rag_pass@localhost:5434/ecommerce")
    #
    # psycopg3 sync:
    #   import psycopg
    #   conn = psycopg.connect("postgresql://rag_user:rag_pass@localhost:5434/ecommerce")
    #
    # psycopg3 async:
    #   import psycopg
    #   conn = await psycopg.AsyncConnection.connect("...")
    #
    # -------------------------------------------------------------------------
    # PARAMETERIZED QUERIES
    # -------------------------------------------------------------------------
    #
    # asyncpg — PostgreSQL-native $N placeholders:
    #   await conn.fetch("SELECT * FROM users WHERE id = $1", 1)
    #
    # psycopg3 — uses %s (DB-API 2.0 style):
    #   cursor.execute("SELECT * FROM users WHERE id = %s", (1,))
    #
    # psycopg3 also supports named params with %(name)s:
    #   cursor.execute("SELECT * FROM users WHERE id = %(uid)s", {"uid": 1})
    #
    # -------------------------------------------------------------------------
    # FETCH METHODS
    # -------------------------------------------------------------------------
    #
    # asyncpg                          psycopg3
    # ───────────────────────────────  ─────────────────────────────────────
    # await conn.fetch(sql, *args)     cursor.execute(sql, args); cursor.fetchall()
    # await conn.fetchrow(sql, *args)  cursor.execute(sql, args); cursor.fetchone()
    # await conn.fetchval(sql, *args)  cursor.execute(sql, args); cursor.fetchone()[0]
    # await conn.execute(sql, *args)   cursor.execute(sql, args)   [no return value]
    #
    # asyncpg returns asyncpg.Record; psycopg3 returns tuple by default.
    # psycopg3 with row_factory=psycopg.rows.dict_row returns dicts.
    #
    # -------------------------------------------------------------------------
    # TRANSACTIONS
    # -------------------------------------------------------------------------
    #
    # asyncpg:
    #   async with conn.transaction():
    #       await conn.execute(...)
    #
    # psycopg3 sync:
    #   with conn.transaction():
    #       cursor.execute(...)
    #   # or just: conn.autocommit = False; conn.commit() / conn.rollback()
    #
    # psycopg3 async:
    #   async with conn.transaction():
    #       await cursor.execute(...)
    #
    # -------------------------------------------------------------------------
    # COPY
    # -------------------------------------------------------------------------
    #
    # asyncpg:
    #   await conn.copy_records_to_table("users", records=[...], columns=[...])
    #   await conn.copy_to_table("users", source=file_obj)
    #
    # psycopg3:
    #   with cursor.copy("COPY users (email, name) FROM STDIN") as copy:
    #       copy.write_row(("alice@example.com", "Alice"))
    #
    # -------------------------------------------------------------------------
    # KEY DIFFERENCES SUMMARY
    # -------------------------------------------------------------------------
    #
    # Feature               asyncpg                     psycopg3
    # ─────────────────────────────────────────────────────────────────────────
    # Protocol              Binary (faster)             Text by default
    # Return type           asyncpg.Record              tuple (or dict w/ factory)
    # Sync support          No                          Yes (and async)
    # Placeholders          $1, $2, …                   %s or %(name)s
    # Prepared stmts        Explicit conn.prepare()     Automatic via server_side
    # COPY API              copy_records_to_table()     cursor.copy(COPY …)
    # Type registration     set_type_codec()            register_adapter()
    # Pool                  asyncpg.create_pool()       psycopg_pool.AsyncConnectionPool()
    # DB-API 2.0 compat     No                          Yes
    pass


# ---------------------------------------------------------------------------
# Entrypoint — wire everything together for a quick smoke test
# ---------------------------------------------------------------------------

SEED_SQL = """
TRUNCATE users, products, orders, order_items, reviews, job_queue, categories
    RESTART IDENTITY CASCADE;

INSERT INTO users (email, name)
    VALUES ('alice@example.com', 'Alice');

INSERT INTO products (sku, name, description, price, category, stock_qty)
    VALUES ('SEED-001', 'Seed Widget', 'A demo widget for reference examples',
            9.99, 'widgets', 100);

INSERT INTO orders (user_id, status) VALUES (1, 'pending');

INSERT INTO order_items (order_id, product_id, qty, unit_price)
    VALUES (1, 1, 1, 9.99);

INSERT INTO job_queue (payload) VALUES ('{}');
"""


async def main() -> None:
    dsn = "postgresql://rag_user:rag_pass@localhost:5434/ecommerce"

    # Create pool
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)

    async with pool.acquire() as conn:
        # Set up schema then seed deterministic base data so all demos can use
        # hardcoded id=1 references (user_id=1, product_id=1, order_id=1, etc.)
        await conn.execute(SCHEMA_SQL)
        await conn.execute(SEED_SQL)

        # Run demos (each is safe to call in sequence)
        await demo_fetch_methods(conn)
        await demo_parameterized(conn)
        await demo_returning(conn)
        await demo_transactions(conn)
        await demo_savepoints(conn)
        await demo_row_locking(conn)
        await demo_mvcc(conn)
        await demo_cursors(conn)
        await demo_bulk(conn)
        await demo_upsert(conn)
        await demo_full_text_search(conn)
        await demo_jsonb(conn)
        await demo_jsonb_advanced(conn)
        await demo_bytea(conn)
        await demo_window_functions(conn)
        await demo_partition_by(conn)
        await demo_table_partitioning(conn)
        await demo_ctes(conn)
        await demo_lateral(conn)
        await demo_advisory_locks(conn)
        await demo_prepared_statements(conn)
        await demo_change_stream(conn, dsn)
        await demo_read_after_write(conn, pool)

    await demo_listen_notify(dsn)
    await demo_type_codecs(dsn)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
