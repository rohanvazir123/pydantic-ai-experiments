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

async def demo_connections(dsn: str = "postgresql://localhost/ecommerce") -> None:
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


async def demo_single_connection(dsn: str = "postgresql://localhost/ecommerce") -> None:
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
    csv_output = await conn.copy_from_query(
        "SELECT email, name FROM users ORDER BY id",
        format="csv",
        header=True,
    )
    print(csv_output.decode())  # CSV text


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

async def demo_listen_notify(dsn: str = "postgresql://localhost/ecommerce") -> None:
    """
    LISTEN/NOTIFY is PostgreSQL's lightweight pub/sub.
    asyncpg callbacks fire in the event loop when a notification arrives.
    Use a dedicated connection (not from the pool) — it must stay open.
    """

    notifications_received: list[asyncpg.Notification] = []

    def on_order_event(
        conn: asyncpg.Connection,
        pid: int,           # PID of the notifying backend
        channel: str,       # channel name
        payload: str,       # arbitrary string payload
    ) -> None:
        print(f"[{channel}] from PID {pid}: {payload}")
        notifications_received.append(
            asyncpg.Notification(pid=pid, channel=channel, payload=payload)
        )

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
# 18. Prepared Statements
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
# 19. Type Codecs
# ---------------------------------------------------------------------------

async def demo_type_codecs(dsn: str = "postgresql://localhost/ecommerce") -> None:
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
    #   conn = await asyncpg.connect("postgresql://localhost/ecommerce")
    #
    # psycopg3 sync:
    #   import psycopg
    #   conn = psycopg.connect("postgresql://localhost/ecommerce")
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

async def main() -> None:
    dsn = "postgresql://localhost/ecommerce"

    # Create pool
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)

    async with pool.acquire() as conn:
        # Set up schema
        await conn.execute(SCHEMA_SQL)

        # Run demos (each is safe to call in sequence)
        await demo_fetch_methods(conn)
        await demo_parameterized(conn)
        await demo_returning(conn)
        await demo_transactions(conn)
        await demo_savepoints(conn)
        await demo_row_locking(conn)
        await demo_cursors(conn)
        await demo_bulk(conn)
        await demo_upsert(conn)
        await demo_full_text_search(conn)
        await demo_jsonb(conn)
        await demo_window_functions(conn)
        await demo_ctes(conn)
        await demo_lateral(conn)
        await demo_advisory_locks(conn)
        await demo_prepared_statements(conn)

    await demo_listen_notify(dsn)
    await demo_type_codecs(dsn)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
