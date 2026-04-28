"""
GCS-backed Delta Lake example using real orders data from gs://eagerbeaver-1.

Data layout in GCS:
  gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet
    columns  : order_id, user_id, product_name, quantity, order_date
    total    : 600 rows across 4 date partitions (2025-07-25/26/28/29)
    key note : order_id is NOT unique globally — same order_id can appear
               on different dates, so the natural key is (order_id, order_date)

Pipeline:
  1. Seed   — load the two earliest date partitions into a local Delta table (v0)
  2. Merge  — bring in the two later partitions as "new arrivals" + update 5
               existing orders with corrected quantities (v1)
  3. Travel — read version 0 to confirm original data is unchanged
  4. History — show every commit
"""
import datetime
import os
from pathlib import Path

import duckdb
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from dotenv import load_dotenv

# Load GCS HMAC credentials — looks for .env in the same directory as this file
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# GCS paths
# ---------------------------------------------------------------------------

ORDERS_PATH = "gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet"

# ---------------------------------------------------------------------------
# Schema
#
# Matches the GCS Parquet columns exactly, plus ingested_at.
# DuckDB returns VARCHAR as utf8 (pa.string()), so we use that here.
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    pa.field("order_id",     pa.int64(),   nullable=True),
    pa.field("user_id",      pa.int64(),   nullable=True),
    pa.field("product_name", pa.utf8(),    nullable=True),
    pa.field("quantity",     pa.int64(),   nullable=True),
    pa.field("order_date",   pa.date32(),  nullable=True),
    pa.field("ingested_at",  pa.timestamp("us", tz="UTC"), nullable=True),
])


# ---------------------------------------------------------------------------
# DuckDB + GCS connection
# ---------------------------------------------------------------------------

def _gcs_con() -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB connection authenticated for GCS.

    httpfs lets DuckDB read gs:// paths directly.
    The named secret is picked up automatically by every subsequent
    parquet_scan() in this connection.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE OR REPLACE SECRET gcs_secret (
          TYPE    gcs,
          KEY_ID  '{os.environ["GCS_HMAC_ID"]}',
          SECRET  '{os.environ["GCS_HMAC_SECRET"]}'
        );
    """)
    return con


def _read_orders(
    con: duckdb.DuckDBPyConnection,
    where: str = "",
    limit: int | None = None,
) -> pa.Table:
    """
    Run a SELECT against the GCS Parquet files and return a PyArrow Table.

    where  — optional SQL predicate (no WHERE keyword), e.g. "order_date = '2025-07-25'"
    limit  — optional row cap applied after the WHERE filter

    ingested_at is added here — it records when this batch was loaded,
    not when the order was placed.
    """
    filter_clause = f"WHERE {where}" if where else ""
    limit_clause  = f"LIMIT {limit}" if limit else ""
    tbl = con.execute(f"""
        SELECT
            order_id,
            user_id,
            product_name,
            quantity,
            order_date,
            current_timestamp AT TIME ZONE 'UTC' AS ingested_at
        FROM parquet_scan('{ORDERS_PATH}')
        {filter_clause}
        ORDER BY order_date, order_id
        {limit_clause}
    """).arrow().read_all()

    # Cast to SCHEMA so the timezone is normalised and types are consistent
    # regardless of which DuckDB version is in use.
    return tbl.cast(SCHEMA)


# ---------------------------------------------------------------------------
# Step 1 — Seed: load the two earliest date partitions
# ---------------------------------------------------------------------------

def seed(uri: str) -> None:
    """
    Write the historical backfill (Jul 25 + Jul 26) into the Delta table.
    This becomes version 0 — 300 rows, two date partitions.

    In production this would be a one-time historical load before you
    switch to daily incremental merges.
    """
    con = _gcs_con()
    data = _read_orders(con, where="order_date IN ('2025-07-25', '2025-07-26')")
    write_deltalake(uri, data, mode="overwrite")
    print(f"[seed] {len(data)} rows (Jul 25 + Jul 26) -> version 0")


# ---------------------------------------------------------------------------
# Step 2 — Incoming batch: next two days + 5 quantity corrections
# ---------------------------------------------------------------------------

def incoming_batch() -> pa.Table:
    """
    Simulate what arrives in the daily incremental run:

      a) New partitions: Jul 28 (134 rows) + Jul 29 (166 rows)
         These order_ids do not exist in the target -> INSERTED

      b) Quantity corrections: 5 orders from Jul 25 with doubled quantity
         These match on (order_id, order_date) and quantity differs -> UPDATED

    In production, both of these would come from GCS. Here we build
    the corrections manually so the example is fully self-contained.
    """
    con = _gcs_con()

    # New date partitions — none of these (order_id, order_date) pairs exist yet
    new_dates = _read_orders(con, where="order_date IN ('2025-07-28', '2025-07-29')")

    # Quantity corrections for the first 5 Jul 25 orders
    corrections = _read_orders(con, where="order_date = '2025-07-25'", limit=5)
    # Double the quantity to trigger the update predicate (s.quantity != t.quantity)
    corrections = corrections.set_column(
        corrections.schema.get_field_index("quantity"),
        "quantity",
        pa.array([q * 2 for q in corrections.column("quantity").to_pylist()], type=pa.int64()),
    )

    # Stack new dates and corrections into one source batch
    return pa.concat_tables([new_dates, corrections])


# ---------------------------------------------------------------------------
# Step 3 — Merge
# ---------------------------------------------------------------------------

def run_merge(uri: str, source: pa.Table) -> None:
    """
    Atomic MERGE on the composite key (order_id, order_date).

    We use a composite key because order_id alone is not globally unique —
    the same order_id can appear on different dates as separate orders.
    """

    # Load the latest snapshot from _delta_log/
    dt = DeltaTable(uri)

    (
        dt.merge(
            source=source,
            # Composite join: match on BOTH order_id and order_date.
            # A row is "matched" only if both values align — this is safe
            # because (order_id, order_date) is unique in our dataset.
            predicate="t.order_id = s.order_id AND t.order_date = s.order_date",
            source_alias="s",
            target_alias="t",
        )

        # WHEN MATCHED — same (order_id, order_date) in both source and target
        # Only rewrite the row if the quantity actually changed.
        # The 5 correction rows have doubled quantity, so they pass this predicate.
        # All other Jul 25/26 rows that happen to appear in source are skipped.
        .when_matched_update_all(predicate="s.quantity != t.quantity")

        # WHEN NOT MATCHED — (order_id, order_date) pair is new
        # The 300 Jul 28/29 rows don't exist in the target at all -> inserted.
        .when_not_matched_insert_all()

        # One atomic _delta_log commit covering all three groups of rows.
        .execute()
    )
    print("[merge] committed -> version 1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(dt: DeltaTable, label: str, n: int = 10) -> None:
    rows = dt.to_pyarrow_table().sort_by([("order_date", "ascending"), ("order_id", "ascending")]).to_pylist()
    print(f"\n-- {label}  (v{dt.version()},  {len(rows)} rows) --")
    for r in rows[:n]:
        print(
            f"  {r['order_date']}  order_id={r['order_id']:>3}  "
            f"product={r['product_name']:<12}  qty={r['quantity']}"
        )
    if len(rows) > n:
        print(f"  ... {len(rows) - n} more rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Fixed local path so you can inspect _delta_log/ and Parquet files after the run.
    # The seed step uses mode="overwrite" so re-running always starts fresh.
    uri = str(Path(__file__).parent / "output" / "orders_delta")

    # 1. Historical backfill -> version 0
    seed(uri)
    _print(DeltaTable(uri), "after seed")

    # 2. Build today's incoming batch
    batch = incoming_batch()
    print(f"\n[batch] {len(batch)} rows  ({len(batch) - 5} new date partitions + 5 corrections)")

    # 3. Merge -> version 1
    run_merge(uri, batch)
    _print(DeltaTable(uri), "after merge")

    # 4. Time travel: read version 0 (Jul 25/26 data with original quantities)
    _print(DeltaTable(uri, version=0), "time travel -> v0")

    # 5. Commit history
    print("\n-- commit history --")
    for entry in DeltaTable(uri).history():
        print(f"  v{entry['version']}  {entry['operation']:<20}  {entry.get('operationParameters', {})}")


if __name__ == "__main__":
    main()
