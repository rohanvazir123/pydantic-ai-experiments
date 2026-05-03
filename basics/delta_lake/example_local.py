"""
Fully working local Delta Lake example — no GCS or cloud credentials needed.

Delta Lake is an open-source storage layer that adds ACID transactions, versioning,
and schema enforcement on top of plain Parquet files. Every write creates a new
entry in the _delta_log/ directory; reads can target any past version.

This example walks through the four core operations:
  1. write_deltalake  — seed initial rows (version 0)
  2. DeltaTable.merge — upsert + conditional delete in one atomic commit (version 1)
  3. Time travel      — read version 0 after version 1 exists
  4. History          — list all commits and their operation metadata

Expected output after merge:
  - id=2  INACTIVE -> ACTIVE   (matched, status changed  -> updated)
  - id=3  ACTIVE   -> ACTIVE   (matched, status same     -> skipped by predicate)
  - id=5  new record           (not in target, flag=true -> inserted)
  - id=6  INACTIVE, absent from source                  -> deleted
  - id=1, id=4  untouched (absent from source, but not INACTIVE -> kept)
"""
import datetime
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from deltalake.exceptions import CommitFailedError, DeltaError
except ImportError:
    from deltalake._internal import CommitFailedError, DeltaError  # type: ignore[no-redef]

from basics.delta_lake.dt_merge import validate_target_schema


# ---------------------------------------------------------------------------
# Schema
#
# We define a single PyArrow schema that both the seed data and the incoming
# merge batch must conform to. Delta Lake stores this schema in the transaction
# log and will enforce it on every write — mismatched types raise an error
# before any data is committed.
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    pa.field("id",               pa.int64(),                   nullable=False),
    pa.field("status",           pa.utf8(),                    nullable=False),
    pa.field("last_updated",     pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("new_feature_flag", pa.bool_(),                   nullable=False),
])

# Single timestamp shared across all rows in this run — keeps the example output clean.
_NOW = datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch(ids: list[int], statuses: list[str], flags: list[bool]) -> pa.Table:
    """Build a typed PyArrow table from plain Python lists."""
    n = len(ids)
    return pa.table(
        {
            "id":               pa.array(ids,        type=pa.int64()),
            "status":           pa.array(statuses,   type=pa.utf8()),
            "last_updated":     pa.array([_NOW] * n, type=pa.timestamp("us", tz="UTC")),
            "new_feature_flag": pa.array(flags,      type=pa.bool_()),
        },
        schema=SCHEMA,
    )


def _print(dt: DeltaTable, label: str) -> None:
    """Read the full table at its current version and pretty-print every row."""
    rows = dt.to_pyarrow_table().sort_by("id").to_pylist()
    print(f"\n-- {label}  (v{dt.version()},  {len(rows)} rows) --")
    for r in rows:
        print(f"  id={r['id']:>2}  status={r['status']:<10}  flag={r['new_feature_flag']}")


# ---------------------------------------------------------------------------
# Step 1 — Seed
# ---------------------------------------------------------------------------

def seed(uri: str) -> None:
    """
    Write the initial Delta table.

    write_deltalake() creates the _delta_log/ directory and commits version 0.
    mode="overwrite" drops any existing data at that path first.
    """
    data = _batch(
        #         id=1       id=2         id=3       id=4         id=6
        ids=      [1,        2,           3,         4,           6         ],
        statuses= ["ACTIVE", "INACTIVE",  "ACTIVE",  "PENDING",   "INACTIVE"],
        flags=    [True,     True,        True,      True,        True      ],
    )
    write_deltalake(uri, data, mode="overwrite")
    print(f"[seed] {len(data)} rows written -> version 0")


# ---------------------------------------------------------------------------
# Step 2 — Incoming batch (simulates the GCS/DuckDB source in dt_merge.py)
# ---------------------------------------------------------------------------

def incoming_batch() -> pa.Table:
    """
    In production this batch arrives from GCS via DuckDB (see dt_merge.py).
    Here we build it by hand to keep the example self-contained.

    Only ids 2, 3, 5 are present in the source.
    ids 1, 4, 6 are absent — the merge will decide what to do with them
    based on the when_not_matched_by_source_delete predicate.
    """
    return _batch(
        ids=      [2,        3,        5       ],
        statuses= ["ACTIVE", "ACTIVE", "ACTIVE"],
        flags=    [True,     True,     True    ],
    )


# ---------------------------------------------------------------------------
# Step 3 — Merge
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((DeltaError, CommitFailedError)),
    reraise=True,
)
def run_merge(uri: str, source: pa.Table) -> None:
    """
    Perform an atomic MERGE from source into the Delta table at uri.

    Follows the same pipeline as run_pydantic_merge() in dt_merge.py:
      1. Load Delta table & validate schema
      2. Register source in DuckDB (zero-copy)
      3. Transform via SQL, stream as RecordBatchReader
      4. Cast each batch lazily to SCHEMA
      5. Merge with schema evolution & predicates; retry on commit conflicts
    """
    # 1. Load latest snapshot & validate schema
    dt = DeltaTable(uri)
    validate_target_schema(dt)

    # 2. Register in-memory PyArrow table in DuckDB (zero-copy)
    con = duckdb.connect()
    con.register("source_tbl", source)

    # 3. Transform via SQL, stream as RecordBatchReader (one batch in RAM at a time)
    source_reader = con.execute("""
        SELECT
            id,
            status,
            current_timestamp AT TIME ZONE 'UTC' AS last_updated,
            new_feature_flag
        FROM source_tbl
    """).fetch_record_batch(rows_per_batch=10_000)

    # 4. Cast each batch lazily to SCHEMA — schema mismatch aborts & triggers retry
    source_reader = pa.RecordBatchReader.from_batches(
        SCHEMA, (batch.cast(SCHEMA) for batch in source_reader)
    )

    # 5. Atomic MERGE:
    #   id=2: INACTIVE != ACTIVE  -> updated
    #   id=3: ACTIVE   == ACTIVE  -> skipped (predicate false)
    #   id=5: not in target, flag=true -> inserted
    #   id=6: INACTIVE, not in source -> deleted
    #   id=1, id=4: absent from source but not INACTIVE -> kept
    (
        dt.merge(
            source=source_reader,
            predicate="t.id = s.id",
            source_alias="s",
            target_alias="t",
            merge_schema=True,
        )
        .when_matched_update_all(predicate="s.status != t.status")
        .when_not_matched_insert_all(predicate="s.new_feature_flag = true")
        .when_not_matched_by_source_delete(predicate="t.status = 'INACTIVE'")
        .execute()
    )
    print("[merge] committed -> version 1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # TemporaryDirectory gives us a clean local path that is deleted on exit.
    # In production, uri would be a GCS path: "gs://my-bucket/orders_delta"
    with tempfile.TemporaryDirectory() as tmp:
        uri = str(Path(tmp) / "orders_delta")

        # 1. Write version 0
        seed(uri)
        _print(DeltaTable(uri), "after seed")

        # 2. Merge source batch into the table -> produces version 1
        run_merge(uri, incoming_batch())
        _print(DeltaTable(uri), "after merge")

        # 3. Time travel: pass version=0 to read the state before the merge.
        #    The Parquet files from version 0 are still on disk — Delta never
        #    deletes old files until you explicitly run VACUUM.
        _print(DeltaTable(uri, version=0), "time travel -> v0")

        # 4. History: every entry in _delta_log/ is one commit.
        #    operationParameters records exactly what predicates were used.
        print("\n-- commit history --")
        for entry in DeltaTable(uri).history():
            params = entry.get("operationParameters", {})
            print(f"  v{entry['version']}  {entry['operation']:<20}  {params}")


if __name__ == "__main__":
    main()
