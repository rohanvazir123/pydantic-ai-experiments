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

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake


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

def run_merge(uri: str, source: pa.Table) -> None:
    """
    Perform an atomic MERGE from source into the Delta table at uri.

    A MERGE is a single atomic commit that can update, insert, and delete rows
    in one pass — no separate UPDATE/INSERT/DELETE transactions needed.
    Delta Lake writes all changes to Parquet files and records them in the
    _delta_log/ as a single version, so readers either see the full new state
    or the full old state — never a partial update.
    """

    # Load the current version of the Delta table from disk.
    # DeltaTable reads the _delta_log/ to find the latest snapshot.
    dt = DeltaTable(uri)

    (
        dt.merge(
            source=source,           # the incoming PyArrow table (or RecordBatchReader)
            predicate="t.id = s.id", # JOIN key: rows are "matched" when their ids are equal
            source_alias="s",        # alias for the source table used in predicates below
            target_alias="t",        # alias for the target (Delta) table
        )

        # WHEN MATCHED (id exists in both source and target)
        # Only update the row if the status value actually changed.
        # Without this predicate, every matched row would be rewritten even if
        # nothing changed — wasteful on large tables.
        #   id=2: INACTIVE != ACTIVE  -> updated
        #   id=3: ACTIVE   == ACTIVE  -> skipped (predicate is false)
        .when_matched_update_all(predicate="s.status != t.status")

        # WHEN NOT MATCHED (id is in source but NOT in target — a brand-new row)
        # Only insert if the new_feature_flag is set. This lets us filter out
        # source rows we don't want to land in the target yet.
        #   id=5: not in target, flag=true -> inserted
        .when_not_matched_insert_all(predicate="s.new_feature_flag = true")

        # WHEN NOT MATCHED BY SOURCE (id is in target but NOT in source — row disappeared)
        # Delete stale target rows, but only if they are INACTIVE.
        # Rows that are ACTIVE or PENDING but absent from the source are left alone —
        # they may simply not have been included in this batch.
        #   id=6: INACTIVE, not in source -> deleted
        #   id=1: ACTIVE,   not in source -> kept
        #   id=4: PENDING,  not in source -> kept
        .when_not_matched_by_source_delete(predicate="t.status = 'INACTIVE'")

        # Commit all three operations atomically.
        # This writes new Parquet files and appends a single entry to _delta_log/.
        # On failure (e.g. concurrent writer), the whole merge is rolled back.
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
