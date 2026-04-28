import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import duckdb
from deltalake import DeltaTable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from deltalake.exceptions import DeltaError, CommitFailedError
except ImportError:
    from deltalake._internal import DeltaError, CommitFailedError  # type: ignore[no-redef]

# Pydantic-generated Arrow schema (mock)
MY_SCHEMA = pa.schema([
    pa.field("id",               pa.int64(),                   nullable=False),
    pa.field("status",           pa.utf8(),                    nullable=False),
    pa.field("last_updated",     pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("new_feature_flag", pa.bool_(),                   nullable=False),
])


def validate_target_schema(dt: DeltaTable) -> None:
    """Raise ValueError if the target Delta table is missing any columns from MY_SCHEMA."""
    target_fields = {f.name for f in dt.schema().to_pyarrow()}
    missing = {f.name for f in MY_SCHEMA} - target_fields
    if missing:
        raise ValueError(f"Target Delta table missing columns: {missing}")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((DeltaError, CommitFailedError)),
    reraise=True,
)
def run_pydantic_merge(target_uri: str, parquet_source: str) -> None:
    """
    Streaming merge from GCS Parquet source into a Delta table.

    Pipeline:
        1. Load Delta table & validate schema against Pydantic model
        2. Register PyArrow GCS dataset in DuckDB (zero-copy)
        3. Transform via DuckDB SQL, stream as RecordBatchReader
        4. Cast each batch lazily to Pydantic-generated schema
        5. Merge into Delta table with schema evolution & predicates

    Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to a
    service-account JSON file with read access to parquet_source.
    """
    # 1. Load latest Delta table & validate schema
    dt = DeltaTable(target_uri)
    validate_target_schema(dt)

    # 2. Register PyArrow GCS dataset in DuckDB (zero-copy, no RAM spike)
    gcs = pafs.GcsFileSystem()
    source_dataset = ds.dataset(parquet_source, format="parquet", filesystem=gcs)
    con = duckdb.connect()
    con.register("source_ds", source_dataset)

    # 3. Transform via DuckDB SQL, stream as RecordBatchReader
    source_arrow = con.execute("""
        SELECT
            id,
            upper(status)                        AS status,
            current_timestamp AT TIME ZONE 'UTC' AS last_updated,
            true                                 AS new_feature_flag
        FROM source_ds
    """).fetch_record_batch(rows_per_batch=10_000)

    # 4. Cast each batch lazily to target schema
    #    One batch in memory at a time — no RAM spike on large datasets.
    #    Schema mismatch on any batch aborts the merge and triggers a retry.
    source_arrow = pa.RecordBatchReader.from_batches(
        MY_SCHEMA, (batch.cast(MY_SCHEMA) for batch in source_arrow)
    )

    # 5. Atomic MERGE into Delta table
    (
        dt.merge(
            source=source_arrow,
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
