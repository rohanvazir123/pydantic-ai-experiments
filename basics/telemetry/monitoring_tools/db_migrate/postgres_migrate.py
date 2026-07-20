import os
from datetime import datetime
from prefect import task, flow
from prefect.client.schemas.schedules import CronSchedule
import psycopg
from psycopg.rows import dict_row

# Configuration Matrix
TABLES_TO_MIGRATE = ["users", "orders", "transactions"]
SRC_SCHEMA = "production"
TGT_SCHEMA = "clean_analytics"

# Database Connection URIs (Best practice is to load these from env variables)
SRC_DB_URI = os.getenv("SRC_DB_URI", "postgresql://user:pass@host:5432/src_db")
TGT_DB_URI = os.getenv("TGT_DB_URI", "postgresql://user:pass@host:5432/tgt_db")

'''
No Pipeline Stopping on Errors: By running return_state=True inside the for table loop, 
Prefect registers the specific table task execution failure in red on your UI dashboard, but seamlessly steps directly to the next table item in your python list without halting execution.

No Shared Memory Bottlenecks (OOM): 
The use of Psycopg 3's named server-side cursor (name="server_side_cursor") streams records from Postgres sequentially in micro-batches of 2000. 
It transforms them in memory chunks and flashes them downstream, allowing you to move 100-gigabyte tables effortlessly on minimal server resources.

Easy Triage: 
If the transactions table fails, you can visually locate it on your http://localhost:4200 browser page and click Rerun on that single item, or just open your 
local terminal and test it directly by running migrate_table_data.fn("transactions") instantly.
'''


# 1. DATA TRANSFORMATION LAYER
@task(name="Transform Table Row")
def transform_row(row: dict, table_name: str) -> dict:
    """Modifies database rows on-the-fly inside memory buffers."""
    # Data Cleaning: Strip out PII/sensitive info
    row.pop("password_hash", None)
    row.pop("ssn", None)
    
    # Audit Trail: Inject pipeline metadata into every target row
    row["migrated_at"] = datetime.utcnow()
    
    # Schema Evolution: Modify structures conditionally based on table context
    if table_name == "transactions" and "amt" in row:
        row["amount_usd"] = float(row.pop("amt"))
        
    return row


# 2. DATA EXTRACT-TRANSFORM-LOAD MIGRATION TASK
# Automatically retries up to 3 times if network cuts out, waiting 15 seconds between runs
@task(name="Migrate Table Data", retries=3, retry_delay_seconds=15)
def migrate_table_data(table_name: str):
    """Fetches, transforms, and bulk-inserts table data via memory streams."""
    print(f"Starting data migration for table: {SRC_SCHEMA}.{table_name}")
    
    # Establish distinct streaming connections using Psycopg 3
    with psycopg.connect(SRC_DB_URI, row_factory=dict_row) as src_conn, \
         psycopg.connect(TGT_DB_URI) as tgt_conn:
        
        with src_conn.cursor(name="server_side_cursor") as src_curr, \
             tgt_conn.cursor() as tgt_curr:
            
            # Step A: Dynamically extract column structure from the first source row
            src_curr.execute(f"SELECT * FROM {SRC_SCHEMA}.{table_name} LIMIT 1")
            sample_row = src_curr.fetchone()
            
            if not sample_row:
                print(f"Table {SRC_SCHEMA}.{table_name} is empty. Skipping data copy.")
                return

            # Apply transform logic to find the new target schema key layout
            transformed_sample = transform_row.fn(sample_row, table_name)
            columns = list(transformed_sample.keys())
            
            # Step B: Ensure the target schema and table structure exist
            tgt_curr.execute(f"CREATE SCHEMA IF NOT EXISTS {TGT_SCHEMA};")
            
            # This generates a basic target table clone matching your key names
            column_definitions = ", ".join([f'"{col}" TEXT' if "at" not in col else f'"{col}" TIMESTAMP' for col in columns])
            tgt_curr.execute(f'CREATE TABLE IF NOT EXISTS "{TGT_SCHEMA}"."{table_name}" ({column_definitions});')
            tgt_curr.execute(f'TRUNCATE TABLE "{TGT_SCHEMA}"."{table_name}";')
            
            # Step C: Stream data using a server-side cursor to prevent OOM memory errors
            src_curr.execute(f"SELECT * FROM {SRC_SCHEMA}.{table_name}")
            
            batch_size = 2000
            insert_query = f'INSERT INTO "{TGT_SCHEMA}"."{table_name}" ({", ".join([f'"{c}"' for c in columns])}) VALUES ({", ".join(["%s"] * len(columns))})'
            
            while True:
                rows = src_curr.fetchmany(batch_size)
                if not rows:
                    break
                
                transformed_batch = []
                for row in rows:
                    # Execute .fn directly to process data smoothly inside the loop iteration
                    clean_row = transform_row.fn(row, table_name)
                    transformed_batch.append(tuple(clean_row.get(col) for col in columns))
                
                # Bulk copy the chunk arrays into Postgres instantly
                tgt_curr.executemany(insert_query, transformed_batch)
            
            tgt_conn.commit()
    print(f"Finished data migration for table: {TGT_SCHEMA}.{table_name}")


# 3. SCHEMA & INDEX SYNCHRONIZATION TASK
@task(name="Sync Table Indexes")
def sync_table_indexes(table_name: str):
    """Clones production index structures over to your analytic replicas."""
    print(f"Analyzing index structures for table: {table_name}")
    
    # Query to fetch index definitions excluding primary keys
    index_lookup_sql = """
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE schemaname = %s AND tablename = %s AND indexname NOT LIKE '%%_pkey';
    """
    
    with psycopg.connect(SRC_DB_URI) as src_conn, \
         psycopg.connect(TGT_DB_URI) as tgt_conn:
         
        with src_conn.cursor() as src_curr, tgt_conn.cursor() as tgt_curr:
            # Fetch Source Indexes
            src_curr.execute(index_lookup_sql, (SRC_SCHEMA, table_name))
            src_indexes = {row[0]: row[1] for row in src_curr.fetchall()}
            
            # Fetch Target Indexes
            tgt_curr.execute(index_lookup_sql, (TGT_SCHEMA, table_name))
            tgt_indexes = {row[0]: row[1] for row in tgt_curr.fetchall()}
            
            # Step A: Drop target indexes that were removed in production
            for idx_name in tgt_indexes:
                if idx_name not in src_indexes:
                    print(f"Dropping obsolete target index: {idx_name}")
                    tgt_curr.execute(f'DROP INDEX IF EXISTS "{TGT_SCHEMA}"."{idx_name}";')
            
            # Step B: Build missing indexes using background rules safely
            for idx_name, idx_def in src_indexes.items():
                if idx_name not in tgt_indexes:
                    print(f"Deploying missing target index: {idx_name}")
                    # Rewrite the index definition to target the new schema location
                    clean_def = idx_def.replace(f" {SRC_SCHEMA}.", f" {TGT_SCHEMA}.")
                    tgt_curr.execute(clean_def)
            
            tgt_conn.commit()


# 4. MASTER FLOW LOGIC
@flow(name="Postgres Production Table Sync Engine")
def postgres_sync_flow():
    failed_tables = []
    
    for table in TABLES_TO_MIGRATE:
        # return_state=True stops an isolated error from blowing up the remaining iterations [1]
        migration_state = migrate_table_data(table_name=table, return_state=True)
        
        if migration_state.is_failed():
            print(f"❌ Error encountered migrating data for '{table}'. Continuing pipeline...")
            failed_tables.append(table)
            continue  # Skip index matching for this specific broken table layout
            
        # If data loaded fine, process index adjustments [1]
        index_state = sync_table_indexes(table_name=table, return_state=True)
        if index_state.is_failed():
            print(f"⚠️ Index optimization failed for '{table}', data was migrated successfully.")

    # Single aggregation point for errors
    if failed_tables:
        print(f"CRITICAL SYSTEM SUMMARY: Pipeline finished with table failures: {failed_tables}")
        # You can tie Prefect Automations here to alert Slack, PagerDuty, or Email


# Trigger Deployment Config
if __name__ == "__main__":
    # To deploy as a Background Daemon Cron Engine (Runs daily at midnight UTC)
    postgres_sync_flow.serve(
        name="postgres-table-sync-deployment",
        schedule=CronSchedule(cron="0 0 * * *", timezone="UTC")
    )
