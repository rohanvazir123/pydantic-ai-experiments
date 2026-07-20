'''
Key Differences to Note (Airflow vs Prefect)No with DAG(...) 
Context Managers: Prefect uses the @flow decorator 
directly over a standard Python function wrapper.
No PythonOperator wrappers: 
You simply call the python tasks directly inside your flow function 
loop (migrate_collection_data(col)). 
Prefect automatically intercepts the execution to handle logging, 
retries, and UI state tracking.transform_document.fn() usage:
To call a @task function inside another task 
without creating nested UI structures, you use the .fn attribute, 
which exposes the raw underlying Python function.
'''

from datetime import datetime
from pymongo import MongoClient
from prefect import task, flow
from prefect.client.schemas.schedules import CronSchedule

# Configuration Constants
COLLECTIONS = ["users", "transactions"]
SRC_URI = "mongodb://username:password@source_host:27017/"
TGT_URI = "mongodb://username:password@target_host:27017/"
SRC_DB = "production_db"
TGT_DB = "clean_analytics_db"


# 1. TRANSFORMATION LOGIC TASK
@task(name="Transform Document Data")
def transform_document(doc: dict, collection_name: str) -> dict:
    """Applies structural changes to a document in flight."""
    doc.pop("password_hash", None)
    doc.pop("session_tokens", None)
    
    doc["migrated_at"] = datetime.utcnow()
    
    if collection_name == "transactions" and "amt" in doc:
        doc["amount_usd"] = float(doc.pop("amt"))
            
    return doc


# 2. DATA MIGRATION TASK
@task(name="Migrate Collection Data", retries=2, retry_delay_seconds=30)
def migrate_collection_data(collection_name: str):
    """Streams data, transforms it, and loads it to the target in batches."""
    src_client = MongoClient(SRC_URI)
    tgt_client = MongoClient(TGT_URI)
    
    src_coll = src_client[SRC_DB][collection_name]
    tgt_coll = tgt_client[TGT_DB][collection_name]
    
    batch = []
    batch_size = 500
    
    for doc in src_coll.find():
        # Call transformation function directly inside the task
        transformed_doc = transform_document.fn(doc, collection_name)
        batch.append(transformed_doc)
        
        if len(batch) >= batch_size:
            tgt_coll.insert_many(batch)
            batch = []
            
    if batch:
        tgt_coll.insert_many(batch)
        
    src_client.close()
    tgt_client.close()


# 3. INDEX SYNC TASK
@task(name="Sync Collection Indexes")
def sync_collection_indexes(collection_name: str):
    """Compares production and target index schemas, dropping or adding changes."""
    src_client = MongoClient(SRC_URI)
    tgt_client = MongoClient(TGT_URI)
    
    src_coll = src_client[SRC_DB][collection_name]
    tgt_coll = tgt_client[TGT_DB][collection_name]
    
    src_indexes = src_coll.index_information()
    tgt_indexes = tgt_coll.index_information()
    
    # Drop target indexes no longer present in production
    for index_name in list(tgt_indexes.keys()):
        if index_name == "_id_":
            continue
        if index_name not in src_indexes:
            print(f"Dropping stale index: {index_name} from {collection_name}")
            tgt_coll.drop_index(index_name)
            
    # Recreate missing or new indexes from production
    for index_name, index_info in src_indexes.items():
        if index_name == "_id_":
            continue
        if index_name not in tgt_indexes:
            print(f"Creating new index: {index_name} on {collection_name}")
            index_keys = index_info["key"]
            index_options = {k: v for k, v in index_info.items() if k not in ["key", "v", "ns"]}
            tgt_coll.create_index(index_keys, name=index_name, **index_options)
            
    src_client.close()
    tgt_client.close()


# 4. THE MASTER FLOW (The DAG equivalent)
@flow(name="MongoDB Production Maintenance & Sync")
def mongodb_maintenance_flow():
    for col in COLLECTIONS:
        # Run migration and index syncing in clean, sequentially trackable task blocks
        migrate_collection_data(collection_name=col)
        sync_collection_indexes(collection_name=col)

from prefect import task, flow

def resilient_prefect_flow():
    failed_collections = []
    
    for col in ["users", "transactions", "logs"]:
        # return_state=True prevents the exception from crashing the entire script loop
        state = migrate_collection_data(collection_name=col, return_state=True)
        
        if state.is_failed():
            print(f"CRITICAL ERROR: {col} failed! Logging issue and continuing loop.")
            failed_collections.append(col)
        else:
            print(f"Task completed cleanly for {col}")

    # Send a single comprehensive custom alert summary at the very end
    if failed_collections:
        print(f"ALERT SENDING: The following tables failed and need manual fixing: {failed_collections}")

if __name__ == "__main__":
    resilient_prefect_flow()


'''
if __name__ == "__main__":
    # To run manually right now:
    # mongodb_maintenance_flow()
    
    # To deploy this script with a Cron schedule (Runs daily at midnight)
    mongodb_maintenance_flow.serve(
        name="daily-mongo-sync-deployment",
        schedule=CronSchedule(cron="0 0 * * *", timezone="UTC")
    )
'''

