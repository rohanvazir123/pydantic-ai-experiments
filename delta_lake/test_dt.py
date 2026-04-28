import pprint
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

con = duckdb.connect(database=":memory:")

# Enable httpfs extension if using GCS
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# Register your HMAC credentials as a DuckDB secret
con.execute(f"""
CREATE OR REPLACE SECRET gcs_secret (
  TYPE gcs,
  KEY_ID '{os.environ["GCS_HMAC_ID"]}',
  SECRET '{os.environ["GCS_HMAC_SECRET"]}'
);
""")


# orders table: point to all Parquet files across partitions
orders_path = "gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet"

con.execute(f"""
CREATE OR REPLACE TABLE orders AS
SELECT * FROM parquet_scan('{orders_path}');
""")


# Users table: point to all Parquet files across partitions
users_path = "gs://eagerbeaver-1/partitioned_data/users/*/*.parquet"

con.execute(f"""
CREATE OR REPLACE TABLE users AS
SELECT * FROM parquet_scan('{users_path}');
""")

print("\nTables in DuckDB:", con.execute("SHOW TABLES;").fetchall())
sample_orders = con.execute("SELECT * FROM orders LIMIT 10;").fetchall()
sample_users = con.execute("SELECT * FROM users LIMIT 10;").fetchall()

print(
    "\nSample orders:",
)
pprint.pprint(sample_orders)

print(
    "\nSample users:",
)
pprint.pprint(sample_users)
