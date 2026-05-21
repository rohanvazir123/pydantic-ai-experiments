# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick inspection script for GCS partitioned data."""
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"""
CREATE OR REPLACE SECRET gcs_secret (
  TYPE    gcs,
  KEY_ID  '{os.environ["GCS_HMAC_ID"]}',
  SECRET  '{os.environ["GCS_HMAC_SECRET"]}'
);
""")

print("=== orders ===")
print("schema:")
for row in con.execute("DESCRIBE SELECT * FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet') LIMIT 1").fetchall():
    print(" ", row)

print("\ndistinct order_dates:")
for row in con.execute("SELECT DISTINCT order_date, COUNT(1) FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet') GROUP BY 1 ORDER BY 1").fetchall():
    print(" ", row)

print("\ntotal rows:", con.execute("SELECT COUNT(1) FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet')").fetchone())
print("distinct order_ids:", con.execute("SELECT COUNT(DISTINCT order_id) FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet')").fetchone())

print("\nsample (5 rows):")
for row in con.execute("SELECT * FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/orders/*/*.parquet') ORDER BY order_id LIMIT 5").fetchall():
    print(" ", row)

print("\n=== users ===")
print("total rows:", con.execute("SELECT COUNT(1) FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/users/*/*.parquet')").fetchone())
print("distinct user_ids:", con.execute("SELECT COUNT(DISTINCT user_id) FROM parquet_scan('gs://eagerbeaver-1/partitioned_data/users/*/*.parquet')").fetchone())
