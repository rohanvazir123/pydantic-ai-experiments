"""DuckDB in-memory fixture for NL2SQL eval.

Extracted from test_nlp_sql_postgres_v2.py so the eval runner can reuse
the same schema and data without importing pytest machinery.
"""

from __future__ import annotations

import duckdb

SCHEMA_TEXT = """\
Table: sales
Columns:
  product  VARCHAR   -- product name, e.g. 'Laptop', 'Monitor'
  user_id  INTEGER   -- buyer identifier
  quantity INTEGER   -- units purchased in this transaction
  revenue  DOUBLE    -- transaction revenue in USD

Sample data (4 rows):
  product   user_id  quantity  revenue
  Laptop    1        3         3000.0
  Laptop    2        1         1000.0
  Monitor   1        2          600.0
  Monitor   3        5         1500.0

Rules:
  - GCS Parquet tables  → bare table name,        e.g. FROM sales
  - rag_db tables       → rag.main.<table>,        e.g. FROM rag.main.documents
  - local_pg tables     → local_pg.main.<table>,   e.g. FROM local_pg.main.baby_names
  - Return only SELECT queries. Never use DROP, DELETE, INSERT, UPDATE, or ALTER.\
"""


def build_sales_fixture() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection pre-loaded with the sales table."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE sales (
            product  VARCHAR,
            user_id  INTEGER,
            quantity INTEGER,
            revenue  DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO sales VALUES
            ('Laptop',  1, 3, 3000.0),
            ('Laptop',  2, 1, 1000.0),
            ('Monitor', 1, 2,  600.0),
            ('Monitor', 3, 5, 1500.0)
    """)
    return conn
