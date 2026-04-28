"""
Tests for nlp_sql_postgres_v1.py

Strategy:
- DuckDB is used for real (in-memory) -- no mocking needed
- GCS and PostgreSQL connections are mocked so tests run offline
- Pydantic AI agent is mocked to return controlled SQL strings
- ConversationManager is tested with real DuckDB execution
"""

import hashlib
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from nlp_sql_postgres_v1 import (
    ConversationManager,
    PostgresDB,
    UnifiedDataSource,
    strip_sql_fences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection with a sample sales table."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE sales (
            product VARCHAR,
            user_id INTEGER,
            quantity INTEGER,
            revenue DOUBLE
        )
    """)
    c.execute("""
        INSERT INTO sales VALUES
            ('Laptop',  1, 3, 3000.0),
            ('Laptop',  2, 1, 1000.0),
            ('Monitor', 1, 2,  600.0),
            ('Monitor', 3, 5, 1500.0)
    """)
    yield c
    c.close()


def _mock_agent(sql: str) -> MagicMock:
    """Return a mock Pydantic AI agent whose run_sync always returns `sql`."""
    run_result = MagicMock()
    run_result.output = sql
    agent = MagicMock()
    agent.run_sync.return_value = run_result
    return agent


def _manager(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    schema: str = "Table: sales\n  - product (VARCHAR)",
) -> ConversationManager:
    return ConversationManager(conn, _mock_agent(sql), schema)


# ---------------------------------------------------------------------------
# strip_sql_fences
# ---------------------------------------------------------------------------
class TestStripSqlFences:
    def test_plain_sql_unchanged(self):
        assert strip_sql_fences("SELECT 1") == "SELECT 1"

    def test_removes_sql_fences(self):
        assert strip_sql_fences("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_removes_plain_fences(self):
        assert strip_sql_fences("```\nSELECT 1\n```") == "SELECT 1"

    def test_strips_whitespace(self):
        assert strip_sql_fences("  SELECT 1  ") == "SELECT 1"

    def test_multiline_query_preserved(self):
        result = strip_sql_fences("```sql\nSELECT a,\n  b\nFROM t\n```")
        assert "SELECT a," in result
        assert "FROM t" in result


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------
class TestConversationManager:
    def test_run_query_returns_results(self, conn):
        mgr = _manager(conn, "SELECT product, SUM(quantity) FROM sales GROUP BY product")
        result = mgr.run_query("Total quantity per product?")
        assert result is not None
        assert len(result) == 2

    def test_result_added_to_history(self, conn):
        mgr = _manager(conn, "SELECT COUNT(*) FROM sales")
        mgr.run_query("How many rows?")
        assert len(mgr.history) == 1
        assert mgr.history[0][0] == "How many rows?"

    def test_nl_cache_hit_skips_agent(self, conn):
        agent = _mock_agent("SELECT COUNT(*) FROM sales")
        mgr = ConversationManager(conn, agent, "schema")
        mgr.run_query("How many rows?")
        mgr.run_query("How many rows?")  # exact NL match -> agent not called again
        assert agent.run_sync.call_count == 1

    def test_sql_hash_cache_skips_reexecution(self, conn):
        sql = "SELECT COUNT(*) FROM sales"
        run_result = MagicMock()
        run_result.output = sql
        agent = MagicMock()
        agent.run_sync.return_value = run_result

        mgr = ConversationManager(conn, agent, "schema")
        mgr.run_query("Question A?")
        cache_after_first = dict(mgr.query_cache)
        mgr.run_query("Question B?")  # different NL, same SQL -> cache hit

        assert mgr.query_cache == cache_after_first

    def test_bad_sql_returns_none(self, conn):
        mgr = _manager(conn, "SELECT * FROM nonexistent_table_xyz")
        assert mgr.run_query("Anything?") is None

    def test_history_context_limited_to_n_turns(self, conn):
        mgr = _manager(conn, "SELECT 1")
        for i in range(5):
            mgr.run_query(f"Query {i}?")
        context = mgr._history_context(n=3)
        assert "Query 4?" in context
        assert "Query 1?" not in context

    def test_lru_cache_bounded(self, conn):
        mgr = ConversationManager(conn, MagicMock(), "schema", cache_size=3)
        for i in range(5):
            h = hashlib.md5(f"SELECT {i}".encode()).hexdigest()
            mgr.query_cache[h] = [(i,)]
            if len(mgr.query_cache) > mgr.cache_size:
                mgr.query_cache.popitem(last=False)
        assert len(mgr.query_cache) <= 3

    def test_show_history_does_not_raise(self, conn):
        mgr = _manager(conn, "SELECT 1")
        mgr.run_query("A?")
        mgr.show_history()

    def test_prompt_includes_schema(self, conn):
        agent = _mock_agent("SELECT 1")
        schema = "Table: sales\n  - product (VARCHAR)"
        mgr = ConversationManager(conn, agent, schema)
        mgr.run_query("A?")
        prompt = agent.run_sync.call_args[0][0]
        assert "Table: sales" in prompt

    def test_prompt_includes_nl_query(self, conn):
        agent = _mock_agent("SELECT 1")
        mgr = ConversationManager(conn, agent, "schema")
        mgr.run_query("How many laptops were sold?")
        prompt = agent.run_sync.call_args[0][0]
        assert "How many laptops were sold?" in prompt

    def test_prompt_includes_history_on_second_turn(self, conn):
        agent = _mock_agent("SELECT 1")
        mgr = ConversationManager(conn, agent, "schema")
        mgr.run_query("First question?")
        mgr.run_query("Second question?")
        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "First question?" in second_prompt


# ---------------------------------------------------------------------------
# UnifiedDataSource.generate_schema (no GCS or Postgres connection needed)
# ---------------------------------------------------------------------------
class TestUnifiedDataSourceSchema:
    def _source_with_views(
        self,
        conn: duckdb.DuckDBPyConnection,
        view_names: list[str],
    ) -> UnifiedDataSource:
        for name in view_names:
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT 'a' AS col1, 1 AS col2"
            )
        src = UnifiedDataSource(
            conn=conn,
            gcs_bucket="test-bucket",
            gcs_prefix="data/",
            gcs_user_project="test-project",
        )
        src._gcs_views = {n: n for n in view_names}
        return src

    def test_schema_contains_view_names(self, conn):
        src = self._source_with_views(conn, ["orders", "users"])
        schema = src.generate_schema()
        assert "orders" in schema
        assert "users" in schema

    def test_schema_contains_column_names(self, conn):
        src = self._source_with_views(conn, ["orders"])
        schema = src.generate_schema()
        assert "col1" in schema
        assert "col2" in schema

    def test_schema_has_gcs_section_header(self, conn):
        src = self._source_with_views(conn, ["orders"])
        assert "GCS Parquet" in src.generate_schema()

    def test_schema_stored_on_instance(self, conn):
        src = self._source_with_views(conn, ["orders"])
        schema = src.generate_schema()
        assert src.schema_text == schema

    def test_conversation_manager_raises_without_agent_or_schema(self, conn):
        src = UnifiedDataSource(
            conn=conn, gcs_bucket="b", gcs_prefix="p/", gcs_user_project="proj"
        )
        with pytest.raises(ValueError):
            src.conversation_manager()

    def test_conversation_manager_raises_without_schema(self, conn):
        src = UnifiedDataSource(
            conn=conn, gcs_bucket="b", gcs_prefix="p/", gcs_user_project="proj"
        )
        src.agent = MagicMock()
        with pytest.raises(ValueError):
            src.conversation_manager()


# ---------------------------------------------------------------------------
# PostgresDB dataclass
# ---------------------------------------------------------------------------
class TestPostgresDB:
    def test_fields_stored(self):
        db = PostgresDB(alias="rag", connection_string="postgresql://u:p@h:5434/db")
        assert db.alias == "rag"
        assert "5434" in db.connection_string

    def test_two_dbs_independent(self):
        db1 = PostgresDB(alias="rag", connection_string="postgresql://u:p@h:5434/db1")
        db2 = PostgresDB(alias="local_pg", connection_string="postgresql://u:p@h:5432/db2")
        assert db1.alias != db2.alias


# ---------------------------------------------------------------------------
# End-to-end: ConversationManager over a real DuckDB table
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_query_pipeline(self, conn):
        sql = "SELECT product, SUM(revenue) AS total FROM sales GROUP BY product ORDER BY total DESC"
        mgr = _manager(conn, sql)
        result = mgr.run_query("Total revenue per product?")
        assert result is not None
        products = [row[0] for row in result]
        assert "Laptop" in products
        assert "Monitor" in products

    def test_aggregate_correct_value(self, conn):
        mgr = _manager(conn, "SELECT SUM(quantity) FROM sales")
        assert mgr.run_query("Total quantity?") == [(11,)]

    def test_filter_query(self, conn):
        mgr = _manager(conn, "SELECT product FROM sales WHERE revenue > 2000")
        result = mgr.run_query("High revenue products?")
        assert result == [("Laptop",)]

    def test_follow_up_includes_first_turn_in_prompt(self, conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT SUM(revenue) FROM sales"
        r2.output = "SELECT SUM(revenue) FROM sales WHERE product = 'Laptop'"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema")
        mgr.run_query("Total revenue?")
        mgr.run_query("Only for Laptop?")

        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Total revenue?" in second_prompt


# ---------------------------------------------------------------------------
# Complex JOIN queries across multiple simulated data sources
#
# DuckDB treats each table as a separate "source" here:
#   sales      -> GCS parquet view
#   customers  -> local_pg equivalent
#   regions    -> rag_db equivalent
#
# The tests verify that the ConversationManager executes multi-table SQL
# correctly regardless of which source generated the JOIN.
# ---------------------------------------------------------------------------
@pytest.fixture
def multi_conn():
    """DuckDB connection with three tables simulating cross-source data."""
    c = duckdb.connect(":memory:")

    # GCS parquet equivalent: sales fact table
    c.execute("""
        CREATE TABLE sales (
            order_id   INTEGER,
            customer_id INTEGER,
            product    VARCHAR,
            region_id  INTEGER,
            quantity   INTEGER,
            revenue    DOUBLE
        )
    """)
    c.execute("""
        INSERT INTO sales VALUES
            (1, 10, 'Laptop',  1, 2, 2000.0),
            (2, 11, 'Monitor', 2, 3,  900.0),
            (3, 10, 'Laptop',  2, 1, 1000.0),
            (4, 12, 'Tablet',  1, 5, 1250.0),
            (5, 11, 'Tablet',  3, 2,  500.0)
    """)

    # local_pg equivalent: customers dimension
    c.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name        VARCHAR,
            country     VARCHAR,
            tier        VARCHAR
        )
    """)
    c.execute("""
        INSERT INTO customers VALUES
            (10, 'Alice',   'US',  'Gold'),
            (11, 'Bob',     'UK',  'Silver'),
            (12, 'Charlie', 'US',  'Bronze')
    """)

    # rag_db equivalent: regions dimension
    c.execute("""
        CREATE TABLE regions (
            region_id   INTEGER PRIMARY KEY,
            region_name VARCHAR,
            gdp_index   DOUBLE
        )
    """)
    c.execute("""
        INSERT INTO regions VALUES
            (1, 'North America', 1.0),
            (2, 'Europe',        0.9),
            (3, 'Asia',          0.7)
    """)

    yield c
    c.close()


MULTI_SCHEMA = """
=== GCS Parquet tables (use bare name) ===
Table: sales
  - order_id (INTEGER)
  - customer_id (INTEGER)
  - product (VARCHAR)
  - region_id (INTEGER)
  - quantity (INTEGER)
  - revenue (DOUBLE)

=== local_pg tables (prefix: local_pg.main.<table>) ===
Table: local_pg.main.customers
  - customer_id (INTEGER)
  - name (VARCHAR)
  - country (VARCHAR)
  - tier (VARCHAR)

=== rag tables (prefix: rag.main.<table>) ===
Table: rag.main.regions
  - region_id (INTEGER)
  - region_name (VARCHAR)
  - gdp_index (DOUBLE)
"""


class TestJoinQueries:
    """
    Each test represents an NL question the LLM would translate to a JOIN.
    The 'agent' mock returns the SQL; DuckDB executes it for real.
    Table names use bare names (simulating all tables in the same DuckDB session).
    """

    def test_inner_join_sales_customers(self, multi_conn):
        sql = """
            SELECT c.name, SUM(s.revenue) AS total_revenue
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            GROUP BY c.name
            ORDER BY total_revenue DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("What is the total revenue per customer name?")
        assert result is not None
        names = [row[0] for row in result]
        assert "Alice" in names    # orders 1 + 3 = 3000
        assert "Bob" in names      # orders 2 + 5 = 1400
        assert "Charlie" in names  # order 4 = 1250

    def test_three_way_join(self, multi_conn):
        sql = """
            SELECT c.name, r.region_name, SUM(s.revenue) AS total
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            JOIN regions r   ON s.region_id   = r.region_id
            GROUP BY c.name, r.region_name
            ORDER BY total DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Show revenue by customer and region?")
        assert result is not None
        assert len(result) > 0
        # Alice in North America: order 1 = 2000
        assert any(row[0] == "Alice" and row[1] == "North America" for row in result)

    def test_join_with_filter(self, multi_conn):
        sql = """
            SELECT c.name, s.product, s.revenue
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            WHERE c.country = 'US'
            ORDER BY s.revenue DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Show all sales for US customers?")
        assert result is not None
        # US customers: Alice (10) and Charlie (12)
        names = {row[0] for row in result}
        assert names == {"Alice", "Charlie"}

    def test_join_with_aggregation_and_having(self, multi_conn):
        sql = """
            SELECT c.tier, COUNT(DISTINCT s.order_id) AS num_orders, SUM(s.revenue) AS total
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            GROUP BY c.tier
            HAVING SUM(s.revenue) > 1000
            ORDER BY total DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Which customer tiers have total revenue above 1000?")
        assert result is not None
        tiers = [row[0] for row in result]
        assert "Gold" in tiers     # Alice: 3000
        assert "Silver" in tiers   # Bob: 1400
        # Bronze (Charlie): 1250 > 1000, so also present
        assert "Bronze" in tiers

    def test_join_with_subquery(self, multi_conn):
        sql = """
            SELECT c.name, top_sales.product, top_sales.revenue
            FROM (
                SELECT customer_id, product, revenue
                FROM sales
                WHERE revenue = (SELECT MAX(revenue) FROM sales)
            ) top_sales
            JOIN customers c ON top_sales.customer_id = c.customer_id
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Who made the highest single sale and what was the product?")
        assert result is not None
        assert len(result) == 1
        assert result[0][0] == "Alice"
        assert result[0][1] == "Laptop"
        assert result[0][2] == 2000.0

    def test_join_with_window_function(self, multi_conn):
        sql = """
            SELECT
                c.name,
                s.product,
                s.revenue,
                RANK() OVER (PARTITION BY s.product ORDER BY s.revenue DESC) AS rank
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Rank customers by revenue within each product?")
        assert result is not None
        # Laptop has 2 rows (Alice 2000 rank 1, Alice 1000 rank 2)
        laptop_rows = [r for r in result if r[1] == "Laptop"]
        assert len(laptop_rows) == 2
        assert laptop_rows[0][3] == 1  # highest revenue first

    def test_left_join_shows_all_customers(self, multi_conn):
        sql = """
            SELECT c.name, COALESCE(SUM(s.revenue), 0) AS total_revenue
            FROM customers c
            LEFT JOIN sales s ON c.customer_id = s.customer_id
            GROUP BY c.name
            ORDER BY c.name
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Show total revenue for all customers, including those with no sales?")
        assert result is not None
        # All 3 customers returned (LEFT JOIN keeps all)
        assert len(result) == 3

    def test_join_with_region_gdp_weighting(self, multi_conn):
        sql = """
            SELECT r.region_name, SUM(s.revenue * r.gdp_index) AS gdp_weighted_revenue
            FROM sales s
            JOIN regions r ON s.region_id = r.region_id
            GROUP BY r.region_name
            ORDER BY gdp_weighted_revenue DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("What is the GDP-weighted revenue by region?")
        assert result is not None
        region_names = [row[0] for row in result]
        assert "North America" in region_names
        assert "Europe" in region_names

    def test_cte_with_join(self, multi_conn):
        sql = """
            WITH customer_totals AS (
                SELECT customer_id, SUM(revenue) AS total
                FROM sales
                GROUP BY customer_id
            )
            SELECT c.name, c.tier, ct.total
            FROM customer_totals ct
            JOIN customers c ON ct.customer_id = c.customer_id
            WHERE ct.total > 1500
            ORDER BY ct.total DESC
        """
        mgr = _manager(multi_conn, sql, MULTI_SCHEMA)
        result = mgr.run_query("Which customers have spent more than 1500 in total?")
        assert result is not None
        names = [row[0] for row in result]
        assert "Alice" in names   # 3000 > 1500
        assert "Bob" not in names  # 1400 < 1500

    def test_multi_turn_join_refinement(self, multi_conn):
        """Second question narrows the first JOIN query."""
        r1 = MagicMock()
        r1.output = "SELECT c.name, SUM(s.revenue) AS total FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.name ORDER BY total DESC"
        r2 = MagicMock()
        r2.output = "SELECT c.name, SUM(s.revenue) AS total FROM sales s JOIN customers c ON s.customer_id = c.customer_id WHERE c.country = 'US' GROUP BY c.name ORDER BY total DESC"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(multi_conn, agent, MULTI_SCHEMA)
        result1 = mgr.run_query("Revenue per customer?")
        result2 = mgr.run_query("Only US customers?")

        assert result1 is not None and len(result1) == 3   # all 3 customers
        assert result2 is not None and len(result2) == 2   # Alice + Charlie only

        # History from turn 1 should appear in turn 2 prompt
        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Revenue per customer?" in second_prompt
