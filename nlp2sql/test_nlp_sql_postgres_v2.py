"""
Tests for nlp_sql_postgres_v2.py

Strategy:
- DuckDB is used for real (in-memory) -- no mocking needed for execution
- GCS and PostgreSQL connections are mocked so tests run offline
- Pydantic AI agent is mocked to return controlled SQL strings
- max_retries=1 in most fixtures to keep mock setup simple;
  retry-specific tests use their own agents with side_effect lists
"""

import hashlib
from unittest.mock import MagicMock

import duckdb
import pytest

from nlp_sql_postgres_v2 import (
    ConversationManager,
    PostgresDB,
    QueryResult,
    UnifiedDataSource,
    _apply_row_cap,
    _check_readonly,
    strip_sql_fences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn():
    """In-memory DuckDB connection with a sample sales table."""
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
    run_result = MagicMock()
    run_result.output = sql
    agent = MagicMock()
    agent.run_sync.return_value = run_result
    return agent


def _manager(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    schema: str = "Table: sales\n  - product (VARCHAR)",
    max_retries: int = 1,
) -> ConversationManager:
    """max_retries=1 keeps most tests to a single agent call."""
    return ConversationManager(conn, _mock_agent(sql), schema, max_retries=max_retries)


# ---------------------------------------------------------------------------
# strip_sql_fences (unchanged from v1)
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
# QueryResult dataclass
# ---------------------------------------------------------------------------
class TestQueryResult:
    def test_success_when_no_error(self):
        qr = QueryResult(nl_query="Q?", sql="SELECT 1", columns=["n"], rows=[(1,)])
        assert qr.success

    def test_not_success_when_error(self):
        qr = QueryResult(nl_query="Q?", sql="BAD", columns=[], rows=[], error="no such table")
        assert not qr.success

    def test_default_not_cached(self):
        qr = QueryResult(nl_query="Q?", sql="SELECT 1", columns=["n"], rows=[(1,)])
        assert not qr.cached

    def test_default_attempts_one(self):
        qr = QueryResult(nl_query="Q?", sql="SELECT 1", columns=["n"], rows=[(1,)])
        assert qr.attempts == 1

    def test_pretty_print_shows_header_and_row(self, capsys):
        qr = QueryResult(nl_query="Q?", sql="SELECT 1 AS n", columns=["n"], rows=[(42,)])
        qr.pretty_print()
        out = capsys.readouterr().out
        assert "n" in out
        assert "42" in out

    def test_pretty_print_shows_error(self, capsys):
        qr = QueryResult(nl_query="Q?", sql="BAD", columns=[], rows=[], error="no such table")
        qr.pretty_print()
        out = capsys.readouterr().out
        assert "no such table" in out

    def test_pretty_print_no_rows(self, capsys):
        qr = QueryResult(nl_query="Q?", sql="SELECT 1 WHERE FALSE", columns=["n"], rows=[])
        qr.pretty_print()
        assert "no rows" in capsys.readouterr().out

    def test_pretty_print_truncates_at_max_rows(self, capsys):
        rows = [(i,) for i in range(25)]
        qr = QueryResult(nl_query="Q?", sql="SELECT x", columns=["x"], rows=rows)
        qr.pretty_print(max_rows=10)
        out = capsys.readouterr().out
        assert "15 more rows" in out

    def test_pretty_print_multi_column(self, capsys):
        qr = QueryResult(
            nl_query="Q?",
            sql="SELECT a, b",
            columns=["alpha", "beta"],
            rows=[("foo", 99)],
        )
        qr.pretty_print()
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        assert "foo" in out
        assert "99" in out


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------
class TestConversationManager:
    def test_run_query_returns_query_result(self, conn):
        mgr = _manager(conn, "SELECT COUNT(*) FROM sales")
        result = mgr.run_query("How many rows?")
        assert isinstance(result, QueryResult)

    def test_successful_result_has_rows(self, conn):
        mgr = _manager(conn, "SELECT product, SUM(quantity) FROM sales GROUP BY product")
        result = mgr.run_query("Quantity per product?")
        assert result.success
        assert len(result.rows) == 2

    def test_result_has_column_names(self, conn):
        mgr = _manager(conn, "SELECT product, SUM(quantity) AS total FROM sales GROUP BY product")
        result = mgr.run_query("Quantity per product?")
        assert result.columns == ["product", "total"]

    def test_result_added_to_history(self, conn):
        mgr = _manager(conn, "SELECT COUNT(*) FROM sales")
        mgr.run_query("How many rows?")
        assert len(mgr.history) == 1
        assert mgr.history[0][0] == "How many rows?"

    def test_nl_cache_hit_skips_agent(self, conn):
        agent = _mock_agent("SELECT COUNT(*) FROM sales")
        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("How many rows?")
        mgr.run_query("How many rows?")  # exact match -> cache hit
        assert agent.run_sync.call_count == 1

    def test_nl_cache_normalized_case(self, conn):
        """Lowercase variant should hit the same cache entry."""
        agent = _mock_agent("SELECT COUNT(*) FROM sales")
        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("How many rows?")
        mgr.run_query("how many rows?")
        assert agent.run_sync.call_count == 1

    def test_nl_cache_normalized_whitespace(self, conn):
        """Extra internal whitespace should still hit the cache."""
        agent = _mock_agent("SELECT COUNT(*) FROM sales")
        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("How many rows?")
        mgr.run_query("How  many  rows?")
        assert agent.run_sync.call_count == 1

    def test_sql_hash_cache_skips_reexecution(self, conn):
        sql = "SELECT COUNT(*) FROM sales"
        r1, r2 = MagicMock(), MagicMock()
        r1.output = sql
        r2.output = sql
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("Question A?")
        cache_size_after_first = len(mgr._sql_cache)
        mgr.run_query("Question B?")  # different NL, same SQL -> SQL cache hit

        assert len(mgr._sql_cache) == cache_size_after_first

    def test_bad_sql_returns_error_result(self, conn):
        mgr = _manager(conn, "SELECT * FROM nonexistent_table_xyz", max_retries=1)
        result = mgr.run_query("Anything?")
        assert not result.success
        assert result.error is not None

    def test_failed_query_still_in_history(self, conn):
        mgr = _manager(conn, "SELECT * FROM nonexistent_table_xyz", max_retries=1)
        mgr.run_query("Bad query?")
        assert len(mgr.history) == 1
        assert not mgr.history[0][2].success

    def test_history_context_excludes_failed_turns(self, conn):
        """Failed SQL should not pollute the history shown to the model."""
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT * FROM bad_table"
        r2.output = "SELECT 1"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("Bad question?")
        mgr.run_query("Good question?")

        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Bad question?" not in second_prompt

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
            qr = QueryResult(f"Q{i}", f"SELECT {i}", ["n"], [(i,)])
            mgr._cache_put(mgr._sql_cache, h, qr)
        assert len(mgr._sql_cache) <= 3

    def test_show_history_does_not_raise(self, conn):
        mgr = _manager(conn, "SELECT 1")
        mgr.run_query("A?")
        mgr.show_history()

    def test_prompt_includes_schema(self, conn):
        agent = _mock_agent("SELECT 1")
        schema = "Table: sales\n  - product (VARCHAR)"
        mgr = ConversationManager(conn, agent, schema, max_retries=1)
        mgr.run_query("A?")
        prompt = agent.run_sync.call_args[0][0]
        assert "Table: sales" in prompt

    def test_prompt_includes_nl_query(self, conn):
        agent = _mock_agent("SELECT 1")
        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("How many laptops were sold?")
        prompt = agent.run_sync.call_args[0][0]
        assert "How many laptops were sold?" in prompt

    def test_prompt_includes_history_on_second_turn(self, conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT 1"
        r2.output = "SELECT 2"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("First question?")
        mgr.run_query("Second question?")
        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "First question?" in second_prompt


# ---------------------------------------------------------------------------
# Self-correcting retry behavior
# ---------------------------------------------------------------------------
class TestRetryBehavior:
    def test_succeeds_on_second_attempt(self, conn):
        """Agent returns bad SQL first, correct SQL second -- should succeed."""
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT * FROM nonexistent_xyz"
        r2.output = "SELECT COUNT(*) FROM sales"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=3)
        result = mgr.run_query("How many sales?")

        assert result.success
        assert result.attempts == 2
        assert agent.run_sync.call_count == 2

    def test_correction_prompt_contains_error(self, conn):
        """The second prompt must include the SQL error from attempt 1."""
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT * FROM nonexistent_xyz"
        r2.output = "SELECT 1"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=3)
        mgr.run_query("Test query?")

        correction_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Error" in correction_prompt or "error" in correction_prompt.lower()

    def test_correction_prompt_contains_bad_sql(self, conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT * FROM bad_table"
        r2.output = "SELECT 1"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=3)
        mgr.run_query("Test query?")

        correction_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "bad_table" in correction_prompt

    def test_all_retries_exhausted_returns_error_result(self, conn):
        agent = _mock_agent("SELECT * FROM nonexistent_xyz")
        mgr = ConversationManager(conn, agent, "schema", max_retries=2)
        result = mgr.run_query("Impossible query?")

        assert not result.success
        assert result.error is not None
        assert result.attempts == 2
        assert agent.run_sync.call_count == 2

    def test_failed_result_recorded_in_history(self, conn):
        mgr = ConversationManager(
            conn, _mock_agent("SELECT * FROM bad_table"), "schema", max_retries=1
        )
        mgr.run_query("Bad query?")
        assert len(mgr.history) == 1
        assert not mgr.history[0][2].success

    def test_successful_retry_not_counted_as_cached(self, conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT * FROM bad"
        r2.output = "SELECT COUNT(*) FROM sales"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=3)
        result = mgr.run_query("Count sales?")
        assert not result.cached


# ---------------------------------------------------------------------------
# UnifiedDataSource.generate_schema (no live GCS or Postgres needed)
# ---------------------------------------------------------------------------
class TestUnifiedDataSourceSchema:
    def _source_with_views(self, conn, view_names):
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
# PostgresDB
# ---------------------------------------------------------------------------
class TestPostgresDB:
    def test_fields_stored(self):
        db = PostgresDB(alias="rag", connection_string="postgresql://u:p@h:5434/db")
        assert db.alias == "rag"
        assert "5434" in db.connection_string

    def test_two_dbs_independent(self):
        db1 = PostgresDB(alias="rag",      connection_string="postgresql://u:p@h:5434/db1")
        db2 = PostgresDB(alias="local_pg", connection_string="postgresql://u:p@h:5432/db2")
        assert db1.alias != db2.alias


# ---------------------------------------------------------------------------
# End-to-end: real DuckDB execution through ConversationManager
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_query_pipeline(self, conn):
        sql = "SELECT product, SUM(revenue) AS total FROM sales GROUP BY product ORDER BY total DESC"
        result = _manager(conn, sql).run_query("Total revenue per product?")
        assert result.success
        products = [row[0] for row in result.rows]
        assert "Laptop" in products
        assert "Monitor" in products

    def test_result_columns_present(self, conn):
        mgr = _manager(conn, "SELECT product, SUM(revenue) AS total FROM sales GROUP BY product")
        result = mgr.run_query("Revenue per product?")
        assert "product" in result.columns
        assert "total" in result.columns

    def test_aggregate_correct_value(self, conn):
        result = _manager(conn, "SELECT SUM(quantity) AS s FROM sales").run_query("Total quantity?")
        assert result.rows == [(11,)]

    def test_filter_query(self, conn):
        result = _manager(conn, "SELECT product FROM sales WHERE revenue > 2000").run_query(
            "High revenue products?"
        )
        assert result.rows == [("Laptop",)]

    def test_follow_up_includes_first_turn_in_prompt(self, conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT SUM(revenue) AS s FROM sales"
        r2.output = "SELECT SUM(revenue) AS s FROM sales WHERE product = 'Laptop'"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(conn, agent, "schema", max_retries=1)
        mgr.run_query("Total revenue?")
        mgr.run_query("Only for Laptop?")

        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Total revenue?" in second_prompt


# ---------------------------------------------------------------------------
# JOIN queries across multiple simulated data sources
# ---------------------------------------------------------------------------
@pytest.fixture
def multi_conn():
    c = duckdb.connect(":memory:")

    c.execute("""
        CREATE TABLE sales (
            order_id    INTEGER,
            customer_id INTEGER,
            product     VARCHAR,
            region_id   INTEGER,
            quantity    INTEGER,
            revenue     DOUBLE
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
            (10, 'Alice',   'US', 'Gold'),
            (11, 'Bob',     'UK', 'Silver'),
            (12, 'Charlie', 'US', 'Bronze')
    """)

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
    def test_inner_join_sales_customers(self, multi_conn):
        sql = """
            SELECT c.name, SUM(s.revenue) AS total_revenue
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            GROUP BY c.name ORDER BY total_revenue DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Revenue per customer?")
        assert result.success
        names = [row[0] for row in result.rows]
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_result_columns_in_join(self, multi_conn):
        sql = "SELECT c.name, SUM(s.revenue) AS total FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.name"
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Revenue per customer?")
        assert result.columns == ["name", "total"]

    def test_three_way_join(self, multi_conn):
        sql = """
            SELECT c.name, r.region_name, SUM(s.revenue) AS total
            FROM sales s
            JOIN customers c ON s.customer_id = c.customer_id
            JOIN regions r   ON s.region_id   = r.region_id
            GROUP BY c.name, r.region_name ORDER BY total DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Revenue by customer and region?")
        assert result.success
        assert any(row[0] == "Alice" and row[1] == "North America" for row in result.rows)

    def test_join_with_filter(self, multi_conn):
        sql = """
            SELECT c.name, s.product, s.revenue
            FROM sales s JOIN customers c ON s.customer_id = c.customer_id
            WHERE c.country = 'US' ORDER BY s.revenue DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("US customer sales?")
        assert result.success
        names = {row[0] for row in result.rows}
        assert names == {"Alice", "Charlie"}

    def test_join_with_aggregation_and_having(self, multi_conn):
        sql = """
            SELECT c.tier, COUNT(DISTINCT s.order_id) AS num_orders, SUM(s.revenue) AS total
            FROM sales s JOIN customers c ON s.customer_id = c.customer_id
            GROUP BY c.tier HAVING SUM(s.revenue) > 1000 ORDER BY total DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Tiers with revenue > 1000?")
        assert result.success
        tiers = [row[0] for row in result.rows]
        assert "Gold" in tiers
        assert "Silver" in tiers
        assert "Bronze" in tiers

    def test_join_with_subquery(self, multi_conn):
        sql = """
            SELECT c.name, top_sales.product, top_sales.revenue
            FROM (
                SELECT customer_id, product, revenue FROM sales
                WHERE revenue = (SELECT MAX(revenue) FROM sales)
            ) top_sales
            JOIN customers c ON top_sales.customer_id = c.customer_id
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Highest single sale?")
        assert result.success
        assert len(result.rows) == 1
        assert result.rows[0][0] == "Alice"
        assert result.rows[0][2] == 2000.0

    def test_join_with_window_function(self, multi_conn):
        sql = """
            SELECT c.name, s.product, s.revenue,
                   RANK() OVER (PARTITION BY s.product ORDER BY s.revenue DESC) AS rank
            FROM sales s JOIN customers c ON s.customer_id = c.customer_id
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Rank by product revenue?")
        assert result.success
        laptop_rows = [r for r in result.rows if r[1] == "Laptop"]
        assert len(laptop_rows) == 2
        assert laptop_rows[0][3] == 1

    def test_left_join_shows_all_customers(self, multi_conn):
        sql = """
            SELECT c.name, COALESCE(SUM(s.revenue), 0) AS total_revenue
            FROM customers c LEFT JOIN sales s ON c.customer_id = s.customer_id
            GROUP BY c.name ORDER BY c.name
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("All customer revenues?")
        assert result.success
        assert len(result.rows) == 3

    def test_join_with_gdp_weighting(self, multi_conn):
        sql = """
            SELECT r.region_name, SUM(s.revenue * r.gdp_index) AS gdp_weighted
            FROM sales s JOIN regions r ON s.region_id = r.region_id
            GROUP BY r.region_name ORDER BY gdp_weighted DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("GDP-weighted revenue?")
        assert result.success
        region_names = [row[0] for row in result.rows]
        assert "North America" in region_names
        assert "Europe" in region_names

    def test_cte_with_join(self, multi_conn):
        sql = """
            WITH customer_totals AS (
                SELECT customer_id, SUM(revenue) AS total FROM sales GROUP BY customer_id
            )
            SELECT c.name, c.tier, ct.total
            FROM customer_totals ct JOIN customers c ON ct.customer_id = c.customer_id
            WHERE ct.total > 1500 ORDER BY ct.total DESC
        """
        result = _manager(multi_conn, sql, MULTI_SCHEMA).run_query("Top spenders?")
        assert result.success
        names = [row[0] for row in result.rows]
        assert "Alice" in names
        assert "Bob" not in names

    def test_multi_turn_join_refinement(self, multi_conn):
        r1, r2 = MagicMock(), MagicMock()
        r1.output = (
            "SELECT c.name, SUM(s.revenue) AS total FROM sales s "
            "JOIN customers c ON s.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY total DESC"
        )
        r2.output = (
            "SELECT c.name, SUM(s.revenue) AS total FROM sales s "
            "JOIN customers c ON s.customer_id = c.customer_id "
            "WHERE c.country = 'US' GROUP BY c.name ORDER BY total DESC"
        )
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(multi_conn, agent, MULTI_SCHEMA, max_retries=1)
        result1 = mgr.run_query("Revenue per customer?")
        result2 = mgr.run_query("Only US customers?")

        assert result1.success and len(result1.rows) == 3
        assert result2.success and len(result2.rows) == 2

        second_prompt = agent.run_sync.call_args_list[1][0][0]
        assert "Revenue per customer?" in second_prompt

    def test_retry_corrects_wrong_join_column(self, multi_conn):
        """Agent references a nonexistent column first, corrects on retry."""
        r1, r2 = MagicMock(), MagicMock()
        r1.output = "SELECT c.name FROM sales s JOIN customers c ON s.nonexistent_col = c.customer_id"
        r2.output = "SELECT c.name FROM sales s JOIN customers c ON s.customer_id = c.customer_id"
        agent = MagicMock()
        agent.run_sync.side_effect = [r1, r2]

        mgr = ConversationManager(multi_conn, agent, MULTI_SCHEMA, max_retries=3)
        result = mgr.run_query("Customer names in sales?")

        assert result.success
        assert result.attempts == 2


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class TestCheckReadonly:
    @pytest.mark.parametrize("keyword", ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"])
    def test_blocks_write_keywords(self, keyword):
        assert _check_readonly(f"{keyword} TABLE foo") is not None

    @pytest.mark.parametrize("keyword", ["drop", "delete", "insert"])
    def test_blocks_lowercase(self, keyword):
        assert _check_readonly(f"{keyword} into foo") is not None

    def test_allows_select(self):
        assert _check_readonly("SELECT * FROM sales") is None

    def test_allows_with_cte(self):
        assert _check_readonly("WITH cte AS (SELECT 1) SELECT * FROM cte") is None

    def test_error_message_contains_keyword(self):
        err = _check_readonly("DELETE FROM sales")
        assert "DELETE" in err


class TestApplyRowCap:
    def test_adds_limit_when_missing(self):
        result = _apply_row_cap("SELECT * FROM sales", 500)
        assert "LIMIT 500" in result

    def test_does_not_duplicate_existing_limit(self):
        sql = "SELECT * FROM sales LIMIT 10"
        result = _apply_row_cap(sql, 500)
        assert result.count("LIMIT") == 1
        assert "LIMIT 10" in result

    def test_limit_case_insensitive(self):
        sql = "SELECT * FROM sales limit 5"
        result = _apply_row_cap(sql, 500)
        assert result.count("imit") == 1  # only one LIMIT token

    def test_strips_trailing_semicolon_before_appending(self):
        result = _apply_row_cap("SELECT * FROM sales;", 100)
        assert result.endswith("LIMIT 100")
        assert ";" not in result

    def test_multiline_query(self):
        sql = "SELECT *\nFROM sales\nWHERE quantity > 1"
        result = _apply_row_cap(sql, 250)
        assert "LIMIT 250" in result


class TestReadonlyGuardrailIntegration:
    """SELECT-only guardrail blocks write SQL inside run_query."""

    def test_write_sql_returns_error_result(self, conn):
        mgr = _manager(conn, "DROP TABLE sales", max_retries=1)
        result = mgr.run_query("Delete everything?")
        assert not result.success
        assert result.error is not None

    def test_write_sql_error_mentions_keyword(self, conn):
        mgr = _manager(conn, "DELETE FROM sales", max_retries=1)
        result = mgr.run_query("Clear sales?")
        assert "DELETE" in result.error

    def test_write_sql_not_executed(self, conn):
        """Guardrail fires before DuckDB — table should still exist after."""
        mgr = _manager(conn, "DROP TABLE sales", max_retries=1)
        mgr.run_query("Drop the table?")
        # If DROP had executed, this SELECT would fail
        count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        assert count == 4


class TestRowCapIntegration:
    """Row cap is applied; query still succeeds."""

    def test_result_capped_when_no_limit(self, conn):
        # Insert many rows
        conn.execute("INSERT INTO sales SELECT 'X', i, 1, 1.0 FROM range(1, 20) t(i)")
        mgr = ConversationManager(
            conn, _mock_agent("SELECT * FROM sales"), "schema",
            max_retries=1, max_result_rows=5,
        )
        result = mgr.run_query("All rows?")
        assert result.success
        assert len(result.rows) <= 5

    def test_existing_limit_respected(self, conn):
        mgr = ConversationManager(
            conn, _mock_agent("SELECT * FROM sales LIMIT 2"), "schema",
            max_retries=1, max_result_rows=10_000,
        )
        result = mgr.run_query("Two rows?")
        assert result.success
        assert len(result.rows) == 2
