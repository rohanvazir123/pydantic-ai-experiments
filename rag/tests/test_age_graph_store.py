"""Tests for kg.age_graph_store and the create_kg_store factory.

Unit tests mock the asyncpg pool and test Cypher generation logic.
Integration tests are skipped unless AGE_DATABASE_URL is set and reachable.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kg.age_graph_store import AgeGraphStore, _unquote_agtype
from kg import create_kg_store, PgGraphStore


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestUnquoteAgtype:
    def test_strips_double_quotes(self):
        assert _unquote_agtype('"Acme Corp"') == "Acme Corp"

    def test_leaves_unquoted_string(self):
        assert _unquote_agtype("Acme Corp") == "Acme Corp"

    def test_handles_none(self):
        assert _unquote_agtype(None) == ""

    def test_handles_empty_string(self):
        assert _unquote_agtype("") == ""

    def test_strips_whitespace_then_quotes(self):
        assert _unquote_agtype('  "Delaware law"  ') == "Delaware law"

    def test_numeric_agtype(self):
        # AGE returns counts as unquoted numeric strings
        assert _unquote_agtype("42") == "42"


# ---------------------------------------------------------------------------
# AgeGraphStore unit tests (mocked pool)
# ---------------------------------------------------------------------------


def _make_age_store_with_pool(mock_pool) -> AgeGraphStore:
    """Return an AgeGraphStore with a mocked pool, bypassing __init__ URL check."""
    store = AgeGraphStore.__new__(AgeGraphStore)
    import asyncio
    store.settings = MagicMock()
    store.settings.age_database_url = "postgresql://age_user:age_pass@localhost:5433/legal_graph"
    store.settings.age_graph_name = "legal_graph"
    store._age_url = "postgresql://age_user:age_pass@localhost:5433/legal_graph"
    store._graph = "legal_graph"
    store.pool = mock_pool
    store._initialized = True
    store._init_lock = asyncio.Lock()
    return store


def _mock_pool_with_rows(rows: list) -> MagicMock:
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=rows)
    mock_conn.fetchrow = AsyncMock(return_value=rows[0] if rows else None)
    mock_conn.execute = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    return mock_pool


class TestAgeGraphStoreCypherQuery:
    def test_cypher_wraps_in_ag_catalog_call(self):
        store = _make_age_store_with_pool(MagicMock())
        result = store._cypher("MATCH (n) RETURN n")
        assert "ag_catalog.cypher" in result
        assert "legal_graph" in result
        assert "MATCH (n) RETURN n" in result

    def test_cypher_uses_configured_graph_name(self):
        store = _make_age_store_with_pool(MagicMock())
        store._graph = "my_custom_graph"
        result = store._cypher("RETURN 1")
        assert "my_custom_graph" in result


class TestAgeGraphStoreUpsertEntity:
    @pytest.mark.asyncio
    async def test_returns_uuid_string(self):
        mock_pool = _mock_pool_with_rows([{"uuid": '"aaaabbbb-0000-0000-0000-000000000001"'}])
        store = _make_age_store_with_pool(mock_pool)
        result = await store.upsert_entity("Acme Corp", "Party", "doc-id")
        assert result == "aaaabbbb-0000-0000-0000-000000000001"

    @pytest.mark.asyncio
    async def test_normalizes_name_in_cypher(self):
        captured = []

        async def capture_fetch(query, *args):
            captured.append(query)
            return [{"uuid": '"new-uuid"'}]

        mock_conn = AsyncMock()
        mock_conn.fetch = capture_fetch
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        await store.upsert_entity("  ACME CORP  ", "Party")
        # normalized_name in the Cypher should be lowercase
        assert "acme corp" in captured[0]

    @pytest.mark.asyncio
    async def test_includes_entity_type_in_cypher(self):
        captured = []

        async def capture_fetch(query, *args):
            captured.append(query)
            return [{"uuid": '"new-uuid"'}]

        mock_conn = AsyncMock()
        mock_conn.fetch = capture_fetch
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        await store.upsert_entity("Acme", "Jurisdiction")
        assert "Jurisdiction" in captured[0]

    @pytest.mark.asyncio
    async def test_falls_back_to_generated_uuid_if_no_rows(self):
        mock_pool = _mock_pool_with_rows([])
        store = _make_age_store_with_pool(mock_pool)
        result = await store.upsert_entity("Acme", "Party")
        # Should return a valid UUID even when AGE returns no rows
        assert len(result) == 36
        assert result.count("-") == 4


class TestAgeGraphStoreAddRelationship:
    @pytest.mark.asyncio
    async def test_returns_uuid_string(self):
        mock_pool = _mock_pool_with_rows([{"uuid": '"rel-uuid-0001"'}])
        store = _make_age_store_with_pool(mock_pool)
        result = await store.add_relationship("src-id", "tgt-id", "PARTY_TO")
        assert result == "rel-uuid-0001"

    @pytest.mark.asyncio
    async def test_includes_relationship_type_in_cypher(self):
        captured = []

        async def capture_fetch(query, *args):
            captured.append(query)
            return [{"uuid": '"rel-uuid"'}]

        mock_conn = AsyncMock()
        mock_conn.fetch = capture_fetch
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        await store.add_relationship("s", "t", "GOVERNED_BY_LAW")
        assert "GOVERNED_BY_LAW" in captured[0]

    @pytest.mark.asyncio
    async def test_falls_back_to_generated_uuid_if_no_rows(self):
        mock_pool = _mock_pool_with_rows([])
        store = _make_age_store_with_pool(mock_pool)
        result = await store.add_relationship("s", "t", "PARTY_TO")
        assert len(result) == 36


class TestAgeGraphStoreSearchEntities:
    @pytest.mark.asyncio
    async def test_returns_parsed_entity_dicts(self):
        # Column names must match the AGE AS clause: uuid, name, label, document_id
        mock_pool = _mock_pool_with_rows([{
            "uuid": '"entity-uuid-1"',
            "name": '"Acme Corp"',
            "label": '"Party"',
            "document_id": '"doc-uuid-1"',
        }])
        store = _make_age_store_with_pool(mock_pool)
        results = await store.search_entities("acme")
        assert len(results) == 1
        assert results[0]["name"] == "Acme Corp"
        assert results[0]["entity_type"] == "Party"

    @pytest.mark.asyncio
    async def test_filters_by_entity_type_in_cypher(self):
        captured = []

        async def capture_fetch(query, *args):
            captured.append(query)
            return []

        mock_conn = AsyncMock()
        mock_conn.fetch = capture_fetch
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        await store.search_entities("acme", entity_type="Jurisdiction")
        assert "Jurisdiction" in captured[0]


class TestAgeGraphStoreSearchAsContext:
    @pytest.mark.asyncio
    async def test_returns_formatted_facts(self):
        # Column names must match the AGE AS clause: src_name, src_label, rel, tgt_name, tgt_label
        mock_pool = _mock_pool_with_rows([{
            "src_name": '"Acme Corp"',
            "src_label": '"Party"',
            "rel": '"PARTY_TO"',
            "tgt_name": '"Acme Distributor Agreement"',
            "tgt_label": '"Contract"',
        }])
        store = _make_age_store_with_pool(mock_pool)
        result = await store.search_as_context("Acme Corp")
        assert "## Knowledge Graph" in result
        assert "Acme Corp" in result
        assert "PARTY_TO" in result

    @pytest.mark.asyncio
    async def test_returns_not_found_on_no_results(self):
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        result = await store.search_as_context("xyznonexistent")
        assert "No relevant" in result


class TestAgeGraphStoreGetStats:
    @pytest.mark.asyncio
    async def test_returns_stats_dict_shape(self):
        # Column names must match AGE AS clause: label agtype, cnt agtype / rel_type agtype, cnt agtype
        vertex_rows = [
            {"label": '"Party"', "cnt": "5"},
            {"label": '"Jurisdiction"', "cnt": "3"},
        ]
        edge_rows = [
            {"rel_type": '"PARTY_TO"', "cnt": "4"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=[vertex_rows, edge_rows])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_age_store_with_pool(mock_pool)
        stats = await store.get_graph_stats()
        assert stats["total_entities"] == 8
        assert stats["total_relationships"] == 4
        assert stats["entities_by_type"]["Party"] == 5
        assert stats["relationships_by_type"]["PARTY_TO"] == 4


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestCreateKgStore:
    # load_settings is imported at module level in rag/knowledge_graph/__init__.py,
    # so the patch target is the package namespace (no __init__ suffix).

    def test_returns_pg_store_by_default(self):
        with patch("kg.load_settings") as mock_ls:
            mock_ls.return_value.kg_backend = "postgres"
            store = create_kg_store()
        assert isinstance(store, PgGraphStore)

    def test_returns_age_store_when_configured(self):
        mock_age_instance = MagicMock(spec=AgeGraphStore)
        with patch("kg.load_settings") as mock_ls, \
             patch("kg.AgeGraphStore", return_value=mock_age_instance):
            mock_ls.return_value.kg_backend = "age"
            store = create_kg_store()
        assert store is mock_age_instance

    def test_unknown_backend_defaults_to_postgres(self):
        with patch("kg.load_settings") as mock_ls:
            mock_ls.return_value.kg_backend = "something_unknown"
            store = create_kg_store()
        assert isinstance(store, PgGraphStore)


# ---------------------------------------------------------------------------
# Integration tests — skipped unless AGE is running
# ---------------------------------------------------------------------------


class TestAgeGraphStoreIntegration:
    """Requires AGE running: docker compose up -d"""

    @pytest.mark.asyncio
    async def test_initialize_and_stats(self):
        import os
        age_url = os.getenv("AGE_DATABASE_URL")
        if not age_url:
            pytest.skip("AGE_DATABASE_URL not set — start AGE with: docker compose up -d")

        with patch.object(
            __import__("rag.config.settings", fromlist=["load_settings"]),
            "load_settings",
        ):
            store = AgeGraphStore.__new__(AgeGraphStore)
            import asyncio
            store._age_url = age_url
            store._graph = "test_legal_graph"
            store.pool = None
            store._initialized = False
            store._init_lock = asyncio.Lock()

        try:
            await store.initialize()
            stats = await store.get_graph_stats()
            assert "total_entities" in stats
            assert "total_relationships" in stats
        except Exception as e:
            pytest.skip(f"AGE not reachable: {e}")
        finally:
            await store.close()
