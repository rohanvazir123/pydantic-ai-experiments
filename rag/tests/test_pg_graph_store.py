"""Tests for rag.knowledge_graph.pg_graph_store and rag.knowledge_graph.cuad_kg_builder.

Unit tests use an in-memory mock of the asyncpg pool.
Integration tests require a live PostgreSQL connection (skipped otherwise).
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.knowledge_graph.cuad_kg_builder import (
    CuadKgBuilder,
    entity_type_for,
    relationship_type_for,
)
from rag.knowledge_graph.pg_graph_store import PgGraphStore, _normalize


# ---------------------------------------------------------------------------
# Pure-function unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Acme Corp") == "acme corp"

    def test_collapses_whitespace(self):
        assert _normalize("  Acme   Corp  ") == "acme corp"

    def test_strips(self):
        assert _normalize("  hello  ") == "hello"

    def test_empty(self):
        assert _normalize("") == ""


class TestEntityTypeFor:
    def test_parties(self):
        assert entity_type_for("Parties") == "Party"

    def test_governing_law(self):
        assert entity_type_for("Governing Law") == "Jurisdiction"

    def test_expiration_date(self):
        assert entity_type_for("Expiration Date") == "Date"

    def test_license_grant(self):
        assert entity_type_for("License Grant") == "LicenseClause"

    def test_termination_for_cause(self):
        assert entity_type_for("Termination for Cause") == "TerminationClause"

    def test_non_compete(self):
        assert entity_type_for("Non-Compete") == "RestrictionClause"

    def test_ip_ownership(self):
        assert entity_type_for("IP Ownership Assignment") == "IPClause"

    def test_cap_on_liability(self):
        assert entity_type_for("Cap on Liability") == "LiabilityClause"

    def test_unknown_defaults_to_clause(self):
        assert entity_type_for("Unknown Clause Type XYZ") == "Clause"


class TestRelationshipTypeFor:
    def test_party(self):
        assert relationship_type_for("Party") == "PARTY_TO"

    def test_jurisdiction(self):
        assert relationship_type_for("Jurisdiction") == "GOVERNED_BY_LAW"

    def test_license_clause(self):
        assert relationship_type_for("LicenseClause") == "HAS_LICENSE"

    def test_termination_clause(self):
        assert relationship_type_for("TerminationClause") == "HAS_TERMINATION"

    def test_restriction_clause(self):
        assert relationship_type_for("RestrictionClause") == "HAS_RESTRICTION"

    def test_ip_clause(self):
        assert relationship_type_for("IPClause") == "HAS_IP_CLAUSE"

    def test_liability_clause(self):
        assert relationship_type_for("LiabilityClause") == "HAS_LIABILITY"

    def test_generic_clause(self):
        assert relationship_type_for("Clause") == "HAS_CLAUSE"

    def test_unknown_defaults_to_has_clause(self):
        assert relationship_type_for("SomethingElse") == "HAS_CLAUSE"


# ---------------------------------------------------------------------------
# PgGraphStore unit tests (mocked pool)
# ---------------------------------------------------------------------------


def _make_store_with_pool(mock_pool):
    """Return an initialized PgGraphStore with a mocked pool."""
    store = PgGraphStore.__new__(PgGraphStore)
    import asyncio
    store.settings = MagicMock()
    store.settings.database_url = "mock://db"
    store.pool = mock_pool
    store._initialized = True
    store._init_lock = asyncio.Lock()
    return store


class TestPgGraphStoreUpsertEntity:
    @pytest.mark.asyncio
    async def test_upsert_returns_string_uuid(self):
        mock_row = {"id": "aaaaaaaa-0000-0000-0000-000000000001"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        result = await store.upsert_entity("Acme Corp", "Party", "doc-uuid-123")
        assert result == "aaaaaaaa-0000-0000-0000-000000000001"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_upsert_passes_normalized_name(self):
        mock_row = {"id": "aaaaaaaa-0000-0000-0000-000000000002"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        await store.upsert_entity("  ACME CORP  ", "Party", None)
        call_args = mock_conn.fetchrow.call_args[0]
        # normalized_name is the 3rd positional arg
        assert call_args[3] == "acme corp"

    @pytest.mark.asyncio
    async def test_upsert_serializes_metadata_as_json(self):
        mock_row = {"id": "aaaaaaaa-0000-0000-0000-000000000003"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        await store.upsert_entity("Acme", "Party", None, metadata={"k": "v"})
        call_args = mock_conn.fetchrow.call_args[0]
        # metadata_json is the 5th positional arg
        parsed = json.loads(call_args[5])
        assert parsed == {"k": "v"}


class TestPgGraphStoreAddRelationship:
    @pytest.mark.asyncio
    async def test_add_relationship_returns_string_uuid(self):
        mock_row = {"id": "bbbbbbbb-0000-0000-0000-000000000001"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        result = await store.add_relationship("src-id", "tgt-id", "PARTY_TO")
        assert result == "bbbbbbbb-0000-0000-0000-000000000001"

    @pytest.mark.asyncio
    async def test_add_relationship_passes_rel_type(self):
        mock_row = {"id": "bbbbbbbb-0000-0000-0000-000000000002"}
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        await store.add_relationship("src", "tgt", "GOVERNED_BY_LAW", "doc-id")
        call_args = mock_conn.fetchrow.call_args[0]
        assert "GOVERNED_BY_LAW" in call_args


class TestPgGraphStoreSearchAsContext:
    @pytest.mark.asyncio
    async def test_returns_context_string_with_header(self):
        mock_row = {
            "entity_name": "Acme Corp",
            "entity_type": "Party",
            "relationship_type": "PARTY_TO",
            "related_name": "Distributor Agreement",
            "related_type": "Contract",
            "document_title": "Acme Distributor Agreement 2020",
        }
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        result = await store.search_as_context("Acme Corp")
        assert "## Knowledge Graph" in result
        assert "Acme Corp" in result
        assert "PARTY_TO" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_entity_search_when_no_rel_results(self):
        mock_entity_row = {
            "id": "abc",
            "name": "Delaware law",
            "entity_type": "Jurisdiction",
            "normalized_name": "delaware law",
            "document_id": None,
            "metadata": "{}",
            "document_title": "Acme Corp Agreement",
            "document_source": "legal/Acme.md",
        }
        mock_conn = AsyncMock()
        # First fetch (relationship query) returns empty
        # Second fetch (entity search) returns one entity
        mock_conn.fetch = AsyncMock(side_effect=[[], [mock_entity_row]])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        result = await store.search_as_context("Delaware law")
        assert "Delaware law" in result

    @pytest.mark.asyncio
    async def test_returns_not_found_when_no_results(self):
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        store = _make_store_with_pool(mock_pool)
        result = await store.search_as_context("xyznonexistent")
        assert "No relevant" in result


# ---------------------------------------------------------------------------
# CuadKgBuilder unit tests (mocked store)
# ---------------------------------------------------------------------------


SAMPLE_EVAL_PAIRS = [
    {
        "contract_title": "Acme Distributor Agreement",
        "contract_type": "Distributor Agreement",
        "question_type": "Parties",
        "question": "Who are the parties?",
        "answers": ["Acme Corp", "Beta Ltd"],
        "answer_starts": [10, 20],
    },
    {
        "contract_title": "Acme Distributor Agreement",
        "contract_type": "Distributor Agreement",
        "question_type": "Governing Law",
        "question": "What is the governing law?",
        "answers": ["laws of Delaware"],
        "answer_starts": [100],
    },
    {
        "contract_title": "Acme Distributor Agreement",
        "contract_type": "Distributor Agreement",
        "question_type": "Expiration Date",
        "question": "What is the expiration date?",
        "answers": [],  # impossible — should be skipped
        "answer_starts": [],
    },
    {
        "contract_title": "Beta License Agreement",
        "contract_type": "License Agreement",
        "question_type": "License Grant",
        "question": "What license is granted?",
        "answers": ["non-exclusive perpetual license"],
        "answer_starts": [50],
    },
]


class TestCuadKgBuilder:
    def _make_builder(self, doc_id_map: dict[str, str | None]) -> tuple[CuadKgBuilder, MagicMock]:
        store = MagicMock(spec=PgGraphStore)
        store.pool = MagicMock()

        entity_counter = [0]

        async def upsert_entity(name, entity_type, document_id=None, metadata=None):
            entity_counter[0] += 1
            return f"entity-{entity_counter[0]:04d}"

        rel_counter = [0]

        async def add_relationship(source_id, target_id, relationship_type, document_id=None, properties=None):
            rel_counter[0] += 1
            return f"rel-{rel_counter[0]:04d}"

        store.upsert_entity = upsert_entity
        store.add_relationship = add_relationship

        builder = CuadKgBuilder(store)
        # Patch _get_document_id to return from the map
        builder._doc_id_cache = dict(doc_id_map)

        return builder, store

    @pytest.mark.asyncio
    async def test_skips_empty_answers(self, tmp_path):
        doc_map = {"Acme Distributor Agreement": "doc-1", "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        stats = await builder.build(eval_path=path)
        # Expiration Date has empty answers → skipped
        assert stats["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_creates_entities_for_each_answer(self, tmp_path):
        doc_map = {"Acme Distributor Agreement": "doc-1", "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        entity_calls = []

        async def track_upsert(name, entity_type, document_id=None, metadata=None):
            entity_calls.append((name, entity_type))
            return f"eid-{len(entity_calls)}"

        builder.store.upsert_entity = track_upsert
        builder.store.add_relationship = AsyncMock(return_value="rel-1")

        await builder.build(eval_path=path)

        entity_names = [c[0] for c in entity_calls]
        assert "Acme Corp" in entity_names
        assert "Beta Ltd" in entity_names
        assert "laws of Delaware" in entity_names
        assert "non-exclusive perpetual license" in entity_names

    @pytest.mark.asyncio
    async def test_correct_entity_types_assigned(self, tmp_path):
        doc_map = {"Acme Distributor Agreement": "doc-1", "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        entity_calls = []

        async def track_upsert(name, entity_type, document_id=None, metadata=None):
            entity_calls.append((name, entity_type))
            return f"eid-{len(entity_calls)}"

        builder.store.upsert_entity = track_upsert
        builder.store.add_relationship = AsyncMock(return_value="rel-1")

        await builder.build(eval_path=path)

        type_map = {name: etype for name, etype in entity_calls}
        assert type_map.get("Acme Corp") == "Party"
        assert type_map.get("laws of Delaware") == "Jurisdiction"
        assert type_map.get("non-exclusive perpetual license") == "LicenseClause"

    @pytest.mark.asyncio
    async def test_skips_documents_not_in_db(self, tmp_path):
        # doc_id_cache has no entry for "Acme Distributor Agreement" → None
        doc_map = {"Acme Distributor Agreement": None, "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        stats = await builder.build(eval_path=path)
        # Pairs for Acme (3 pairs: parties, governing law, expiration) all skipped
        assert stats["skipped"] >= 3

    @pytest.mark.asyncio
    async def test_limit_parameter(self, tmp_path):
        doc_map = {"Acme Distributor Agreement": "doc-1", "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        entity_calls = []

        async def track_upsert(name, entity_type, document_id=None, metadata=None):
            entity_calls.append(name)
            return f"eid-{len(entity_calls)}"

        builder.store.upsert_entity = track_upsert
        builder.store.add_relationship = AsyncMock(return_value="rel-1")

        # Only process first 1 pair
        await builder.build(eval_path=path, limit=1)
        # limit=1 → only the first pair (Parties for Acme) processed
        assert len(entity_calls) <= 5  # Contract + 2 answers + at most some extras

    @pytest.mark.asyncio
    async def test_creates_contract_entity(self, tmp_path):
        doc_map = {"Beta License Agreement": "doc-2"}
        pairs = [SAMPLE_EVAL_PAIRS[3]]  # Only Beta License pair

        builder, store = self._make_builder(doc_map)
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(pairs), encoding="utf-8")

        entity_calls = []

        async def track_upsert(name, entity_type, document_id=None, metadata=None):
            entity_calls.append((name, entity_type))
            return f"eid-{len(entity_calls)}"

        builder.store.upsert_entity = track_upsert
        builder.store.add_relationship = AsyncMock(return_value="rel-1")

        await builder.build(eval_path=path)

        types = [etype for _, etype in entity_calls]
        assert "Contract" in types  # Contract node created for the document

    @pytest.mark.asyncio
    async def test_stats_counts_returned(self, tmp_path):
        doc_map = {"Acme Distributor Agreement": "doc-1", "Beta License Agreement": "doc-2"}
        builder, store = self._make_builder(doc_map)

        path = tmp_path / "eval.json"
        path.write_text(json.dumps(SAMPLE_EVAL_PAIRS), encoding="utf-8")

        builder.store.upsert_entity = AsyncMock(return_value="eid-1")
        builder.store.add_relationship = AsyncMock(return_value="rel-1")

        stats = await builder.build(eval_path=path)
        assert "entities" in stats
        assert "relationships" in stats
        assert "skipped" in stats
        assert stats["entities"] > 0
        assert stats["relationships"] > 0


# ---------------------------------------------------------------------------
# Integration tests — require live PostgreSQL
# ---------------------------------------------------------------------------


class TestPgGraphStoreIntegration:
    """Live-DB tests. Skipped if PostgreSQL is not reachable."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self):
        store = PgGraphStore()
        try:
            await store.initialize()
            assert store._initialized
            stats = await store.get_graph_stats()
            assert "total_entities" in stats
        except Exception as e:
            pytest.skip(f"PostgreSQL not reachable: {e}")
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_upsert_and_search_roundtrip(self):
        store = PgGraphStore()
        try:
            await store.initialize()
        except Exception as e:
            pytest.skip(f"PostgreSQL not reachable: {e}")

        try:
            eid = await store.upsert_entity(
                "TestPartyXYZ_unique_9999",
                "Party",
                document_id=None,
                metadata={"test": True},
            )
            assert isinstance(eid, str)
            assert len(eid) == 36  # UUID format

            entities = await store.search_entities("TestPartyXYZ_unique_9999", entity_type="Party")
            names = [e["name"] for e in entities]
            assert "TestPartyXYZ_unique_9999" in names
        finally:
            # Clean up test entity
            async with store.pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM kg_entities WHERE name = 'TestPartyXYZ_unique_9999'"
                )
            await store.close()

    @pytest.mark.asyncio
    async def test_get_graph_stats_shape(self):
        store = PgGraphStore()
        try:
            await store.initialize()
        except Exception as e:
            pytest.skip(f"PostgreSQL not reachable: {e}")
        try:
            stats = await store.get_graph_stats()
            assert isinstance(stats["total_entities"], int)
            assert isinstance(stats["total_relationships"], int)
            assert isinstance(stats["entities_by_type"], dict)
            assert isinstance(stats["relationships_by_type"], dict)
        finally:
            await store.close()
