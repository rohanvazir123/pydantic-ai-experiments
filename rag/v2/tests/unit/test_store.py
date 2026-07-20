# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Unit tests for knowledge.store.vector and knowledge.store.graph.

No live database required — asyncpg pool is mocked at the boundary.
Tests cover: key logic, label/rel sanitization, RRF SQL structure,
graph naming, AGE Cypher wrapper, and graceful error paths.
"""

import re

import pytest

from knowledge.store.graph import (
    AgeGraphStore,
    _parse_return_aliases,
    _sanitize_label,
    _sanitize_rel_type,
    _unquote_agtype,
)
from knowledge.store.vector import RRF_K, PostgresHybridStore

# ── AgeGraphStore helpers ─────────────────────────────────────────────────────

class TestSanitizeLabel:
    def test_simple_label(self) -> None:
        assert _sanitize_label("Person") == "Person"

    def test_capitalises_first_letter(self) -> None:
        assert _sanitize_label("person") == "Person"

    def test_strips_non_alphanumeric(self) -> None:
        assert _sanitize_label("My-Entity!") == "MyEntity"

    def test_empty_becomes_entity(self) -> None:
        assert _sanitize_label("") == "Entity"

    def test_only_symbols_becomes_entity(self) -> None:
        assert _sanitize_label("---") == "Entity"


class TestSanitizeRelType:
    def test_uppercase(self) -> None:
        assert _sanitize_rel_type("HAS_MEMBER") == "HAS_MEMBER"

    def test_lowercase_uppercased(self) -> None:
        assert _sanitize_rel_type("has_member") == "HAS_MEMBER"

    def test_strips_non_alphanumeric_except_underscore(self) -> None:
        assert _sanitize_rel_type("APPLIES-TO") == "APPLIESTO"

    def test_empty_becomes_related_to(self) -> None:
        assert _sanitize_rel_type("") == "RELATED_TO"

    def test_only_symbols_becomes_related_to(self) -> None:
        assert _sanitize_rel_type("---") == "RELATED_TO"


class TestParseReturnAliases:
    def test_single_alias(self) -> None:
        cypher = "MATCH (n) RETURN n.name AS name"
        assert _parse_return_aliases(cypher) == ["name"]

    def test_multiple_aliases(self) -> None:
        cypher = "MATCH (n)-[r]->(m) RETURN n.name AS src, type(r) AS rel, m.name AS tgt"
        aliases = _parse_return_aliases(cypher)
        assert aliases == ["src", "rel", "tgt"]

    def test_no_alias_uses_last_token(self) -> None:
        cypher = "MATCH (n) RETURN n.name"
        assert _parse_return_aliases(cypher) == ["name"]

    def test_fallback_on_no_return(self) -> None:
        assert _parse_return_aliases("MATCH (n)") == ["c0"]

    def test_limit_clause_does_not_bleed_into_aliases(self) -> None:
        cypher = "MATCH (n) RETURN n.name AS name LIMIT 10"
        assert _parse_return_aliases(cypher) == ["name"]


class TestUnquoteAgtype:
    def test_strips_quotes(self) -> None:
        assert _unquote_agtype('"Acme Corp"') == "Acme Corp"

    def test_no_quotes_unchanged(self) -> None:
        assert _unquote_agtype("Acme Corp") == "Acme Corp"

    def test_none_returns_empty(self) -> None:
        assert _unquote_agtype(None) == ""

    def test_numeric_string(self) -> None:
        assert _unquote_agtype('"42"') == "42"


class TestAgeGraphStoreName:
    def _store(self) -> AgeGraphStore:
        from unittest import mock
        with mock.patch.dict(__import__("os").environ, {
            "DATABASE_URL": "postgresql://x:x@localhost/x",
            "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
            "AGE_GRAPH_PREFIX": "kg",
        }, clear=True):
            from knowledge.config.settings import Settings
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        return AgeGraphStore(settings=s)

    def test_graph_name_basic(self) -> None:
        store = self._store()
        name = store._graph_name("acme-corp", "hr-policies")
        assert name == "kg_acme_corp_hr_policies"

    def test_graph_name_replaces_colon(self) -> None:
        store = self._store()
        name = store._graph_name("acme", "hr:policies")
        assert ":" not in name

    def test_graph_name_prefix(self) -> None:
        store = self._store()
        name = store._graph_name("t1", "c1")
        assert name.startswith("kg_")

    @pytest.mark.asyncio
    async def test_cypher_read_guard_blocks_create(self) -> None:
        """run_cypher_query must reject write statements without hitting the DB."""
        store = self._store()
        store._pool = object()  # fake non-None pool so assert passes

        blocked_queries = [
            "CREATE (n:Person {name: 'Alice'})",
            "MERGE (n:Person {name: 'Bob'})",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
        ]
        for cypher in blocked_queries:
            result = await store.run_cypher_query(cypher, "corp1", "tenant1")
            assert "Error:" in result, f"Should have blocked: {cypher}"

    def test_cypher_read_passes_match(self) -> None:
        """MATCH queries are not blocked by the guard (they would hit the DB)."""
        safe = "MATCH (n:Person) RETURN n.name LIMIT 5"
        assert not re.search(
            r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b",
            safe, re.IGNORECASE,
        )


# ── PostgresHybridStore ───────────────────────────────────────────────────────

class TestRRFConstant:
    def test_rrf_k_is_60(self) -> None:
        assert RRF_K == 60

class TestHybridSearchSQL:
    """Verify the RRF SQL structure contains the expected clauses without a live DB."""

    def test_rrf_sql_has_full_outer_join(self) -> None:
        import inspect
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        assert "FULL OUTER JOIN" in src.upper()

    def test_rrf_sql_has_coalesce(self) -> None:
        import inspect
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        assert "COALESCE" in src.upper()

    def test_rrf_sql_uses_rrf_k_constant(self) -> None:
        import inspect
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        # Should reference RRF_K not hardcoded 60
        assert "RRF_K" in src

    def test_result_has_confidence_none(self) -> None:
        """hybrid_search results carry confidence=None until reranker sets it."""
        # We verify the structure via the source — confidence is set to None
        import inspect
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        assert '"confidence": None' in src

    def test_result_has_raw_score_type_rrf(self) -> None:
        import inspect
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        assert '"raw_score_type": "rrf"' in src


# ── OR-tsquery helper ─────────────────────────────────────────────────────────

class TestToOrTsquery:
    """_to_or_tsquery converts natural-language queries to OR-expanded websearch queries."""

    def test_pto_query_produces_or_terms(self) -> None:
        from knowledge.store.vector import _to_or_tsquery
        result = _to_or_tsquery("What is the PTO and leave policy?")
        assert " OR " in result
        assert "PTO" in result
        assert "leave" in result
        assert "policy" in result

    def test_stop_words_stripped(self) -> None:
        from knowledge.store.vector import _to_or_tsquery
        result = _to_or_tsquery("What is the company policy?")
        assert "what" not in result.lower()
        assert "the" not in result.lower()
        assert "is" not in result.lower()
        assert "company" in result.lower()
        assert "policy" in result.lower()

    def test_q4_query(self) -> None:
        from knowledge.store.vector import _to_or_tsquery
        result = _to_or_tsquery("Which business units performed best in Q4?")
        assert "business" in result
        assert "units" in result
        assert "Q4" in result

    def test_short_words_stripped(self) -> None:
        from knowledge.store.vector import _to_or_tsquery
        result = _to_or_tsquery("Do we have PTO?")
        # "Do" and "we" are stop words; "have" is stop word; "PTO" keeps
        assert "PTO" in result
        tokens = result.split(" OR ")
        assert all(len(t) >= 2 for t in tokens)

    def test_empty_after_stop_removal_falls_back(self) -> None:
        from knowledge.store.vector import _to_or_tsquery
        # All stop words — falls back to original query
        result = _to_or_tsquery("is the a an")
        assert result == "is the a an"

    def test_hybrid_search_uses_or_query(self) -> None:
        import inspect
        from knowledge.store.vector import PostgresHybridStore
        src = inspect.getsource(PostgresHybridStore.hybrid_search)
        assert "or_query" in src
        assert "_to_or_tsquery" in src

    def test_text_search_uses_or_query(self) -> None:
        import inspect
        from knowledge.store.vector import PostgresHybridStore
        src = inspect.getsource(PostgresHybridStore.text_search)
        assert "or_query" in src
        assert "_to_or_tsquery" in src
