"""
End-to-end integration test: Bronze → Silver → Gold → Risk → NL2Cypher.

Mocks the 5 LLM passes (_run_agent) to return deterministic fixture data,
so Ollama is NOT required.  Uses real PostgreSQL (Bronze/Silver) and real
AGE (Gold) — skips if either is unreachable.

Run:
    pytest rag/tests/knowledge_graph/test_extraction_pipeline_e2e.py -m integration -v
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

# ---------------------------------------------------------------------------
# Deterministic test fixtures (no LLM)
# ---------------------------------------------------------------------------

_CONTRACT_ID = "e2e00000-0000-0000-0000-000000000001"
_TITLE = "E2E Test — Acme + Beta LLC"
_CHUNK = (
    "Acme Corp and Beta LLC hereby agree to this contract. "
    "This agreement is governed by the laws of Delaware. "
    "Acme Corp shall indemnify Beta LLC against all third-party claims."
)

# Fixed IDs — relationships reference entity IDs, so they must match
_E_ACME = "ent-e2e-0001"
_E_BETA = "ent-e2e-0002"
_E_DEL  = "ent-e2e-0003"
_R_GOV  = "rel-e2e-0001"
_R_IND  = "rel-e2e-0002"

# Return value for each of the 5 LLM passes (called in order per chunk)
_ENTITY_RESP = {
    "entities": [
        {"entity_id": _E_ACME, "label": "Party",        "canonical_name": "Acme Corp", "text_span": "Acme Corp", "confidence": 0.95},
        {"entity_id": _E_BETA, "label": "Party",        "canonical_name": "Beta LLC",  "text_span": "Beta LLC",  "confidence": 0.95},
        {"entity_id": _E_DEL,  "label": "Jurisdiction", "canonical_name": "Delaware",  "text_span": "Delaware",  "confidence": 0.95},
    ]
}
_REL_RESP = {
    "relationships": [
        {
            "relationship_id": _R_GOV,
            "source_entity_id": _E_ACME,
            "target_entity_id": _E_DEL,
            "relationship_type": "GOVERNED_BY",
            "evidence_text": "governed by the laws of Delaware",
            "confidence": 0.90,
        },
        {
            "relationship_id": _R_IND,
            "source_entity_id": _E_ACME,
            "target_entity_id": _E_BETA,
            "relationship_type": "INDEMNIFIES",
            "evidence_text": "Acme Corp shall indemnify Beta LLC",
            "confidence": 0.90,
        },
    ]
}
_HIER_RESP   = {"nodes": [], "edges": []}
_XREF_RESP   = {"references": []}
_VALID_RESP  = {
    "valid_relationships": [{"relationship_id": _R_GOV}, {"relationship_id": _R_IND}]
}

# 5 responses per chunk in call order: entities → relationships → hierarchy → cross_refs → validate
_ONE_CHUNK_RESPONSES = [_ENTITY_RESP, _REL_RESP, _HIER_RESP, _XREF_RESP, _VALID_RESP]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_pool() -> asyncpg.Pool | None:
    from rag.config.settings import load_settings
    try:
        s = load_settings()
        if not s.database_url:
            return None
        return await asyncpg.create_pool(s.database_url, min_size=1, max_size=2)
    except Exception:
        return None


async def _setup_document(pool: asyncpg.Pool) -> None:
    """Insert a test document row so the FK on kg_raw_extractions is satisfied."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, title, source, content)
            VALUES ($1::uuid, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING
            """,
            _CONTRACT_ID, _TITLE, "e2e-test-source", _CHUNK,
        )


async def _cleanup(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM kg_raw_extractions          WHERE contract_id = $1", _CONTRACT_ID)
        await conn.execute("DELETE FROM kg_staging_entities          WHERE contract_id = $1::uuid", _CONTRACT_ID)
        await conn.execute("DELETE FROM kg_staging_relationships     WHERE contract_id = $1::uuid", _CONTRACT_ID)
        await conn.execute("DELETE FROM kg_canonical_entities        WHERE contract_id = $1::uuid", _CONTRACT_ID)
        await conn.execute("DELETE FROM kg_canonical_relationships   WHERE contract_id = $1::uuid", _CONTRACT_ID)
        await conn.execute("DELETE FROM documents                    WHERE id = $1::uuid", _CONTRACT_ID)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_bronze_row_written():
    """Bronze table receives exactly one row per chunk."""
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_raw_extractions WHERE contract_id = $1",
                _CONTRACT_ID,
            )
        assert count == 1, f"expected 1 Bronze row, got {count}"

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_silver_canonical_entities_and_relationships():
    """Silver dedup produces 3 canonical entities and 2 canonical relationships."""
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            result = await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        assert result["canonical_entities"] == 3, f"expected 3 canonical entities: {result}"
        assert result["canonical_relationships"] == 2, f"expected 2 canonical relationships: {result}"

        async with pool.acquire() as conn:
            ents = await conn.fetch(
                "SELECT label, canonical_name FROM kg_canonical_entities WHERE contract_id = $1::uuid",
                _CONTRACT_ID,
            )
        names = {r["canonical_name"] for r in ents}
        assert "Acme Corp" in names
        assert "Beta LLC" in names
        assert "Delaware" in names

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_gold_vertices_queryable_in_age():
    """After Gold projection, entities are searchable in AGE."""
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            result = await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        assert result["age_entities"] >= 2, f"expected ≥2 AGE entities, got {result}"

        parties = await age.search_entities("Acme Corp", entity_type="Party")
        assert any(e["name"] == "Acme Corp" for e in parties), "Acme Corp vertex not in AGE"

        jurisdictions = await age.search_entities("Delaware", entity_type="Jurisdiction")
        assert any(e["name"] == "Delaware" for e in jurisdictions), "Delaware vertex not in AGE"

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_nl2cypher_round_trip():
    """INDEMNIFIES edge written by Gold is queryable via NL2Cypher."""
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store, NL2CypherConverter
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        converter = NL2CypherConverter()
        cypher = await converter.convert("Which parties indemnify each other?")
        result_str = await age.run_cypher_query(cypher)
        assert "Acme" in result_str, (
            f"INDEMNIFIES edge not found via NL2Cypher.\nCypher: {cypher}\nResult: {result_str}"
        )

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_project_contract_replay_no_llm():
    """
    project_contract() replays Silver+Gold from Bronze — _run_agent must NOT be called.

    Step 1: write Bronze via process_contract (mocked LLM).
    Step 2: project_contract() with mock that raises on call — must succeed without raising.
    """
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        # Step 1: write Bronze
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        # Step 2: replay — if LLM is called, the test raises
        async def _should_not_be_called(*_a, **_kw):
            raise AssertionError("_run_agent called during project_contract replay")

        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", side_effect=_should_not_be_called):
            replay = await pipeline.project_contract(_CONTRACT_ID)

        assert "error" not in replay, f"project_contract failed: {replay}"
        assert replay["canonical_entities"] == 3

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_bronze_row_idempotent_on_rerun():
    """
    Running process_contract twice for the same contract_id produces exactly 1 Bronze row
    (ON CONFLICT DO UPDATE, not INSERT duplicate).
    """
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()
    await _setup_document(pool)

    try:
        for _ in range(2):
            with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
                m.side_effect = list(_ONE_CHUNK_RESPONSES)
                await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_raw_extractions WHERE contract_id = $1",
                _CONTRACT_ID,
            )
        assert count == 1, f"Bronze row not idempotent — expected 1, got {count}"

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()


@pytest.mark.asyncio
async def test_bronze_json_file_written(tmp_path, monkeypatch):
    """_save_json writes a JSON file containing all 5-pass LLM outputs."""
    pool = await _make_pool()
    if pool is None:
        pytest.skip("PostgreSQL not reachable")

    from kg import create_kg_store
    from kg.legal.ingestion import extraction_pipeline as ep
    from kg.legal.ingestion.extraction_pipeline import ExtractionPipeline

    age = create_kg_store()
    try:
        await age.initialize()
    except Exception as exc:
        await pool.close()
        pytest.skip(f"AGE not reachable: {exc}")

    pipeline = ExtractionPipeline(pool, age)
    await pipeline.initialize()

    # Redirect JSON output to tmp_path
    monkeypatch.setattr(ep.ExtractionPipeline, "_JSON_DIR", tmp_path / "jsons")
    await _setup_document(pool)

    try:
        with patch("kg.legal.ingestion.extraction_pipeline._run_agent", new_callable=AsyncMock) as m:
            m.side_effect = list(_ONE_CHUNK_RESPONSES)
            await pipeline.process_contract(_CONTRACT_ID, _TITLE, [_CHUNK])

        json_files = list((tmp_path / "jsons").glob("*.json"))
        assert len(json_files) == 1, f"expected 1 JSON file, found {json_files}"

        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert payload["contract_id"] == _CONTRACT_ID
        assert len(payload["chunks"]) == 1
        chunk0 = payload["chunks"][0]
        assert len(chunk0["entities"]) == 3
        assert len(chunk0["relationships"]) == 2
        entity_names = [e["canonical_name"] for e in chunk0["entities"]]
        assert "Acme Corp" in entity_names
        assert "Delaware" in entity_names

    finally:
        await _cleanup(pool)
        await pool.close()
        await age.close()
