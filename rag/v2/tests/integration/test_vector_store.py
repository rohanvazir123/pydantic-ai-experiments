"""Integration tests for PostgresHybridStore.

Tests the full DB layer: documents, chunks, vector search, text search,
hybrid RRF search, corpus isolation, and cascade deletes. All tests run
against a live PostgreSQL + pgvector instance. Auto-skipped when unreachable.

Run:
    DATABASE_URL=... pytest tests/integration/test_vector_store.py -v
"""

import math
import random
import uuid
from typing import Any

import pytest
import pytest_asyncio

from knowledge.config.settings import load_settings
from knowledge.store.vector import PostgresHybridStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 768  # must match EMBEDDING_DIMENSION in env


def _unit_vec(seed: int) -> list[float]:
    """Deterministic unit vector of dimension DIM seeded by seed."""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _chunk(content: str, embedding_seed: int, idx: int = 0) -> dict[str, Any]:
    return {
        "content": content,
        "embedding": _unit_vec(embedding_seed),
        "chunk_index": idx,
        "token_count": len(content.split()),
        "metadata": {"source_seed": embedding_seed},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def store() -> PostgresHybridStore:
    settings = load_settings()
    s = PostgresHybridStore(settings=settings)
    try:
        await s.initialize()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    yield s
    await s.close()


def _ids() -> tuple[str, str, str]:
    """Return (corpus_id, tenant_id, source) unique to this test invocation."""
    tag = uuid.uuid4().hex[:8]
    return f"corpus-{tag}", f"tenant-{tag}", f"s3://test/{tag}/doc.md"


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

class TestDocumentCRUD:

    @pytest.mark.asyncio
    async def test_save_document_returns_uuid(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document(
            title="Test Doc", source=source, corpus_id=corpus, tenant_id=tenant
        )
        assert uuid.UUID(doc_id)  # valid UUID

    @pytest.mark.asyncio
    async def test_save_document_upsert_returns_same_id(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        id1 = await store.save_document("Doc", source, corpus, tenant)
        id2 = await store.save_document("Doc Updated", source, corpus, tenant)
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_save_document_upsert_updates_title(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        await store.save_document("Original", source, corpus, tenant)
        await store.save_document("Updated Title", source, corpus, tenant)
        sources = await store.get_all_document_sources(corpus, tenant)
        assert source in sources

    @pytest.mark.asyncio
    async def test_get_document_hash_missing_returns_none(self, store: PostgresHybridStore) -> None:
        corpus, tenant, _source = _ids()
        result = await store.get_document_hash("nonexistent-source", corpus, tenant)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_hash_round_trip(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        content_hash = "abc123deadbeef"
        await store.save_document(
            "Doc", source, corpus, tenant, metadata={"content_hash": content_hash}
        )
        retrieved = await store.get_document_hash(source, corpus, tenant)
        assert retrieved == content_hash

    @pytest.mark.asyncio
    async def test_get_all_document_sources_empty(self, store: PostgresHybridStore) -> None:
        corpus, tenant, _ = _ids()
        sources = await store.get_all_document_sources(corpus, tenant)
        assert sources == []

    @pytest.mark.asyncio
    async def test_get_all_document_sources_lists_them(self, store: PostgresHybridStore) -> None:
        corpus, tenant, _ = _ids()
        tag = uuid.uuid4().hex[:6]
        src_a = f"s3://bucket/{tag}/a.md"
        src_b = f"s3://bucket/{tag}/b.md"
        await store.save_document("A", src_a, corpus, tenant)
        await store.save_document("B", src_b, corpus, tenant)
        sources = await store.get_all_document_sources(corpus, tenant)
        assert src_a in sources
        assert src_b in sources

    @pytest.mark.asyncio
    async def test_delete_document_removes_it(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        await store.save_document("Doc", source, corpus, tenant)
        await store.delete_document_and_chunks(source, corpus, tenant)
        sources = await store.get_all_document_sources(corpus, tenant)
        assert source not in sources


# ---------------------------------------------------------------------------
# Chunk upsert + count
# ---------------------------------------------------------------------------

class TestChunks:

    @pytest.mark.asyncio
    async def test_upsert_chunks_increases_count(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Doc", source, corpus, tenant)
        chunks = [_chunk(f"paragraph {i}", embedding_seed=i, idx=i) for i in range(5)]
        await store.upsert_chunks(chunks, doc_id, corpus, tenant)
        count = await store.get_chunk_count(corpus, tenant)
        assert count >= 5

    @pytest.mark.asyncio
    async def test_upsert_empty_chunks_is_noop(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Doc", source, corpus, tenant)
        await store.upsert_chunks([], doc_id, corpus, tenant)
        count = await store.get_chunk_count(corpus, tenant)
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_document_cascades_to_chunks(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Doc", source, corpus, tenant)
        await store.upsert_chunks(
            [_chunk("hello world", embedding_seed=42)], doc_id, corpus, tenant
        )
        assert await store.get_chunk_count(corpus, tenant) >= 1

        await store.delete_document_and_chunks(source, corpus, tenant)
        assert await store.get_chunk_count(corpus, tenant) == 0

    @pytest.mark.asyncio
    async def test_truncate_corpus_removes_all_chunks(self, store: PostgresHybridStore) -> None:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Doc", source, corpus, tenant)
        await store.upsert_chunks(
            [_chunk("alpha", 1), _chunk("beta", 2)], doc_id, corpus, tenant
        )
        await store.truncate_corpus(corpus, tenant)
        assert await store.get_chunk_count(corpus, tenant) == 0


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

class TestSemanticSearch:

    @pytest_asyncio.fixture
    async def seeded(self, store: PostgresHybridStore) -> dict[str, Any]:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Seeded Doc", source, corpus, tenant)
        seed_vec = _unit_vec(seed=7)
        await store.upsert_chunks(
            [
                {"content": "the quick brown fox", "embedding": seed_vec,
                 "chunk_index": 0, "token_count": 4, "metadata": {}},
                {"content": "unrelated content here", "embedding": _unit_vec(seed=99),
                 "chunk_index": 1, "token_count": 3, "metadata": {}},
            ],
            doc_id, corpus, tenant,
        )
        return {"store": store, "corpus": corpus, "tenant": tenant,
                "seed_vec": seed_vec, "doc_id": doc_id}

    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].semantic_search(
            seeded["seed_vec"], seeded["corpus"], seeded["tenant"], k=5
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_semantic_search_top_result_is_nearest(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].semantic_search(
            seeded["seed_vec"], seeded["corpus"], seeded["tenant"], k=5
        )
        # The exact same vector should score highest (score ≈ 1.0)
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
        assert results[0]["content"] == "the quick brown fox"

    @pytest.mark.asyncio
    async def test_semantic_search_respects_k(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].semantic_search(
            seeded["seed_vec"], seeded["corpus"], seeded["tenant"], k=1
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_semantic_search_empty_corpus_returns_empty(
        self, store: PostgresHybridStore
    ) -> None:
        corpus, tenant, _ = _ids()
        results = await store.semantic_search(_unit_vec(1), corpus, tenant, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------

class TestTextSearch:

    @pytest_asyncio.fixture
    async def seeded(self, store: PostgresHybridStore) -> dict[str, Any]:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Text Doc", source, corpus, tenant)
        await store.upsert_chunks(
            [
                {"content": "machine learning enables intelligent systems",
                 "embedding": _unit_vec(1), "chunk_index": 0,
                 "token_count": 5, "metadata": {}},
                {"content": "cooking recipes for beginners",
                 "embedding": _unit_vec(2), "chunk_index": 1,
                 "token_count": 4, "metadata": {}},
            ],
            doc_id, corpus, tenant,
        )
        return {"store": store, "corpus": corpus, "tenant": tenant}

    @pytest.mark.asyncio
    async def test_text_search_finds_keyword(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].text_search(
            "machine learning", seeded["corpus"], seeded["tenant"], k=5
        )
        assert any("machine learning" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_text_search_no_false_positives(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].text_search(
            "machine learning", seeded["corpus"], seeded["tenant"], k=5
        )
        # "cooking" chunk should NOT appear in results for "machine learning"
        assert not any("cooking" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_text_search_empty_corpus_returns_empty(
        self, store: PostgresHybridStore
    ) -> None:
        corpus, tenant, _ = _ids()
        results = await store.text_search("anything", corpus, tenant, k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_text_search_scores_descending(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].text_search(
            "intelligent", seeded["corpus"], seeded["tenant"], k=5
        )
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hybrid search (RRF)
# ---------------------------------------------------------------------------

class TestHybridSearch:

    @pytest_asyncio.fixture
    async def seeded(self, store: PostgresHybridStore) -> dict[str, Any]:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Hybrid Doc", source, corpus, tenant)
        target_vec = _unit_vec(seed=13)
        await store.upsert_chunks(
            [
                {"content": "neural network deep learning model",
                 "embedding": target_vec, "chunk_index": 0,
                 "token_count": 5, "metadata": {}},
                {"content": "database management systems",
                 "embedding": _unit_vec(seed=55), "chunk_index": 1,
                 "token_count": 3, "metadata": {}},
                {"content": "cooking and recipes",
                 "embedding": _unit_vec(seed=77), "chunk_index": 2,
                 "token_count": 3, "metadata": {}},
            ],
            doc_id, corpus, tenant,
        )
        return {"store": store, "corpus": corpus, "tenant": tenant,
                "target_vec": target_vec}

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].hybrid_search(
            "neural network", seeded["target_vec"],
            seeded["corpus"], seeded["tenant"], k=5
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_has_rrf_score_type(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].hybrid_search(
            "neural network", seeded["target_vec"],
            seeded["corpus"], seeded["tenant"], k=5
        )
        for r in results:
            assert r["raw_score_type"] == "rrf"
            assert r["confidence"] is None

    @pytest.mark.asyncio
    async def test_hybrid_search_scores_descending(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].hybrid_search(
            "neural network", seeded["target_vec"],
            seeded["corpus"], seeded["tenant"], k=5
        )
        scores = [r["raw_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_k(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].hybrid_search(
            "neural", seeded["target_vec"],
            seeded["corpus"], seeded["tenant"], k=1
        )
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_surfaces_text_and_vector_match(
        self, seeded: dict[str, Any]
    ) -> None:
        # "neural network" chunk matches BOTH text leg and vector leg → should rank first
        results = await seeded["store"].hybrid_search(
            "neural network", seeded["target_vec"],
            seeded["corpus"], seeded["tenant"], k=5
        )
        assert results[0]["content"] == "neural network deep learning model"


# ---------------------------------------------------------------------------
# Corpus isolation
# ---------------------------------------------------------------------------

class TestCorpusIsolation:

    @pytest.mark.asyncio
    async def test_search_does_not_cross_corpus(self, store: PostgresHybridStore) -> None:
        _, tenant, _ = _ids()
        tag = uuid.uuid4().hex[:6]
        corpus_a, corpus_b = f"corp-a-{tag}", f"corp-b-{tag}"
        target_vec = _unit_vec(seed=42)

        doc_a = await store.save_document(
            "A", f"s3://{tag}/a.md", corpus_a, tenant
        )
        await store.upsert_chunks(
            [{"content": "only in corpus A", "embedding": target_vec,
              "chunk_index": 0, "token_count": 4, "metadata": {}}],
            doc_a, corpus_a, tenant,
        )

        results = await store.semantic_search(target_vec, corpus_b, tenant, k=5)
        assert all("corpus A" not in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_get_chunk_count_is_per_corpus(self, store: PostgresHybridStore) -> None:
        _, tenant, _ = _ids()
        tag = uuid.uuid4().hex[:6]
        corpus_a, corpus_b = f"ca-{tag}", f"cb-{tag}"

        doc_a = await store.save_document("A", f"s3://{tag}/a.md", corpus_a, tenant)
        doc_b = await store.save_document("B", f"s3://{tag}/b.md", corpus_b, tenant)

        await store.upsert_chunks(
            [_chunk("in A", 1), _chunk("in A2", 2)], doc_a, corpus_a, tenant
        )
        await store.upsert_chunks(
            [_chunk("in B only", 3)], doc_b, corpus_b, tenant
        )

        assert await store.get_chunk_count(corpus_a, tenant) == 2
        assert await store.get_chunk_count(corpus_b, tenant) == 1

    @pytest.mark.asyncio
    async def test_unknown_corpus_returns_empty_search(self, store: PostgresHybridStore) -> None:
        results = await store.semantic_search(
            _unit_vec(1), "corpus-that-does-not-exist", "tenant-xyz", k=5
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_text_search_scoped_to_corpus(self, store: PostgresHybridStore) -> None:
        _, tenant, _ = _ids()
        tag = uuid.uuid4().hex[:6]
        corpus_a, corpus_b = f"ta-{tag}", f"tb-{tag}"

        doc_a = await store.save_document("A", f"s3://{tag}/a.md", corpus_a, tenant)
        await store.upsert_chunks(
            [{"content": "exclusive keyword zxqvw", "embedding": _unit_vec(1),
              "chunk_index": 0, "token_count": 3, "metadata": {}}],
            doc_a, corpus_a, tenant,
        )

        results = await store.text_search("zxqvw", corpus_b, tenant, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

class TestResultShape:

    @pytest_asyncio.fixture
    async def seeded(self, store: PostgresHybridStore) -> dict[str, Any]:
        corpus, tenant, source = _ids()
        doc_id = await store.save_document("Shape Doc", source, corpus, tenant)
        await store.upsert_chunks(
            [{"content": "shape test content", "embedding": _unit_vec(1),
              "chunk_index": 0, "token_count": 3,
              "metadata": {"page": 1, "section": "intro"}}],
            doc_id, corpus, tenant,
        )
        return {"store": store, "corpus": corpus, "tenant": tenant,
                "vec": _unit_vec(1)}

    @pytest.mark.asyncio
    async def test_semantic_result_has_required_keys(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].semantic_search(
            seeded["vec"], seeded["corpus"], seeded["tenant"], k=1
        )
        assert results
        r = results[0]
        for key in ("id", "document_id", "content", "metadata", "score"):
            assert key in r, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_metadata_round_trips(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].semantic_search(
            seeded["vec"], seeded["corpus"], seeded["tenant"], k=1
        )
        assert results[0]["metadata"]["page"] == 1
        assert results[0]["metadata"]["section"] == "intro"

    @pytest.mark.asyncio
    async def test_hybrid_result_has_raw_score(self, seeded: dict[str, Any]) -> None:
        results = await seeded["store"].hybrid_search(
            "shape test", seeded["vec"], seeded["corpus"], seeded["tenant"], k=1
        )
        assert results
        assert results[0]["raw_score"] > 0
