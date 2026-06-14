"""Integration tests for RedisCache (L2 cache).

Tests embedding cache, search result cache, document fingerprint cache,
health cache, and corpus invalidation against a live Redis instance.
Auto-skipped when Redis is unreachable.

Run:
    REDIS_URL=... pytest tests/integration/test_cache.py -v
"""

import uuid
from typing import Any

import pytest
import pytest_asyncio

from knowledge.config.settings import load_settings
from knowledge.store.cache import RedisCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def cache() -> RedisCache:
    settings = load_settings()
    c = RedisCache(settings=settings)
    try:
        await c.connect()
    except Exception as exc:
        pytest.skip(f"Redis unreachable: {exc}")
    yield c
    await c.close()


def _tag() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: RedisCache) -> None:
        result = await cache.get_embedding(f"text-that-does-not-exist-{_tag()}")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_returns_vector(self, cache: RedisCache) -> None:
        text = f"unique-text-{_tag()}"
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        await cache.set_embedding(text, vec)
        result = await cache.get_embedding(text)
        assert result == pytest.approx(vec)

    @pytest.mark.asyncio
    async def test_different_texts_are_independent(self, cache: RedisCache) -> None:
        tag = _tag()
        text_a, text_b = f"text-a-{tag}", f"text-b-{tag}"
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        await cache.set_embedding(text_a, vec_a)
        await cache.set_embedding(text_b, vec_b)
        assert await cache.get_embedding(text_a) == pytest.approx(vec_a)
        assert await cache.get_embedding(text_b) == pytest.approx(vec_b)

    @pytest.mark.asyncio
    async def test_large_vector_round_trips(self, cache: RedisCache) -> None:
        text = f"large-vec-{_tag()}"
        vec = [float(i) / 1000.0 for i in range(768)]
        await cache.set_embedding(text, vec)
        result = await cache.get_embedding(text)
        assert result == pytest.approx(vec, abs=1e-5)


# ---------------------------------------------------------------------------
# Search result cache
# ---------------------------------------------------------------------------

class TestSearchCache:

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: RedisCache) -> None:
        result = await cache.get_search(
            f"unique-query-{_tag()}", ["corpus-xyz"], filters=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_returns_results(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"what is rag {tag}"
        corpus_ids = [f"corpus-{tag}"]
        results = [
            {"id": "abc", "content": "RAG is retrieval augmented generation",
             "score": 0.9},
        ]
        await cache.set_search(query, corpus_ids, results)
        cached = await cache.get_search(query, corpus_ids)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["content"] == results[0]["content"]

    @pytest.mark.asyncio
    async def test_corpus_ids_order_independent(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"query {tag}"
        results = [{"id": "x", "score": 0.5}]
        await cache.set_search(query, ["b", "a"], results)
        # Retrieve with different order — should still hit
        cached = await cache.get_search(query, ["a", "b"])
        assert cached is not None

    @pytest.mark.asyncio
    async def test_different_corpus_is_cache_miss(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"query {tag}"
        await cache.set_search(query, [f"corpus-a-{tag}"], [{"id": "x"}])
        cached = await cache.get_search(query, [f"corpus-b-{tag}"])
        assert cached is None

    @pytest.mark.asyncio
    async def test_filters_affect_cache_key(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"query {tag}"
        corpus = [f"c-{tag}"]
        await cache.set_search(query, corpus, [{"id": "x"}],
                                filters={"doc_type": "pdf"})
        # No filters — different key → miss
        assert await cache.get_search(query, corpus, filters=None) is None
        # Same filters — hit
        assert await cache.get_search(query, corpus,
                                       filters={"doc_type": "pdf"}) is not None

    @pytest.mark.asyncio
    async def test_nx_does_not_overwrite_existing(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"query {tag}"
        corpus = [f"c-{tag}"]
        first = [{"id": "first", "score": 1.0}]
        second = [{"id": "second", "score": 0.5}]
        await cache.set_search(query, corpus, first)
        await cache.set_search(query, corpus, second)  # NX: should not overwrite
        cached = await cache.get_search(query, corpus)
        assert cached is not None
        assert cached[0]["id"] == "first"


# ---------------------------------------------------------------------------
# Fingerprint cache
# ---------------------------------------------------------------------------

class TestFingerprintCache:

    @pytest.mark.asyncio
    async def test_unknown_hash_returns_false(self, cache: RedisCache) -> None:
        result = await cache.get_fingerprint(f"nonexistent-{_tag()}")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_then_get_returns_true(self, cache: RedisCache) -> None:
        content_hash = f"sha256-{_tag()}"
        await cache.set_fingerprint(content_hash)
        assert await cache.get_fingerprint(content_hash) is True

    @pytest.mark.asyncio
    async def test_delete_removes_fingerprint(self, cache: RedisCache) -> None:
        content_hash = f"sha256-{_tag()}"
        await cache.set_fingerprint(content_hash)
        assert await cache.get_fingerprint(content_hash) is True
        await cache.delete_fingerprint(content_hash)
        assert await cache.get_fingerprint(content_hash) is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, cache: RedisCache) -> None:
        content_hash = f"sha256-nonexistent-{_tag()}"
        await cache.delete_fingerprint(content_hash)  # must not raise
        assert await cache.get_fingerprint(content_hash) is False

    @pytest.mark.asyncio
    async def test_independent_hashes_are_isolated(self, cache: RedisCache) -> None:
        h1 = f"hash-a-{_tag()}"
        h2 = f"hash-b-{_tag()}"
        await cache.set_fingerprint(h1)
        assert await cache.get_fingerprint(h1) is True
        assert await cache.get_fingerprint(h2) is False


# ---------------------------------------------------------------------------
# Health cache
# ---------------------------------------------------------------------------

class TestHealthCache:

    @pytest.mark.asyncio
    async def test_get_health_miss_returns_none(self, cache: RedisCache) -> None:
        result = await cache.get_health(f"svc-{_tag()}")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_health(self, cache: RedisCache) -> None:
        svc = f"svc-{_tag()}"
        data: dict[str, Any] = {"status": "healthy", "latency_ms": 12.5}
        await cache.set_health(svc, data)
        result = await cache.get_health(svc)
        assert result is not None
        assert result["status"] == "healthy"
        assert result["latency_ms"] == pytest.approx(12.5)

    @pytest.mark.asyncio
    async def test_different_services_are_independent(self, cache: RedisCache) -> None:
        tag = _tag()
        svc_a, svc_b = f"svc-a-{tag}", f"svc-b-{tag}"
        await cache.set_health(svc_a, {"status": "healthy"})
        assert await cache.get_health(svc_a) is not None
        assert await cache.get_health(svc_b) is None


# ---------------------------------------------------------------------------
# Corpus invalidation
# ---------------------------------------------------------------------------

class TestCorpusInvalidation:

    @pytest.mark.asyncio
    async def test_invalidate_clears_search_cache(self, cache: RedisCache) -> None:
        tag = _tag()
        query = f"query-{tag}"
        corpus = f"corp-{tag}"
        await cache.set_search(query, [corpus], [{"id": "x"}])
        assert await cache.get_search(query, [corpus]) is not None

        await cache.invalidate_corpus(corpus, "tenant-x")

        # After invalidation, cache miss (all search keys were purged)
        assert await cache.get_search(query, [corpus]) is None

    @pytest.mark.asyncio
    async def test_invalidate_returns_count(self, cache: RedisCache) -> None:
        tag = _tag()
        for i in range(3):
            await cache.set_search(f"q{i}-{tag}", [f"c-{tag}"], [{"id": str(i)}])

        deleted = await cache.invalidate_corpus(f"c-{tag}", "t")
        assert deleted >= 3

    @pytest.mark.asyncio
    async def test_invalidate_empty_corpus_returns_zero(self, cache: RedisCache) -> None:
        tag = _tag()
        # Only set fingerprint and embedding keys, no search keys
        await cache.set_fingerprint(f"hash-{tag}")
        await cache.set_embedding(f"text-{tag}", [0.1, 0.2])

        deleted = await cache.invalidate_corpus(f"nonexistent-corpus-{tag}", "t")
        # search keys scanned, none matching → 0 deleted (fingerprint/embed untouched)
        assert deleted == 0
