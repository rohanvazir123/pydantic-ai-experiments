"""Unit tests for knowledge.store.cache.

Uses fakeredis — no live Redis required.
"""

import pytest
import pytest_asyncio
import fakeredis.aioredis as fakeredis

from knowledge.store.cache import RedisCache, _embed_key, _search_key, _fingerprint_key
from knowledge.config.settings import Settings


def make_cache() -> RedisCache:
    from unittest import mock
    with mock.patch.dict(__import__("os").environ, {
        "DATABASE_URL": "postgresql://x:x@localhost/x",
        "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
    }, clear=True):
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
    return RedisCache(settings=settings)


@pytest_asyncio.fixture
async def cache() -> RedisCache:
    c = make_cache()
    c._client = fakeredis.FakeRedis(decode_responses=False)
    return c


# ── Key construction ──────────────────────────────────────────────────────────

class TestKeyConstruction:
    def test_embed_key_prefix(self) -> None:
        assert _embed_key("hello").startswith("cache:embed:")

    def test_embed_key_deterministic(self) -> None:
        assert _embed_key("hello") == _embed_key("hello")

    def test_embed_key_different_inputs(self) -> None:
        assert _embed_key("hello") != _embed_key("world")

    def test_search_key_prefix(self) -> None:
        assert _search_key("q", ["c1"], None).startswith("cache:search:")

    def test_search_key_corpus_order_independent(self) -> None:
        k1 = _search_key("q", ["c1", "c2"], None)
        k2 = _search_key("q", ["c2", "c1"], None)
        assert k1 == k2

    def test_search_key_filter_changes_key(self) -> None:
        k1 = _search_key("q", ["c1"], None)
        k2 = _search_key("q", ["c1"], {"doc_type": "pdf"})
        assert k1 != k2

    def test_fingerprint_key_prefix(self) -> None:
        assert _fingerprint_key("abc123").startswith("cache:doc_fingerprint:")


# ── Embedding cache ───────────────────────────────────────────────────────────

class TestEmbeddingCache:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: RedisCache) -> None:
        result = await cache.get_embedding("nonexistent text")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get(self, cache: RedisCache) -> None:
        vector = [0.1, 0.2, 0.3]
        await cache.set_embedding("hello", vector)
        result = await cache.get_embedding("hello")
        assert result == pytest.approx(vector)

    @pytest.mark.asyncio
    async def test_different_texts_different_entries(self, cache: RedisCache) -> None:
        await cache.set_embedding("foo", [1.0, 2.0])
        await cache.set_embedding("bar", [3.0, 4.0])
        assert await cache.get_embedding("foo") == pytest.approx([1.0, 2.0])
        assert await cache.get_embedding("bar") == pytest.approx([3.0, 4.0])

    @pytest.mark.asyncio
    async def test_no_client_returns_none(self) -> None:
        c = make_cache()
        # _client is None (not connected)
        assert await c.get_embedding("text") is None

    @pytest.mark.asyncio
    async def test_no_client_set_is_noop(self) -> None:
        c = make_cache()
        await c.set_embedding("text", [1.0])  # must not raise


# ── Search result cache ───────────────────────────────────────────────────────

class TestSearchCache:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache: RedisCache) -> None:
        result = await cache.get_search("q", ["corp1"])
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "abc", "content": "hello", "score": 0.9}]
        await cache.set_search("q", ["corp1"], results)
        cached = await cache.get_search("q", ["corp1"])
        assert cached == results

    @pytest.mark.asyncio
    async def test_corpus_order_independent(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "x"}]
        await cache.set_search("q", ["c2", "c1"], results)
        # Different order — should still hit the same key
        cached = await cache.get_search("q", ["c1", "c2"])
        assert cached == results

    @pytest.mark.asyncio
    async def test_different_corpus_is_miss(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "x"}]
        await cache.set_search("q", ["corp1"], results)
        assert await cache.get_search("q", ["corp2"]) is None

    @pytest.mark.asyncio
    async def test_nx_does_not_overwrite(self, cache: RedisCache) -> None:
        first = [{"chunk_id": "first"}]
        second = [{"chunk_id": "second"}]
        await cache.set_search("q", ["corp1"], first)
        await cache.set_search("q", ["corp1"], second)  # NX — should not overwrite
        cached = await cache.get_search("q", ["corp1"])
        assert cached == first


# ── Document fingerprint cache ────────────────────────────────────────────────

class TestFingerprintCache:
    @pytest.mark.asyncio
    async def test_unknown_hash_returns_false(self, cache: RedisCache) -> None:
        assert await cache.get_fingerprint("unknownhash") is False

    @pytest.mark.asyncio
    async def test_set_then_get(self, cache: RedisCache) -> None:
        await cache.set_fingerprint("abc123hash")
        assert await cache.get_fingerprint("abc123hash") is True

    @pytest.mark.asyncio
    async def test_delete_removes_entry(self, cache: RedisCache) -> None:
        await cache.set_fingerprint("deleteme")
        await cache.delete_fingerprint("deleteme")
        assert await cache.get_fingerprint("deleteme") is False

    @pytest.mark.asyncio
    async def test_no_client_returns_false(self) -> None:
        c = make_cache()
        assert await c.get_fingerprint("hash") is False


# ── Graceful degradation (no client) ─────────────────────────────────────────

class TestGracefulDegradation:
    """All methods must be no-ops or return None/False when client is not connected.
    None of them should raise."""

    @pytest.mark.asyncio
    async def test_invalidate_corpus_no_client(self) -> None:
        c = make_cache()
        count = await c.invalidate_corpus("corp1", "tenant1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_health_no_client(self) -> None:
        c = make_cache()
        assert await c.get_health("postgres") is None

    @pytest.mark.asyncio
    async def test_set_health_no_client(self) -> None:
        c = make_cache()
        await c.set_health("postgres", {"status": "ok"})  # must not raise
