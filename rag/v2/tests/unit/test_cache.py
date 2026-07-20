# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Unit tests for knowledge.store.cache.

Uses fakeredis — no live Redis required.
"""

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from knowledge.config.settings import Settings
from knowledge.retrieval.normalizer import normalize_query
from knowledge.store.cache import RedisCache, _embed_key, _fingerprint_key, _search_key


# ── spaCy availability guard ──────────────────────────────────────────────────

def _spacy_model_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


requires_spacy = pytest.mark.skipif(
    not _spacy_model_available(),
    reason="spaCy en_core_web_sm not installed — run: python -m spacy download en_core_web_sm",
)


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


# ── Normalized key invariants ─────────────────────────────────────────────────

class TestNormalizedKeyInvariant:
    """_search_key must return the same hash for queries that normalize identically.

    Two sub-cases:
      - Basic (always): case folding and whitespace collapse
      - spaCy (conditional): lemmatization and punctuation stripping
    """

    def test_case_variants_produce_same_key(self) -> None:
        # "pto policy" vs "PTO Policy" — both lowercase to "pto policy"
        assert _search_key("pto policy", ["c1"], None) == _search_key("PTO Policy", ["c1"], None)

    def test_extra_whitespace_produces_same_key(self) -> None:
        assert _search_key("pto  policy", ["c1"], None) == _search_key("pto policy", ["c1"], None)

    def test_leading_trailing_whitespace(self) -> None:
        assert _search_key("  pto policy  ", ["c1"], None) == _search_key("pto policy", ["c1"], None)

    def test_mixed_case_and_whitespace(self) -> None:
        assert _search_key("  PTO  Policy  ", ["c1"], None) == _search_key("pto policy", ["c1"], None)

    @requires_spacy
    def test_punctuation_stripped(self) -> None:
        # spaCy drops punctuation tokens; trailing "?" must not change the key
        assert _search_key("What is the PTO policy?", ["c1"], None) == \
               _search_key("What is the PTO policy", ["c1"], None)

    @requires_spacy
    def test_lemmatization_singular_plural(self) -> None:
        # "policies" lemmatizes to "policy"
        assert _search_key("PTO policies", ["c1"], None) == _search_key("PTO policy", ["c1"], None)

    @requires_spacy
    def test_lemmatization_verb_form(self) -> None:
        # "serves" → "serve"; key for both sentence variants must match
        # normalize_query("What industries does NeuralFlow serve?")
        # normalize_query("What industry does NeuralFlow serve")
        k1 = normalize_query("What industries does NeuralFlow serve?")
        k2 = normalize_query("What industry does NeuralFlow serve")
        assert k1 == k2

    def test_different_queries_different_keys(self) -> None:
        assert _search_key("pto policy", ["c1"], None) != _search_key("benefits plan", ["c1"], None)

    def test_different_corpus_different_keys(self) -> None:
        assert _search_key("pto policy", ["c1"], None) != _search_key("pto policy", ["c2"], None)


# ── Normalized cache hit (set/get with variant queries) ───────────────────────

class TestNormalizedSearchCacheHit:
    """Storing under one query form should be retrievable with a normalized variant."""

    @pytest.mark.asyncio
    async def test_case_variant_hits_cache(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "abc", "content": "PTO is 15 days", "score": 0.9}]
        await cache.set_search("PTO Policy", ["corp1"], results)
        # Lowercase variant must hit the same entry
        cached = await cache.get_search("pto policy", ["corp1"])
        assert cached == results

    @pytest.mark.asyncio
    async def test_whitespace_variant_hits_cache(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "def", "content": "remote work policy", "score": 0.8}]
        await cache.set_search("remote  work  policy", ["corp1"], results)
        cached = await cache.get_search("remote work policy", ["corp1"])
        assert cached == results

    @requires_spacy
    @pytest.mark.asyncio
    async def test_plural_hits_singular_cached_entry(self, cache: RedisCache) -> None:
        # Store under "PTO policy", retrieve with "PTO policies" — spaCy lemmatizes both to "policy"
        results = [{"chunk_id": "ghi", "content": "PTO accrual rules", "score": 0.85}]
        await cache.set_search("PTO policy", ["corp1"], results)
        cached = await cache.get_search("PTO policies", ["corp1"])
        assert cached == results

    @requires_spacy
    @pytest.mark.asyncio
    async def test_punctuation_variant_hits_cache(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "jkl", "content": "expense form link", "score": 0.7}]
        await cache.set_search("expense report process", ["corp1"], results)
        # Trailing question mark should normalize away
        cached = await cache.get_search("expense report process?", ["corp1"])
        assert cached == results

    @pytest.mark.asyncio
    async def test_different_corpus_is_still_miss(self, cache: RedisCache) -> None:
        results = [{"chunk_id": "mno", "content": "something", "score": 0.6}]
        await cache.set_search("pto policy", ["corp1"], results)
        # Corpus differs — must NOT hit
        assert await cache.get_search("PTO Policy", ["corp2"]) is None
