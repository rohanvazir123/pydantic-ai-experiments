"""Redis L2 cache store.

Manages three distinct cache namespaces:
  - Embedding cache      cache:embed:{sha256(text)}          24h  msgpack vector
  - Search result cache  cache:search:{sha256(query+…)}      5min msgpack list
  - Document fingerprint cache:doc_fingerprint:{sha256(file)} 7d  string "1"

All keys are prefixed with the namespace so a single SCAN pattern can
identify and bulk-delete any namespace.

Serialisation: msgpack (faster than JSON for binary float arrays).
"""

import hashlib
import logging
from typing import Any

import msgpack
import redis.asyncio as aioredis

from knowledge.config.settings import Settings, load_settings
from knowledge.retrieval.normalizer import normalize_query

logger = logging.getLogger(__name__)

# TTL constants (seconds)
_TTL_EMBED: int = 86_400        # 24 h
_TTL_SEARCH: int = 300          # 5 min
_TTL_FINGERPRINT: int = 604_800 # 7 d
_TTL_HEALTH: int = 30           # 30 s


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _embed_key(text: str) -> str:
    return f”cache:embed:{_sha256(text)}”


def _search_key(query: str, corpus_ids: list[str], filters: dict[str, Any] | None) -> str:
    parts = normalize_query(query) + “|” + “,”.join(sorted(corpus_ids))
    if filters:
        parts += “|” + str(sorted(filters.items()))
    return f”cache:search:{_sha256(parts)}”


def _fingerprint_key(content_hash: str) -> str:
    return f"cache:doc_fingerprint:{content_hash}"


def _health_key(service: str) -> str:
    return f"cache:health:{service}"


class RedisCache:
    """L2 distributed cache backed by Redis.

    All methods are async and safe to call concurrently. None of them raise
    on a Redis failure — they log a warning and return None/False so the
    caller can fall through to the next layer.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            self._settings.redis_url,
            max_connections=self._settings.redis_max_connections,
            decode_responses=False,  # we handle bytes ourselves via msgpack
        )
        await self._client.ping()
        logger.info("RedisCache connected to %s", self._settings.redis_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()  # type: ignore[attr-defined]
            self._client = None

    # ── Embedding cache ───────────────────────────────────────────────────────

    async def get_embedding(self, text: str) -> list[float] | None:
        """Return cached embedding vector or None on miss."""
        if not self._client:
            return None
        try:
            raw = await self._client.get(_embed_key(text))
            if raw is None:
                return None
            return list(msgpack.unpackb(raw, raw=False))
        except Exception as exc:
            logger.warning("RedisCache.get_embedding failed: %s", exc)
            return None

    async def set_embedding(self, text: str, vector: list[float]) -> None:
        """Cache an embedding vector."""
        if not self._client:
            return
        try:
            await self._client.set(
                _embed_key(text),
                msgpack.packb(vector, use_bin_type=True),
                ex=_TTL_EMBED,
            )
        except Exception as exc:
            logger.warning("RedisCache.set_embedding failed: %s", exc)

    # ── Search result cache ───────────────────────────────────────────────────

    async def get_search(
        self,
        query: str,
        corpus_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return cached search results or None on miss."""
        if not self._client:
            return None
        try:
            raw = await self._client.get(_search_key(query, corpus_ids, filters))
            if raw is None:
                return None
            return list(msgpack.unpackb(raw, raw=False))
        except Exception as exc:
            logger.warning("RedisCache.get_search failed: %s", exc)
            return None

    async def set_search(
        self,
        query: str,
        corpus_ids: list[str],
        results: list[dict[str, Any]],
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Cache search results. Uses SET NX to avoid overwriting a concurrent write."""
        if not self._client:
            return
        try:
            await self._client.set(
                _search_key(query, corpus_ids, filters),
                msgpack.packb(results, use_bin_type=True),
                ex=_TTL_SEARCH,
                nx=True,  # don't overwrite if two requests race
            )
        except Exception as exc:
            logger.warning("RedisCache.set_search failed: %s", exc)

    async def invalidate_corpus(self, corpus_id: str, tenant_id: str) -> int:
        """Delete all search cache entries for a corpus.

        Uses SCAN + pipeline DELETE — scoped, not a full flush.
        Returns the number of keys deleted.
        """
        if not self._client:
            return 0
        deleted = 0
        try:
            # We can't encode corpus_id into the hash after the fact, so we
            # scan all search keys and delete them all for this corpus/tenant.
            # This is intentionally conservative — better to over-invalidate.
            async for key in self._client.scan_iter("cache:search:*", count=100):
                await self._client.delete(key)
                deleted += 1
            logger.info(
                "RedisCache.invalidate_corpus: deleted %d search keys (corpus=%s)",
                deleted, corpus_id,
            )
        except Exception as exc:
            logger.warning("RedisCache.invalidate_corpus failed: %s", exc)
        return deleted

    # ── Document fingerprint cache ────────────────────────────────────────────

    async def get_fingerprint(self, content_hash: str) -> bool:
        """Return True if this document hash is in the cache (already ingested)."""
        if not self._client:
            return False
        try:
            return await self._client.exists(_fingerprint_key(content_hash)) == 1
        except Exception as exc:
            logger.warning("RedisCache.get_fingerprint failed: %s", exc)
            return False

    async def set_fingerprint(self, content_hash: str) -> None:
        """Mark a document hash as ingested."""
        if not self._client:
            return
        try:
            await self._client.set(
                _fingerprint_key(content_hash), "1", ex=_TTL_FINGERPRINT
            )
        except Exception as exc:
            logger.warning("RedisCache.set_fingerprint failed: %s", exc)

    async def delete_fingerprint(self, content_hash: str) -> None:
        """Remove a fingerprint (called when a document is deleted or replaced)."""
        if not self._client:
            return
        try:
            await self._client.delete(_fingerprint_key(content_hash))
        except Exception as exc:
            logger.warning("RedisCache.delete_fingerprint failed: %s", exc)

    # ── Health cache ──────────────────────────────────────────────────────────

    async def get_health(self, service: str) -> dict[str, Any] | None:
        if not self._client:
            return None
        try:
            import json
            raw = await self._client.get(_health_key(service))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("RedisCache.get_health failed: %s", exc)
            return None

    async def set_health(self, service: str, data: dict[str, Any]) -> None:
        if not self._client:
            return
        try:
            import json
            await self._client.set(
                _health_key(service), json.dumps(data), ex=_TTL_HEALTH
            )
        except Exception as exc:
            logger.warning("RedisCache.set_health failed: %s", exc)
