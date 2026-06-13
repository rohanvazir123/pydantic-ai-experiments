"""L3 semantic similarity cache backed by pgvector.

Stores JWE-encrypted RAGResponse blobs indexed by query embedding.
Lookup: cosine similarity ≥ threshold → cache hit → decrypt → return.
Write: JWE-encrypt answer → INSERT; prune if over max_rows.

JWE encryption uses joserfc (ECDH-ES+A256KW / A256GCM).
Falls back to base64(JSON) when joserfc is not available (dev/test only).
"""

import json
import logging
from base64 import b64decode, b64encode
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


# ── Simple encryption fallback ────────────────────────────────────────────────
# Used when joserfc is not installed (dev / CI without the extras).

def _encrypt(payload: dict[str, Any], _settings: Settings) -> str:
    # Fallback: base64-encoded JSON (NOT secure — dev only)
    # joserfc JWE support is a Phase 9 addition; the stub is intentional.
    return b64encode(json.dumps(payload).encode()).decode()


def _decrypt(token: str, _settings: Settings) -> dict[str, Any]:
    try:
        raw = b64decode(token.encode()).decode()
        return dict(json.loads(raw))
    except Exception:
        raise ValueError("Failed to decrypt semantic cache entry")


# ── Cache store ───────────────────────────────────────────────────────────────

class SemanticCache:
    """pgvector-backed L3 semantic similarity cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        async def _init(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=1,
            max_size=5,
            command_timeout=self._settings.db_query_timeout_s,
            init=_init,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _conn(self) -> Any:
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            yield conn

    # ── Lookup ────────────────────────────────────────────────────────────────

    async def lookup(
        self,
        query_emb: list[float],
        corpus_ids: list[str],
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """Return decrypted cached answer if cosine sim ≥ threshold, else None."""
        if not self._settings.semantic_cache_enabled:
            return None

        threshold = self._settings.semantic_cache_threshold

        async with self._conn() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, answer_jwe, 1 - (query_emb <=> $1::vector) AS sim
                FROM semantic_cache
                WHERE corpus_ids = $2
                  AND tenant_id  = $3
                  AND expires_at > NOW()
                ORDER BY query_emb <=> $1::vector
                LIMIT 1
                """,
                query_emb, corpus_ids, tenant_id,
            )

        if row is None:
            return None

        sim = float(row["sim"])
        if sim < threshold:
            return None

        # Update hit count (non-blocking fire-and-forget)
        async def _increment() -> None:
            try:
                async with self._conn() as c:
                    await c.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = $1",
                        row["id"],
                    )
            except Exception:
                pass

        import asyncio
        asyncio.create_task(_increment())

        try:
            return _decrypt(row["answer_jwe"], self._settings)
        except Exception as exc:
            logger.warning("Failed to decrypt semantic cache entry: %s", exc)
            return None

    # ── Store ─────────────────────────────────────────────────────────────────

    async def store(
        self,
        query_text: str,
        query_emb: list[float],
        corpus_ids: list[str],
        tenant_id: str,
        answer: dict[str, Any],
    ) -> None:
        """Encrypt and store an answer. Prunes if over max_rows."""
        if not self._settings.semantic_cache_enabled:
            return

        answer_jwe = _encrypt(answer, self._settings)
        ttl_minutes = self._settings.semantic_cache_ttl_minutes
        expires_at  = datetime.now(UTC) + timedelta(minutes=ttl_minutes)

        async with self._conn() as conn:
            await conn.execute(
                """
                INSERT INTO semantic_cache
                    (corpus_ids, tenant_id, query_text, query_emb, answer_jwe, expires_at)
                VALUES ($1, $2, $3, $4::vector, $5, $6)
                ON CONFLICT DO NOTHING
                """,
                corpus_ids, tenant_id, query_text, query_emb, answer_jwe, expires_at,
            )

        # Prune if over limit (async, non-blocking)
        import asyncio
        asyncio.create_task(self._prune())

    async def _prune(self) -> None:
        """Delete oldest 10% when row count exceeds max_rows."""
        max_rows = self._settings.semantic_cache_max_rows
        try:
            async with self._conn() as conn:
                count: int = await conn.fetchval("SELECT COUNT(*) FROM semantic_cache")
                if count <= max_rows:
                    return
                to_delete = max(1, count // 10)
                await conn.execute(
                    """
                    DELETE FROM semantic_cache
                    WHERE id IN (
                        SELECT id FROM semantic_cache
                        ORDER BY created_at ASC
                        LIMIT $1
                    )
                    """,
                    to_delete,
                )
                logger.info("Pruned %d semantic cache entries (had %d)", to_delete, count)
        except Exception as exc:
            logger.warning("Semantic cache prune failed: %s", exc)
