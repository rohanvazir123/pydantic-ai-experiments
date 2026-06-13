"""PostgreSQL hybrid vector store.

Manages the `documents` and `chunks` tables using pgvector HNSW (cosine ANN)
and tsvector GIN (BM25 full-text). All queries are scoped to a single
corpus_id + tenant_id pair via Row-Level Security.

RLS enforcement: caller must run
    SET LOCAL app.tenant_id = '<tenant_id>'
before every transaction. This is handled by _conn() automatically.

Hybrid search uses Reciprocal Rank Fusion (k=60) to combine the two legs:
    score = Σ  1 / (60 + rank_i)  for each search leg that returned the row
"""

import json
import logging
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Any, cast

import asyncpg
from pgvector.asyncpg import register_vector

from knowledge.config.settings import Settings, load_settings
from knowledge.store.cache import RedisCache

logger = logging.getLogger(__name__)

RRF_K: int = 60
OVERFETCH_FACTOR: int = 3   # fetch k x OVERFETCH_FACTOR before reranking


class PostgresHybridStore:
    """Async hybrid vector + full-text search store backed by PostgreSQL + pgvector."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache: RedisCache | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._cache = cache
        self._pool: asyncpg.Pool | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        async def _init(conn: asyncpg.Connection) -> None:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
                format="text",
            )
            await conn.set_type_codec(
                "json",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
                format="text",
            )

        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=self._settings.db_query_timeout_s,
            init=_init,
        )
        logger.info("PostgresHybridStore initialised")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _conn(self, tenant_id: str) -> Any:
        """Acquire a connection inside an explicit transaction with RLS context.

        SET LOCAL only persists for the duration of the current transaction.
        Without an explicit BEGIN the implicit per-statement transactions would
        reset app.tenant_id before each subsequent statement, breaking RLS.
        """
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"SET LOCAL app.tenant_id = '{tenant_id}'"
                )
                yield conn

    # ── Documents ─────────────────────────────────────────────────────────────

    async def save_document(
        self,
        title: str,
        source: str,
        corpus_id: str,
        tenant_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert a document record; return its UUID."""
        doc_id = str(_uuid.uuid4())
        async with self._conn(tenant_id) as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (id, title, source, content, corpus_id, tenant_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (source) DO UPDATE
                    SET title      = EXCLUDED.title,
                        content    = EXCLUDED.content,
                        metadata   = EXCLUDED.metadata
                RETURNING id
                """,
                doc_id, title, source, content, corpus_id, tenant_id,
                metadata or {},
            )
        return str(row["id"])

    async def get_document_hash(self, source: str, corpus_id: str, tenant_id: str) -> str | None:
        """Return the content_hash from document metadata, or None if not found."""
        async with self._conn(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM documents WHERE source = $1 AND corpus_id = $2",
                source, corpus_id,
            )
        if row is None:
            return None
        return cast("str | None", row["metadata"].get("content_hash"))

    async def delete_document_and_chunks(self, source: str, corpus_id: str, tenant_id: str) -> None:
        """Delete a document and all its chunks (cascades via FK)."""
        async with self._conn(tenant_id) as conn:
            await conn.execute(
                "DELETE FROM documents WHERE source = $1 AND corpus_id = $2",
                source, corpus_id,
            )

    async def get_all_document_sources(self, corpus_id: str, tenant_id: str) -> list[str]:
        """Return all document source paths for a corpus."""
        async with self._conn(tenant_id) as conn:
            rows = await conn.fetch(
                "SELECT source FROM documents WHERE corpus_id = $1",
                corpus_id,
            )
        return [r["source"] for r in rows]

    # ── Chunks ────────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_id: str,
        corpus_id: str,
        tenant_id: str,
    ) -> None:
        """Bulk-upsert chunks for a document.

        Each chunk dict must have: content, embedding, chunk_index, token_count, metadata.
        Uses executemany for efficiency.
        """
        if not chunks:
            return
        records = [
            (
                str(_uuid.uuid4()),
                document_id,
                c["content"],
                c["embedding"],
                c["chunk_index"],
                c.get("token_count"),
                corpus_id,
                tenant_id,
                c.get("metadata", {}),
            )
            for c in chunks
        ]
        async with self._conn(tenant_id) as conn:
            await conn.executemany(
                """
                INSERT INTO chunks
                    (id, document_id, content, embedding, chunk_index, token_count,
                     corpus_id, tenant_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
                """,
                records,
            )
        logger.debug("Upserted %d chunks for document %s", len(chunks), document_id)

    async def delete_chunks_for_document(self, document_id: str, tenant_id: str) -> None:
        async with self._conn(tenant_id) as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1", document_id
            )

    # ── Search ────────────────────────────────────────────────────────────────

    async def semantic_search(
        self,
        query_embedding: list[float],
        corpus_id: str,
        tenant_id: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """HNSW cosine ANN search. Returns up to k results ordered by cosine distance."""
        async with self._conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND corpus_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                query_embedding, corpus_id, k,
            )
        return [dict(r) for r in rows]

    async def text_search(
        self,
        query: str,
        corpus_id: str,
        tenant_id: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """tsvector GIN BM25 search via ts_rank."""
        async with self._conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, metadata,
                       ts_rank(content_tsv, websearch_to_tsquery('english', $1)) AS score
                FROM chunks
                WHERE content_tsv @@ websearch_to_tsquery('english', $1)
                  AND corpus_id = $2
                ORDER BY score DESC
                LIMIT $3
                """,
                query, corpus_id, k,
            )
        return [dict(r) for r in rows]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        corpus_id: str,
        tenant_id: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + cosine ANN search fused with RRF (k=60).

        Fetches k x OVERFETCH_FACTOR candidates from each leg, fuses with RRF,
        and returns the top k. raw_score is the RRF score; raw_score_type = 'rrf'.
        confidence is set to None here — populated by the CrossEncoder reranker.
        """
        fetch = k * OVERFETCH_FACTOR

        async with self._conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                WITH
                text_ranked AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank(content_tsv,
                                   websearch_to_tsquery('english', $1)) DESC
                           ) AS rn
                    FROM chunks
                    WHERE content_tsv @@ websearch_to_tsquery('english', $1)
                      AND corpus_id = $3
                ),
                vec_ranked AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY embedding <=> $2::vector ASC
                           ) AS rn
                    FROM chunks
                    WHERE embedding IS NOT NULL
                      AND corpus_id = $3
                    LIMIT $4
                ),
                rrf AS (
                    SELECT
                        COALESCE(t.id, v.id) AS id,
                        COALESCE(1.0 / ($5 + t.rn), 0) +
                        COALESCE(1.0 / ($5 + v.rn), 0) AS score
                    FROM text_ranked t
                    FULL OUTER JOIN vec_ranked v ON t.id = v.id
                )
                SELECT
                    c.id, c.document_id, c.content, c.metadata,
                    r.score AS raw_score
                FROM rrf r
                JOIN chunks c ON c.id = r.id
                ORDER BY r.score DESC
                LIMIT $6
                """,
                query, query_embedding, corpus_id,
                fetch, RRF_K, k,
            )

        return [
            {
                **dict(r),
                "raw_score_type": "rrf",
                "confidence": None,    # set by CrossEncoder reranker
            }
            for r in rows
        ]

    async def get_chunk_count(self, corpus_id: str, tenant_id: str) -> int:
        async with self._conn(tenant_id) as conn:
            return cast("int", await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE corpus_id = $1", corpus_id
            ))

    async def truncate_corpus(self, corpus_id: str, tenant_id: str) -> None:
        """Delete all documents and chunks for a corpus."""
        async with self._conn(tenant_id) as conn:
            await conn.execute(
                "DELETE FROM documents WHERE corpus_id = $1", corpus_id
            )
