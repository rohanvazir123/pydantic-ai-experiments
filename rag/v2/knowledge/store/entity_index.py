# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Entity shadow index for Apache AGE entities.

AGE does not support tsvector GIN or pgvector HNSW indexes — every CONTAINS
scan in AGE is O(n). This module maintains a `kg_entity_index` shadow table
in the main PostgreSQL database so both BM25 and cosine ANN are available.

Hybrid search uses RRF (k=60) to combine:
    - BM25 rank from  name_tsv @@ websearch_to_tsquery(...)
    - Cosine rank from  embedding <=> query_vector

Ported from kg/entity_index.py with these v2 additions:
    - corpus_id + tenant_id columns for scoped queries and deletes
    - HNSW index (was IVFFlat in v1)
"""

import logging

import asyncpg
from pgvector.asyncpg import register_vector

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

RRF_K: int = 60

_SQL_UPSERT = """
INSERT INTO kg_entity_index (age_uuid, name, label, corpus_id, tenant_id, document_id, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (age_uuid)
DO UPDATE SET
    name        = EXCLUDED.name,
    label       = EXCLUDED.label,
    corpus_id   = EXCLUDED.corpus_id,
    tenant_id   = EXCLUDED.tenant_id,
    document_id = EXCLUDED.document_id,
    embedding   = EXCLUDED.embedding
"""

_SQL_HYBRID = """
WITH
text_ranked AS (
    SELECT age_uuid,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(name_tsv, websearch_to_tsquery('english', $1)) DESC
           ) AS rn
    FROM kg_entity_index
    WHERE name_tsv @@ websearch_to_tsquery('english', $1)
      AND corpus_id = $3
      AND tenant_id = $4
    {label_filter}
),
vec_ranked AS (
    SELECT age_uuid,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector ASC) AS rn
    FROM kg_entity_index
    WHERE embedding IS NOT NULL
      AND corpus_id = $3
      AND tenant_id = $4
    {label_filter}
    LIMIT 60
),
rrf AS (
    SELECT COALESCE(t.age_uuid, v.age_uuid) AS age_uuid,
           (COALESCE(1.0 / ({k} + t.rn), 0) + COALESCE(1.0 / ({k} + v.rn), 0)) AS score
    FROM text_ranked t
    FULL OUTER JOIN vec_ranked v ON t.age_uuid = v.age_uuid
)
SELECT r.age_uuid, e.name, e.label, e.document_id, r.score
FROM rrf r
JOIN kg_entity_index e ON e.age_uuid = r.age_uuid
ORDER BY r.score DESC
LIMIT $5
"""

_SQL_VEC_ONLY = """
SELECT age_uuid, name, label, document_id,
       1 - (embedding <=> $1::vector) AS score
FROM kg_entity_index
WHERE embedding IS NOT NULL
  AND corpus_id = $2
  AND tenant_id = $3
{label_filter}
ORDER BY embedding <=> $1::vector
LIMIT $4
"""


class EntityIndex:
    """Hybrid BM25 + cosine ANN index for knowledge graph entities.

    Backed by kg_entity_index in the main PostgreSQL DB (DATABASE_URL).
    Used by AgeGraphStore.search_entities() as a fast alternative to the
    O(n) CONTAINS scan that AGE would otherwise require.
    """

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
        logger.info("EntityIndex initialised")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert(
        self,
        age_uuid: str,
        name: str,
        label: str,
        corpus_id: str,
        tenant_id: str,
        document_id: str = "",
        embedding: list[float] | None = None,
    ) -> None:
        """Insert or update one entity row."""
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(
                _SQL_UPSERT,
                age_uuid, name, label, corpus_id, tenant_id, document_id, embedding,
            )

    async def upsert_batch(
        self,
        entities: list[dict],
        corpus_id: str,
        tenant_id: str,
    ) -> None:
        """Bulk-upsert a list of entity dicts (age_uuid, name, label, document_id, embedding)."""
        assert self._pool, "Call initialize() first"
        records = [
            (
                e["age_uuid"], e["name"], e["label"],
                corpus_id, tenant_id,
                e.get("document_id", ""),
                e.get("embedding"),
            )
            for e in entities
        ]
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            await conn.executemany(_SQL_UPSERT, records)

    async def delete_for_document(self, document_id: str, corpus_id: str, tenant_id: str) -> None:
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM kg_entity_index WHERE document_id = $1 AND corpus_id = $2 AND tenant_id = $3",
                document_id, corpus_id, tenant_id,
            )

    async def delete_for_corpus(self, corpus_id: str, tenant_id: str) -> None:
        assert self._pool, "Call initialize() first"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM kg_entity_index WHERE corpus_id = $1 AND tenant_id = $2",
                corpus_id, tenant_id,
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        corpus_id: str,
        tenant_id: str,
        label: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """RRF hybrid BM25 + cosine search.

        Falls back to vector-only if tsvector has no matches
        (e.g. query is all stopwords).
        """
        assert self._pool, "Call initialize() first"
        label_clause = f"AND label = '{label}'" if label else ""

        sql = _SQL_HYBRID.format(label_filter=label_clause, k=RRF_K)

        async with self._pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(sql, query, query_embedding, corpus_id, tenant_id, limit)

        if rows:
            return [
                {
                    "age_uuid":    r["age_uuid"],
                    "name":        r["name"],
                    "label":       r["label"],
                    "document_id": r["document_id"],
                    "score":       float(r["score"]),
                }
                for r in rows
            ]

        # tsvector had no hits — fall back to vector-only
        sql_vec = _SQL_VEC_ONLY.format(label_filter=label_clause)
        async with self._pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(sql_vec, query_embedding, corpus_id, tenant_id, limit)

        return [
            {
                "age_uuid":    r["age_uuid"],
                "name":        r["name"],
                "label":       r["label"],
                "document_id": r["document_id"],
                "score":       float(r["score"]),
            }
            for r in rows
        ]
