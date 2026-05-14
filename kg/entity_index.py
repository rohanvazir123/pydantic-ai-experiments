"""
Shadow table for entity full-text + vector search in main PostgreSQL DB.

AGE does not expose tsvector/GIN or pgvector indexes — every CONTAINS scan
in AGE is O(n) with no index support.  This module maintains a lightweight
mirror table (kg_entity_index) in the main PostgreSQL DB (DATABASE_URL) so
that both tsvector GIN and pgvector IVFFlat are available.

Search uses RRF (k=60) to combine:
  - BM25 rank from  name_tsv @@ websearch_to_tsquery(...)
  - Cosine rank  from  embedding <=> query_vector

Table schema
------------
    age_uuid    TEXT PRIMARY KEY       -- AGE vertex uuid property
    name        TEXT NOT NULL
    label       TEXT NOT NULL          -- vertex label (Party, Contract, …)
    document_id TEXT NOT NULL DEFAULT ''
    name_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', name)) STORED
    embedding   vector(N)              -- N = settings.embedding_dimension

Indexes
-------
    GIN on name_tsv          (full-text)
    B-tree on label           (label filter)
    IVFFlat on embedding      (ANN, created lazily once ≥ 100 rows exist)
"""

import asyncio
import logging

import asyncpg
from pgvector.asyncpg import register_vector

from rag.config.settings import load_settings
from rag.ingestion.embedder import EmbeddingGenerator

logger = logging.getLogger(__name__)

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS kg_entity_index (
    age_uuid    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    label       TEXT NOT NULL,
    document_id TEXT NOT NULL DEFAULT '',
    name_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', name)) STORED,
    embedding   vector({dim})
)
"""

_DDL_GIN = """
CREATE INDEX IF NOT EXISTS kg_entity_index_tsv_gin
ON kg_entity_index USING GIN (name_tsv)
"""

_DDL_LABEL = """
CREATE INDEX IF NOT EXISTS kg_entity_index_label
ON kg_entity_index (label)
"""

_DDL_IVFFLAT = """
CREATE INDEX IF NOT EXISTS kg_entity_index_ivfflat
ON kg_entity_index USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50)
"""

_SQL_UPSERT = """
INSERT INTO kg_entity_index (age_uuid, name, label, document_id, embedding)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (age_uuid)
DO UPDATE SET
    name        = EXCLUDED.name,
    label       = EXCLUDED.label,
    document_id = EXCLUDED.document_id,
    embedding   = EXCLUDED.embedding
"""

_SQL_HYBRID = """
WITH
text_ranked AS (
    SELECT age_uuid,
           ROW_NUMBER() OVER (ORDER BY ts_rank(name_tsv, websearch_to_tsquery('english', $1)) DESC) AS rn
    FROM kg_entity_index
    WHERE name_tsv @@ websearch_to_tsquery('english', $1)
    {label_filter}
),
vec_ranked AS (
    SELECT age_uuid,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $2::vector ASC) AS rn
    FROM kg_entity_index
    WHERE embedding IS NOT NULL
    {label_filter}
    LIMIT 60
),
rrf AS (
    SELECT COALESCE(t.age_uuid, v.age_uuid) AS age_uuid,
           (COALESCE(1.0 / (60.0 + t.rn), 0) + COALESCE(1.0 / (60.0 + v.rn), 0)) AS score
    FROM text_ranked t
    FULL OUTER JOIN vec_ranked v ON t.age_uuid = v.age_uuid
)
SELECT r.age_uuid, e.name, e.label, e.document_id, r.score
FROM rrf r
JOIN kg_entity_index e ON e.age_uuid = r.age_uuid
ORDER BY r.score DESC
LIMIT {limit}
"""


class EntityIndex:
    """
    Hybrid full-text + vector search index for KG entities.

    Backed by kg_entity_index in the main PostgreSQL DB (DATABASE_URL).
    Used by AgeGraphStore.search_entities() as a drop-in replacement for
    the O(n) CONTAINS scan that AGE would otherwise perform.

    Lifecycle:
        index = EntityIndex()
        await index.initialize()        # idempotent — safe to call repeatedly
        await index.upsert(...)         # mirror each AGE vertex write
        results = await index.hybrid_search(query)
        await index.close()
    """

    def __init__(self) -> None:
        self.settings = load_settings()
        self._db_url = self.settings.database_url
        self._dim = self.settings.embedding_dimension
        self.pool: asyncpg.Pool | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._embedder = EmbeddingGenerator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await self._do_initialize()

    async def _do_initialize(self) -> None:
        async def _init_conn(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        self.pool = await asyncpg.create_pool(
            self._db_url,
            min_size=1,
            max_size=5,
            init=_init_conn,
        )
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(_DDL_TABLE.format(dim=self._dim))
            await conn.execute(_DDL_GIN)
            await conn.execute(_DDL_LABEL)
            await self._maybe_ivfflat(conn)

        self._initialized = True
        logger.info("EntityIndex initialized (dim=%d)", self._dim)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        age_uuid: str,
        name: str,
        label: str,
        document_id: str = "",
    ) -> None:
        """Insert or update an entity row including its name embedding."""
        assert self.pool, "Call initialize() first"
        embedding = await self._embedder.embed_query(name)
        async with self.pool.acquire() as conn:
            await conn.execute(_SQL_UPSERT, age_uuid, name, label, document_id, embedding)

    async def delete_for_document(self, document_id: str) -> None:
        """Remove all index rows for a given document_id."""
        assert self.pool, "Call initialize() first"
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM kg_entity_index WHERE document_id = $1", document_id
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query: str,
        label: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        RRF hybrid search combining tsvector BM25 rank and cosine vector rank.

        Returns list of dicts: age_uuid, name, label, document_id, score.
        Falls back to vector-only if the text query has no tsvector matches.
        """
        assert self.pool, "Call initialize() first"
        q_embedding = await self._embedder.embed_query(query)

        params: list = [query, q_embedding]
        if label:
            label_filter = "AND label = $3"
            params.append(label)
        else:
            label_filter = ""

        sql = _SQL_HYBRID.format(label_filter=label_filter, limit=limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        if not rows:
            # tsvector matched nothing — fall back to pure vector search
            rows = await self._vector_only(q_embedding, label, limit)

        return [
            {
                "age_uuid": r["age_uuid"],
                "name": r["name"],
                "label": r["label"],
                "document_id": r["document_id"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    async def _vector_only(
        self,
        embedding: list[float],
        label: str | None,
        limit: int,
    ) -> list[asyncpg.Record]:
        """Pure cosine ANN search — used when text query has no tsvector hits."""
        label_filter = "AND label = $2" if label else ""
        params: list = [embedding]
        if label:
            params.append(label)
        sql = f"""
        SELECT age_uuid, name, label, document_id,
               (1.0 - (embedding <=> $1::vector)) AS score
        FROM kg_entity_index
        WHERE embedding IS NOT NULL
        {label_filter}
        ORDER BY embedding <=> $1::vector ASC
        LIMIT {limit}
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, *params)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _maybe_ivfflat(self, conn: asyncpg.Connection) -> None:
        """Create IVFFlat index once we have ≥ 100 rows (idempotent)."""
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM kg_entity_index")
            if count >= 100:
                await conn.execute(_DDL_IVFFLAT)
        except Exception as exc:
            logger.debug("IVFFlat index skipped: %s", exc)
