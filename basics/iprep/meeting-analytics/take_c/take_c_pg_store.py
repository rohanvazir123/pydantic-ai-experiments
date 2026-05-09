"""
PostgreSQL persistence for Take C semantic clustering.


Tables added to meeting_analytics schema:
  semantic_clusters        cluster_id, theme_title, audience, rationale, phrase_count
  semantic_phrases         canonical text + vector(768) embedding + tsvector index
  semantic_meeting_themes  meeting_id, cluster_id, is_primary, call_type, sentiment

Hybrid search (pgvector + tsvector + RRF):
  semantic_search_phrases()      cosine similarity on phrase embeddings (IVFFlat)
  text_search_phrases()          tsvector plainto_tsquery on canonical text (GIN)
  hybrid_search_phrases()        RRF merge of both signals (k=60, same as RAG store)

Insight queries (for dashboard / slide deck):
  insight_theme_sentiment()        avg sentiment score by theme
  insight_churn_by_theme()         churn_signal count per theme (joins key_moments)
  insight_call_type_theme_matrix() call_type x theme meeting counts
  insight_feature_gap_themes()     feature_gap signal count per theme

Connection: reads DATABASE_URL first, then falls back to PG_HOST/PORT/USER/PASSWORD/DATABASE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

SCHEMA = "meeting_analytics"

# IVFFlat lists: sqrt(~343 phrases) ≈ 18.  Kept small; same index type as RAG store.
_IVFFLAT_LISTS = 15

# Embedding dimension — must match the model used in take_c_semantic_clustering.py
EMBEDDING_DIMENSION = 768


# ---------------------------------------------------------------------------
# Connection helpers  (same env-var pattern as generate_rule_based_taxonomy.py)
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    script_dir = Path(__file__).resolve().parent
    for env_file in (script_dir.parents[2] / ".env", script_dir / ".env"):
        if not env_file.exists():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv()


def _build_dsn() -> str:
    """Prefer DATABASE_URL; fall back to individual PG_* vars (Take A pattern)."""
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "postgres")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PhraseSearchResult:
    canonical: str
    cluster_id: int
    theme_title: str
    score: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SemanticClusterStore:
    """
    Hybrid pgvector + tsvector store for Take C semantic phrases.

    Follows the same pool + register_vector + IVFFlat/GIN + RRF pattern
    as rag/storage/vector_store/postgres.py.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or _build_dsn()
        self.pool: asyncpg.Pool | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await self._do_initialize()

    async def _do_initialize(self) -> None:
        # Enable pgvector before creating the pool so register_vector works
        # on the very first connection — same guard as RAG postgres.py.
        temp = await asyncpg.connect(self._dsn)
        try:
            await temp.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await temp.close()

        self.pool = await asyncpg.create_pool(
            self._dsn,
            min_size=1,
            max_size=5,
            command_timeout=60,
            init=register_vector,  # registers vector codec once per connection
        )

        async with self.pool.acquire() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            await self._create_tables(conn)
            await self._create_indexes(conn)

        self._initialized = True
        logger.info("SemanticClusterStore initialised — schema %s", SCHEMA)

    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.semantic_clusters (
                cluster_id   INTEGER PRIMARY KEY,
                theme_title  TEXT NOT NULL,
                audience     TEXT NOT NULL,
                rationale    TEXT,
                phrase_count INTEGER DEFAULT 0
            )
        """)

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.semantic_phrases (
                id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                canonical    TEXT NOT NULL UNIQUE,
                aliases      TEXT[] DEFAULT '{{}}',
                cluster_id   INTEGER NOT NULL
                                 REFERENCES {SCHEMA}.semantic_clusters(cluster_id),
                embedding    vector({EMBEDDING_DIMENSION}),
                content_tsv  tsvector
                                 GENERATED ALWAYS AS (to_tsvector('english', canonical)) STORED
            )
        """)

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.semantic_meeting_themes (
                meeting_id       TEXT NOT NULL,
                cluster_id       INTEGER NOT NULL
                                     REFERENCES {SCHEMA}.semantic_clusters(cluster_id),
                is_primary       BOOLEAN NOT NULL DEFAULT false,
                call_type        TEXT,
                call_confidence  TEXT,
                sentiment_score  NUMERIC,
                overall_sentiment TEXT,
                PRIMARY KEY (meeting_id, cluster_id)
            )
        """)

    async def _create_indexes(self, conn: asyncpg.Connection) -> None:
        # IVFFlat on phrase embeddings (cosine) — same pattern as RAG store
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS semantic_phrases_embedding_idx
            ON {SCHEMA}.semantic_phrases
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {_IVFFLAT_LISTS})
        """)

        # GIN on tsvector — full-text search
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS semantic_phrases_tsv_idx
            ON {SCHEMA}.semantic_phrases
            USING GIN (content_tsv)
        """)

        # B-tree on cluster_id for fast grouping queries
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS semantic_phrases_cluster_idx
            ON {SCHEMA}.semantic_phrases (cluster_id)
        """)

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS semantic_meeting_themes_meeting_idx
            ON {SCHEMA}.semantic_meeting_themes (meeting_id)
        """)

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS semantic_meeting_themes_cluster_idx
            ON {SCHEMA}.semantic_meeting_themes (cluster_id)
        """)

    async def reset_semantic_tables(self) -> None:
        """Drop and recreate the three semantic tables (leaves Take A tables intact)."""
        await self.initialize()
        async with self.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.semantic_meeting_themes CASCADE")
            await conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.semantic_phrases CASCADE")
            await conn.execute(f"DROP TABLE IF EXISTS {SCHEMA}.semantic_clusters CASCADE")
            await self._create_tables(conn)
            await self._create_indexes(conn)
        logger.info("Semantic tables reset")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Write helpers  (imported data models are passed as plain dicts to
    # avoid coupling this file to take_c_semantic_clustering's Pydantic models)
    # ------------------------------------------------------------------

    async def save_cluster_labels(self, labels: list[dict[str, Any]]) -> None:
        """
        Persist cluster label dicts.
        Each dict: {cluster_id, theme_title, audience, rationale, phrase_count}
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {SCHEMA}.semantic_clusters
                    (cluster_id, theme_title, audience, rationale, phrase_count)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (cluster_id) DO UPDATE
                    SET theme_title  = EXCLUDED.theme_title,
                        audience     = EXCLUDED.audience,
                        rationale    = EXCLUDED.rationale,
                        phrase_count = EXCLUDED.phrase_count
                """,
                [
                    (
                        lb["cluster_id"],
                        lb["theme_title"],
                        lb["audience"],
                        lb.get("rationale", ""),
                        lb.get("phrase_count", 0),
                    )
                    for lb in labels
                ],
            )
        logger.info("Saved %d cluster labels", len(labels))

    async def save_phrases(self, phrases: list[dict[str, Any]]) -> None:
        """
        Batch-insert phrase dicts.
        Each dict: {canonical, aliases, cluster_id, embedding}
        Follows the executemany pattern from rag/storage/vector_store/postgres.py.
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {SCHEMA}.semantic_phrases
                    (canonical, aliases, cluster_id, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (canonical) DO UPDATE
                    SET aliases    = EXCLUDED.aliases,
                        cluster_id = EXCLUDED.cluster_id,
                        embedding  = EXCLUDED.embedding
                """,
                [
                    (
                        p["canonical"],
                        p.get("aliases", []),
                        p["cluster_id"],
                        p["embedding"],  # list[float] — register_vector handles encoding
                    )
                    for p in phrases
                ],
            )
        logger.info("Saved %d phrases", len(phrases))

    async def save_meeting_themes(self, assignments: list[dict[str, Any]]) -> None:
        """
        Persist meeting theme assignment dicts.
        Each dict: {meeting_id, theme_ids, primary_theme_id,
                    inferred_call_type, call_confidence, sentiment_score, overall_sentiment}
        Expands theme_ids list into one row per (meeting_id, cluster_id).
        """
        await self.initialize()
        rows: list[tuple[Any, ...]] = []
        for a in assignments:
            for cid in a["theme_ids"]:
                rows.append((
                    a["meeting_id"],
                    cid,
                    cid == a["primary_theme_id"],
                    a.get("inferred_call_type", "unknown"),
                    a.get("call_confidence", "low"),
                    a.get("sentiment_score"),
                    a.get("overall_sentiment"),
                ))

        async with self.pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {SCHEMA}.semantic_meeting_themes
                    (meeting_id, cluster_id, is_primary, call_type,
                     call_confidence, sentiment_score, overall_sentiment)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (meeting_id, cluster_id) DO UPDATE
                    SET is_primary        = EXCLUDED.is_primary,
                        call_type         = EXCLUDED.call_type,
                        call_confidence   = EXCLUDED.call_confidence,
                        sentiment_score   = EXCLUDED.sentiment_score,
                        overall_sentiment = EXCLUDED.overall_sentiment
                """,
                rows,
            )
        logger.info("Saved %d meeting-theme rows for %d meetings", len(rows), len(assignments))

    # ------------------------------------------------------------------
    # Hybrid search  (pgvector + tsvector + RRF)
    # Copied from rag/storage/vector_store/postgres.py and adapted for phrases.
    # ------------------------------------------------------------------

    async def semantic_search_phrases(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[PhraseSearchResult]:
        """Cosine similarity on phrase embeddings via IVFFlat index."""
        await self.initialize()
        async with self.pool.acquire() as conn:
            # Increase IVFFlat probes for better recall — same as RAG store
            await conn.execute("SET ivfflat.probes = 10")
            rows = await conn.fetch(
                f"""
                SELECT
                    sp.canonical,
                    sp.cluster_id,
                    sc.theme_title,
                    1 - (sp.embedding <=> $1::vector) AS score
                FROM {SCHEMA}.semantic_phrases sp
                JOIN {SCHEMA}.semantic_clusters sc USING (cluster_id)
                ORDER BY sp.embedding <=> $1::vector
                LIMIT $2
                """,
                query_embedding,
                top_k,
            )
        return [
            PhraseSearchResult(
                canonical=r["canonical"],
                cluster_id=r["cluster_id"],
                theme_title=r["theme_title"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def text_search_phrases(
        self, query: str, top_k: int = 10
    ) -> list[PhraseSearchResult]:
        """tsvector plainto_tsquery full-text search on canonical phrase text."""
        await self.initialize()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    sp.canonical,
                    sp.cluster_id,
                    sc.theme_title,
                    ts_rank(sp.content_tsv, plainto_tsquery('english', $1)) AS score
                FROM {SCHEMA}.semantic_phrases sp
                JOIN {SCHEMA}.semantic_clusters sc USING (cluster_id)
                WHERE sp.content_tsv @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                top_k * 2,  # over-fetch for RRF — same pattern as RAG store
            )
        return [
            PhraseSearchResult(
                canonical=r["canonical"],
                cluster_id=r["cluster_id"],
                theme_title=r["theme_title"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def hybrid_search_phrases(
        self, query: str, query_embedding: list[float], top_k: int = 10
    ) -> list[PhraseSearchResult]:
        """
        Merge semantic + text results with Reciprocal Rank Fusion (k=60).
        Same RRF algorithm as rag/storage/vector_store/postgres.py.
        """
        semantic, text = await asyncio.gather(
            self.semantic_search_phrases(query_embedding, top_k * 2),
            self.text_search_phrases(query, top_k * 2),
            return_exceptions=True,
        )
        if isinstance(semantic, Exception):
            logger.warning("Semantic search error: %s", semantic)
            semantic = []
        if isinstance(text, Exception):
            logger.warning("Text search error: %s", text)
            text = []

        return self._reciprocal_rank_fusion([semantic, text], k=60)[:top_k]

    def _reciprocal_rank_fusion(
        self, result_lists: list[list[PhraseSearchResult]], k: int = 60
    ) -> list[PhraseSearchResult]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.
        Copied verbatim from rag/storage/vector_store/postgres.py and adapted
        for PhraseSearchResult (keyed by canonical phrase text).
        """
        rrf_scores: dict[str, float] = {}
        phrase_map: dict[str, PhraseSearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                key = result.canonical
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in phrase_map:
                    phrase_map[key] = result

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            PhraseSearchResult(
                canonical=phrase_map[key].canonical,
                cluster_id=phrase_map[key].cluster_id,
                theme_title=phrase_map[key].theme_title,
                score=score,
            )
            for key, score in merged
        ]

    # ------------------------------------------------------------------
    # Insight queries
    # ------------------------------------------------------------------

    async def insight_theme_sentiment(self) -> list[dict[str, Any]]:
        """
        Average sentiment score per theme (primary assignments only).
        Answers: which themes have the most negative / positive meetings?
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    sc.cluster_id,
                    sc.theme_title,
                    sc.audience,
                    COUNT(smt.meeting_id)           AS meeting_count,
                    ROUND(AVG(smt.sentiment_score)::numeric, 2) AS avg_sentiment,
                    ROUND(MIN(smt.sentiment_score)::numeric, 2) AS min_sentiment,
                    ROUND(MAX(smt.sentiment_score)::numeric, 2) AS max_sentiment
                FROM {SCHEMA}.semantic_clusters sc
                LEFT JOIN {SCHEMA}.semantic_meeting_themes smt
                    ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true
                GROUP BY sc.cluster_id, sc.theme_title, sc.audience
                ORDER BY avg_sentiment ASC NULLS LAST
            """)
        return [dict(r) for r in rows]

    async def insight_churn_by_theme(self) -> list[dict[str, Any]]:
        """
        Churn signal count per theme, joining with Take A key_moments table.
        Answers: which themes carry the highest churn / financial risk?
        Gracefully returns empty list if key_moments table does not exist.
        """
        await self.initialize()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT
                        sc.theme_title,
                        sc.audience,
                        COUNT(DISTINCT smt.meeting_id)       AS meeting_count,
                        COUNT(km.moment_index)               AS churn_signal_count,
                        ROUND(
                            COUNT(km.moment_index)::numeric
                            / NULLIF(COUNT(DISTINCT smt.meeting_id), 0),
                            2
                        )                                    AS churn_per_meeting
                    FROM {SCHEMA}.semantic_clusters sc
                    JOIN {SCHEMA}.semantic_meeting_themes smt
                        ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true
                    LEFT JOIN {SCHEMA}.key_moments km
                        ON smt.meeting_id = km.meeting_id
                        AND km.moment_type = 'churn_signal'
                    GROUP BY sc.theme_title, sc.audience
                    ORDER BY churn_signal_count DESC
                """)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("insight_churn_by_theme: %s (Take A tables may not exist)", exc)
            return []

    async def insight_call_type_theme_matrix(self) -> list[dict[str, Any]]:
        """
        Meeting count for every (call_type, theme_title) pair (primary assignments).
        Answers: what do support/external/internal calls talk about most?
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    smt.call_type,
                    sc.theme_title,
                    sc.audience,
                    COUNT(*) AS meeting_count
                FROM {SCHEMA}.semantic_meeting_themes smt
                JOIN {SCHEMA}.semantic_clusters sc USING (cluster_id)
                WHERE smt.is_primary = true
                GROUP BY smt.call_type, sc.theme_title, sc.audience
                ORDER BY smt.call_type, meeting_count DESC
            """)
        return [dict(r) for r in rows]

    async def insight_feature_gap_themes(self) -> list[dict[str, Any]]:
        """
        Feature gap signal count per theme, joining with Take A key_moments.
        Answers: which themes have the most unmet product needs?
        Gracefully returns empty list if key_moments table does not exist.
        """
        await self.initialize()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT
                        sc.theme_title,
                        sc.audience,
                        COUNT(DISTINCT smt.meeting_id)  AS meeting_count,
                        COUNT(km.moment_index)          AS feature_gap_count
                    FROM {SCHEMA}.semantic_clusters sc
                    JOIN {SCHEMA}.semantic_meeting_themes smt
                        ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true
                    LEFT JOIN {SCHEMA}.key_moments km
                        ON smt.meeting_id = km.meeting_id
                        AND km.moment_type = 'feature_gap'
                    GROUP BY sc.theme_title, sc.audience
                    ORDER BY feature_gap_count DESC
                """)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("insight_feature_gap_themes: %s (Take A tables may not exist)", exc)
            return []

    async def insight_sentiment_distribution_by_theme(self) -> list[dict[str, Any]]:
        """
        Count of each overall_sentiment category per theme (primary assignments).
        Answers: for a given theme, how many meetings are positive vs negative?
        e.g. Customer Retention: 8 mixed-negative, 4 negative, 2 neutral, 2 mixed-positive
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    sc.theme_title,
                    sc.audience,
                    smt.overall_sentiment,
                    COUNT(*)                                        AS meeting_count,
                    ROUND(AVG(smt.sentiment_score)::numeric, 2)    AS avg_score
                FROM {SCHEMA}.semantic_meeting_themes smt
                JOIN {SCHEMA}.semantic_clusters sc USING (cluster_id)
                WHERE smt.is_primary = true
                  AND smt.overall_sentiment IS NOT NULL
                GROUP BY sc.theme_title, sc.audience, smt.overall_sentiment
                ORDER BY sc.theme_title, meeting_count DESC
            """)
        return [dict(r) for r in rows]

    async def insight_signal_counts_by_theme(self) -> list[dict[str, Any]]:
        """
        All key-moment signal type counts pivoted per theme (primary assignments).
        Answers: for each theme, how many churn signals / concerns / feature gaps /
                 praise moments / pricing offers / technical issues?
        Requires Take A key_moments table. Returns empty list if not available.
        """
        await self.initialize()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT
                        sc.theme_title,
                        sc.audience,
                        COUNT(DISTINCT smt.meeting_id)                              AS meeting_count,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'churn_signal')    AS churn_signal,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'concern')         AS concern,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'feature_gap')     AS feature_gap,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'technical_issue') AS technical_issue,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'praise')          AS praise,
                        COUNT(km.moment_index) FILTER (WHERE km.moment_type = 'pricing_offer')   AS pricing_offer
                    FROM {SCHEMA}.semantic_clusters sc
                    JOIN {SCHEMA}.semantic_meeting_themes smt
                        ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true
                    LEFT JOIN {SCHEMA}.key_moments km USING (meeting_id)
                    GROUP BY sc.theme_title, sc.audience
                    ORDER BY churn_signal DESC, concern DESC
                """)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("insight_signal_counts_by_theme: %s (Take A tables may not exist)", exc)
            return []

    async def insight_theme_cooccurrence(self, top_n: int = 15) -> list[dict[str, Any]]:
        """
        Which theme pairs appear together most often in the same meeting?
        Answers: e.g. "Customer Retention" + "Incidents & Reliability" co-occurring
                 frequently means outages are driving churn — an actionable finding.
        """
        await self.initialize()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    sc1.theme_title  AS theme_a,
                    sc2.theme_title  AS theme_b,
                    COUNT(*)         AS co_occurrence_count
                FROM {SCHEMA}.semantic_meeting_themes smt1
                JOIN {SCHEMA}.semantic_meeting_themes smt2
                    ON  smt1.meeting_id  = smt2.meeting_id
                    AND smt1.cluster_id  < smt2.cluster_id
                JOIN {SCHEMA}.semantic_clusters sc1 ON smt1.cluster_id = sc1.cluster_id
                JOIN {SCHEMA}.semantic_clusters sc2 ON smt2.cluster_id = sc2.cluster_id
                GROUP BY sc1.theme_title, sc2.theme_title
                ORDER BY co_occurrence_count DESC
                LIMIT $1
            """, top_n)
        return [dict(r) for r in rows]

    async def insight_high_risk_meetings(self, sentiment_threshold: float = 3.0) -> list[dict[str, Any]]:
        """
        Meetings with churn signals AND low sentiment — the highest financial risk.
        Answers: which specific meetings should leadership follow up on immediately?
        Requires Take A key_moments table. Returns empty list if not available.
        """
        await self.initialize()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT
                        smt.meeting_id,
                        sc.theme_title,
                        smt.call_type,
                        smt.sentiment_score,
                        smt.overall_sentiment,
                        COUNT(km.moment_index)  AS churn_signals
                    FROM {SCHEMA}.semantic_meeting_themes smt
                    JOIN {SCHEMA}.semantic_clusters sc
                        ON smt.cluster_id = sc.cluster_id AND smt.is_primary = true
                    JOIN {SCHEMA}.key_moments km
                        ON smt.meeting_id = km.meeting_id
                        AND km.moment_type = 'churn_signal'
                    WHERE smt.sentiment_score < $1
                    GROUP BY smt.meeting_id, sc.theme_title, smt.call_type,
                             smt.sentiment_score, smt.overall_sentiment
                    ORDER BY churn_signals DESC, smt.sentiment_score ASC
                """, sentiment_threshold)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("insight_high_risk_meetings: %s (Take A tables may not exist)", exc)
            return []

    async def row_counts(self) -> dict[str, int]:
        """Sanity-check row counts for the three semantic tables."""
        await self.initialize()
        async with self.pool.acquire() as conn:
            counts: dict[str, int] = {}
            for table in ("semantic_clusters", "semantic_phrases", "semantic_meeting_themes"):
                row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {SCHEMA}.{table}")
                counts[table] = int(row["n"])
        return counts
