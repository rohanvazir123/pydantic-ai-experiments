"""
Load Take C outputs (JSON/CSV) into Postgres without re-running the pipeline.

Reads from take_c/outputs/:
  semantic_clusters.json  -> meeting_analytics.semantic_clusters
  phrase_clusters.csv     -> meeting_analytics.semantic_phrases  (embedding=NULL)
  meeting_themes.csv      -> meeting_analytics.semantic_meeting_themes

Usage (from repo root):
  python basics/iprep/meeting-analytics/take_c/load_outputs_to_pg.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SCHEMA = "meeting_analytics"
EMBEDDING_DIMENSION = 768


def _load_dotenv() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _build_dsn() -> str:
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    database = os.getenv("PG_DATABASE", "postgres")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


async def create_tables(conn: asyncpg.Connection) -> None:
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

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
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            canonical   TEXT NOT NULL UNIQUE,
            aliases     TEXT[] DEFAULT '{{}}',
            cluster_id  INTEGER NOT NULL
                            REFERENCES {SCHEMA}.semantic_clusters(cluster_id),
            embedding   vector({EMBEDDING_DIMENSION}),
            content_tsv tsvector
                            GENERATED ALWAYS AS (to_tsvector('english', canonical)) STORED
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.semantic_meeting_themes (
            meeting_id        TEXT NOT NULL,
            cluster_id        INTEGER NOT NULL
                                  REFERENCES {SCHEMA}.semantic_clusters(cluster_id),
            is_primary        BOOLEAN NOT NULL DEFAULT false,
            call_type         TEXT,
            call_confidence   TEXT,
            sentiment_score   NUMERIC,
            overall_sentiment TEXT,
            PRIMARY KEY (meeting_id, cluster_id)
        )
    """)

    # GIN index for full-text search on phrases
    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS semantic_phrases_tsv_idx
        ON {SCHEMA}.semantic_phrases USING GIN (content_tsv)
    """)
    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS semantic_phrases_cluster_idx
        ON {SCHEMA}.semantic_phrases (cluster_id)
    """)
    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS semantic_meeting_themes_meeting_idx
        ON {SCHEMA}.semantic_meeting_themes (meeting_id)
    """)


async def load_clusters(conn: asyncpg.Connection) -> int:
    data = json.loads((OUTPUTS / "semantic_clusters.json").read_text(encoding="utf-8"))
    rows = [
        (c["cluster_id"], c["theme_title"], c["audience"],
         c.get("rationale", ""), c["phrase_count"])
        for c in data
    ]
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.semantic_clusters
                (cluster_id, theme_title, audience, rationale, phrase_count)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (cluster_id) DO UPDATE
                SET theme_title=EXCLUDED.theme_title,
                    audience=EXCLUDED.audience,
                    rationale=EXCLUDED.rationale,
                    phrase_count=EXCLUDED.phrase_count""",
        rows,
    )
    return len(rows)


async def load_phrases(conn: asyncpg.Connection) -> int:
    rows: list[tuple] = []
    with open(OUTPUTS / "phrase_clusters.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cluster_id = int(row["cluster_id"])
            canonical = row["canonical"].strip()
            aliases_raw = row.get("aliases", "").strip()
            aliases = [a.strip() for a in aliases_raw.split(";") if a.strip()] if aliases_raw else []
            rows.append((canonical, aliases, cluster_id))

    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.semantic_phrases (canonical, aliases, cluster_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (canonical) DO UPDATE
                SET aliases=EXCLUDED.aliases,
                    cluster_id=EXCLUDED.cluster_id""",
        rows,
    )
    return len(rows)


async def load_meeting_themes(conn: asyncpg.Connection) -> int:
    rows: list[tuple] = []
    with open(OUTPUTS / "meeting_themes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meeting_id = row["meeting_id"].strip()
            primary_id = int(row["primary_theme_id"])
            call_type = row.get("call_type", "").strip() or None
            call_confidence = row.get("call_confidence", "").strip() or None
            sentiment_score_raw = row.get("sentiment_score", "").strip()
            sentiment_score = float(sentiment_score_raw) if sentiment_score_raw else None
            overall_sentiment = row.get("overall_sentiment", "").strip() or None

            all_ids_raw = row.get("all_theme_ids", "").strip()
            all_ids = [int(x.strip()) for x in all_ids_raw.split(";") if x.strip()]
            if not all_ids:
                all_ids = [primary_id]

            for cid in all_ids:
                rows.append((
                    meeting_id, cid, cid == primary_id,
                    call_type, call_confidence, sentiment_score, overall_sentiment,
                ))

    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.semantic_meeting_themes
                (meeting_id, cluster_id, is_primary, call_type, call_confidence,
                 sentiment_score, overall_sentiment)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (meeting_id, cluster_id) DO UPDATE
                SET is_primary=EXCLUDED.is_primary,
                    call_type=EXCLUDED.call_type,
                    call_confidence=EXCLUDED.call_confidence,
                    sentiment_score=EXCLUDED.sentiment_score,
                    overall_sentiment=EXCLUDED.overall_sentiment""",
        rows,
    )
    return len(rows)


async def main() -> None:
    _load_dotenv()
    dsn = _build_dsn()

    # Enable pgvector extension first
    temp = await asyncpg.connect(dsn)
    try:
        await temp.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await temp.close()

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, init=register_vector)
    try:
        async with pool.acquire() as conn:
            print("\n[1/4] Creating tables and indexes...")
            await create_tables(conn)

            print("[2/4] Loading semantic_clusters...")
            n = await load_clusters(conn)
            print(f"      {n} clusters loaded")

            print("[3/4] Loading semantic_phrases (embedding=NULL)...")
            n = await load_phrases(conn)
            print(f"      {n} phrases loaded")

            print("[4/4] Loading semantic_meeting_themes...")
            n = await load_meeting_themes(conn)
            print(f"      {n} meeting-theme rows loaded")

        # Quick verification
        async with pool.acquire() as conn:
            r = await conn.fetchrow(f"""
                SELECT
                  (SELECT count(*) FROM {SCHEMA}.semantic_clusters) AS clusters,
                  (SELECT count(*) FROM {SCHEMA}.semantic_phrases) AS phrases,
                  (SELECT count(*) FROM {SCHEMA}.semantic_meeting_themes) AS meeting_themes,
                  (SELECT count(DISTINCT meeting_id) FROM {SCHEMA}.semantic_meeting_themes) AS meetings
            """)
            print(f"\nVerification:")
            print(f"  semantic_clusters:       {r['clusters']} rows")
            print(f"  semantic_phrases:        {r['phrases']} rows")
            print(f"  semantic_meeting_themes: {r['meeting_themes']} rows ({r['meetings']} distinct meetings)")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
