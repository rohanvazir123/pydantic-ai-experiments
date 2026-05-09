"""
Load Take B (TF-IDF / KMeans) outputs into Postgres.

Reads from take_b/outputs/:
  cluster_summary.json  -> meeting_analytics.kmeans_clusters
  cluster_terms.csv     -> meeting_analytics.kmeans_cluster_terms
  meeting_clusters.csv  -> meeting_analytics.kmeans_meeting_clusters

Tables created:
  kmeans_clusters         cluster_id, label, meeting_count, silhouette_score
  kmeans_cluster_terms    cluster_id, rank, term  (top-12 centroid terms)
  kmeans_meeting_clusters meeting_id, cluster_id  (hard single-cluster assignment)

Usage (from repo root):
  python basics/iprep/meeting-analytics/take_b/load_outputs_to_pg.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path

import asyncpg

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SCHEMA = "meeting_analytics"


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
    port = os.getenv("PG_PORT", "5434")
    user = os.getenv("PG_USER", "rag_user")
    password = os.getenv("PG_PASSWORD", "rag_pass")
    database = os.getenv("PG_DATABASE", "rag_db")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


async def create_tables(conn: asyncpg.Connection) -> None:
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.kmeans_clusters (
            cluster_id      INTEGER PRIMARY KEY,
            label           TEXT NOT NULL,
            meeting_count   INTEGER DEFAULT 0,
            silhouette_score NUMERIC
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.kmeans_cluster_terms (
            cluster_id  INTEGER NOT NULL
                            REFERENCES {SCHEMA}.kmeans_clusters(cluster_id),
            rank        INTEGER NOT NULL,
            term        TEXT NOT NULL,
            PRIMARY KEY (cluster_id, rank)
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.kmeans_meeting_clusters (
            meeting_id  TEXT PRIMARY KEY,
            cluster_id  INTEGER NOT NULL
                            REFERENCES {SCHEMA}.kmeans_clusters(cluster_id)
        )
    """)

    await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS kmeans_meeting_clusters_cluster_idx
        ON {SCHEMA}.kmeans_meeting_clusters (cluster_id)
    """)


async def load_clusters(conn: asyncpg.Connection) -> int:
    summary = json.loads((OUTPUTS / "cluster_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUTS / "cluster_metrics.json").read_text(encoding="utf-8"))
    silhouette = metrics.get("silhouette_score")

    rows = [
        (c["cluster_id"], c["generated_label"], c["meeting_count"], silhouette)
        for c in summary
    ]
    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.kmeans_clusters
                (cluster_id, label, meeting_count, silhouette_score)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cluster_id) DO UPDATE
                SET label=EXCLUDED.label,
                    meeting_count=EXCLUDED.meeting_count,
                    silhouette_score=EXCLUDED.silhouette_score""",
        rows,
    )
    return len(rows)


async def load_terms(conn: asyncpg.Connection) -> int:
    rows: list[tuple] = []
    with open(OUTPUTS / "cluster_terms.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["cluster_id"]), int(row["rank"]), row["term"].strip()))

    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.kmeans_cluster_terms (cluster_id, rank, term)
            VALUES ($1, $2, $3)
            ON CONFLICT (cluster_id, rank) DO UPDATE SET term=EXCLUDED.term""",
        rows,
    )
    return len(rows)


async def load_meeting_clusters(conn: asyncpg.Connection) -> int:
    rows: list[tuple] = []
    with open(OUTPUTS / "meeting_clusters.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["meeting_id"].strip(), int(row["cluster_id"])))

    await conn.executemany(
        f"""INSERT INTO {SCHEMA}.kmeans_meeting_clusters (meeting_id, cluster_id)
            VALUES ($1, $2)
            ON CONFLICT (meeting_id) DO UPDATE SET cluster_id=EXCLUDED.cluster_id""",
        rows,
    )
    return len(rows)


async def main() -> None:
    _load_dotenv()

    conn = await asyncpg.connect(_build_dsn())
    try:
        print("\n[1/4] Creating Take B tables...")
        await create_tables(conn)

        print("[2/4] Loading kmeans_clusters...")
        n = await load_clusters(conn)
        print(f"      {n} clusters")

        print("[3/4] Loading kmeans_cluster_terms...")
        n = await load_terms(conn)
        print(f"      {n} terms")

        print("[4/4] Loading kmeans_meeting_clusters...")
        n = await load_meeting_clusters(conn)
        print(f"      {n} meeting assignments")

        r = await conn.fetchrow(f"""
            SELECT
              (SELECT count(*) FROM {SCHEMA}.kmeans_clusters) AS clusters,
              (SELECT count(*) FROM {SCHEMA}.kmeans_cluster_terms) AS terms,
              (SELECT count(*) FROM {SCHEMA}.kmeans_meeting_clusters) AS meetings
        """)
        print(f"\nVerification:")
        print(f"  kmeans_clusters:         {r['clusters']} rows")
        print(f"  kmeans_cluster_terms:    {r['terms']} rows")
        print(f"  kmeans_meeting_clusters: {r['meetings']} rows")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
