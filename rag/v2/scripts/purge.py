# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Purge the default corpus and clear all caches for a forced full re-ingestion.

Run via:  make purge-corpus   (followed automatically by make seed)

Steps:
  1. Run purge_default_corpus.sql against PostgreSQL (removes documents, chunks,
     entity index, semantic cache)
  2. Clear Redis fingerprint cache keys for this corpus so incremental check
     does not skip files on next ingest
  3. Drop and recreate the Apache AGE graph for this corpus (if AGE is running)
  4. Print a report with row counts before and after

After this script, run `make seed` (or `make purge-corpus` which does both)
to re-ingest all documents from scratch with mode='full'.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent
SQL_FILE = Path(__file__).parent / "purge_default_corpus.sql"

TENANT_ID = "default"
CORPUS_ID = "neuralflow"


async def purge_postgres() -> None:
    """Run the SQL purge file against the main PostgreSQL DB."""
    import os
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        from knowledge.config.settings import load_settings
        db_url = load_settings().database_url

    print(f"  Running {SQL_FILE.name} against PostgreSQL...")
    result = subprocess.run(
        ["psql", db_url, "-f", str(SQL_FILE), "--no-psqlrc"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ psql error:\n{result.stderr}")
        sys.exit(1)

    # Print the summary SELECT output
    for line in result.stdout.splitlines():
        if line.strip() and not line.startswith("--"):
            print(f"    {line}")
    print("  ✓ PostgreSQL content purged")


async def purge_redis_fingerprints() -> None:
    """Delete all doc_fingerprint cache keys for this corpus.

    The fingerprint cache is keyed by sha256(file_content) — not by corpus —
    so we delete ALL fingerprint keys. This forces the pipeline to re-hash every
    file and check the DB on next ingest. Fingerprints for other corpora will
    be re-populated transparently on their next ingest run.
    """
    import redis.asyncio as aioredis
    from knowledge.config.settings import load_settings

    s = load_settings()
    r = aioredis.from_url(s.redis_url, decode_responses=False)

    pattern  = "cache:doc_fingerprint:*"
    deleted  = 0
    async for key in r.scan_iter(pattern, count=200):
        await r.delete(key)
        deleted += 1

    # Also clear search result cache for this corpus (now stale)
    async for key in r.scan_iter("cache:search:*", count=200):
        await r.delete(key)
        deleted += 1

    await r.aclose()
    print(f"  ✓ Redis: {deleted} cache keys deleted (fingerprints + search results)")


async def purge_age_graph() -> None:
    """Drop the AGE graph for the default corpus (if AGE is running)."""
    try:
        import asyncpg
        from knowledge.config.settings import load_settings
        from knowledge.store.graph import AgeGraphStore

        s     = load_settings()
        store = AgeGraphStore(settings=s)
        await store.initialize()
        graph_name = store._graph_name(TENANT_ID, CORPUS_ID)
        try:
            await store.delete_corpus_graph(TENANT_ID, CORPUS_ID)
            print(f"  ✓ AGE graph '{graph_name}' dropped")
        except Exception as exc:
            if "does not exist" in str(exc).lower():
                print(f"  ✓ AGE graph '{graph_name}' did not exist (nothing to drop)")
            else:
                print(f"  ⚠ AGE graph drop failed: {exc} (continuing)")
        finally:
            await store.close()
    except Exception as exc:
        print(f"  ⚠ AGE not available ({exc}) — skipping graph drop")


async def main() -> None:
    print(f"\n=== Purging corpus {TENANT_ID}:{CORPUS_ID} ===\n")

    print("1/3  PostgreSQL — documents, chunks, entity index, semantic cache")
    await purge_postgres()

    print("\n2/3  Redis — fingerprint cache + search result cache")
    await purge_redis_fingerprints()

    print("\n3/3  Apache AGE — knowledge graph")
    await purge_age_graph()

    print(f"\n=== Purge complete ===")
    print(f"\nAll content for {TENANT_ID}:{CORPUS_ID} has been removed.")
    print("Fingerprint cache cleared — next ingest will process ALL files.")
    print("\nNext step:")
    print("  make seed     # re-ingest ../../rag/documents/ (mode=full)")


if __name__ == "__main__":
    asyncio.run(main())
