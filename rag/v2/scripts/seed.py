"""Dev bootstrap seed script.

Run via:  make seed   (or: uv run python scripts/seed.py)

Steps:
  1. Connectivity checks — DB, Redis, Ollama
  2. Default corpus config written to .env if CORPUS_CONFIGS_JSON is empty
  3. Sample documents ingested from ../../rag/documents/ into default:neuralflow
  4. Summary printed

This script is idempotent — safe to run multiple times.
Incremental ingestion skips unchanged files on subsequent runs.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the rag/v2/ directory is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_DOCS = ROOT.parent.parent / "rag" / "documents"  # ../../rag/documents

DEFAULT_CORPUS_CONFIG = [
    {
        "id": "neuralflow",
        "display_name": "NeuralFlow AI Docs",
        "source_folders": [str(SAMPLE_DOCS)],
        "allowed_roles": ["reader", "writer", "admin"],
        "enable_graph_extraction": False,
        "metadata_tags": {"corpus": "neuralflow", "env": "dev"},
    }
]

TENANT_ID = "default"
CORPUS_ID = "neuralflow"


# ── Step 1: connectivity checks ───────────────────────────────────────────────

async def check_postgres() -> None:
    import asyncpg
    from knowledge.config.settings import load_settings
    s = load_settings()
    try:
        conn = await asyncpg.connect(s.database_url, timeout=5)
        await conn.fetchval("SELECT 1")
        await conn.close()
        print("  ✓ PostgreSQL connected")
    except Exception as exc:
        print(f"  ✗ PostgreSQL: {exc}")
        print("    Start with: docker compose up postgres -d")
        sys.exit(1)


async def check_redis() -> None:
    import redis.asyncio as aioredis
    from knowledge.config.settings import load_settings
    s = load_settings()
    try:
        r = aioredis.from_url(s.redis_url)
        await r.ping()
        await r.aclose()
        print("  ✓ Redis connected")
    except Exception as exc:
        print(f"  ✗ Redis: {exc}")
        print("    Start with: docker compose up redis -d")
        sys.exit(1)


async def check_ollama() -> None:
    import httpx
    from knowledge.config.settings import load_settings
    s = load_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{s.embedding_base_url}/models")
            if resp.status_code < 500:
                print(f"  ✓ Ollama connected ({s.embedding_base_url})")
            else:
                raise RuntimeError(f"HTTP {resp.status_code}")
    except Exception as exc:
        print(f"  ✗ Ollama: {exc}")
        print("    Start with: ollama serve && make pull-models")
        sys.exit(1)


# ── Step 2: write default corpus config if missing ────────────────────────────

def ensure_corpus_config() -> None:
    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"

    # Read current .env (or .env.example as fallback)
    source = env_file if env_file.exists() else env_example
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []

    # Check if CORPUS_CONFIGS_JSON is already set to something non-empty
    for line in lines:
        if line.startswith("CORPUS_CONFIGS_JSON="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value and value != "[]":
                print(f"  ✓ CORPUS_CONFIGS_JSON already set ({value[:60]}...)")
                # Also ensure the env var is exported for this process
                os.environ.setdefault("CORPUS_CONFIGS_JSON", value)
                return

    # Write or update .env
    corpus_json = json.dumps(DEFAULT_CORPUS_CONFIG)
    new_line = f'CORPUS_CONFIGS_JSON={corpus_json}'

    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if "CORPUS_CONFIGS_JSON=" in content:
            import re
            content = re.sub(r"CORPUS_CONFIGS_JSON=.*", new_line, content)
        else:
            content += f"\n{new_line}\n"
        env_file.write_text(content, encoding="utf-8")
    else:
        # Create .env from .env.example with the corpus line appended
        base = env_example.read_text(encoding="utf-8") if env_example.exists() else ""
        (ROOT / ".env").write_text(base + f"\n{new_line}\n", encoding="utf-8")

    os.environ["CORPUS_CONFIGS_JSON"] = corpus_json
    print(f"  ✓ Default corpus config written to {env_file.name}")
    print(f"    corpus_id: {CORPUS_ID} → {SAMPLE_DOCS}")


# ── Step 3: ingest sample documents ──────────────────────────────────────────

async def ingest_sample_docs() -> int:
    """Run DocumentIngestionPipeline directly (no Redis worker needed).

    Deduplication — a file is ingested at most once per content hash:
      Layer 1: Redis fingerprint cache  cache:doc_fingerprint:{sha256(content)}
               → SKIP if exists (hash matches → file unchanged)
      Layer 2: PostgreSQL documents.metadata->>'content_hash'
               → SKIP if DB hash matches (handles Redis flush / cold start)

    Both layers use SHA-256(file_content) — NOT mtime or filename.
    A renamed but content-identical file is correctly detected as a duplicate.
    A content-changed file (same name) is correctly re-ingested.
    """
    if not SAMPLE_DOCS.exists():
        print(f"  ✗ Sample docs not found: {SAMPLE_DOCS}")
        print("    Run from the rag/v2/ directory inside the monorepo.")
        return 0

    from knowledge.config.settings import load_settings
    from knowledge.ingestion.pipeline import DocumentIngestionPipeline
    from knowledge.store.cache import RedisCache
    from knowledge.store.vector import PostgresHybridStore
    from knowledge.bus.schemas import IngestJob

    settings   = load_settings()
    vs         = PostgresHybridStore(settings=settings)
    cache      = RedisCache(settings=settings)

    await vs.initialize()
    await cache.connect()

    pipeline = DocumentIngestionPipeline(
        settings=settings,
        vector_store=vs,
        cache=cache,
    )

    job = IngestJob(
        tenant_id=TENANT_ID,
        corpus_id=CORPUS_ID,
        source_path=str(SAMPLE_DOCS),
        mode="incremental",          # idempotent — skips unchanged files
        enable_graph_extraction=False,
    )

    print(f"  Ingesting {SAMPLE_DOCS} ...")
    result = await pipeline.run(job)

    await vs.close()
    await cache.close()

    if result.errors:
        for err in result.errors:
            print(f"  ⚠ {err}")

    if result.skipped and result.chunks_ingested == 0:
        print(f"  ✓ All documents unchanged (incremental skip)")
    else:
        print(f"  ✓ {result.chunks_ingested} chunks ingested ({len(result.errors)} errors)")

    return result.chunks_ingested


# ── Step 4: verify ────────────────────────────────────────────────────────────

async def verify_seed() -> None:
    from knowledge.config.settings import load_settings
    from knowledge.store.vector import PostgresHybridStore

    settings = load_settings()
    vs       = PostgresHybridStore(settings=settings)
    await vs.initialize()

    count = await vs.get_chunk_count(CORPUS_ID, TENANT_ID)
    await vs.close()

    if count > 0:
        print(f"  ✓ Verified: {count} chunks in {TENANT_ID}:{CORPUS_ID}")
    else:
        print(f"  ✗ No chunks found in {TENANT_ID}:{CORPUS_ID} — check ingestion errors above")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n=== RAG v2 Seed ===\n")

    print("1/4  Connectivity checks")
    await check_postgres()
    await check_redis()
    await check_ollama()

    print("\n2/4  Corpus config")
    ensure_corpus_config()

    # Reload settings now that CORPUS_CONFIGS_JSON is in env
    from knowledge.config.settings import load_settings, Settings
    load_settings.cache_clear()

    print("\n3/4  Sample document ingestion")
    await ingest_sample_docs()

    print("\n4/4  Verification")
    await verify_seed()

    print("\n=== Seed complete ===")
    print(f"\nDefault corpus: {TENANT_ID}:{CORPUS_ID}")
    print(f"Sample docs:    {SAMPLE_DOCS}")
    print("\nNext steps:")
    print("  make test-unit        # unit tests (no services)")
    print("  make test-integration # integration tests (requires running services)")
    print("  uv run uvicorn knowledge.api.app:app --reload  # start API")


if __name__ == "__main__":
    asyncio.run(main())
