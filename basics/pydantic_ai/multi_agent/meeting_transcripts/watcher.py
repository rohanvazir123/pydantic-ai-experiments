"""
Automated Ingestion Watcher
============================

Watches the dataset/ directory for new meeting folders and automatically runs
the pipeline on any unprocessed meeting.

Two strategies (both run in the same loop):
  1. Polling — checks every INGEST_POLL_INTERVAL seconds (default 60s)
  2. Event-driven — uses watchfiles on platforms that support it; falls back
     to polling-only if watchfiles is not installed

A meeting is considered "already processed" if its ID appears in
history.json OR (when --db-url is set) in the PostgreSQL meetings table.

Usage:
    # Polling mode (no extra deps)
    OLLAMA_BASE_URL=http://localhost:11434/v1 \\
    python basics/pydantic_ai/multi_agent/meeting_transcripts/watcher.py

    # Custom poll interval
    INGEST_POLL_INTERVAL=30 \\
    OLLAMA_BASE_URL=http://localhost:11434/v1 \\
    python basics/pydantic_ai/multi_agent/meeting_transcripts/watcher.py

    # Also persist to PostgreSQL after each successful pipeline run
    OLLAMA_BASE_URL=http://localhost:11434/v1 \\
    DATABASE_URL=postgresql://user:pass@host/dbname \\
    python basics/pydantic_ai/multi_agent/meeting_transcripts/watcher.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import PipelineOutput, get_history, run_pipeline  # noqa: E402

try:
    from ingestion import ingest  # noqa: E402

    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.getenv("INGEST_POLL_INTERVAL", "60"))
DB_URL = os.getenv("DATABASE_URL", "")
DATASET_DIR = Path(__file__).parent / "dataset"

# ---------------------------------------------------------------------------
# Processed-meeting registry
# ---------------------------------------------------------------------------


def is_processed(meeting_id: str) -> bool:
    """Return True if the meeting has already been ingested."""
    return get_history(meeting_id) is not None


def discover_unprocessed(dataset_dir: Path) -> list[str]:
    """Return meeting IDs that have a transcript but have not been processed."""
    unprocessed: list[str] = []
    for meeting_dir in sorted(dataset_dir.iterdir()):
        if not meeting_dir.is_dir():
            continue
        if not (meeting_dir / "transcript.json").exists():
            continue
        if not is_processed(meeting_dir.name):
            unprocessed.append(meeting_dir.name)
    return unprocessed


# ---------------------------------------------------------------------------
# Single-meeting ingestion
# ---------------------------------------------------------------------------


async def process_meeting(meeting_id: str, dataset_dir: Path) -> None:
    logger.info("Processing new meeting: %s", meeting_id)
    try:
        output: PipelineOutput = await run_pipeline(
            meeting_id, dataset_dir, force=False
        )
        if DB_URL and _HAS_ASYNCPG:
            from ingestion import ingest  # local import to avoid circular at module level

            await ingest(meeting_id, output, DB_URL)
            logger.info("Persisted %s to PostgreSQL", meeting_id)
        else:
            if DB_URL and not _HAS_ASYNCPG:
                logger.warning("DATABASE_URL set but asyncpg not installed; skipping DB write")
    except Exception as exc:
        logger.error("Failed to process %s: %s", meeting_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


async def poll_loop(dataset_dir: Path) -> None:
    """Check for unprocessed meetings every POLL_INTERVAL seconds."""
    logger.info("Polling loop started (interval=%ds, dataset=%s)", POLL_INTERVAL, dataset_dir)
    while True:
        unprocessed = discover_unprocessed(dataset_dir)
        if unprocessed:
            logger.info("Found %d unprocessed meeting(s): %s", len(unprocessed), unprocessed)
            for meeting_id in unprocessed:
                await process_meeting(meeting_id, dataset_dir)
        else:
            logger.debug("No new meetings found")
        await asyncio.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Event-driven loop (watchfiles — optional)
# ---------------------------------------------------------------------------


async def event_driven_loop(dataset_dir: Path) -> None:
    """Watch dataset/ for new directories using watchfiles (inotify/FSEvents/kqueue)."""
    try:
        from watchfiles import awatch  # type: ignore[import-not-found]
    except ImportError:
        logger.info("watchfiles not installed; using polling-only mode")
        await poll_loop(dataset_dir)
        return

    logger.info("Event-driven watcher started (dataset=%s)", dataset_dir)
    async for changes in awatch(str(dataset_dir)):
        changed_paths = {Path(p) for _, p in changes}
        # A new meeting appears as a new directory with at least transcript.json
        new_meetings = {
            p.parent.name
            for p in changed_paths
            if p.name == "transcript.json" and p.parent.parent == dataset_dir
        }
        for meeting_id in new_meetings:
            if not is_processed(meeting_id):
                await process_meeting(meeting_id, dataset_dir)


# ---------------------------------------------------------------------------
# Combined entrypoint: event-driven + polling fallback heartbeat
# ---------------------------------------------------------------------------


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not DATASET_DIR.exists():
        logger.error("Dataset directory does not exist: %s", DATASET_DIR)
        raise SystemExit(1)

    # Run both loops concurrently:
    # - event_driven_loop fires immediately on new files (if watchfiles installed)
    # - poll_loop is the safety net that catches anything the event loop missed
    await asyncio.gather(
        event_driven_loop(DATASET_DIR),
        poll_loop(DATASET_DIR),
    )


if __name__ == "__main__":
    asyncio.run(main())
