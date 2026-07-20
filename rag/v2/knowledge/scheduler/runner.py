# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""APScheduler-based periodic ingest scheduler.

Polls scheduled_jobs every 60s; publishes IngestJob to Redis for due jobs.
Started in FastAPI lifespan. Graceful shutdown on SIGTERM.
"""

import asyncio
import logging

from knowledge.bus.publisher import Publisher
from knowledge.bus.schemas import IngestJob
from knowledge.scheduler.job_store import ScheduledJobStore

logger = logging.getLogger(__name__)


async def scheduler_tick(
    job_store: ScheduledJobStore,
    publisher: Publisher,
    max_concurrent: int = 5,
) -> None:
    """Single tick: find due jobs, publish IngestJob for each, advance next_run_at."""
    due = await job_store.get_due_jobs()
    if not due:
        return

    sem = asyncio.Semaphore(max_concurrent)

    async def _publish_one(row: dict) -> None:
        async with sem:
            import json
            src_cfg = row.get("source_config", {})
            if isinstance(src_cfg, str):
                src_cfg = json.loads(src_cfg)

            job = IngestJob(
                tenant_id=row["tenant_id"],
                corpus_id=row["corpus_id"],
                source_path=src_cfg.get("path"),
                source_url=src_cfg.get("url"),
                enable_graph_extraction=bool(row.get("enable_graph_extraction", False)),
                mode=row.get("mode", "incremental"),
            )
            try:
                await publisher.publish_ingest_job(job)
                await job_store.update_next_run_at(str(row["id"]), row["cron_expr"])
                await job_store.update_last_run(str(row["id"]), "triggered", job.job_id)
                logger.info(
                    "Scheduled job '%s' triggered → ingest job %s",
                    row["name"], job.job_id,
                )
            except Exception as exc:
                logger.error("Failed to trigger scheduled job '%s': %s", row["name"], exc)

    await asyncio.gather(*[_publish_one(r) for r in due])


async def run_scheduler(
    job_store: ScheduledJobStore,
    publisher: Publisher,
    interval_s: int = 60,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Scheduler main loop. Runs until stop_event is set."""
    logger.info("Scheduler started (interval=%ds)", interval_s)
    while not (stop_event and stop_event.is_set()):
        try:
            await scheduler_tick(job_store, publisher)
        except Exception as exc:
            logger.error("Scheduler tick error: %s", exc)
        await asyncio.sleep(interval_s)
    logger.info("Scheduler stopped")
