# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Ingest worker entrypoint.

Run as:
    python -m knowledge.ingestion.worker

Connects all stores, creates the Redis consumer group, and enters the consume
loop. Handles SIGTERM gracefully: finishes the current job then exits.
"""

import asyncio
import logging
import signal
import sys

from knowledge.bus.consumer import consume_loop
from knowledge.bus.publisher import Publisher
from knowledge.bus.schemas import IngestJob
from knowledge.config.settings import load_settings
from knowledge.ingestion.pipeline import DocumentIngestionPipeline
from knowledge.store.cache import RedisCache
from knowledge.store.graph import AgeGraphStore
from knowledge.store.vector import PostgresHybridStore

logger = logging.getLogger(__name__)

_STREAM  = "knowledge:ingest"
_GROUP   = "ingest-workers"


async def _main() -> None:
    settings   = load_settings()
    worker_id  = f"ingest-worker-{settings.redis_url.split('/')[-1]}-{id(settings)}"

    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=False,
    )

    vector_store = PostgresHybridStore(settings=settings)
    age_store    = AgeGraphStore(settings=settings)
    cache        = RedisCache(settings=settings)
    publisher    = Publisher(redis_client)

    await vector_store.initialize()
    await age_store.initialize()
    await cache.connect()

    pipeline = DocumentIngestionPipeline(
        settings=settings,
        vector_store=vector_store,
        age_store=age_store,
        cache=cache,
        publisher=publisher,
    )

    async def handler(job: IngestJob) -> None:
        await publisher.update_job_status(job.job_id, "running", progress=0)
        result = await pipeline.run(job)
        status = "completed" if not result.errors else "failed"
        await publisher.update_job_status(
            job.job_id, status,
            progress=100,
            chunks_ingested=result.chunks_ingested,
            error="; ".join(result.errors) if result.errors else None,
        )

    stop_event = asyncio.Event()

    def _handle_sigterm(*_: object) -> None:
        logger.info("SIGTERM received — finishing current job then stopping")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info("Ingest worker '%s' starting", worker_id)
    try:
        await consume_loop(
            redis=redis_client,
            stream=_STREAM,
            group=_GROUP,
            worker_id=worker_id,
            job_model=IngestJob,
            handler=handler,
            job_timeout_s=settings.job_timeout_s,
            _stop_event=stop_event,
        )
    finally:
        await vector_store.close()
        await age_store.close()
        await cache.close()
        await redis_client.aclose()  # type: ignore[attr-defined]
        logger.info("Ingest worker '%s' stopped cleanly", worker_id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
