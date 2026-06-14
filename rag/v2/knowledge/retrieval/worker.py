"""Retrieval worker entrypoint — handles async / bulk search requests.

Interactive queries bypass this worker and call the retriever directly
in the API process. This worker handles bulk search batches submitted
to knowledge:search via POST /v1/search?async=true (future feature).

Run as:
    python -m knowledge.retrieval.worker
"""

import asyncio
import logging
import signal
import sys

import redis.asyncio as aioredis

from knowledge.bus.consumer import consume_loop
from knowledge.bus.schemas import SearchRequest
from knowledge.config.settings import load_settings
from knowledge.ingestion.embedder import Embedder
from knowledge.retrieval.retriever import Retriever
from knowledge.store.cache import RedisCache
from knowledge.store.vector import PostgresHybridStore

logger = logging.getLogger(__name__)

_STREAM = "knowledge:search"
_GROUP  = "retrieval-workers"


async def _main() -> None:
    settings  = load_settings()
    worker_id = f"retrieval-worker-{id(settings)}"

    redis_client = aioredis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=False,
    )

    vector_store = PostgresHybridStore(settings=settings)
    cache        = RedisCache(settings=settings)
    embedder     = Embedder(settings=settings)

    await vector_store.initialize()
    await cache.connect()

    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        cache=cache,
        settings=settings,
    )

    async def handler(req: SearchRequest) -> None:
        results = await retriever.retrieve(
            query=req.query,
            corpus_ids=req.corpus_ids,
            tenant_id=req.tenant_id,
            k=req.k,
        )
        if req.callback_key:
            import json
            serialised = json.dumps([retriever._result_to_dict(r) for r in results])
            await redis_client.lpush(req.callback_key, serialised)
            await redis_client.expire(req.callback_key, 3600)

    stop_event = asyncio.Event()

    def _handle_sigterm(*_: object) -> None:
        logger.info("SIGTERM received — stopping retrieval worker")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info("Retrieval worker '%s' starting", worker_id)
    try:
        await consume_loop(
            redis=redis_client,
            stream=_STREAM,
            group=_GROUP,
            worker_id=worker_id,
            job_model=SearchRequest,
            handler=handler,
            job_timeout_s=settings.job_timeout_s,
            _stop_event=stop_event,
        )
    finally:
        await vector_store.close()
        await cache.close()
        await redis_client.aclose()  # type: ignore[attr-defined]
        logger.info("Retrieval worker '%s' stopped", worker_id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
