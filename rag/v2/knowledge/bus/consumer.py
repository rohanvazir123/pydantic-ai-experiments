"""Redis Streams consumer base loop.

Every worker (ingest, retrieval, eval) runs consume_loop() as its main
coroutine. The loop:
  1. XREADGROUP to claim one message at a time (block 5s).
  2. Deserialise the payload into the appropriate Pydantic model.
  3. Call the handler coroutine with a timeout (JOB_TIMEOUT_S).
  4. On success: XACK.
  5. On retriable failure: exponential backoff, re-enqueue with attempt+1, XACK original.
  6. After MAX_RETRIES: move to DLQ stream, XACK original, emit WorkerEvent.

Heartbeat: published to knowledge:events every HEARTBEAT_INTERVAL_S.

Retriable exceptions: anything not in NON_RETRIABLE; includes asyncio.TimeoutError.
Non-retriable: ValueError, TypeError, json.JSONDecodeError (corrupt message — retry useless).
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import redis.asyncio as aioredis

from knowledge.bus.backoff import exponential_backoff
from knowledge.bus.schemas import IngestJob, WorkerEvent

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES          = 3
HEARTBEAT_INTERVAL_S = 10
WORKER_HEARTBEAT_TTL = 30   # seconds; absence means worker is dead

_NON_RETRIABLE = (ValueError, TypeError, json.JSONDecodeError, KeyError)

Handler = Callable[[Any], Awaitable[None]]


async def _send_heartbeat(
    redis: aioredis.Redis,
    worker_id: str,
    stream_events: str,
) -> None:
    key = f"worker:{worker_id}:heartbeat"
    await redis.set(key, str(time.time()), ex=WORKER_HEARTBEAT_TTL)
    event = WorkerEvent(
        event_type="heartbeat",
        worker_id=worker_id,
    )
    await redis.xadd(stream_events, {"payload": event.model_dump_json()})


async def _move_to_dlq(
    redis: aioredis.Redis,
    stream: str,
    job_id: str,
    payload_raw: str,
    exc: Exception,
    permanent: bool,
) -> None:
    dlq_stream = f"{stream}:dlq"
    await redis.xadd(dlq_stream, {
        "job_id":    job_id,
        "payload":   payload_raw,
        "error":     str(exc),
        "permanent": "1" if permanent else "0",
    })
    logger.error(
        "Job %s moved to DLQ (%s): %s",
        job_id, "permanent" if permanent else "exhausted retries", exc,
    )


async def _re_enqueue(
    redis: aioredis.Redis,
    stream: str,
    job: Any,
    next_attempt: int,
) -> None:
    """Re-publish the job with incremented attempt counter."""
    job_copy = job.model_copy(update={"attempt": next_attempt})
    await redis.xadd(stream, {"payload": job_copy.model_dump_json()})


async def ensure_consumer_group(
    redis: aioredis.Redis,
    stream: str,
    group: str,
) -> None:
    """Create the consumer group if it does not already exist (MKSTREAM)."""
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
        logger.info("Created consumer group '%s' on stream '%s'", group, stream)
    except Exception as exc:
        if "BUSYGROUP" in str(exc):
            logger.debug("Consumer group '%s' already exists", group)
        else:
            raise


async def _execute_with_retry(
    redis: aioredis.Redis,
    stream: str,
    msg_id: bytes,
    payload_raw: str,
    job_model: type,
    handler: Handler,
    job_timeout_s: float,
    max_retries: int = MAX_RETRIES,
) -> None:
    """Deserialise, run handler, ACK on success; retry or DLQ on failure."""
    try:
        job = job_model.model_validate_json(payload_raw)
    except Exception as exc:
        # Corrupt payload — non-retriable; ACK and DLQ immediately
        await redis.xack(stream, stream.split(":")[0], msg_id)
        await _move_to_dlq(redis, stream, "unknown", payload_raw, exc, permanent=True)
        return

    job_id  = getattr(job, "job_id", None) or getattr(job, "run_id", "unknown")
    attempt = getattr(job, "attempt", 1)

    try:
        await asyncio.wait_for(handler(job), timeout=job_timeout_s)
        await redis.xack(stream, _group_from_stream(stream), msg_id)
        logger.info("Job %s completed (attempt %d)", job_id, attempt)

    except _NON_RETRIABLE as exc:
        # Permanent failure — ACK original, DLQ immediately
        await redis.xack(stream, _group_from_stream(stream), msg_id)
        await _move_to_dlq(redis, stream, job_id, payload_raw, exc, permanent=True)

    except (Exception, asyncio.TimeoutError) as exc:
        if attempt >= max_retries:
            # Exhausted retries — ACK original, DLQ
            await redis.xack(stream, _group_from_stream(stream), msg_id)
            await _move_to_dlq(redis, stream, job_id, payload_raw, exc, permanent=False)
        else:
            # Transient failure — backoff, re-enqueue with attempt+1, ACK original
            backoff_s = exponential_backoff(attempt)
            logger.warning(
                "Job %s failed (attempt %d/%d), retrying in %.1fs: %s",
                job_id, attempt, max_retries, backoff_s, exc,
            )
            await asyncio.sleep(backoff_s)
            await _re_enqueue(redis, stream, job, attempt + 1)
            await redis.xack(stream, _group_from_stream(stream), msg_id)


def _group_from_stream(stream: str) -> str:
    """Derive a default group name from the stream name.

    knowledge:ingest → ingest-workers
    knowledge:search → retrieval-workers
    knowledge:eval   → eval-workers
    """
    mapping = {
        "knowledge:ingest": "ingest-workers",
        "knowledge:search": "retrieval-workers",
        "knowledge:eval":   "eval-workers",
    }
    return mapping.get(stream, stream.replace(":", "-"))


async def consume_loop(
    redis: aioredis.Redis,
    stream: str,
    group: str,
    worker_id: str,
    job_model: type,
    handler: Handler,
    job_timeout_s: float = 300.0,
    max_retries: int = MAX_RETRIES,
    _stop_event: asyncio.Event | None = None,   # for testing: stop after N iterations
) -> None:
    """Main consumer loop. Runs until the process is killed or _stop_event set.

    XREADGROUP COUNT 1 BLOCK 5000 ensures:
      - One message processed at a time per worker (backpressure).
      - 5s block allows heartbeats to fire regularly without busy-wait.
    """
    await ensure_consumer_group(redis, stream, group)

    last_heartbeat = 0.0
    stream_events  = "knowledge:events"

    while True:
        if _stop_event and _stop_event.is_set():
            break

        # Heartbeat every HEARTBEAT_INTERVAL_S
        now = time.monotonic()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            try:
                await _send_heartbeat(redis, worker_id, stream_events)
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)
            last_heartbeat = now

        try:
            messages = await redis.xreadgroup(
                group, worker_id, {stream: ">"}, count=1, block=5000
            )
        except Exception as exc:
            logger.error("XREADGROUP error on '%s': %s", stream, exc)
            await asyncio.sleep(1)
            continue

        if not messages:
            continue

        for _stream_name, entries in messages:
            for msg_id, fields in entries:
                payload_raw = fields.get(b"payload", b"{}").decode()
                await _execute_with_retry(
                    redis, stream, msg_id, payload_raw,
                    job_model, handler, job_timeout_s, max_retries,
                )
