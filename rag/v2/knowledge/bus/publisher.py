"""Redis Streams publisher helpers.

Every publish call:
  1. XADD the serialised job to the stream (returns message ID).
  2. HSET a job status hash for status polling via GET /ingest/{id}/status.

The stream carries only the job_id as a reference; callers poll the job hash
for progress updates. Workers update the hash directly.
"""

import logging
from datetime import datetime, UTC

import redis.asyncio as aioredis

from knowledge.bus.schemas import EvalJob, IngestJob, SearchRequest, WorkerEvent

logger = logging.getLogger(__name__)

_STREAM_INGEST = "knowledge:ingest"
_STREAM_SEARCH = "knowledge:search"
_STREAM_EVAL   = "knowledge:eval"
_STREAM_EVENTS = "knowledge:events"

_JOB_HASH_TTL = 48 * 3600  # 48 hours


class Publisher:
    """Thin async wrapper around Redis XADD for all stream types."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ── Ingest ────────────────────────────────────────────────────────────────

    async def publish_ingest_job(self, job: IngestJob) -> str:
        """XADD IngestJob to knowledge:ingest; initialise job status hash.

        Returns the Redis Stream message ID (not the job_id).
        """
        msg_id = await self._redis.xadd(
            _STREAM_INGEST,
            {"payload": job.model_dump_json()},
        )
        await self._init_job_hash(
            job.job_id,
            status="queued",
            corpus_id=job.corpus_id,
            tenant_id=job.tenant_id,
        )
        logger.info(
            "Published IngestJob job_id=%s corpus=%s msg_id=%s",
            job.job_id, job.corpus_id, msg_id,
        )
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    # ── Search ────────────────────────────────────────────────────────────────

    async def publish_search_request(self, req: SearchRequest) -> str:
        msg_id = await self._redis.xadd(
            _STREAM_SEARCH,
            {"payload": req.model_dump_json()},
        )
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    # ── Eval ──────────────────────────────────────────────────────────────────

    async def publish_eval_job(self, job: EvalJob) -> str:
        msg_id = await self._redis.xadd(
            _STREAM_EVAL,
            {"payload": job.model_dump_json()},
        )
        logger.info("Published EvalJob run_id=%s corpus=%s", job.run_id, job.corpus_id)
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    # ── Events ────────────────────────────────────────────────────────────────

    async def publish_event(self, event: WorkerEvent) -> str:
        msg_id = await self._redis.xadd(
            _STREAM_EVENTS,
            {"payload": event.model_dump_json()},
        )
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    # ── Job hash helpers ──────────────────────────────────────────────────────

    async def _init_job_hash(
        self,
        job_id: str,
        status: str,
        corpus_id: str,
        tenant_id: str,
    ) -> None:
        """Create initial job status hash; expires after 48h."""
        key = f"job:{job_id}"
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping={
            "status":     status,
            "corpus_id":  corpus_id,
            "tenant_id":  tenant_id,
            "progress":   "0",
            "submitted_at": datetime.now(UTC).isoformat(),
        })
        pipe.expire(key, _JOB_HASH_TTL)
        await pipe.execute()

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: int = 0,
        error: str | None = None,
        chunks_ingested: int | None = None,
    ) -> None:
        """Update the job status hash. Called by workers during/after processing."""
        key = f"job:{job_id}"
        mapping: dict[str, str] = {
            "status":   status,
            "progress": str(progress),
        }
        if error is not None:
            mapping["error"] = error
        if chunks_ingested is not None:
            mapping["chunks_ingested"] = str(chunks_ingested)
        if status in ("completed", "failed"):
            from datetime import datetime, UTC
            mapping["completed_at"] = datetime.now(UTC).isoformat()
        await self._redis.hset(key, mapping=mapping)
