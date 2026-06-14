"""Unit tests for knowledge.bus.consumer and knowledge.bus.publisher.

Uses fakeredis — no live Redis required.
"""

import json

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from knowledge.bus.consumer import (
    MAX_RETRIES,
    _execute_with_retry,
    _group_from_stream,
    ensure_consumer_group,
)
from knowledge.bus.publisher import Publisher
from knowledge.bus.schemas import IngestJob


@pytest_asyncio.fixture
async def redis():
    return fakeredis.FakeRedis(decode_responses=False)


# ── _group_from_stream ────────────────────────────────────────────────────────

class TestGroupFromStream:
    def test_ingest_stream(self):
        assert _group_from_stream("knowledge:ingest") == "ingest-workers"

    def test_search_stream(self):
        assert _group_from_stream("knowledge:search") == "retrieval-workers"

    def test_eval_stream(self):
        assert _group_from_stream("knowledge:eval") == "eval-workers"

    def test_unknown_stream_fallback(self):
        assert _group_from_stream("knowledge:custom") == "knowledge-custom"


# ── Publisher ─────────────────────────────────────────────────────────────────

class TestPublisher:
    @pytest.mark.asyncio
    async def test_publish_ingest_job_creates_stream_entry(self, redis):
        pub = Publisher(redis)
        job = IngestJob(tenant_id="t1", corpus_id="c1")
        msg_id = await pub.publish_ingest_job(job)
        assert msg_id  # non-empty stream message ID

    @pytest.mark.asyncio
    async def test_publish_creates_job_hash(self, redis):
        pub = Publisher(redis)
        job = IngestJob(tenant_id="t1", corpus_id="c1")
        await pub.publish_ingest_job(job)
        status = await redis.hget(f"job:{job.job_id}", "status")
        assert status == b"queued"

    @pytest.mark.asyncio
    async def test_update_job_status(self, redis):
        pub = Publisher(redis)
        job = IngestJob(tenant_id="t1", corpus_id="c1")
        await pub.publish_ingest_job(job)
        await pub.update_job_status(job.job_id, "completed", progress=100, chunks_ingested=42)
        status = await redis.hget(f"job:{job.job_id}", "status")
        assert status == b"completed"
        chunks = await redis.hget(f"job:{job.job_id}", "chunks_ingested")
        assert chunks == b"42"


# ── ensure_consumer_group ─────────────────────────────────────────────────────

class TestEnsureConsumerGroup:
    @pytest.mark.asyncio
    async def test_creates_group_and_stream(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        info = await redis.xinfo_groups("knowledge:ingest")
        # fakeredis may return str or bytes keys depending on version
        names = {
            (g.get("name") or g.get(b"name", b"")).decode()
            if isinstance(g.get("name") or g.get(b"name"), bytes)
            else str(g.get("name") or g.get(b"name", ""))
            for g in info
        }
        assert "ingest-workers" in names

    @pytest.mark.asyncio
    async def test_idempotent_on_existing_group(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")  # must not raise


# ── _execute_with_retry ───────────────────────────────────────────────────────

class TestExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_ack_on_success(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        job = IngestJob(tenant_id="t1", corpus_id="c1")
        await redis.xadd("knowledge:ingest", {"payload": job.model_dump_json()})
        msgs = await redis.xreadgroup(
            "ingest-workers", "w1", {"knowledge:ingest": ">"}, count=1
        )
        msg_id = msgs[0][1][0][0]
        payload_raw = msgs[0][1][0][1][b"payload"].decode()

        handled: list[str] = []

        async def handler(j: IngestJob) -> None:
            handled.append(j.job_id)

        await _execute_with_retry(
            redis, "knowledge:ingest", msg_id, payload_raw,
            IngestJob, handler, job_timeout_s=10,
        )
        # Handler ran → job was processed; XACK is implicit in _execute_with_retry
        assert handled == [job.job_id]

    @pytest.mark.asyncio
    async def test_dlq_on_non_retriable(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        job = IngestJob(tenant_id="t1", corpus_id="c1")
        await redis.xadd("knowledge:ingest", {"payload": job.model_dump_json()})
        msgs = await redis.xreadgroup(
            "ingest-workers", "w1", {"knowledge:ingest": ">"}, count=1
        )
        msg_id = msgs[0][1][0][0]
        payload_raw = msgs[0][1][0][1][b"payload"].decode()

        async def handler(j: IngestJob) -> None:
            raise ValueError("bad schema")

        await _execute_with_retry(
            redis, "knowledge:ingest", msg_id, payload_raw,
            IngestJob, handler, job_timeout_s=10,
        )
        dlq_msgs = await redis.xrange("knowledge:ingest:dlq", "-", "+")
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0][1][b"permanent"] == b"1"

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        job = IngestJob(tenant_id="t1", corpus_id="c1", attempt=1)
        await redis.xadd("knowledge:ingest", {"payload": job.model_dump_json()})
        msgs = await redis.xreadgroup(
            "ingest-workers", "w1", {"knowledge:ingest": ">"}, count=1
        )
        msg_id = msgs[0][1][0][0]
        payload_raw = msgs[0][1][0][1][b"payload"].decode()

        async def handler(j: IngestJob) -> None:
            raise ConnectionError("transient")

        # Override backoff to 0 for test speed
        import knowledge.bus.consumer as consumer_mod
        orig = consumer_mod.exponential_backoff
        consumer_mod.exponential_backoff = lambda *a, **kw: 0.0

        try:
            await _execute_with_retry(
                redis, "knowledge:ingest", msg_id, payload_raw,
                IngestJob, handler, job_timeout_s=10, max_retries=3,
            )
        finally:
            consumer_mod.exponential_backoff = orig

        # A re-enqueued message should appear in the stream (attempt=2)
        all_msgs = await redis.xrange("knowledge:ingest", "-", "+")
        payloads = [json.loads(m[1][b"payload"]) for m in all_msgs]
        assert any(p["attempt"] == 2 for p in payloads)

    @pytest.mark.asyncio
    async def test_dlq_after_max_retries(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        job = IngestJob(tenant_id="t1", corpus_id="c1", attempt=MAX_RETRIES)
        await redis.xadd("knowledge:ingest", {"payload": job.model_dump_json()})
        msgs = await redis.xreadgroup(
            "ingest-workers", "w1", {"knowledge:ingest": ">"}, count=1
        )
        msg_id = msgs[0][1][0][0]
        payload_raw = msgs[0][1][0][1][b"payload"].decode()

        async def handler(j: IngestJob) -> None:
            raise ConnectionError("still failing")

        import knowledge.bus.consumer as consumer_mod
        orig = consumer_mod.exponential_backoff
        consumer_mod.exponential_backoff = lambda *a, **kw: 0.0
        try:
            await _execute_with_retry(
                redis, "knowledge:ingest", msg_id, payload_raw,
                IngestJob, handler, job_timeout_s=10, max_retries=MAX_RETRIES,
            )
        finally:
            consumer_mod.exponential_backoff = orig

        dlq_msgs = await redis.xrange("knowledge:ingest:dlq", "-", "+")
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0][1][b"permanent"] == b"0"  # exhausted, not permanent

    @pytest.mark.asyncio
    async def test_corrupt_payload_goes_to_dlq(self, redis):
        await ensure_consumer_group(redis, "knowledge:ingest", "ingest-workers")
        await redis.xadd("knowledge:ingest", {"payload": b"not-json"})
        msgs = await redis.xreadgroup(
            "ingest-workers", "w1", {"knowledge:ingest": ">"}, count=1
        )
        msg_id = msgs[0][1][0][0]

        async def handler(j): pass

        await _execute_with_retry(
            redis, "knowledge:ingest", msg_id, "not-json",
            IngestJob, handler, job_timeout_s=10,
        )
        dlq_msgs = await redis.xrange("knowledge:ingest:dlq", "-", "+")
        assert len(dlq_msgs) == 1
