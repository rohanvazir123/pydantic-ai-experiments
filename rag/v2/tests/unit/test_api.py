"""Unit tests for Phase 8 — API schemas, middleware, quota.

No live services or FastAPI test client required — tests cover
Pydantic model validation and middleware/quota helper logic.
"""

import os
import uuid
from unittest import mock

import pytest
import pytest_asyncio
import fakeredis.aioredis as fakeredis

from knowledge.api.schemas import (
    APIResponse,
    ChatRequest,
    ErrorDetail,
    FeedbackRequest,
    HealthResponse,
    IngestRequest,
    SearchRequest,
    ScheduledJobRequest,
    TokenRequest,
)
from knowledge.api.quota import QuotaExceeded, enforce_quota
from knowledge.api.timeout import TimeoutBudget


# ── APIResponse envelope ──────────────────────────────────────────────────────

class TestAPIResponse:
    def test_success_has_data_no_error(self) -> None:
        resp = APIResponse(request_id="req-1", data={"key": "value"})
        assert resp.data == {"key": "value"}
        assert resp.error is None

    def test_error_response(self) -> None:
        err = ErrorDetail(code="NOT_FOUND", message="Resource not found.", status_code=404)
        resp = APIResponse[None](request_id="req-2", error=err)
        assert resp.data is None
        assert resp.error.code == "NOT_FOUND"

    def test_cache_hit_field(self) -> None:
        resp = APIResponse(request_id="r", data="ok", cache_hit="l2")
        assert resp.cache_hit == "l2"

    def test_generic_typed_response(self) -> None:
        from knowledge.api.schemas import HealthResponse
        hr = HealthResponse(status="healthy")
        resp = APIResponse[HealthResponse](request_id="r", data=hr)
        assert resp.data.status == "healthy"


class TestErrorDetail:
    def test_minimal(self) -> None:
        err = ErrorDetail(code="X", message="msg")
        assert err.retry_after_s is None
        assert err.details == {}

    def test_with_retry(self) -> None:
        err = ErrorDetail(code="RATE_LIMIT_EXCEEDED", message="Too many requests.", retry_after_s=60)
        assert err.retry_after_s == 60


# ── ChatRequest ───────────────────────────────────────────────────────────────

class TestChatRequest:
    def test_required_fields(self) -> None:
        req = ChatRequest(
            query="What is the PTO policy?",
            corpus_ids=["c1"],
            session_id="sess-abc",
        )
        assert req.model_tier == "auto"
        assert req.message_history is None

    def test_session_id_required(self) -> None:
        with pytest.raises(Exception):
            ChatRequest(query="q", corpus_ids=["c1"])   # missing session_id

    def test_valid_model_tier(self) -> None:
        req = ChatRequest(query="q", corpus_ids=["c1"], session_id="s", model_tier="large")
        assert req.model_tier == "large"

    def test_invalid_model_tier(self) -> None:
        with pytest.raises(Exception):
            ChatRequest(query="q", corpus_ids=["c1"], session_id="s", model_tier="huge")


# ── SearchRequest ─────────────────────────────────────────────────────────────

class TestSearchRequest:
    def test_defaults(self) -> None:
        req = SearchRequest(query="q", corpus_ids=["c1"])
        assert req.k == 5
        assert req.search_type == "hybrid"
        assert req.include_graph is False

    def test_k_bounds(self) -> None:
        with pytest.raises(Exception):
            SearchRequest(query="q", corpus_ids=["c1"], k=0)
        with pytest.raises(Exception):
            SearchRequest(query="q", corpus_ids=["c1"], k=100)


# ── IngestRequest ─────────────────────────────────────────────────────────────

class TestIngestRequest:
    def test_required_corpus_id(self) -> None:
        with pytest.raises(Exception):
            IngestRequest()  # missing corpus_id

    def test_defaults(self) -> None:
        req = IngestRequest(corpus_id="c1", source_path="/tmp/docs")
        assert req.mode == "incremental"
        assert req.enable_graph_extraction is False


# ── ScheduledJobRequest ───────────────────────────────────────────────────────

class TestScheduledJobRequest:
    def test_valid_source_type(self) -> None:
        req = ScheduledJobRequest(
            name="daily sync",
            source_type="local",
            source_config={"path": "/mnt/docs"},
            corpus_id="c1",
            cron_expr="0 2 * * *",
        )
        assert req.mode == "incremental"

    def test_invalid_source_type(self) -> None:
        with pytest.raises(Exception):
            ScheduledJobRequest(
                name="x", source_type="ftp",  # not in Literal
                source_config={}, corpus_id="c1", cron_expr="* * * * *",
            )


# ── HealthResponse ────────────────────────────────────────────────────────────

class TestHealthResponse:
    def test_healthy(self) -> None:
        h = HealthResponse(
            status="healthy",
            components={"postgres": "healthy", "redis": "healthy"},
        )
        assert h.degraded_modes == []

    def test_degraded(self) -> None:
        h = HealthResponse(
            status="degraded",
            degraded_modes=["no_graph"],
            components={"postgres": "healthy", "age_graph": "circuit_open"},
        )
        assert "no_graph" in h.degraded_modes


# ── TimeoutBudget ─────────────────────────────────────────────────────────────

class TestTimeoutBudget:
    def test_defaults(self) -> None:
        b = TimeoutBudget()
        assert b.total_s == 30.0
        assert b.embedding_s == 5.0

    def test_custom_values(self) -> None:
        b = TimeoutBudget(total_s=60.0, generation_s=30.0)
        assert b.total_s == 60.0
        assert b.generation_s == 30.0


# ── Quota enforcement ─────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def redis():
    return fakeredis.FakeRedis(decode_responses=False)


class TestEnforceQuota:
    @pytest.mark.asyncio
    async def test_no_limits_passes(self, redis) -> None:
        headers = await enforce_quota("t1", redis, max_per_day=0, max_per_minute=0)
        assert headers.rate_limit == 0
        assert headers.daily_limit == 0

    @pytest.mark.asyncio
    async def test_daily_limit_exceeded(self, redis) -> None:
        import datetime
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        await redis.set(f"quota:t1:queries:{today}", "100")

        with pytest.raises(QuotaExceeded) as exc_info:
            await enforce_quota("t1", redis, max_per_day=50, max_per_minute=0)
        assert exc_info.value.code == "DAILY_QUOTA_EXCEEDED"

    @pytest.mark.asyncio
    async def test_rpm_limit_exceeded(self, redis) -> None:
        import time
        bucket = int(time.time() // 60)
        await redis.set(f"quota:t1:rpm:{bucket}", "60")

        with pytest.raises(QuotaExceeded) as exc_info:
            await enforce_quota("t1", redis, max_per_day=0, max_per_minute=30)
        assert exc_info.value.code == "RATE_LIMIT_EXCEEDED"
        assert exc_info.value.retry_after_s == 60

    @pytest.mark.asyncio
    async def test_free_tier_blocks_chat(self, redis) -> None:
        with pytest.raises(QuotaExceeded) as exc_info:
            await enforce_quota(
                "t1", redis, max_per_day=0, max_per_minute=0,
                request_type="chat", llm_enabled=False,
            )
        assert exc_info.value.code == "LLM_NOT_ENABLED_ON_FREE_TIER"

    @pytest.mark.asyncio
    async def test_free_tier_allows_search(self, redis) -> None:
        # search is allowed even when llm_enabled=False
        headers = await enforce_quota(
            "t1", redis, max_per_day=0, max_per_minute=0,
            request_type="search", llm_enabled=False,
        )
        assert headers is not None

    @pytest.mark.asyncio
    async def test_headers_returned_on_success(self, redis) -> None:
        headers = await enforce_quota("t1", redis, max_per_day=100, max_per_minute=60)
        assert headers.daily_used == 1
        assert headers.rate_remaining == 59

    @pytest.mark.asyncio
    async def test_counter_incremented_each_call(self, redis) -> None:
        for _ in range(3):
            await enforce_quota("t1", redis, max_per_day=100, max_per_minute=100)
        import datetime
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        val = int(await redis.get(f"quota:t1:queries:{today}") or 0)
        assert val == 3
