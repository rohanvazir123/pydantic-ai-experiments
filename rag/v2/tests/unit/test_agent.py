"""Unit tests for Phase 6 — agent, judge, cost_guard, pipeline.

No live LLM or services required. Pydantic AI agents are mocked at the
run() boundary so we test pipeline logic, not model output.
"""

import uuid
from unittest import mock

import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from knowledge.agent.agent import CitationCheck, GenerationResult
from knowledge.agent.cost_guard import (
    SystemBudgetExceeded,
    TenantBudgetExceeded,
    check_cost_circuit_breaker,
    record_cost,
)
from knowledge.agent.judge import JudgeResult
from knowledge.agent.pipeline import (
    _PARTIAL_NOTE,
    ConfidenceAwarePipeline,
    PipelineStatus,
)
from knowledge.config.settings import Settings
from knowledge.ingestion.models import Citation, SearchResult

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(**overrides: str) -> Settings:
    import os
    base = {
        "DATABASE_URL":     "postgresql://x:x@localhost/x",
        "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
    }
    base.update(overrides)
    with mock.patch.dict(os.environ, base, clear=True):
        return Settings(_env_file=None)   # type: ignore[call-arg]


def _sr(confidence: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(),
        document_title="Doc", document_source="doc.md",
        content="Some content.", raw_score=0.9, raw_score_type="rrf",
        confidence=confidence,
    )


def _citation() -> Citation:
    return Citation(
        chunk_id=uuid.uuid4(), document_title="Doc",
        document_source="doc.md", relevance_score=0.85,
        excerpt="Some excerpt.",
    )


# ── Cost guard ────────────────────────────────────────────────────────────────

class TestCostGuard:
    @pytest_asyncio.fixture
    async def redis(self):
        return fakeredis.FakeRedis(decode_responses=False)

    @pytest.mark.asyncio
    async def test_no_limits_passes(self, redis) -> None:
        s = _make_settings(SYSTEM_DAILY_COST_LIMIT_USD="0.0")
        await check_cost_circuit_breaker("t1", redis, tenant_limit=0.0, settings=s)

    @pytest.mark.asyncio
    async def test_tenant_budget_exceeded(self, redis) -> None:
        import datetime
        month = datetime.datetime.now(datetime.UTC).strftime("%Y-%m")
        await redis.set(f"quota:t1:cost_usd:{month}", "10.0")

        s = _make_settings(SYSTEM_DAILY_COST_LIMIT_USD="0.0")
        with pytest.raises(TenantBudgetExceeded) as exc_info:
            await check_cost_circuit_breaker("t1", redis, tenant_limit=5.0, settings=s)
        assert exc_info.value.tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_system_budget_exceeded(self, redis) -> None:
        await redis.set("system:cost_usd:daily", "100.0")

        s = _make_settings(SYSTEM_DAILY_COST_LIMIT_USD="50.0")
        with pytest.raises(SystemBudgetExceeded):
            await check_cost_circuit_breaker("t1", redis, tenant_limit=0.0, settings=s)

    @pytest.mark.asyncio
    async def test_record_cost_increments_counters(self, redis) -> None:
        await record_cost("t1", redis, 0.01)
        import datetime
        month = datetime.datetime.now(datetime.UTC).strftime("%Y-%m")
        val = float(await redis.get(f"quota:t1:cost_usd:{month}") or 0)
        assert val == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_record_cost_zero_is_noop(self, redis) -> None:
        await record_cost("t1", redis, 0.0)
        import datetime
        month = datetime.datetime.now(datetime.UTC).strftime("%Y-%m")
        val = await redis.get(f"quota:t1:cost_usd:{month}")
        assert val is None


# ── ConfidenceAwarePipeline ───────────────────────────────────────────────────

def _make_pipeline(settings: Settings | None = None) -> ConfidenceAwarePipeline:
    mock_retriever = mock.AsyncMock()
    return ConfidenceAwarePipeline(
        retriever=mock_retriever,
        settings=settings or _make_settings(),
    )


class TestPipelineLayer1Gate:
    @pytest.mark.asyncio
    async def test_abstains_when_no_results(self) -> None:
        pipeline = _make_pipeline()
        pipeline._retriever.retrieve_with_confidence.return_value = []

        resp = await pipeline.run("q", ["c1"], "t1")
        assert resp.status == PipelineStatus.ABSTAINED_RETRIEVAL
        assert resp.abstention_layer == 1

    @pytest.mark.asyncio
    async def test_layer1_hook_fires_on_abstain(self) -> None:
        from knowledge.hooks.registry import HookPoint, registry

        fired: list[str] = []

        async def capture(ctx):
            fired.append("validation_fail")
            return ctx

        registry.register(HookPoint.ON_VALIDATION_FAIL, capture, name="_test_l1")
        try:
            pipeline = _make_pipeline()
            pipeline._retriever.retrieve_with_confidence.return_value = []
            await pipeline.run("q", ["c1"], "t1")
            assert "validation_fail" in fired
        finally:
            registry.clear(HookPoint.ON_VALIDATION_FAIL)


class TestPipelineLayer2Gate:
    @pytest.mark.asyncio
    async def test_abstains_on_uncited_claims(self) -> None:
        pipeline = _make_pipeline()
        pipeline._retriever.retrieve_with_confidence.return_value = [_sr()]

        gen = GenerationResult(
            answer="Some answer.",
            citations=[],
            citation_check=CitationCheck(is_trustworthy=False, uncited_claims=["claim 1"]),
        )

        mock_usage = mock.MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50

        mock_result = mock.MagicMock()
        mock_result.output = gen
        mock_result.usage.return_value = mock_usage

        with mock.patch(
            "knowledge.agent.pipeline.traced_agent_run",
            new_callable=mock.AsyncMock,
            return_value=mock_result,
        ):
            resp = await pipeline.run("q", ["c1"], "t1")

        assert resp.status == PipelineStatus.ABSTAINED_CITATION
        assert resp.abstention_layer == 2


class TestPipelineLayer3Gate:
    def _mock_gen_result(self) -> tuple:
        gen = GenerationResult(
            answer="The answer is yes.",
            citations=[_citation()],
            citation_check=CitationCheck(is_trustworthy=True),
        )
        mock_usage = mock.MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50

        mock_result = mock.MagicMock()
        mock_result.output = gen
        mock_result.usage.return_value = mock_usage
        return gen, mock_result

    @pytest.mark.asyncio
    async def test_abstains_on_unsupported_verdict(self) -> None:
        pipeline = _make_pipeline()
        pipeline._retriever.retrieve_with_confidence.return_value = [_sr()]
        _, mock_result = self._mock_gen_result()

        judge_result = JudgeResult(
            verdict="unsupported", confidence=0.9,
            reasoning="Claims not in passages."
        )

        with mock.patch("knowledge.agent.pipeline.traced_agent_run",
                        new_callable=mock.AsyncMock, return_value=mock_result), \
             mock.patch("knowledge.agent.pipeline.run_judge",
                        new_callable=mock.AsyncMock, return_value=judge_result):
            resp = await pipeline.run("q", ["c1"], "t1")

        assert resp.status == PipelineStatus.ABSTAINED_JUDGE
        assert resp.abstention_layer == 3

    @pytest.mark.asyncio
    async def test_answers_on_supported_verdict(self) -> None:
        pipeline = _make_pipeline()
        pipeline._retriever.retrieve_with_confidence.return_value = [_sr()]
        _, mock_result = self._mock_gen_result()

        judge_result = JudgeResult(
            verdict="supported", confidence=0.95,
            reasoning="All claims grounded."
        )

        with mock.patch("knowledge.agent.pipeline.traced_agent_run",
                        new_callable=mock.AsyncMock, return_value=mock_result), \
             mock.patch("knowledge.agent.pipeline.run_judge",
                        new_callable=mock.AsyncMock, return_value=judge_result):
            resp = await pipeline.run("q", ["c1"], "t1")

        assert resp.status == PipelineStatus.ANSWERED
        assert resp.confidence == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_partial_appends_uncertainty_note(self) -> None:
        pipeline = _make_pipeline()
        pipeline._retriever.retrieve_with_confidence.return_value = [_sr()]
        _, mock_result = self._mock_gen_result()

        judge_result = JudgeResult(
            verdict="partial", confidence=0.7,
            reasoning="Mostly grounded."
        )

        with mock.patch("knowledge.agent.pipeline.traced_agent_run",
                        new_callable=mock.AsyncMock, return_value=mock_result), \
             mock.patch("knowledge.agent.pipeline.run_judge",
                        new_callable=mock.AsyncMock, return_value=judge_result):
            resp = await pipeline.run("q", ["c1"], "t1")

        assert resp.status == PipelineStatus.ANSWERED
        assert resp.low_confidence_warning is True
        assert _PARTIAL_NOTE in resp.answer

    @pytest.mark.asyncio
    async def test_abstains_on_low_judge_confidence(self) -> None:
        s = _make_settings(JUDGE_CONFIDENCE_THRESHOLD="0.8")
        pipeline = _make_pipeline(settings=s)
        pipeline._retriever.retrieve_with_confidence.return_value = [_sr()]
        _, mock_result = self._mock_gen_result()

        judge_result = JudgeResult(
            verdict="supported", confidence=0.5,   # below 0.8 threshold
            reasoning="Low confidence verdict."
        )

        with mock.patch("knowledge.agent.pipeline.traced_agent_run",
                        new_callable=mock.AsyncMock, return_value=mock_result), \
             mock.patch("knowledge.agent.pipeline.run_judge",
                        new_callable=mock.AsyncMock, return_value=judge_result):
            resp = await pipeline.run("q", ["c1"], "t1")

        assert resp.status == PipelineStatus.ABSTAINED_JUDGE


class TestPipelineStatusEnum:
    def test_all_statuses_are_strings(self) -> None:
        for status in PipelineStatus:
            assert isinstance(status.value, str)

    def test_answered_is_answered(self) -> None:
        assert PipelineStatus.ANSWERED == "answered"
