"""Unit tests for the retrieval pipeline.

No external services required — CrossEncoder and stores are mocked.
"""

import uuid
from unittest import mock

import fakeredis.aioredis as fakeredis
import pytest

from knowledge.ingestion.models import SearchResult
from knowledge.retrieval.fusion import (
    RRF_K,
    apply_confidence_filter,
    fuse_to_search_results,
    rrf_fuse,
    sigmoid,
)
from knowledge.retrieval.retriever import Retriever

# ── Sigmoid ───────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero_logit_is_half(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self) -> None:
        assert sigmoid(10.0) > 0.99

    def test_large_negative_approaches_zero(self) -> None:
        assert sigmoid(-10.0) < 0.01

    def test_symmetric(self) -> None:
        assert sigmoid(2.0) == pytest.approx(1 - sigmoid(-2.0))

    def test_output_in_0_1(self) -> None:
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert 0.0 < sigmoid(x) < 1.0


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _make_row(id_: str, **extra: object) -> dict:
    return {"id": id_, "content": f"content_{id_}", "metadata": {}, **extra}


class TestRRFFuse:
    def test_single_list_preserves_order(self) -> None:
        rows = [_make_row("a"), _make_row("b"), _make_row("c")]
        result = rrf_fuse([rows])
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_item_in_both_legs_gets_higher_score(self) -> None:
        leg1 = [_make_row("shared"), _make_row("only_1")]
        leg2 = [_make_row("shared"), _make_row("only_2")]
        result = rrf_fuse([leg1, leg2])
        scores = {r["id"]: r["raw_score"] for r in result}
        assert scores["shared"] > scores["only_1"]
        assert scores["shared"] > scores["only_2"]

    def test_rrf_score_formula(self) -> None:
        leg = [_make_row("x")]
        result = rrf_fuse([leg], k=RRF_K)
        # rank=0 (0-indexed) → score = 1 / (60 + 0 + 1) = 1/61
        expected = 1.0 / (RRF_K + 0 + 1)
        assert result[0]["raw_score"] == pytest.approx(expected)

    def test_raw_score_type_is_rrf(self) -> None:
        result = rrf_fuse([[_make_row("a")]])
        assert result[0]["raw_score_type"] == "rrf"

    def test_confidence_is_none_after_fusion(self) -> None:
        result = rrf_fuse([[_make_row("a")]])
        assert result[0]["confidence"] is None

    def test_top_k_limits_output(self) -> None:
        rows = [_make_row(str(i)) for i in range(10)]
        result = rrf_fuse([rows], top_k=3)
        assert len(result) == 3

    def test_empty_input_returns_empty(self) -> None:
        assert rrf_fuse([]) == []


class TestFuseToSearchResults:
    def test_returns_search_result_objects(self) -> None:
        row = {
            "id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "content": "hello",
            "metadata": {"title": "Doc", "source": "doc.md"},
        }
        results = fuse_to_search_results([[row]])
        assert isinstance(results[0], SearchResult)

    def test_confidence_none_after_fusion(self) -> None:
        row = {"id": str(uuid.uuid4()), "content": "test", "metadata": {}}
        results = fuse_to_search_results([[row]])
        assert results[0].confidence is None

    def test_raw_score_type_rrf(self) -> None:
        row = {"id": str(uuid.uuid4()), "content": "test", "metadata": {}}
        results = fuse_to_search_results([[row]])
        assert results[0].raw_score_type == "rrf"


# ── Confidence filter ─────────────────────────────────────────────────────────

def _sr(confidence: float | None) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="T",
        document_source="s",
        content="c",
        raw_score=0.5,
        raw_score_type="rrf",
        confidence=confidence,
    )


class TestApplyConfidenceFilter:
    def test_drops_below_threshold(self) -> None:
        results = [_sr(0.8), _sr(0.3), _sr(0.9)]
        filtered = apply_confidence_filter(results, min_confidence=0.5)
        confs = [r.confidence for r in filtered]
        assert 0.3 not in confs
        assert 0.8 in confs

    def test_keeps_exactly_at_threshold(self) -> None:
        results = [_sr(0.5)]
        filtered = apply_confidence_filter(results, min_confidence=0.5)
        assert len(filtered) == 1

    def test_none_confidence_passes_through(self) -> None:
        # None means reranker hasn't run — don't filter these
        results = [_sr(None), _sr(0.1)]
        filtered = apply_confidence_filter(results, min_confidence=0.5)
        assert any(r.confidence is None for r in filtered)

    def test_all_pass_when_threshold_zero(self) -> None:
        results = [_sr(0.01), _sr(0.001)]
        assert len(apply_confidence_filter(results, 0.0)) == 2


# ── Retriever ─────────────────────────────────────────────────────────────────

def _make_settings():
    with mock.patch.dict("os.environ", {
        "DATABASE_URL":     "postgresql://x:x@localhost/x",
        "AGE_DATABASE_URL": "postgresql://x:x@localhost/x",
    }, clear=True):
        from knowledge.config.settings import Settings
        return Settings(_env_file=None)  # type: ignore[call-arg]


def _make_retriever(**overrides: object) -> Retriever:
    return Retriever(settings=_make_settings(), **overrides)


class TestRetriever:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_stores(self) -> None:
        r = _make_retriever()
        results = await r.retrieve("query", ["c1"], "t1", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_l2_cache_hit_returns_early(self) -> None:
        mock_cache = mock.AsyncMock()
        cached = [{
            "chunk_id":       str(uuid.uuid4()),
            "document_id":    str(uuid.uuid4()),
            "document_title": "T",
            "document_source": "s",
            "content":        "hello",
            "metadata":       {},
            "raw_score":      0.9,
            "raw_score_type": "rrf",
            "confidence":     0.85,
        }]
        mock_cache.get_search.return_value = cached
        r = _make_retriever(cache=mock_cache)
        results = await r.retrieve("query", ["c1"], "t1")
        assert len(results) == 1
        assert results[0].content == "hello"
        mock_cache.get_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_layer1_gate_returns_empty_on_low_confidence(self) -> None:
        settings = _make_settings()
        # Patch the threshold to an impossibly high value on the instance
        r = Retriever(settings=settings)
        r._settings.__dict__["retrieval_confidence_threshold"] = 999.0

        # Patch retrieve so we get results but still fail the gate
        async def _fake_retrieve(*a: object, **kw: object):
            return [_sr(0.1), _sr(0.2)]

        r.retrieve = _fake_retrieve  # type: ignore[method-assign]
        results = await r.retrieve_with_confidence("q", ["c1"], "t1")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_as_context_formats_output(self) -> None:
        r = _make_retriever()

        async def _fake_retrieve(*a: object, **kw: object):
            return [
                SearchResult(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    document_title="HR Handbook",
                    document_source="hr.md",
                    content="Employees get 15 days PTO.",
                    raw_score=0.9,
                    raw_score_type="rrf",
                    confidence=0.85,
                )
            ]

        r.retrieve = _fake_retrieve  # type: ignore[method-assign]
        ctx = await r.retrieve_as_context("PTO policy", ["c1"], "t1")
        assert "chunk_id:" in ctx
        assert "HR Handbook" in ctx
        assert "15 days PTO" in ctx

    @pytest.mark.asyncio
    async def test_retrieve_as_context_no_results(self) -> None:
        r = _make_retriever()

        async def _fake_retrieve(*a: object, **kw: object):
            return []

        r.retrieve = _fake_retrieve  # type: ignore[method-assign]
        ctx = await r.retrieve_as_context("anything", ["c1"], "t1")
        assert "No relevant" in ctx

    def test_result_to_dict_round_trip(self) -> None:
        sr = _sr(0.9)
        d = Retriever._result_to_dict(sr)
        back = Retriever._dicts_to_results([d])
        assert len(back) == 1
        assert back[0].confidence == pytest.approx(0.9)


# ── Circuit breaker degradation ───────────────────────────────────────────────

class TestRetrieverCircuitBreakers:
    """Each retrieval leg degrades gracefully when its circuit breaker opens."""

    @staticmethod
    def _make_vs() -> mock.AsyncMock:
        vs = mock.AsyncMock()
        vs.semantic_search.return_value = []
        vs.text_search.return_value = []
        return vs

    @staticmethod
    def _make_emb(vector: list[float] | None = None) -> mock.AsyncMock:
        emb = mock.AsyncMock()
        emb.embed.return_value = vector or [0.1] * 5
        return emb

    @pytest.mark.asyncio
    async def test_no_redis_creates_no_circuit_breakers(self) -> None:
        r = _make_retriever()
        assert r._embed_cb is None
        assert r._sem_cb is None
        assert r._text_cb is None

    @pytest.mark.asyncio
    async def test_embed_circuit_open_skips_semantic_search(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=False)
        vs = self._make_vs()
        r = _make_retriever(vector_store=vs, embedder=self._make_emb(), redis=redis)

        await r._embed_cb._open()
        await r.retrieve("query", ["corpus1"], "tenant1")

        # embed circuit open → query_emb=[] → semantic leg never reaches the store
        vs.semantic_search.assert_not_called()
        # text leg is unaffected by the embed CB
        vs.text_search.assert_called()

    @pytest.mark.asyncio
    async def test_pgvector_search_circuit_open_skips_semantic_leg(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=False)
        vs = self._make_vs()
        r = _make_retriever(vector_store=vs, embedder=self._make_emb(), redis=redis)

        await r._sem_cb._open()
        await r.retrieve("query", ["corpus1"], "tenant1")

        vs.semantic_search.assert_not_called()
        vs.text_search.assert_called()

    @pytest.mark.asyncio
    async def test_pgvector_text_circuit_open_skips_text_leg(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=False)
        vs = self._make_vs()
        r = _make_retriever(vector_store=vs, embedder=self._make_emb(), redis=redis)

        await r._text_cb._open()
        await r.retrieve("query", ["corpus1"], "tenant1")

        vs.text_search.assert_not_called()
        vs.semantic_search.assert_called()

    @pytest.mark.asyncio
    async def test_embed_failures_open_circuit_after_threshold(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=False)
        vs = self._make_vs()
        emb = mock.AsyncMock()
        emb.embed.side_effect = ConnectionError("Ollama unreachable")
        r = _make_retriever(vector_store=vs, embedder=emb, redis=redis)
        r._embed_cb._open_threshold = 2  # speed up the test

        for _ in range(2):
            await r.retrieve("query", ["c1"], "t1")

        assert await r._embed_cb._get_state() == "OPEN"

    @pytest.mark.asyncio
    async def test_all_circuits_closed_calls_both_legs(self) -> None:
        redis = fakeredis.FakeRedis(decode_responses=False)
        vs = self._make_vs()
        r = _make_retriever(vector_store=vs, embedder=self._make_emb(), redis=redis)

        await r.retrieve("query", ["c1"], "t1")

        vs.semantic_search.assert_called_once()
        vs.text_search.assert_called_once()
