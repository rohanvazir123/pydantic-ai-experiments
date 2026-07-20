# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Unit tests for Phases 9-12 components.

No live services — auth stubs, scheduler math, working memory trim,
evaluation metric formulas.
"""

import uuid

import pytest

from knowledge.api.auth import TokenClaims, _verify_stub, check_corpus_access
from knowledge.evaluation.metrics.performance import estimate_cost
from knowledge.evaluation.metrics.retrieval import (
    compute_all_metrics,
    hit_rate,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)
from knowledge.evaluation.schemas import GoldSample
from knowledge.memory.working_memory import (
    assemble,
    count_tokens,
    format_history,
)
from knowledge.scheduler.job_store import compute_next_run_at

# ── Auth (Phase 9) ────────────────────────────────────────────────────────────

class TestTokenClaims:
    def test_stub_returns_dev_claims(self) -> None:
        claims = _verify_stub("any.token.here")
        assert claims.tenant_id == "default"
        assert "reader" in claims.roles

    def test_stub_decodes_valid_jwt_payload(self) -> None:
        import base64
        import json
        payload = {"sub": "user1", "tenant_id": "acme", "roles": ["admin"], "exp": 9999999999}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        token   = f"header.{encoded}.sig"
        claims  = _verify_stub(token)
        assert claims.sub       == "user1"
        assert claims.tenant_id == "acme"
        assert "admin" in claims.roles

    def test_check_corpus_access_passes(self) -> None:
        claims = TokenClaims(sub="u", tenant_id="t", roles=["reader"], exp=9999999999)
        check_corpus_access(claims, ["reader", "admin"])   # must not raise

    def test_check_corpus_access_denied(self) -> None:
        from fastapi import HTTPException
        claims = TokenClaims(sub="u", tenant_id="t", roles=["reader"], exp=9999999999)
        with pytest.raises(HTTPException) as exc:
            check_corpus_access(claims, ["admin"])
        assert exc.value.status_code == 403


# ── Scheduler (Phase 10) ──────────────────────────────────────────────────────

class TestComputeNextRunAt:
    def test_daily_cron(self) -> None:
        from datetime import UTC, datetime
        base = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)  # 10:00
        nxt  = compute_next_run_at("0 2 * * *", base)       # fires at 02:00 next day
        assert nxt.hour == 2
        assert nxt.day  == 2

    def test_every_minute_cron(self) -> None:
        from datetime import UTC, datetime
        base = datetime(2026, 6, 1, 10, 30, 0, tzinfo=UTC)
        nxt  = compute_next_run_at("* * * * *", base)
        assert nxt > base

    def test_weekly_cron(self) -> None:
        from datetime import UTC, datetime
        base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
        nxt  = compute_next_run_at("0 0 * * 0", base)    # every Sunday midnight
        assert (nxt - base).days <= 7


# ── Working Memory (Phase 10.5) ───────────────────────────────────────────────

class TestWorkingMemory:
    def test_count_tokens_rough_estimate(self) -> None:
        # _rough_token_count = len(text) // 4
        # "hello world " * 100 = 1200 chars → 1200 // 4 = 300 tokens
        text  = "hello world " * 100
        count = count_tokens([text])
        assert count == 300   # deterministic: chars // 4

    def test_count_tokens_multiple_parts(self) -> None:
        # Sum across list
        assert count_tokens(["aaaa", "aaaa"]) == 2  # 4//4=1 each

    def test_assemble_no_trim_needed(self) -> None:
        ctx = assemble(
            system_prompt="You are helpful.",
            user_memories=["User is an engineer"],
            history_messages=[{"role": "user", "content": "Hello"}],
            retrieved_chunks=[],
            query="What is RAG?",
            budget=8192,
        )
        assert ctx.context_truncated is False
        assert "User is an engineer" in ctx.user_memory_prefix

    def test_assemble_trims_chunks_first(self) -> None:
        # Create fake SearchResult-like objects
        class FakeResult:
            def __init__(self, confidence, content):
                self.chunk_id       = uuid.uuid4()
                self.document_title = "Doc"
                self.confidence     = confidence
                self.content        = content

        # Fill budget with a tiny limit to force trimming
        chunks = [FakeResult(0.9, "A" * 400), FakeResult(0.3, "B" * 400)]
        ctx = assemble(
            system_prompt="Sys",
            user_memories=[],
            history_messages=[],
            retrieved_chunks=chunks,
            query="q",
            budget=200,   # very tight
        )
        assert ctx.context_truncated is True

    def test_format_history(self) -> None:
        msgs = [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = format_history(msgs)
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text


# ── Evaluation metrics (Phase 12) ─────────────────────────────────────────────

class TestRetrievalMetrics:
    def test_hit_rate_positive(self) -> None:
        assert hit_rate([0, 1, 0]) == 1.0

    def test_hit_rate_negative(self) -> None:
        assert hit_rate([0, 0, 0]) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)

    def test_reciprocal_rank_second(self) -> None:
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_ndcg_perfect(self) -> None:
        assert ndcg_at_k([1, 1, 0], k=3) == pytest.approx(1.0)

    def test_ndcg_none(self) -> None:
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_compute_all_shape(self) -> None:
        rels   = [[1, 0, 0], [0, 1, 0]]
        totals = [1, 1]
        m      = compute_all_metrics(rels, totals, k=3)
        assert set(m.keys()) == {"hit_rate", "mrr", "precision", "recall", "ndcg"}

    def test_percentile_median(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


class TestCostEstimation:
    def test_local_model_zero_cost(self) -> None:
        assert estimate_cost("llama3.2:3b", 1000, 200) == 0.0

    def test_cloud_model_nonzero_cost(self) -> None:
        cost = estimate_cost("claude-haiku-4-5", 1000, 200)
        assert cost > 0

    def test_cost_scales_linearly(self) -> None:
        c1 = estimate_cost("claude-haiku-4-5", 1000, 0)
        c2 = estimate_cost("claude-haiku-4-5", 2000, 0)
        assert c2 == pytest.approx(c1 * 2)


class TestGoldSample:
    def test_default_difficulty(self) -> None:
        s = GoldSample(
            corpus_id="c1",
            query="What is PTO?",
            relevant_doc_sources=["team-handbook"],
        )
        assert s.difficulty == "medium"
        assert s.tags == []
