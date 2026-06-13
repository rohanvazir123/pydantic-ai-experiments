"""Confidence-Aware Pipeline — three-layer gate orchestrator.

Layer 1: Retrieval confidence gate
  aggregate = Σ(confidence for top-K results)
  < threshold → abstained_retrieval (no LLM call)

Layer 2: Citation gate
  len(uncited_claims) > 0 → abstained_citation

Layer 3: Judge gate
  verdict = unsupported OR judge_confidence < threshold → abstained_judge
  verdict = partial → answer proceeds with uncertainty note

Streaming path (run_stream): Layer 1 only — judge is incompatible with
token-by-token streaming. Judge runs offline via eval.

Every gate fires the appropriate hook point.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from knowledge.agent.agent import (
    GenerationResult,
    RAGState,
    agent,
    traced_agent_run,
)
from knowledge.agent.judge import judge as run_judge
from knowledge.config.settings import Settings, load_settings
from knowledge.hooks.context import HookContext
from knowledge.hooks.registry import HookPoint, registry
from knowledge.ingestion.models import Citation, SearchResult

logger = logging.getLogger(__name__)


class PipelineStatus(StrEnum):
    ANSWERED            = "answered"
    ABSTAINED_RETRIEVAL = "abstained_retrieval"
    ABSTAINED_CITATION  = "abstained_citation"
    ABSTAINED_JUDGE     = "abstained_judge"


class RAGResponse(BaseModel):
    answer:                  str
    status:                  PipelineStatus
    confidence:              float | None          = None
    citations:               list[Citation] | None = None
    low_confidence_warning:  bool                  = False
    pipeline_latency_ms:     dict[str, int]        = {}
    # Cost fields
    estimated_cost_usd:      float                 = 0.0
    model_tier_used:         str                   = "small"
    prompt_tokens:           int                   = 0
    completion_tokens:       int                   = 0
    cache_hit:               str | None            = None
    # Observability
    request_id:              str                   = ""
    trace_url:               str | None            = None
    # Abstention details
    abstention_layer:        int | None            = None
    abstention_reason:       str | None            = None


_PARTIAL_NOTE = "\n\nNote: This answer may be incomplete based on the available context."

# Corpus-configurable abstention strings (never LLM-generated)
_ABSTAIN_MSG = {
    PipelineStatus.ABSTAINED_RETRIEVAL: (
        "I could not find relevant information for your question in this knowledge base."
    ),
    PipelineStatus.ABSTAINED_CITATION: (
        "The generated answer could not be fully attributed to the available sources. "
        "Please rephrase your question."
    ),
    PipelineStatus.ABSTAINED_JUDGE: (
        "The answer could not be verified against the source material. "
        "Please try a more specific question."
    ),
}


class ConfidenceAwarePipeline:
    """Orchestrates all three gates and returns a RAGResponse."""

    def __init__(
        self,
        retriever: Any,    # Retriever
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings  = settings or load_settings()

    # ── Blocking run (POST /chat) ─────────────────────────────────────────────

    async def run(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        user_id: str = "",
        session_id: str = "",
        model_tier: str = "small",
        message_history: list | None = None,
        request_id: str = "",
    ) -> RAGResponse:
        """Full 3-gate pipeline. Returns RAGResponse (may be an abstention)."""
        ctx = HookContext(
            request_id=request_id,
            query=query,
            corpus_ids=corpus_ids,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            model_tier=model_tier,
        )
        t: dict[str, int] = {}

        # ── Layer 1: Retrieval confidence gate ────────────────────────────────
        t0 = time.monotonic()
        results = await self._retriever.retrieve_with_confidence(
            query, corpus_ids, tenant_id, k=self._settings.judge_k
        )
        t["retrieval"] = int((time.monotonic() - t0) * 1000)

        ctx.retrieved_chunks     = results
        ctx.aggregate_confidence = sum(
            r.confidence for r in results if r.confidence is not None
        )
        await registry.fire(HookPoint.POST_RETRIEVE, ctx)

        if not results:
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return RAGResponse(
                answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_RETRIEVAL],
                status=PipelineStatus.ABSTAINED_RETRIEVAL,
                pipeline_latency_ms=t,
                abstention_layer=1,
                abstention_reason="aggregate_confidence_below_threshold",
                request_id=request_id,
            )

        # ── Assemble context for LLM ──────────────────────────────────────────
        context_text = self._format_context(results)
        low_conf = ctx.aggregate_confidence < self._settings.confidence_warn_threshold

        # ── PRE_LLM hook (cost guard fires here) ─────────────────────────────
        await registry.fire(HookPoint.PRE_LLM, ctx)

        # ── Layer 2: Agent + citation gate ────────────────────────────────────
        state = RAGState(
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
            corpus_ids=corpus_ids,
        )
        t0 = time.monotonic()
        try:
            result = await traced_agent_run(
                query, state,
                message_history=message_history,
                low_confidence=low_conf,
            )
            gen: GenerationResult = result.output
            usage = result.usage()
        finally:
            await state.close()
        t["generation"] = int((time.monotonic() - t0) * 1000)

        ctx.generation_result = gen
        ctx.llm_response      = gen

        if not gen.citation_check.is_trustworthy:
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return RAGResponse(
                answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_CITATION],
                status=PipelineStatus.ABSTAINED_CITATION,
                pipeline_latency_ms=t,
                abstention_layer=2,
                abstention_reason="uncited_claims",
                request_id=request_id,
            )

        # ── Layer 3: Judge gate ───────────────────────────────────────────────
        t0 = time.monotonic()
        judge_result = await run_judge(
            query=query,
            context=context_text,    # NO chunk_id metadata passed to judge
            answer=gen.answer,
            settings=self._settings,
        )
        t["judge"] = int((time.monotonic() - t0) * 1000)

        await registry.fire(HookPoint.POST_LLM, ctx)

        if (
            judge_result.verdict == "unsupported"
            or judge_result.confidence < self._settings.judge_confidence_threshold
        ):
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return RAGResponse(
                answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_JUDGE],
                status=PipelineStatus.ABSTAINED_JUDGE,
                pipeline_latency_ms=t,
                abstention_layer=3,
                abstention_reason=f"judge:{judge_result.verdict}",
                request_id=request_id,
            )

        # ── Answered ──────────────────────────────────────────────────────────
        answer = gen.answer
        low_conf_warning = judge_result.verdict == "partial"
        if low_conf_warning:
            answer += _PARTIAL_NOTE

        return RAGResponse(
            answer=answer,
            status=PipelineStatus.ANSWERED,
            confidence=judge_result.confidence,
            citations=gen.citations or [],
            low_confidence_warning=low_conf_warning,
            pipeline_latency_ms=t,
            estimated_cost_usd=0.0,   # computed by cost_guard post-run
            model_tier_used=model_tier,
            prompt_tokens=usage.request_tokens or 0,
            completion_tokens=usage.response_tokens or 0,
            request_id=request_id,
        )

    # ── Streaming run (POST /chat/stream) ─────────────────────────────────────

    async def run_stream(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        user_id: str = "",
        session_id: str = "",
        model_tier: str = "small",
        message_history: list | None = None,
    ) -> AsyncGenerator[str]:
        """Streaming path — Layer 1 gate only; judge skipped for latency.

        Yields SSE-formatted data lines.
        """
        ctx = HookContext(
            query=query, corpus_ids=corpus_ids,
            tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        )

        # Layer 1
        results = await self._retriever.retrieve_with_confidence(
            query, corpus_ids, tenant_id, k=self._settings.judge_k
        )
        ctx.retrieved_chunks = results
        await registry.fire(HookPoint.POST_RETRIEVE, ctx)

        if not results:
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            yield _sse({"abstained": True, "layer": 1, "reason": "no_retrieval"})
            return

        await registry.fire(HookPoint.PRE_LLM, ctx)

        state = RAGState(
            user_id=user_id, tenant_id=tenant_id,
            session_id=session_id, corpus_ids=corpus_ids,
        )
        try:
            async with agent.run_stream(
                query,
                deps=state,
                message_history=message_history or [],
            ) as streamed:
                async for delta in streamed.stream_text(delta=True):
                    yield _sse({"delta": delta})

                # Citations from structured output after stream completes
                try:
                    output: GenerationResult = streamed.output
                    citations = [
                        {
                            "chunk_id":       str(c.chunk_id),
                            "document_title": c.document_title,
                            "document_source": c.document_source,
                            "relevance_score": c.relevance_score,
                            "excerpt":        c.excerpt,
                        }
                        for c in (output.citations or [])
                    ]
                except Exception:
                    citations = []

                usage = streamed.usage()
                yield _sse({
                    "done":             True,
                    "citations":        citations,
                    "prompt_tokens":    usage.request_tokens or 0,
                    "completion_tokens": usage.response_tokens or 0,
                })

            await registry.fire(HookPoint.POST_LLM, ctx)
        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            yield _sse({"error": "Internal server error"})
        finally:
            await state.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format chunks for the judge — NO chunk_id metadata."""
        lines = []
        for r in results:
            lines.append(f"Source: {r.document_title}\n{r.content[:800]}")
        return "\n\n---\n\n".join(lines)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
