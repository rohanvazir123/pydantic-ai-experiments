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
    stream_agent,
    traced_agent_run,
)
from knowledge.agent.intent_classifier import classify_intent
from knowledge.agent.judge import judge as run_judge
from knowledge.validation.pipeline import contains_injection
from knowledge.validation.pii_scanner import has_pii, scan_pii
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
    ABSTAINED_PII       = "abstained_pii"


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
    PipelineStatus.ABSTAINED_PII: (
        "Your request contains or would surface personal information. "
        "Please remove personal details and try again."
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

        # ── PII scan: query ───────────────────────────────────────────────────
        if self._settings.pii_scan_enabled:
            t0 = time.monotonic()
            if await has_pii(query):
                t["pii_query"] = int((time.monotonic() - t0) * 1000)
                await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
                return RAGResponse(
                    answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_PII],
                    status=PipelineStatus.ABSTAINED_PII,
                    pipeline_latency_ms=t,
                    abstention_layer=0,
                    abstention_reason="pii_in_query",
                    request_id=request_id,
                )
            t["pii_query"] = int((time.monotonic() - t0) * 1000)

        # ── Intent classification (nano model, 2 s timeout, fallback=factual) ─
        t0 = time.monotonic()
        intent = await classify_intent(query, settings=self._settings)
        ctx.intent = intent
        t["intent"] = int((time.monotonic() - t0) * 1000)

        # ── Layer 1: Retrieval confidence gate ────────────────────────────────
        k_effective = max(1, int(self._settings.judge_k * intent.k_multiplier))
        t0 = time.monotonic()
        results = await self._retriever.retrieve_with_confidence(
            query, corpus_ids, tenant_id,
            k=k_effective,
            include_graph=intent.include_graph,
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

        # ── PII scan: answer ──────────────────────────────────────────────────
        if self._settings.pii_scan_enabled:
            t0 = time.monotonic()
            detected = await scan_pii(gen.answer)
            t["pii_answer"] = int((time.monotonic() - t0) * 1000)
            if detected:
                logger.warning("PII detected in answer (%s) — abstaining", detected)
                await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
                return RAGResponse(
                    answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_PII],
                    status=PipelineStatus.ABSTAINED_PII,
                    pipeline_latency_ms=t,
                    abstention_layer=2,
                    abstention_reason=f"pii_in_answer:{','.join(detected)}",
                    request_id=request_id,
                )

        # ── Layer 3: Judge gate (passthrough when judge_enabled=False) ───────
        judge_confidence: float | None = None
        low_conf_warning = False

        if self._settings.judge_enabled:
            t0 = time.monotonic()
            judge_result = await run_judge(
                query=query,
                context=context_text,    # NO chunk_id metadata passed to judge
                answer=gen.answer,
                settings=self._settings,
            )
            t["judge"] = int((time.monotonic() - t0) * 1000)

            if (
                judge_result.verdict == "unsupported"
                or judge_result.confidence < self._settings.judge_confidence_threshold
            ):
                await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
                await registry.fire(HookPoint.POST_LLM, ctx)
                return RAGResponse(
                    answer=_ABSTAIN_MSG[PipelineStatus.ABSTAINED_JUDGE],
                    status=PipelineStatus.ABSTAINED_JUDGE,
                    pipeline_latency_ms=t,
                    abstention_layer=3,
                    abstention_reason=f"judge:{judge_result.verdict}",
                    request_id=request_id,
                )

            judge_confidence = judge_result.confidence
            low_conf_warning = judge_result.verdict == "partial"

        await registry.fire(HookPoint.POST_LLM, ctx)

        # ── Answered ──────────────────────────────────────────────────────────
        answer = gen.answer
        if low_conf_warning:
            answer += _PARTIAL_NOTE

        return RAGResponse(
            answer=answer,
            status=PipelineStatus.ANSWERED,
            confidence=judge_confidence,
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

        # V1-V6 validation (same gates as blocking path)
        from knowledge.validation.pipeline import ValidationPipeline
        vp = ValidationPipeline(settings=self._settings)
        validation_error = await vp.validate(ctx)
        if validation_error:
            yield _sse({"abstained": True, "layer": 0, "reason": validation_error.message})
            return

        # PII scan: query
        if self._settings.pii_scan_enabled and await has_pii(query):
            yield _sse({"abstained": True, "layer": 0, "reason": "pii_in_query"})
            return

        # Intent classification (nano model, 2 s timeout, fallback=factual)
        intent = await classify_intent(query, settings=self._settings)
        ctx.intent = intent

        # Layer 1
        k_effective = max(1, int(self._settings.judge_k * intent.k_multiplier))
        results = await self._retriever.retrieve_with_confidence(
            query, corpus_ids, tenant_id,
            k=k_effective,
            include_graph=intent.include_graph,
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
        # Build context-augmented prompt for the text streaming agent
        context_text = self._format_context(results)
        augmented_query = (
            f"Use the following source documents to answer the question. "
            f"Each document is enclosed in <document> tags.\n\n"
            f"{context_text}\n\n"
            f"Question: {query}"
        )

        try:
            async with stream_agent.run_stream(
                augmented_query,
                deps=state,
                message_history=message_history or [],
            ) as streamed:
                full_text = ""
                async for delta in streamed.stream_text(delta=True):
                    full_text += delta
                    yield _sse({"delta": delta})

                # Best-effort: extract citations from structured output.
                # Small models (llama3.2:3b) may not produce valid structured
                # output — if parsing fails, stream still delivered the answer.
                citations: list[dict] = []
                prompt_tokens    = 0
                completion_tokens = 0
                try:
                    output: GenerationResult = streamed.output
                    citations = [
                        {
                            "chunk_id":        str(c.chunk_id),
                            "document_title":  c.document_title,
                            "document_source": c.document_source,
                            "relevance_score": c.relevance_score,
                            "excerpt":         c.excerpt,
                        }
                        for c in (output.citations or [])
                    ]
                    usage = streamed.usage()
                    prompt_tokens    = usage.request_tokens or 0
                    completion_tokens = usage.response_tokens or 0
                except Exception:
                    pass  # answer was already streamed; skip citations

                # PII scan: answer (advisory — tokens already streamed)
                pii_warning: list[str] = []
                if self._settings.pii_scan_enabled:
                    t_pii = time.monotonic()
                    pii_warning = await scan_pii(full_text)
                    pii_ms = int((time.monotonic() - t_pii) * 1000)
                    if pii_warning:
                        logger.warning("PII detected in streamed answer (%s) in %dms", pii_warning, pii_ms)

                yield _sse({
                    "done":             True,
                    "citations":        citations,
                    "prompt_tokens":    prompt_tokens,
                    "completion_tokens": completion_tokens,
                    **({"pii_warning": pii_warning} if pii_warning else {}),
                })

            await registry.fire(HookPoint.POST_LLM, ctx)
        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            yield _sse({"error": str(exc)})
        finally:
            await state.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format retrieved chunks as context for the LLM.

        Each chunk is enclosed in <document> tags so the model can distinguish
        document data from prompt instructions.  Chunks whose content matches a
        known injection pattern are silently dropped to prevent poisoned-document
        attacks — a warning is logged so the omission is observable.
        """
        lines = []
        for r in results:
            content = r.content[:2000]
            if contains_injection(content):
                logger.warning(
                    "Injection pattern in chunk %s (%s) — excluded from context",
                    r.chunk_id,
                    r.document_title,
                )
                continue
            lines.append(f'<document title="{r.document_title}">\n{content}\n</document>')
        return "\n\n".join(lines)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
