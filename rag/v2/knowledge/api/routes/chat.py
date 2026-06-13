"""Chat routes — blocking and SSE streaming.

POST /v1/chat         → ConfidenceAwarePipeline.run()  (blocking, full 3-gate)
POST /v1/chat/stream  → ConfidenceAwarePipeline.run_stream() (SSE, Layer 1 only)

Both use POST — the SSE streaming endpoint receives a JSON body (ChatRequest)
which contains session_id, corpus_ids, and message_history. EventSource (GET-only)
cannot carry a JSON body, so we use fetch + ReadableStream on the client.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from knowledge.api.middleware import (
    get_request_id,
    get_tenant_id,
    get_user_id,
    set_session_id,
    set_tenant_id,
)
from knowledge.api.schemas import APIResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_pipeline(request: Request) -> Any:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline


@router.post("", response_model=APIResponse[ChatResponse])
async def chat(body: ChatRequest, request: Request) -> APIResponse[ChatResponse]:
    """Run the confidence-aware pipeline (blocking).

    Returns a structured RAGResponse with citations, confidence, and cost fields.
    """
    pipeline   = _get_pipeline(request)
    request_id = get_request_id() or str(uuid.uuid4())
    set_session_id(body.session_id)   # inject into structlog context for this request
    # tenant_id will be set by JWT auth dependency in Phase 9;
    # for now set it from corpus_ids prefix as a dev stub
    tenant_hint = body.corpus_ids[0].split(":")[0] if body.corpus_ids else "default"
    set_tenant_id(get_tenant_id() or tenant_hint)

    from knowledge.agent.pipeline import RAGResponse as PipelineResponse
    result: PipelineResponse = await pipeline.run(
        query=body.query,
        corpus_ids=body.corpus_ids,
        tenant_id=get_tenant_id() or "default",
        user_id=get_user_id() or "",
        session_id=body.session_id,
        model_tier=body.model_tier if body.model_tier != "auto" else "small",
        message_history=body.message_history,
        request_id=request_id,
    )

    return APIResponse(
        request_id=request_id,
        data=ChatResponse(
            answer=result.answer,
            status=result.status,
            confidence=result.confidence,
            citations=[c.model_dump() for c in (result.citations or [])],
            low_confidence_warning=result.low_confidence_warning,
            pipeline_latency_ms=result.pipeline_latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
            model_tier_used=result.model_tier_used,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cache_hit=result.cache_hit,
            request_id=request_id,
            trace_url=result.trace_url,
            abstention_layer=result.abstention_layer,
            abstention_reason=result.abstention_reason,
        ),
    )


@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """SSE streaming chat — Layer 1 gate only; judge skipped for latency.

    Yields SSE events:
      data: {"delta": "<token>"}
      data: {"citations": [...], "done": true}
      data: {"error": "..."} on failure
      data: {"abstained": true, "layer": 1, "reason": "..."} on gate fire
    """
    pipeline = _get_pipeline(request)

    from collections.abc import AsyncGenerator

    async def _generate() -> AsyncGenerator[str]:
        async for event in pipeline.run_stream(
            query=body.query,
            corpus_ids=body.corpus_ids,
            tenant_id=get_tenant_id() or "default",
            user_id=get_user_id() or "",
            session_id=body.session_id,
            model_tier=body.model_tier if body.model_tier != "auto" else "small",
            message_history=body.message_history,
        ):
            yield event

    return StreamingResponse(_generate(), media_type="text/event-stream")
