"""Ingest routes — submit jobs, poll status, SSE progress stream."""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from knowledge.api.middleware import get_request_id, get_tenant_id
from knowledge.api.schemas import (
    APIResponse,
    IngestJobResponse,
    IngestRequest,
    JobStatusResponse,
)
from knowledge.bus.schemas import IngestJob

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=APIResponse[IngestJobResponse])
async def submit_ingest(body: IngestRequest, request: Request) -> APIResponse[IngestJobResponse]:
    """Submit an ingestion job. Returns job_id immediately.

    The actual ingestion runs asynchronously in the ingest-worker process.
    Poll GET /ingest/{job_id}/status or stream GET /ingest/{job_id}/stream.
    """
    publisher  = getattr(request.app.state, "publisher", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if publisher is None:
        raise HTTPException(status_code=503, detail="Publisher not initialised")

    job = IngestJob(
        tenant_id=get_tenant_id() or "default",
        corpus_id=body.corpus_id,
        source_path=body.source_path,
        source_url=body.source_url,
        enable_graph_extraction=body.enable_graph_extraction,
        mode=body.mode,
    )
    await publisher.publish_ingest_job(job)

    return APIResponse(
        request_id=request_id,
        data=IngestJobResponse(
            job_id=job.job_id,
            status="queued",
            corpus_id=body.corpus_id,
            submitted_at=job.submitted_at.isoformat(),
        ),
    )


@router.get("/{job_id}/status", response_model=APIResponse[JobStatusResponse])
async def job_status(job_id: str, request: Request) -> APIResponse[JobStatusResponse]:
    """Poll job status from the Redis job hash."""
    redis      = getattr(request.app.state, "redis", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not initialised")

    data = await redis.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    def _d(key: bytes) -> str | None:
        v = data.get(key)
        return v.decode() if v else None

    chunks = _d(b"chunks_ingested")
    return APIResponse(
        request_id=request_id,
        data=JobStatusResponse(
            job_id=job_id,
            status=_d(b"status") or "unknown",
            progress=int(_d(b"progress") or 0),
            corpus_id=_d(b"corpus_id") or "",
            chunks_ingested=int(chunks) if chunks else None,
            error=_d(b"error"),
            submitted_at=_d(b"submitted_at"),
            completed_at=_d(b"completed_at"),
        ),
    )


@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream of job progress events filtered by job_id.

    Reads from the knowledge:events Redis stream.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not initialised")

    from collections.abc import AsyncGenerator

    async def _generate() -> AsyncGenerator[str]:
        last_id = "0"
        while True:
            messages = await redis.xread({"knowledge:events": last_id}, count=10, block=2000)
            for _stream, entries in (messages or []):
                for msg_id, fields in entries:
                    last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    payload_raw = fields.get(b"payload", b"{}").decode()
                    try:
                        payload = json.loads(payload_raw)
                        if payload.get("job_id") == job_id:
                            yield f"data: {payload_raw}\n\n"
                            if payload.get("event_type") in ("job_completed", "job_failed"):
                                return
                    except Exception:
                        pass

    return StreamingResponse(_generate(), media_type="text/event-stream")
