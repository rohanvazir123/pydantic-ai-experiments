"""Scheduler routes — CRUD for periodic ingestion jobs."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from knowledge.api.middleware import get_request_id, get_tenant_id
from knowledge.api.schemas import APIResponse, ScheduledJobRequest, ScheduledJobResponse

router = APIRouter(prefix="/scheduler/jobs", tags=["scheduler"])


@router.get("", response_model=APIResponse[list[ScheduledJobResponse]])
async def list_jobs(request: Request) -> APIResponse[list[ScheduledJobResponse]]:
    """List scheduled ingestion jobs for the current tenant."""
    job_store  = getattr(request.app.state, "job_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    jobs = []
    if job_store:
        rows = await job_store.list_by_tenant(get_tenant_id() or "default")
        for r in rows:
            jobs.append(ScheduledJobResponse(
                id=str(r["id"]),
                name=r["name"],
                corpus_id=r["corpus_id"],
                cron_expr=r["cron_expr"],
                mode=r["mode"],
                is_active=r["is_active"],
                next_run_at=str(r.get("next_run_at") or ""),
                last_run_at=str(r.get("last_run_at") or ""),
                last_status=r.get("last_status"),
            ))

    return APIResponse(request_id=request_id, data=jobs)


@router.post("", response_model=APIResponse[ScheduledJobResponse])
async def create_job(body: ScheduledJobRequest, request: Request) -> APIResponse[ScheduledJobResponse]:
    """Create a new scheduled ingestion job."""
    job_store  = getattr(request.app.state, "job_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if job_store is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")

    job_id = await job_store.create(
        tenant_id=get_tenant_id() or "default",
        name=body.name,
        source_type=body.source_type,
        source_config=body.source_config,
        corpus_id=body.corpus_id,
        cron_expr=body.cron_expr,
        mode=body.mode,
        enable_graph_extraction=body.enable_graph_extraction,
    )
    return APIResponse(
        request_id=request_id,
        data=ScheduledJobResponse(
            id=str(job_id),
            name=body.name,
            corpus_id=body.corpus_id,
            cron_expr=body.cron_expr,
            mode=body.mode,
            is_active=True,
        ),
    )


@router.delete("/{job_id}", response_model=APIResponse[dict])
async def delete_job(job_id: str, request: Request) -> APIResponse[dict]:
    """Cancel and remove a scheduled job."""
    job_store  = getattr(request.app.state, "job_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if job_store:
        await job_store.delete(job_id, get_tenant_id() or "default")

    return APIResponse(request_id=request_id, data={"deleted": True})


@router.post("/{job_id}/run-now", response_model=APIResponse[dict])
async def run_now(job_id: str, request: Request) -> APIResponse[dict]:
    """Trigger an immediate one-off ingest run for a scheduled job."""
    job_store  = getattr(request.app.state, "job_store", None)
    publisher  = getattr(request.app.state, "publisher", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if job_store and publisher:
        job = await job_store.get(job_id, get_tenant_id() or "default")
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        from knowledge.bus.schemas import IngestJob
        ingest = IngestJob(
            tenant_id=get_tenant_id() or "default",
            corpus_id=job["corpus_id"],
            source_path=job.get("source_config", {}).get("path"),
            mode=job["mode"],
            enable_graph_extraction=job.get("enable_graph_extraction", False),
        )
        await publisher.publish_ingest_job(ingest)
        return APIResponse(request_id=request_id, data={"job_id": ingest.job_id, "triggered": True})

    raise HTTPException(status_code=503, detail="Scheduler not initialised")
