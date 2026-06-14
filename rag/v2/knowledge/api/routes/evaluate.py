"""Evaluation routes — trigger offline eval runs and view results."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from knowledge.api.middleware import get_request_id, get_tenant_id
from knowledge.api.schemas import APIResponse, EvalRunRequest, EvalRunResponse
from knowledge.bus.schemas import EvalJob

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("/run", response_model=APIResponse[EvalRunResponse])
async def trigger_eval(body: EvalRunRequest, request: Request) -> APIResponse[EvalRunResponse]:
    """Trigger an offline evaluation run. Returns run_id immediately."""
    publisher  = getattr(request.app.state, "publisher", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if publisher is None:
        raise HTTPException(status_code=503, detail="Publisher not initialised")

    job = EvalJob(
        corpus_id=body.corpus_id,
        tenant_id=get_tenant_id() or "default",
        model_tier=body.model_tier,
        search_type=body.search_type,
        k=body.k,
        baseline_run_id=body.baseline_run_id,
    )
    await publisher.publish_eval_job(job)

    return APIResponse(
        request_id=request_id,
        data=EvalRunResponse(
            run_id=job.run_id,
            corpus_id=body.corpus_id,
            status="queued",
        ),
    )


@router.get("/run/{run_id}", response_model=APIResponse[dict])
async def get_eval_run(run_id: str, request: Request) -> APIResponse[dict]:
    """Poll eval run status and aggregated metrics."""
    request_id = get_request_id() or str(uuid.uuid4())
    # Phase 12 (Evaluation): SELECT FROM eval_runs JOIN eval_results
    return APIResponse(
        request_id=request_id,
        data={"run_id": run_id, "status": "queued", "note": "Phase 12 TODO"},
    )


@router.get("/compare", response_model=APIResponse[dict])
async def compare_runs(a: str, b: str, request: Request) -> APIResponse[dict]:
    """Regression diff between two eval runs."""
    request_id = get_request_id() or str(uuid.uuid4())
    # Phase 12 (Evaluation): reporter.generate_report()
    return APIResponse(
        request_id=request_id,
        data={"baseline": a, "current": b, "note": "Phase 12 TODO"},
    )
