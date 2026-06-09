"""Feedback and implicit signal routes."""

import uuid

from fastapi import APIRouter, Request

from knowledge.api.middleware import get_request_id
from knowledge.api.schemas import APIResponse, FeedbackRequest, SignalRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=APIResponse[dict])
async def submit_feedback(body: FeedbackRequest, request: Request) -> APIResponse[dict]:
    """Submit explicit user feedback (thumbs up/down, rating, correction)."""
    request_id = get_request_id() or str(uuid.uuid4())
    # Phase 12 (Evaluation): INSERT INTO user_feedback
    return APIResponse(request_id=request_id, data={"received": True})


@router.post("/signals", response_model=APIResponse[dict])
async def submit_signal(body: SignalRequest, request: Request) -> APIResponse[dict]:
    """Submit implicit behavioural signal (service token only)."""
    request_id = get_request_id() or str(uuid.uuid4())
    # Phase 12 (Evaluation): INSERT INTO implicit_signals
    return APIResponse(request_id=request_id, data={"received": True})
