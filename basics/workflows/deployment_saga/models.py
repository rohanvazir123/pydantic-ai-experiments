from __future__ import annotations

from pydantic import BaseModel


class StepResult(BaseModel):
    stage: str
    success: bool
    message: str


class GoNoGo(BaseModel):
    proceed: bool
    reason: str


class DeploymentReport(BaseModel):
    deployment_id: str
    completed_stages: list[str]
    compensations_run: list[str]
    succeeded: bool
    aborted_at: str | None = None
    final_status: str
