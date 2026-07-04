from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentAlert(BaseModel):
    alert_id: str
    service: str
    error_rate: float       # 0.0–1.0
    latency_p99_ms: int
    description: str


class Triage(BaseModel):
    severity: Severity
    likely_cause: str
    recommended_actions: list[str]  # ordered action names
    reasoning: str


class ActionResult(BaseModel):
    action: str
    success: bool
    error_rate_after: float
    latency_p99_after: int
    notes: str


class IncidentAssessment(BaseModel):
    resolved: bool
    next_action: str | None  # None if resolved or escalating
    escalate: bool
    reasoning: str


class IncidentReport(BaseModel):
    alert_id: str
    severity: Severity
    actions_taken: list[ActionResult]
    compensations: list[str]
    resolved: bool
    escalated: bool
    final_status: str


# ---------------------------------------------------------------------------
# Dataclass wrappers for multi-arg Temporal activities
# ---------------------------------------------------------------------------

@dataclass
class AssessInput:
    alert_json: str
    result_json: str


@dataclass
class PageInput:
    alert_id: str
    summary: str
