"""
Tests for IncidentResponseWorkflow.

Seven scenarios:
  1. Happy path         — restart resolves immediately
  2. Retry then resolve — first action fails (activity error), second resolves
  3. Compensation       — scale_up worsens metrics, scale_down fires, clear_cache resolves
  4. Escalation         — nothing works, page_oncall called after max actions
  5. LLM reroutes       — LLM assessment redirects to rollback_deployment mid-loop
  6. Saga on escalation — scale_up succeeds (in chain), LLM escalates → scale_down compensates
  7. Saga on exhaustion — scale_up in chain, queue empties → compensation fires before page

Uses WorkflowEnvironment (no real Temporal server) and FunctionModel (no real LLM).
One ephemeral server is shared across the module via the temporal_env fixture.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity as _activity

from pydantic_ai.models.test import TestModel

from ..activities import InfraActivities, LLMActivities
from ..models import (
    ActionResult,
    AssessInput,
    IncidentAlert,
    IncidentAssessment,
    IncidentReport,
    PageInput,
    Severity,
    Triage,
)
from ..workflows import IncidentResponseWorkflow
from ..conftest import ALERT_HIGH, ALERT_CRITICAL

# All tests share a module-scoped event loop so the module-scoped temporal_env fixture works.
pytestmark = pytest.mark.asyncio(loop_scope="module")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_QUEUE = "test-incident-queue"


def _triage(actions: list[str], severity: Severity = Severity.HIGH) -> Triage:
    return Triage(
        severity=severity,
        likely_cause="deployment regression",
        recommended_actions=actions,
        reasoning="test triage",
    )


def _assessment(
    resolved: bool,
    next_action: str | None = None,
    escalate: bool = False,
) -> IncidentAssessment:
    return IncidentAssessment(
        resolved=resolved,
        next_action=next_action,
        escalate=escalate,
        reasoning="test assessment",
    )


def _make_llm_activities(
    triage_responses: list[Triage],
    assessment_responses: list[IncidentAssessment],
) -> LLMActivities:
    """Build LLMActivities backed by pre-scripted responses (no real LLM)."""
    triage_iter = iter(triage_responses)
    assess_iter = iter(assessment_responses)

    llm = LLMActivities(model=TestModel())

    async def _triage_run(prompt: str) -> object:
        class _R:
            output = next(triage_iter)
        return _R()

    async def _assess_run(prompt: str) -> object:
        class _R:
            output = next(assess_iter)
        return _R()

    llm._triage_agent.run = _triage_run  # type: ignore[method-assign]
    llm._assess_agent.run = _assess_run  # type: ignore[method-assign]
    return llm


def _base_activities(infra: InfraActivities, llm: LLMActivities) -> list[object]:
    return [
        infra.restart_service, infra.scale_up, infra.scale_down,
        infra.clear_cache, infra.rollback_deployment, infra.page_oncall,
        llm.triage_incident, llm.assess_after_action,
    ]


# ---------------------------------------------------------------------------
# Test 1: Happy path — restart resolves immediately
# ---------------------------------------------------------------------------

async def test_happy_path_restart_resolves(temporal_env: WorkflowEnvironment) -> None:
    """restart_service succeeds and LLM says resolved on first assessment."""
    infra = InfraActivities(scenario={
        "restart_service": {"success": True, "error_rate_delta": -0.44, "latency_delta": -2400},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["restart_service"])],
        assessment_responses=[_assessment(resolved=True)],
    )

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=_base_activities(infra, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_HIGH.model_dump_json(),
            id="test-happy-001",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    assert report.escalated is False
    assert report.final_status == "resolved"
    assert len(report.actions_taken) == 1
    assert report.actions_taken[0].action == "restart_service"
    assert report.compensations == []


# ---------------------------------------------------------------------------
# Test 2: First action fails, second resolves
# ---------------------------------------------------------------------------

async def test_first_action_fails_second_resolves(temporal_env: WorkflowEnvironment) -> None:
    """scale_up fails (activity error), LLM redirects to clear_cache which resolves."""
    infra = InfraActivities(scenario={
        "scale_up": {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "clear_cache": {"success": True, "error_rate_delta": -0.44, "latency_delta": -2300},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up", "clear_cache"])],
        assessment_responses=[
            _assessment(resolved=False, next_action="clear_cache"),
            _assessment(resolved=True),
        ],
    )

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=_base_activities(infra, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_HIGH.model_dump_json(),
            id="test-fail-then-resolve-002",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    assert report.final_status == "resolved"
    assert len(report.actions_taken) == 2
    assert report.actions_taken[0].action == "scale_up"
    assert report.actions_taken[0].success is False
    assert report.actions_taken[1].action == "clear_cache"
    assert report.actions_taken[1].success is True


# ---------------------------------------------------------------------------
# Test 3: Compensation — scale_up worsens metrics, scale_down fires
# ---------------------------------------------------------------------------

async def test_compensation_scale_up_worsens_then_clear_cache_resolves(
    temporal_env: WorkflowEnvironment,
) -> None:
    """scale_up increases error rate (compensation fires), then clear_cache resolves."""
    alert = ALERT_HIGH  # error_rate=0.45
    infra = InfraActivities(scenario={
        # scale_up: error_rate 0.45 → 0.60 (> 0.45 * 1.2 = 0.54 — worsened)
        "scale_up":   {"success": True, "error_rate_delta": +0.15, "latency_delta": +100},
        "scale_down": {"success": True, "error_rate_delta":  0.0,  "latency_delta":   0},
        "clear_cache":{"success": True, "error_rate_delta": -0.44, "latency_delta": -2300},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up", "clear_cache"])],
        assessment_responses=[
            _assessment(resolved=False, next_action="clear_cache"),
            _assessment(resolved=True),
        ],
    )

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=_base_activities(infra, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            alert.model_dump_json(),
            id="test-compensation-003",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    assert "scale_down (compensating scale_up)" in report.compensations
    action_names = [a.action for a in report.actions_taken]
    assert "scale_up" in action_names
    assert "scale_down" in action_names
    assert "clear_cache" in action_names


# ---------------------------------------------------------------------------
# Test 4: Escalation — nothing works, page_oncall called
# ---------------------------------------------------------------------------

async def test_escalation_after_all_actions_fail(temporal_env: WorkflowEnvironment) -> None:
    """All actions fail; LLM escalates after third attempt."""
    infra = InfraActivities(scenario={
        "restart_service":    {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "scale_up":           {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "rollback_deployment":{"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["restart_service", "scale_up", "rollback_deployment"], Severity.CRITICAL)],
        assessment_responses=[
            _assessment(resolved=False, escalate=False, next_action=None),
            _assessment(resolved=False, escalate=False, next_action=None),
            _assessment(resolved=False, escalate=True),
        ],
    )

    paged: list[str] = []

    @_activity.defn(name="page_oncall")
    async def page_oncall_spy(inp: PageInput) -> str:
        paged.append(inp.alert_id)
        return f"paged: {inp.alert_id}"

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=[
            infra.restart_service, infra.scale_up, infra.scale_down,
            infra.clear_cache, infra.rollback_deployment, page_oncall_spy,
            llm.triage_incident, llm.assess_after_action,
        ],
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_CRITICAL.model_dump_json(),
            id="test-escalation-004",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is False
    assert report.escalated is True
    assert report.final_status == "escalated"
    assert ALERT_CRITICAL.alert_id in paged


# ---------------------------------------------------------------------------
# Test 5: LLM reroutes mid-loop to rollback_deployment
# ---------------------------------------------------------------------------

async def test_llm_reroutes_to_rollback(temporal_env: WorkflowEnvironment) -> None:
    """LLM initial triage suggests restart, but assessment redirects to rollback which resolves."""
    infra = InfraActivities(scenario={
        "restart_service":    {"success": True, "error_rate_delta": -0.05, "latency_delta": -50},
        "rollback_deployment":{"success": True, "error_rate_delta": -0.44, "latency_delta": -2400},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["restart_service"])],
        assessment_responses=[
            _assessment(resolved=False, next_action="rollback_deployment"),
            _assessment(resolved=True),
        ],
    )

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=_base_activities(infra, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_HIGH.model_dump_json(),
            id="test-reroute-005",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    action_names = [a.action for a in report.actions_taken]
    assert "restart_service" in action_names
    assert "rollback_deployment" in action_names
    assert report.compensations == []


# ---------------------------------------------------------------------------
# Test 6: Saga chain fires on LLM escalation
# ---------------------------------------------------------------------------

async def test_saga_chain_fires_on_llm_escalation(temporal_env: WorkflowEnvironment) -> None:
    """scale_up succeeds (enters saga chain), LLM immediately escalates → scale_down compensation fires."""
    infra = InfraActivities(scenario={
        "scale_up":   {"success": True, "error_rate_delta": -0.1, "latency_delta": -50},
        "scale_down": {"success": True, "error_rate_delta":  0.0, "latency_delta":  0},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up"], Severity.CRITICAL)],
        assessment_responses=[_assessment(resolved=False, escalate=True)],
    )

    paged: list[str] = []

    @_activity.defn(name="page_oncall")
    async def page_spy(inp: PageInput) -> str:
        paged.append(inp.alert_id)
        return f"paged: {inp.alert_id}"

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=[
            infra.restart_service, infra.scale_up, infra.scale_down,
            infra.clear_cache, infra.rollback_deployment, page_spy,
            llm.triage_incident, llm.assess_after_action,
        ],
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_CRITICAL.model_dump_json(),
            id="test-saga-chain-006",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.escalated is True
    assert report.resolved is False
    assert "scale_down (compensating scale_up)" in report.compensations
    assert ALERT_CRITICAL.alert_id in paged


# ---------------------------------------------------------------------------
# Test 7: Saga chain fires when action queue exhausts (max-actions path)
# ---------------------------------------------------------------------------

async def test_saga_chain_fires_on_queue_exhaustion(temporal_env: WorkflowEnvironment) -> None:
    """scale_up succeeds, queue empties, loop exits → compensation chain fires before page."""
    infra = InfraActivities(scenario={
        "scale_up":   {"success": True, "error_rate_delta": -0.05, "latency_delta": -50},
        "scale_down": {"success": True, "error_rate_delta":  0.0,  "latency_delta":  0},
    })
    # Triage gives only [scale_up]; after assessment (not resolved, no next_action),
    # the queue is empty → loop breaks → "escalated_max_actions" path → compensation chain.
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up"])],
        assessment_responses=[_assessment(resolved=False, next_action=None, escalate=False)],
    )

    paged: list[str] = []

    @_activity.defn(name="page_oncall")
    async def page_spy(inp: PageInput) -> str:
        paged.append(inp.alert_id)
        return f"paged: {inp.alert_id}"

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentResponseWorkflow],
        activities=[
            infra.restart_service, infra.scale_up, infra.scale_down,
            infra.clear_cache, infra.rollback_deployment, page_spy,
            llm.triage_incident, llm.assess_after_action,
        ],
    ):
        result_json = await temporal_env.client.execute_workflow(
            IncidentResponseWorkflow.run,
            ALERT_HIGH.model_dump_json(),
            id="test-queue-exhausted-007",
            task_queue=TASK_QUEUE,
        )

    report = IncidentReport.model_validate_json(result_json)
    assert report.escalated is True
    assert report.final_status == "escalated_max_actions"
    assert "scale_down (compensating scale_up)" in report.compensations
    assert ALERT_HIGH.alert_id in paged
