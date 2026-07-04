"""
Tests for IncidentResponseWorkflow.

Five scenarios:
  1. Happy path         — restart resolves immediately
  2. Retry then resolve — first action fails (activity error), second resolves
  3. Compensation       — scale_up worsens metrics, scale_down fires, clear_cache resolves
  4. Escalation         — nothing works, page_oncall called after max actions
  5. LLM reroutes       — LLM assessment redirects to rollback_deployment mid-loop

Uses WorkflowEnvironment (no real Temporal server) and FunctionModel (no real LLM).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
import pytest_asyncio

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from pydantic_ai.models.test import TestModel
from pydantic_ai import Agent

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
    """
    Build LLMActivities backed by a TestModel that returns pre-scripted responses.

    TestModel returns a fixed structured output; we override via Agent.override()
    in a FunctionModel-style approach using a counter closure over pre-scripted lists.
    """
    triage_iter = iter(triage_responses)
    assess_iter = iter(assessment_responses)

    # Use TestModel with custom_result_text not available — instead we subclass
    # Agent and inject via override context. Simpler: patch the agents directly.
    llm = LLMActivities(model=TestModel())

    # Monkey-patch the run methods to return scripted values
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


# ---------------------------------------------------------------------------
# Test 1: Happy path — restart resolves immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_restart_resolves() -> None:
    """restart_service succeeds and LLM says resolved on first assessment."""
    infra = InfraActivities(scenario={
        "restart_service": {"success": True, "error_rate_delta": -0.44, "latency_delta": -2400},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["restart_service"])],
        assessment_responses=[_assessment(resolved=True)],
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra.restart_service, infra.scale_up, infra.scale_down,
                infra.clear_cache, infra.rollback_deployment, infra.page_oncall,
                llm.triage_incident, llm.assess_after_action,
            ],
        ):
            result_json = await env.client.execute_workflow(
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

@pytest.mark.asyncio
async def test_first_action_fails_second_resolves() -> None:
    """scale_up fails (activity error), LLM redirects to clear_cache which resolves."""
    infra = InfraActivities(scenario={
        "scale_up": {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "clear_cache": {"success": True, "error_rate_delta": -0.44, "latency_delta": -2300},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up", "clear_cache"])],
        # First assessment: scale_up failed, suggest clear_cache
        # Second assessment: clear_cache resolved
        assessment_responses=[
            _assessment(resolved=False, next_action="clear_cache"),
            _assessment(resolved=True),
        ],
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra.restart_service, infra.scale_up, infra.scale_down,
                infra.clear_cache, infra.rollback_deployment, infra.page_oncall,
                llm.triage_incident, llm.assess_after_action,
            ],
        ):
            result_json = await env.client.execute_workflow(
                IncidentResponseWorkflow.run,
                ALERT_HIGH.model_dump_json(),
                id="test-fail-then-resolve-002",
                task_queue=TASK_QUEUE,
            )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    assert report.final_status == "resolved"
    # scale_up (failed) + clear_cache (success)
    assert len(report.actions_taken) == 2
    assert report.actions_taken[0].action == "scale_up"
    assert report.actions_taken[0].success is False
    assert report.actions_taken[1].action == "clear_cache"
    assert report.actions_taken[1].success is True


# ---------------------------------------------------------------------------
# Test 3: Compensation — scale_up worsens metrics, scale_down fires
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compensation_scale_up_worsens_then_clear_cache_resolves() -> None:
    """scale_up increases error rate (compensation needed), then clear_cache resolves."""
    alert = ALERT_HIGH  # error_rate=0.45
    infra = InfraActivities(scenario={
        # scale_up makes things WORSE: error_rate goes from 0.45 → 0.60 (> 0.45 * 1.2 = 0.54)
        "scale_up":   {"success": True, "error_rate_delta": +0.15, "latency_delta": +100},
        "scale_down": {"success": True, "error_rate_delta":  0.0,  "latency_delta":   0},
        "clear_cache":{"success": True, "error_rate_delta": -0.44, "latency_delta": -2300},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["scale_up", "clear_cache"])],
        assessment_responses=[
            # After compensation (scale_down result used): not resolved, try clear_cache
            _assessment(resolved=False, next_action="clear_cache"),
            # After clear_cache: resolved
            _assessment(resolved=True),
        ],
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra.restart_service, infra.scale_up, infra.scale_down,
                infra.clear_cache, infra.rollback_deployment, infra.page_oncall,
                llm.triage_incident, llm.assess_after_action,
            ],
        ):
            result_json = await env.client.execute_workflow(
                IncidentResponseWorkflow.run,
                alert.model_dump_json(),
                id="test-compensation-003",
                task_queue=TASK_QUEUE,
            )

    report = IncidentReport.model_validate_json(result_json)
    assert report.resolved is True
    assert "scale_down (compensating scale_up)" in report.compensations
    # actions_taken: scale_up, scale_down (compensation), clear_cache
    action_names = [a.action for a in report.actions_taken]
    assert "scale_up" in action_names
    assert "scale_down" in action_names
    assert "clear_cache" in action_names


# ---------------------------------------------------------------------------
# Test 4: Escalation — nothing works, page_oncall called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalation_after_all_actions_fail() -> None:
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
    infra_with_page_spy = InfraActivities(scenario={
        "restart_service":    {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "scale_up":           {"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
        "rollback_deployment":{"success": False, "error_rate_delta": 0.0, "latency_delta": 0},
    })

    from temporalio import activity as _activity

    @_activity.defn(name="page_oncall")
    async def page_oncall_spy(inp: PageInput) -> str:
        paged.append(inp.alert_id)
        return f"paged: {inp.alert_id}"

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra_with_page_spy.restart_service,
                infra_with_page_spy.scale_up,
                infra_with_page_spy.scale_down,
                infra_with_page_spy.clear_cache,
                infra_with_page_spy.rollback_deployment,
                page_oncall_spy,
                llm.triage_incident, llm.assess_after_action,
            ],
        ):
            result_json = await env.client.execute_workflow(
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

@pytest.mark.asyncio
async def test_llm_reroutes_to_rollback() -> None:
    """LLM initial triage suggests restart, but assessment redirects to rollback which resolves."""
    infra = InfraActivities(scenario={
        "restart_service":    {"success": True, "error_rate_delta": -0.05, "latency_delta": -50},  # barely helps
        "rollback_deployment":{"success": True, "error_rate_delta": -0.44, "latency_delta": -2400},
    })
    llm = _make_llm_activities(
        triage_responses=[_triage(["restart_service"])],
        assessment_responses=[
            # restart barely helped — LLM says try rollback instead
            _assessment(resolved=False, next_action="rollback_deployment"),
            # rollback resolved it
            _assessment(resolved=True),
        ],
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IncidentResponseWorkflow],
            activities=[
                infra.restart_service, infra.scale_up, infra.scale_down,
                infra.clear_cache, infra.rollback_deployment, infra.page_oncall,
                llm.triage_incident, llm.assess_after_action,
            ],
        ):
            result_json = await env.client.execute_workflow(
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
