"""
Tests for DeploymentSagaWorkflow.

Seven scenarios:
  1. Happy path            — all 5 stages succeed, no compensations
  2. LLM no-go            — integration tests pass but LLM aborts → rollback staging + resources
  3. Staging fails         — deploy_to_staging raises → rollback resources only (chain length 1)
  4. Production fails      — deploy_to_production raises → rollback staging + resources (chain length 2)
  5. DNS fails             — update_dns raises → rollback production + staging + resources (chain length 3)
  6. Provision fails       — first step raises → no saga chain, no compensations, abort immediately
  7. DNS fails order check — exact rollback order: undeploy_production → undeploy_staging → deprovision_resources

One ephemeral Temporal server is shared across the module via the temporal_env fixture.
"""
from __future__ import annotations

import pytest

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity as _activity

from pydantic_ai.models.test import TestModel

from ..activities import DeploymentActivities, LLMReviewActivities
from ..models import DeploymentReport, GoNoGo, StepResult
from ..workflows import DeploymentSagaWorkflow

pytestmark = pytest.mark.asyncio(loop_scope="module")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_QUEUE = "test-deployment-queue"


def _make_llm(responses: list[GoNoGo]) -> LLMReviewActivities:
    """Build LLMReviewActivities whose agent returns pre-scripted GoNoGo responses."""
    it = iter(responses)
    llm = LLMReviewActivities(model=TestModel())

    async def _run(prompt: str) -> object:
        class _R:
            output = next(it)

        return _R()

    llm._agent.run = _run  # type: ignore[method-assign]
    return llm


def _all_activities(
    deploy: DeploymentActivities, llm: LLMReviewActivities
) -> list[object]:
    return [
        deploy.provision_resources,
        deploy.deprovision_resources,
        deploy.deploy_to_staging,
        deploy.undeploy_staging,
        deploy.run_integration_tests,
        deploy.deploy_to_production,
        deploy.undeploy_production,
        deploy.update_dns,
        deploy.revert_dns,
        llm.evaluate_test_results,
    ]


# ---------------------------------------------------------------------------
# Test 1: Happy path — all stages succeed
# ---------------------------------------------------------------------------


async def test_happy_path_all_stages_succeed(temporal_env: WorkflowEnvironment) -> None:
    """All 5 stages succeed; LLM says proceed. Report: succeeded=True, no compensations."""
    deploy = DeploymentActivities()
    llm = _make_llm([GoNoGo(proceed=True, reason="all tests passed")])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-001",
            id="test-saga-001",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is True
    assert report.final_status == "succeeded"
    assert report.compensations_run == []
    assert report.aborted_at is None
    assert len(report.completed_stages) == 5


# ---------------------------------------------------------------------------
# Test 2: LLM no-go → rollback staging + resources
# ---------------------------------------------------------------------------


async def test_llm_nogo_rollbacks_staging_and_resources(temporal_env: WorkflowEnvironment) -> None:
    """LLM says no-go after integration tests → undeploy_staging + deprovision_resources."""
    deploy = DeploymentActivities()
    llm = _make_llm([GoNoGo(proceed=False, reason="flaky tests detected")])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-002",
            id="test-saga-002",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "evaluate_test_results"
    # saga_chain = [provision, staging]; reversed → [undeploy_staging, deprovision_resources]
    assert report.compensations_run == ["undeploy_staging", "deprovision_resources"]


# ---------------------------------------------------------------------------
# Test 3: Staging fails → rollback resources only
# ---------------------------------------------------------------------------


async def test_staging_fails_rollbacks_only_resources(temporal_env: WorkflowEnvironment) -> None:
    """deploy_to_staging raises → saga chain = [provision] → deprovision_resources only."""
    deploy = DeploymentActivities(scenario={"deploy_to_staging": False})
    llm = _make_llm([])  # LLM gate never reached

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-003",
            id="test-saga-003",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "deploy_to_staging"
    assert report.compensations_run == ["deprovision_resources"]


# ---------------------------------------------------------------------------
# Test 4: Production fails → rollback staging + resources
# ---------------------------------------------------------------------------


async def test_production_fails_rollbacks_staging_and_resources(
    temporal_env: WorkflowEnvironment,
) -> None:
    """deploy_to_production raises → saga chain = [provision, staging] → 2 compensations."""
    deploy = DeploymentActivities(scenario={"deploy_to_production": False})
    llm = _make_llm([GoNoGo(proceed=True, reason="tests passed")])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-004",
            id="test-saga-004",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "deploy_to_production"
    # saga reversed: [undeploy_staging, deprovision_resources]
    assert report.compensations_run == ["undeploy_staging", "deprovision_resources"]


# ---------------------------------------------------------------------------
# Test 5: DNS fails → three-stage rollback
# ---------------------------------------------------------------------------


async def test_dns_fails_full_three_stage_rollback(temporal_env: WorkflowEnvironment) -> None:
    """update_dns raises → saga chain = [provision, staging, production] → 3 compensations."""
    deploy = DeploymentActivities(scenario={"update_dns": False})
    llm = _make_llm([GoNoGo(proceed=True, reason="tests passed")])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-005",
            id="test-saga-005",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "update_dns"
    assert report.compensations_run == [
        "undeploy_production",
        "undeploy_staging",
        "deprovision_resources",
    ]


# ---------------------------------------------------------------------------
# Test 6: Provision fails → no compensations, immediate abort
# ---------------------------------------------------------------------------


async def test_provision_fails_no_compensations(temporal_env: WorkflowEnvironment) -> None:
    """provision_resources fails before any stage completes → saga chain empty, no rollback."""
    deploy = DeploymentActivities(scenario={"provision_resources": False})
    llm = _make_llm([])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=_all_activities(deploy, llm),
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-006",
            id="test-saga-006",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    assert report.aborted_at == "provision_resources"
    assert report.compensations_run == []
    assert report.completed_stages == []


# ---------------------------------------------------------------------------
# Test 7: DNS fails — verify exact rollback ORDER via spy activities
# ---------------------------------------------------------------------------


async def test_dns_fails_rollback_order_is_exactly_reversed(
    temporal_env: WorkflowEnvironment,
) -> None:
    """
    When update_dns fails, the compensation chain must run in exact reverse insertion order:
    undeploy_production → undeploy_staging → deprovision_resources.

    Verified via spy activities that record the call sequence.
    Note: run_integration_tests has comp=None, so it never enters the saga chain.
    """
    call_order: list[str] = []

    @_activity.defn(name="deprovision_resources")
    async def spy_deprovision(deployment_id: str) -> str:
        call_order.append("deprovision_resources")
        return StepResult(stage="deprovision_resources", success=True, message="done").model_dump_json()

    @_activity.defn(name="undeploy_staging")
    async def spy_undeploy_staging(deployment_id: str) -> str:
        call_order.append("undeploy_staging")
        return StepResult(stage="undeploy_staging", success=True, message="done").model_dump_json()

    @_activity.defn(name="undeploy_production")
    async def spy_undeploy_production(deployment_id: str) -> str:
        call_order.append("undeploy_production")
        return StepResult(stage="undeploy_production", success=True, message="done").model_dump_json()

    deploy = DeploymentActivities(scenario={"update_dns": False})
    llm = _make_llm([GoNoGo(proceed=True, reason="tests passed")])

    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[DeploymentSagaWorkflow],
        activities=[
            deploy.provision_resources,
            spy_deprovision,
            deploy.deploy_to_staging,
            spy_undeploy_staging,
            deploy.run_integration_tests,
            deploy.deploy_to_production,
            spy_undeploy_production,
            deploy.update_dns,
            deploy.revert_dns,
            llm.evaluate_test_results,
        ],
    ):
        result_json = await temporal_env.client.execute_workflow(
            DeploymentSagaWorkflow.run,
            "deploy-007",
            id="test-saga-007",
            task_queue=TASK_QUEUE,
        )

    report = DeploymentReport.model_validate_json(result_json)
    assert report.succeeded is False
    # Exact order verified via spies:
    # inserted: provision → staging → production  →  reversed: production → staging → provision
    assert call_order == [
        "undeploy_production",
        "undeploy_staging",
        "deprovision_resources",
    ]
    assert report.compensations_run == call_order
