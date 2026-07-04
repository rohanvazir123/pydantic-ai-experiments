"""
Temporal activities for the Deployment Saga workflow.

Two activity classes:
- DeploymentActivities  — simulated deployment operations driven by a scenario dict.
- LLMReviewActivities   — Pydantic AI go/no-go gate after integration tests.
"""
from __future__ import annotations

from temporalio import activity
from pydantic_ai import Agent

from .models import GoNoGo, StepResult

# ---------------------------------------------------------------------------
# Deployment activities
# ---------------------------------------------------------------------------

_DEFAULT_SCENARIO: dict[str, bool] = {
    "provision_resources":   True,
    "deploy_to_staging":     True,
    "run_integration_tests": True,
    "deploy_to_production":  True,
    "update_dns":            True,
}


class DeploymentActivities:
    """Simulated deployment operations. Pass a partial scenario dict to override defaults."""

    def __init__(self, scenario: dict[str, bool] | None = None) -> None:
        self._scenario: dict[str, bool] = {**_DEFAULT_SCENARIO, **(scenario or {})}

    def _run(self, stage: str, deployment_id: str) -> str:
        if not self._scenario.get(stage, True):
            raise RuntimeError(f"{stage} failed for {deployment_id} (simulated)")
        return StepResult(
            stage=stage, success=True, message=f"{stage} completed for {deployment_id}"
        ).model_dump_json()

    def _undo(self, stage: str, deployment_id: str) -> str:
        return StepResult(
            stage=stage, success=True, message=f"{stage} completed for {deployment_id}"
        ).model_dump_json()

    @activity.defn(name="provision_resources")
    async def provision_resources(self, deployment_id: str) -> str:
        return self._run("provision_resources", deployment_id)

    @activity.defn(name="deprovision_resources")
    async def deprovision_resources(self, deployment_id: str) -> str:
        return self._undo("deprovision_resources", deployment_id)

    @activity.defn(name="deploy_to_staging")
    async def deploy_to_staging(self, deployment_id: str) -> str:
        return self._run("deploy_to_staging", deployment_id)

    @activity.defn(name="undeploy_staging")
    async def undeploy_staging(self, deployment_id: str) -> str:
        return self._undo("undeploy_staging", deployment_id)

    @activity.defn(name="run_integration_tests")
    async def run_integration_tests(self, deployment_id: str) -> str:
        return self._run("run_integration_tests", deployment_id)

    @activity.defn(name="deploy_to_production")
    async def deploy_to_production(self, deployment_id: str) -> str:
        return self._run("deploy_to_production", deployment_id)

    @activity.defn(name="undeploy_production")
    async def undeploy_production(self, deployment_id: str) -> str:
        return self._undo("undeploy_production", deployment_id)

    @activity.defn(name="update_dns")
    async def update_dns(self, deployment_id: str) -> str:
        return self._run("update_dns", deployment_id)

    @activity.defn(name="revert_dns")
    async def revert_dns(self, deployment_id: str) -> str:
        return self._undo("revert_dns", deployment_id)


# ---------------------------------------------------------------------------
# LLM review activity
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a deployment gate reviewer. "
    "Given integration test results, decide whether to proceed to production. "
    "Respond with proceed=true only if all critical tests passed."
)


class LLMReviewActivities:
    """LLM-powered go/no-go gate. Inject any Pydantic AI model."""

    def __init__(self, model: object) -> None:
        self._agent: Agent[None, GoNoGo] = Agent(
            model,  # type: ignore[arg-type]
            output_type=GoNoGo,
            system_prompt=_LLM_SYSTEM,
        )

    @activity.defn(name="evaluate_test_results")
    async def evaluate_test_results(self, deployment_id: str) -> str:
        result = await self._agent.run(
            f"Integration tests completed for deployment {deployment_id}. Should we proceed to production?"
        )
        return result.output.model_dump_json()
