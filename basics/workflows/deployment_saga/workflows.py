"""
DeploymentSagaWorkflow — sequential 5-stage deployment pipeline with full cascade rollback.

Saga guarantees:
  - Each completed state-changing step registers a compensation.
  - On any failure or LLM no-go, all registered compensations run in exact reverse order.

Pipeline stages:
  1. provision_resources    → compensation: deprovision_resources
  2. deploy_to_staging      → compensation: undeploy_staging
  3. run_integration_tests  → LLM go/no-go gate  (stateless — no compensation)
  4. deploy_to_production   → compensation: undeploy_production
  5. update_dns             → compensation: revert_dns

The LLM makes exactly one decision: after integration tests, proceed or abort.
Temporal guarantees each step runs durably. Saga chain guarantees rollback order.
"""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from .models import DeploymentReport, GoNoGo

_TIMEOUT = timedelta(seconds=30)
_STEP_RETRY = RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=1))
_COMP_RETRY = RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1))
_LLM_RETRY = RetryPolicy(maximum_attempts=1)

# None = stateless / safe to leave in place — no rollback needed.
COMPENSATIONS: dict[str, str | None] = {
    "provision_resources":   "deprovision_resources",
    "deploy_to_staging":     "undeploy_staging",
    "run_integration_tests": None,
    "deploy_to_production":  "undeploy_production",
    "update_dns":            "revert_dns",
}


@workflow.defn(sandboxed=False)
class DeploymentSagaWorkflow:

    @workflow.run
    async def run(self, deployment_id: str) -> str:
        completed: list[str] = []
        saga_chain: list[tuple[str, str]] = []  # (stage, compensation_activity_name)
        compensations_run: list[str] = []

        async def _step(stage: str) -> None:
            """Execute one pipeline stage; register compensation on success."""
            await workflow.execute_activity(
                stage,
                deployment_id,
                start_to_close_timeout=_TIMEOUT,
                retry_policy=_STEP_RETRY,
            )
            completed.append(stage)
            comp = COMPENSATIONS.get(stage)
            if comp is not None:
                saga_chain.append((stage, comp))

        async def _rollback(aborted_at: str) -> DeploymentReport:
            """Run compensation chain in reverse; return a failed report."""
            for stage_name, comp_name in reversed(saga_chain):
                workflow.logger.info(
                    "Compensation", extra={"for": stage_name, "running": comp_name}
                )
                await workflow.execute_activity(
                    comp_name,
                    deployment_id,
                    start_to_close_timeout=_TIMEOUT,
                    retry_policy=_COMP_RETRY,
                )
                compensations_run.append(comp_name)
            return DeploymentReport(
                deployment_id=deployment_id,
                completed_stages=list(completed),
                compensations_run=list(compensations_run),
                succeeded=False,
                aborted_at=aborted_at,
                final_status="aborted",
            )

        # ── Stage 1: Provision ────────────────────────────────────────────────
        try:
            await _step("provision_resources")
        except ActivityError as exc:
            workflow.logger.error("provision_resources failed", extra={"err": str(exc)})
            return DeploymentReport(
                deployment_id=deployment_id,
                completed_stages=[],
                compensations_run=[],
                succeeded=False,
                aborted_at="provision_resources",
                final_status="aborted",
            ).model_dump_json()

        # ── Stage 2: Staging ──────────────────────────────────────────────────
        try:
            await _step("deploy_to_staging")
        except ActivityError as exc:
            workflow.logger.error("deploy_to_staging failed", extra={"err": str(exc)})
            return (await _rollback("deploy_to_staging")).model_dump_json()

        # ── Stage 3: Integration tests ────────────────────────────────────────
        try:
            await _step("run_integration_tests")
        except ActivityError as exc:
            workflow.logger.error("run_integration_tests failed", extra={"err": str(exc)})
            return (await _rollback("run_integration_tests")).model_dump_json()

        # LLM go/no-go gate — one decision point in the pipeline
        gonogo_json: str = await workflow.execute_activity(
            "evaluate_test_results",
            deployment_id,
            start_to_close_timeout=_TIMEOUT,
            retry_policy=_LLM_RETRY,
        )
        gonogo = GoNoGo.model_validate_json(gonogo_json)
        workflow.logger.info(
            "LLM go/no-go decision",
            extra={"proceed": gonogo.proceed, "reason": gonogo.reason},
        )
        if not gonogo.proceed:
            workflow.logger.warning("LLM aborted deployment", extra={"reason": gonogo.reason})
            return (await _rollback("evaluate_test_results")).model_dump_json()

        # ── Stage 4: Production ───────────────────────────────────────────────
        try:
            await _step("deploy_to_production")
        except ActivityError as exc:
            workflow.logger.error("deploy_to_production failed", extra={"err": str(exc)})
            return (await _rollback("deploy_to_production")).model_dump_json()

        # ── Stage 5: DNS ──────────────────────────────────────────────────────
        try:
            await _step("update_dns")
        except ActivityError as exc:
            workflow.logger.error("update_dns failed", extra={"err": str(exc)})
            return (await _rollback("update_dns")).model_dump_json()

        # ── Success ───────────────────────────────────────────────────────────
        workflow.logger.info("Deployment succeeded", extra={"deployment_id": deployment_id})
        return DeploymentReport(
            deployment_id=deployment_id,
            completed_stages=list(completed),
            compensations_run=[],
            succeeded=True,
            aborted_at=None,
            final_status="succeeded",
        ).model_dump_json()
