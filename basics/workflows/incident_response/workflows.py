"""
IncidentResponseWorkflow — Temporal + Pydantic AI fusion.

Control flow:
  1. LLM triages the alert → severity + ordered action list
  2. Loop (max MAX_ACTIONS):
       a. Execute next action activity (Temporal handles retries durably)
       b. If action succeeded, push onto the saga compensation chain
       c. If action worsened metrics → run compensation chain immediately
       d. LLM assesses outcome → resolved / try next action / escalate
  3. If escalating or loop exhausted → run compensation chain in reverse,
     then page on-call

The LLM drives *what* to do next after every step.
Temporal guarantees *that* each step runs durably and exactly-once.
Saga chain guarantees *compensatory actions cascade in reverse* on abort.
"""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from .models import (
    ActionResult,
    AssessInput,
    IncidentAlert,
    IncidentAssessment,
    IncidentReport,
    PageInput,
    Triage,
)

_ACTIVITY_RETRY = RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=1))
_LLM_RETRY = RetryPolicy(maximum_attempts=1)
_SHORT_TIMEOUT = timedelta(seconds=30)
_PAGE_TIMEOUT = timedelta(seconds=10)

# Saga compensation map — None means the action is stateless/irreversible-safe.
# On abort, completed actions with a non-None compensation are rolled back in reverse.
COMPENSATIONS: dict[str, str | None] = {
    "restart_service":     None,          # stateless — no rollback needed
    "scale_up":            "scale_down",
    "scale_down":          "scale_up",
    "clear_cache":         None,          # irreversible but harmless
    "rollback_deployment": None,          # rollback IS the safe state
}


@workflow.defn(sandboxed=False)
class IncidentResponseWorkflow:
    MAX_ACTIONS = 5

    @workflow.run
    async def run(self, alert_json: str) -> str:
        alert = IncidentAlert.model_validate_json(alert_json)
        # Saga chain: list of (action_name, compensation_name) for completed actions
        _saga_chain: list[tuple[str, str]] = []
        compensations: list[str] = []
        actions_taken: list[ActionResult] = []

        # ── Step 1: LLM triage ──────────────────────────────────────────────
        triage_json: str = await workflow.execute_activity(
            "triage_incident",
            alert_json,
            start_to_close_timeout=_SHORT_TIMEOUT,
            retry_policy=_LLM_RETRY,
        )
        triage = Triage.model_validate_json(triage_json)
        severity = triage.severity
        action_queue: list[str] = list(triage.recommended_actions)

        workflow.logger.info(
            "Triage complete",
            extra={"severity": severity, "actions": action_queue},
        )

        # ── Step 2: action loop ─────────────────────────────────────────────
        for _iteration in range(self.MAX_ACTIONS):
            if not action_queue:
                break

            action_name = action_queue.pop(0)
            workflow.logger.info("Attempting action", extra={"action": action_name})

            try:
                result_json: str = await workflow.execute_activity(
                    action_name,
                    alert_json,
                    start_to_close_timeout=_SHORT_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
                result = ActionResult.model_validate_json(result_json)
            except ActivityError as exc:
                result = ActionResult(
                    action=action_name,
                    success=False,
                    error_rate_after=alert.error_rate,
                    latency_p99_after=alert.latency_p99_ms,
                    notes=f"Activity exhausted retries: {exc}",
                )
                result_json = result.model_dump_json()
                actions_taken.append(result)
                workflow.logger.warning(
                    "Action failed after retries", extra={"action": action_name}
                )
            else:
                actions_taken.append(result)

                # Push onto saga chain if a compensation exists
                comp = COMPENSATIONS.get(action_name)
                if comp is not None:
                    _saga_chain.append((action_name, comp))

                # ── Compensation: action worsened metrics ────────────────────
                if result.error_rate_after > alert.error_rate * 1.2:
                    comp_name = COMPENSATIONS.get(action_name)
                    if comp_name:
                        workflow.logger.warning(
                            "Action worsened metrics — compensating",
                            extra={"action": action_name, "compensation": comp_name},
                        )
                        comp_json: str = await workflow.execute_activity(
                            comp_name,
                            alert_json,
                            start_to_close_timeout=_SHORT_TIMEOUT,
                            retry_policy=_ACTIVITY_RETRY,
                        )
                        comp_result = ActionResult.model_validate_json(comp_json)
                        compensations.append(f"{comp_name} (compensating {action_name})")
                        actions_taken.append(comp_result)
                        # Remove from saga chain — already compensated
                        _saga_chain = [(a, c) for a, c in _saga_chain if a != action_name]
                        result_json = comp_json

            # ── Step 3: LLM assessment ──────────────────────────────────────
            assess_input = AssessInput(alert_json=alert_json, result_json=result_json)
            assessment_json: str = await workflow.execute_activity(
                "assess_after_action",
                assess_input,
                start_to_close_timeout=_SHORT_TIMEOUT,
                retry_policy=_LLM_RETRY,
            )
            assessment = IncidentAssessment.model_validate_json(assessment_json)

            workflow.logger.info(
                "Assessment",
                extra={
                    "resolved": assessment.resolved,
                    "escalate": assessment.escalate,
                    "next_action": assessment.next_action,
                },
            )

            if assessment.resolved:
                return IncidentReport(
                    alert_id=alert.alert_id,
                    severity=severity,
                    actions_taken=actions_taken,
                    compensations=compensations,
                    resolved=True,
                    escalated=False,
                    final_status="resolved",
                ).model_dump_json()

            if assessment.escalate:
                chain_comps = await self._run_compensation_chain(alert_json, _saga_chain)
                compensations.extend(chain_comps)
                await self._page(alert.alert_id, actions_taken)
                return IncidentReport(
                    alert_id=alert.alert_id,
                    severity=severity,
                    actions_taken=actions_taken,
                    compensations=compensations,
                    resolved=False,
                    escalated=True,
                    final_status="escalated",
                ).model_dump_json()

            if assessment.next_action and assessment.next_action not in {
                a.action for a in actions_taken
            }:
                action_queue.insert(0, assessment.next_action)

        # ── Loop exhausted ──────────────────────────────────────────────────
        workflow.logger.error("Max actions reached — escalating")
        chain_comps = await self._run_compensation_chain(alert_json, _saga_chain)
        compensations.extend(chain_comps)
        await self._page(alert.alert_id, actions_taken)
        return IncidentReport(
            alert_id=alert.alert_id,
            severity=severity,
            actions_taken=actions_taken,
            compensations=compensations,
            resolved=False,
            escalated=True,
            final_status="escalated_max_actions",
        ).model_dump_json()

    async def _run_compensation_chain(
        self, alert_json: str, chain: list[tuple[str, str]]
    ) -> list[str]:
        """Run the saga compensation chain in reverse order, skipping None entries."""
        ran: list[str] = []
        for action_name, comp_name in reversed(chain):
            workflow.logger.info(
                "Running compensation", extra={"for": action_name, "comp": comp_name}
            )
            await workflow.execute_activity(
                comp_name,
                alert_json,
                start_to_close_timeout=_SHORT_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            ran.append(f"{comp_name} (compensating {action_name})")
        return ran

    async def _page(self, alert_id: str, actions_taken: list[ActionResult]) -> None:
        summary = f"Not resolved after {len(actions_taken)} action(s): " + ", ".join(
            a.action for a in actions_taken
        )
        await workflow.execute_activity(
            "page_oncall",
            PageInput(alert_id=alert_id, summary=summary),
            start_to_close_timeout=_PAGE_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
