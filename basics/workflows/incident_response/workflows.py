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
    Severity,
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

# Self-healing sequence injected automatically after a non-heal action fails.
# Actions in this list are skipped as triggers (avoids infinite loops).
SELF_HEAL_SEQUENCE: list[str] = ["clear_cache", "restart_service"]


@workflow.defn(sandboxed=False)
class IncidentResponseWorkflow:
    MAX_ACTIONS = 5

    @workflow.run
    async def run(self, alert_json: str) -> str:
        alert = IncidentAlert.model_validate_json(alert_json)
        # SAGA: chain of (action_name, compensation_name) for completed, not-yet-compensated
        # actions. Appended to as actions succeed, pruned as compensations run inline, and
        # replayed in reverse if the incident ends unresolved.
        saga_chain: list[tuple[str, str]] = []
        compensations: list[str] = []
        actions_taken: list[ActionResult] = []
        # Self-heal fires at most once per workflow run to avoid runaway loops.
        _self_heal_attempted: bool = False

        # ── Step 1: LLM triage ──────────────────────────────────────────────
        triage = await self._triage(alert_json)
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

            result, result_json = await self._execute_action(action_name, alert_json, alert)

            if result.success:
                actions_taken.append(result)

                # SAGA: enroll the action in the compensation chain so it can be
                # unwound later if the incident is never resolved.
                comp_name = COMPENSATIONS.get(action_name)
                if comp_name is not None:
                    saga_chain.append((action_name, comp_name))

                # SAGA: an action that makes things worse is compensated immediately,
                # rather than waiting for the chain to unwind at the end.
                comp_result, comp_label = await self._maybe_compensate_inline(
                    action_name, alert_json, result, alert
                )
                if comp_result is not None and comp_label is not None:
                    compensations.append(comp_label)
                    actions_taken.append(comp_result)
                    # SAGA: already compensated — drop it from the chain so the
                    # final unwind (if any) doesn't compensate it a second time.
                    saga_chain = [(a, c) for a, c in saga_chain if a != action_name]
                    result_json = comp_result.model_dump_json()
            else:
                actions_taken.append(result)
                workflow.logger.warning(
                    "Action failed after retries", extra={"action": action_name}
                )
                # Self-heal: if the failed action isn't itself a heal action and the
                # sequence hasn't fired yet, prepend the heal steps to the queue.
                if action_name not in SELF_HEAL_SEQUENCE and not _self_heal_attempted:
                    _self_heal_attempted = True
                    workflow.logger.info("Self-healing sequence triggered after failed action")
                    # Drop any heal steps already pending later in the queue so they
                    # don't run a second time back-to-back.
                    remaining = [a for a in action_queue if a not in SELF_HEAL_SEQUENCE]
                    action_queue = SELF_HEAL_SEQUENCE + remaining

            # ── Step 3: LLM assessment ──────────────────────────────────────
            assessment = await self._assess(alert_json, result_json)

            workflow.logger.info(
                "Assessment",
                extra={
                    "resolved": assessment.resolved,
                    "escalate": assessment.escalate,
                    "next_action": assessment.next_action,
                },
            )

            if assessment.resolved:
                return self._build_report(
                    alert, severity, actions_taken, compensations,
                    resolved=True, escalated=False, final_status="resolved",
                )

            if assessment.escalate:
                # SAGA: unresolved and escalating — unwind everything still on
                # the chain in reverse order before paging.
                compensations.extend(await self._run_compensation_chain(alert_json, saga_chain))
                await self._page(alert.alert_id, actions_taken)
                return self._build_report(
                    alert, severity, actions_taken, compensations,
                    resolved=False, escalated=True, final_status="escalated",
                )

            if assessment.next_action and assessment.next_action not in {
                a.action for a in actions_taken
            }:
                action_queue.insert(0, assessment.next_action)

        # ── Loop exhausted ──────────────────────────────────────────────────
        workflow.logger.error("Max actions reached — escalating")
        # SAGA: ran out of actions without resolving — unwind the chain before paging,
        # same as the explicit-escalate path above.
        compensations.extend(await self._run_compensation_chain(alert_json, saga_chain))
        await self._page(alert.alert_id, actions_taken)
        return self._build_report(
            alert, severity, actions_taken, compensations,
            resolved=False, escalated=True, final_status="escalated_max_actions",
        )

    async def _triage(self, alert_json: str) -> Triage:
        """Run the LLM triage activity and parse its output."""
        triage_json: str = await workflow.execute_activity(
            "triage_incident",
            alert_json,
            start_to_close_timeout=_SHORT_TIMEOUT,
            retry_policy=_LLM_RETRY,
        )
        return Triage.model_validate_json(triage_json)

    async def _execute_action(
        self, action_name: str, alert_json: str, alert: IncidentAlert
    ) -> tuple[ActionResult, str]:
        """Run a single infra action activity, converting an exhausted retry into a
        failed ActionResult instead of letting the workflow raise."""
        try:
            result_json: str = await workflow.execute_activity(
                action_name,
                alert_json,
                start_to_close_timeout=_SHORT_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
            return ActionResult.model_validate_json(result_json), result_json
        except ActivityError as exc:
            result = ActionResult(
                action=action_name,
                success=False,
                error_rate_after=alert.error_rate,
                latency_p99_after=alert.latency_p99_ms,
                notes=f"Activity exhausted retries: {exc}",
            )
            return result, result.model_dump_json()

    async def _maybe_compensate_inline(
        self,
        action_name: str,
        alert_json: str,
        result: ActionResult,
        alert: IncidentAlert,
    ) -> tuple[ActionResult | None, str | None]:
        """SAGA: if `action_name` made error rate meaningfully worse, run its
        compensation right away and return the result + a label for the report.
        Returns (None, None) when no compensation was needed or none is defined."""
        if result.error_rate_after <= alert.error_rate * 1.2:
            return None, None

        comp_name = COMPENSATIONS.get(action_name)
        if not comp_name:
            return None, None

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
        return comp_result, f"{comp_name} (compensating {action_name})"

    async def _assess(self, alert_json: str, result_json: str) -> IncidentAssessment:
        """Run the LLM assessment activity and parse its output."""
        assess_input = AssessInput(alert_json=alert_json, result_json=result_json)
        assessment_json: str = await workflow.execute_activity(
            "assess_after_action",
            assess_input,
            start_to_close_timeout=_SHORT_TIMEOUT,
            retry_policy=_LLM_RETRY,
        )
        return IncidentAssessment.model_validate_json(assessment_json)

    async def _run_compensation_chain(
        self, alert_json: str, chain: list[tuple[str, str]]
    ) -> list[str]:
        """SAGA: run the compensation chain in reverse order (most recent action
        undone first), skipping None entries."""
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
        """Page on-call with a summary of everything attempted so far."""
        summary = f"Not resolved after {len(actions_taken)} action(s): " + ", ".join(
            a.action for a in actions_taken
        )
        await workflow.execute_activity(
            "page_oncall",
            PageInput(alert_id=alert_id, summary=summary),
            start_to_close_timeout=_PAGE_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

    def _build_report(
        self,
        alert: IncidentAlert,
        severity: Severity,
        actions_taken: list[ActionResult],
        compensations: list[str],
        *,
        resolved: bool,
        escalated: bool,
        final_status: str,
    ) -> str:
        """Assemble the final IncidentReport JSON returned by the workflow."""
        return IncidentReport(
            alert_id=alert.alert_id,
            severity=severity,
            actions_taken=actions_taken,
            compensations=compensations,
            resolved=resolved,
            escalated=escalated,
            final_status=final_status,
        ).model_dump_json()
