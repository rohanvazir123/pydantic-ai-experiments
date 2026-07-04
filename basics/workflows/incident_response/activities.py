"""
Temporal activities for the AI Incident Response workflow.

Two activity classes:
- InfraActivities  — simulated infra operations; behaviour driven by a scenario
                     dict so tests can inject any failure/success pattern.
- LLMActivities    — Pydantic AI calls for triage and post-action assessment;
                     accepts an injected model so tests can use FunctionModel.
"""
from __future__ import annotations

from temporalio import activity

from pydantic_ai import Agent

from .models import (
    ActionResult,
    AssessInput,
    IncidentAlert,
    IncidentAssessment,
    PageInput,
    Triage,
)

# ---------------------------------------------------------------------------
# Infra activities
# ---------------------------------------------------------------------------

_DEFAULT_SCENARIO: dict[str, dict] = {
    "restart_service":    {"success": True,  "error_rate_delta": -0.5, "latency_delta": -200},
    "scale_up":           {"success": True,  "error_rate_delta": -0.2, "latency_delta": -100},
    "scale_down":         {"success": True,  "error_rate_delta":  0.0, "latency_delta":   0},
    "clear_cache":        {"success": True,  "error_rate_delta": -0.3, "latency_delta": -150},
    "rollback_deployment":{"success": True,  "error_rate_delta": -0.6, "latency_delta": -300},
    "page_oncall":        {"success": True,  "error_rate_delta":  0.0, "latency_delta":   0},
}


class InfraActivities:
    """Simulated infra operations.  Pass a partial scenario dict to override defaults."""

    def __init__(self, scenario: dict[str, dict] | None = None) -> None:
        self._scenario: dict[str, dict] = {**_DEFAULT_SCENARIO, **(scenario or {})}

    def _apply(self, action: str, service: str, alert_json: str) -> str:
        cfg = self._scenario.get(action, {"success": False, "error_rate_delta": 0.0, "latency_delta": 0})
        alert = IncidentAlert.model_validate_json(alert_json) if alert_json else None

        base_error = alert.error_rate if alert else 0.5
        base_latency = alert.latency_p99_ms if alert else 1000

        if not cfg["success"]:
            raise RuntimeError(f"{action} failed on {service} (simulated failure)")

        new_error = max(0.0, base_error + cfg["error_rate_delta"])
        new_latency = max(0, base_latency + cfg["latency_delta"])

        return ActionResult(
            action=action,
            success=True,
            error_rate_after=round(new_error, 3),
            latency_p99_after=new_latency,
            notes=f"{action} completed on {service}",
        ).model_dump_json()

    @activity.defn(name="restart_service")
    async def restart_service(self, alert_json: str) -> str:
        return self._apply("restart_service", "service", alert_json)

    @activity.defn(name="scale_up")
    async def scale_up(self, alert_json: str) -> str:
        return self._apply("scale_up", "service", alert_json)

    @activity.defn(name="scale_down")
    async def scale_down(self, alert_json: str) -> str:
        return self._apply("scale_down", "service", alert_json)

    @activity.defn(name="clear_cache")
    async def clear_cache(self, alert_json: str) -> str:
        return self._apply("clear_cache", "service", alert_json)

    @activity.defn(name="rollback_deployment")
    async def rollback_deployment(self, alert_json: str) -> str:
        return self._apply("rollback_deployment", "service", alert_json)

    @activity.defn(name="page_oncall")
    async def page_oncall(self, inp: PageInput) -> str:
        return f"on-call paged for {inp.alert_id}: {inp.summary}"


# ---------------------------------------------------------------------------
# LLM activities
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = (
    "You are an SRE incident response agent. "
    "Given a production alert, assess its severity and recommend an ordered list "
    "of remediation actions to try. "
    "Actions MUST be chosen only from: restart_service, scale_up, clear_cache, rollback_deployment."
)

_ASSESS_SYSTEM = (
    "You are assessing whether a remediation action resolved a production incident. "
    "Given the original alert and the action result (with updated metrics), decide: "
    "is the incident resolved? If not, what action to try next (or should we escalate)? "
    "Escalate only if the situation is getting worse or you have run out of ideas."
)


class LLMActivities:
    """LLM-powered triage and assessment.  Inject any Pydantic AI model."""

    def __init__(self, model: object) -> None:
        self._triage_agent: Agent[None, Triage] = Agent(
            model,  # type: ignore[arg-type]
            output_type=Triage,
            system_prompt=_TRIAGE_SYSTEM,
        )
        self._assess_agent: Agent[None, IncidentAssessment] = Agent(
            model,  # type: ignore[arg-type]
            output_type=IncidentAssessment,
            system_prompt=_ASSESS_SYSTEM,
        )

    @activity.defn(name="triage_incident")
    async def triage_incident(self, alert_json: str) -> str:
        alert = IncidentAlert.model_validate_json(alert_json)
        prompt = (
            f"Service: {alert.service}\n"
            f"Error rate: {alert.error_rate:.1%}\n"
            f"p99 latency: {alert.latency_p99_ms}ms\n"
            f"Description: {alert.description}"
        )
        result = await self._triage_agent.run(prompt)
        return result.output.model_dump_json()

    @activity.defn(name="assess_after_action")
    async def assess_after_action(self, inp: AssessInput) -> str:
        alert = IncidentAlert.model_validate_json(inp.alert_json)
        action_result = ActionResult.model_validate_json(inp.result_json)
        prompt = (
            f"Original alert — service: {alert.service}, "
            f"error rate: {alert.error_rate:.1%}, p99: {alert.latency_p99_ms}ms\n"
            f"Action taken: {action_result.action} ({'succeeded' if action_result.success else 'failed'})\n"
            f"Metrics after: error rate {action_result.error_rate_after:.1%}, "
            f"p99 {action_result.latency_p99_after}ms\n"
            f"Notes: {action_result.notes}"
        )
        result = await self._assess_agent.run(prompt)
        return result.output.model_dump_json()
