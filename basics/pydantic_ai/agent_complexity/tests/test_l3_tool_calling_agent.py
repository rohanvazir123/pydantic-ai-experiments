"""Deterministic tests for Level 3 — Tool-Calling Agent.

TestModel proves the tools are wired and exercised; a scripted FunctionModel
proves a realistic call sequence (inspect charges -> issue refund -> finish)
runs end to end and reads injected dependencies.
"""

from __future__ import annotations

import l3_tool_calling_agent as l3
from pydantic_ai import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

EXPECTED_TOOLS = {
    "get_customer_balance",
    "get_recent_charges",
    "check_refund_policy",
    "issue_refund",
}


def test_all_billing_tools_are_registered_and_called() -> None:
    tm = TestModel()  # calls every tool once, then produces structured output
    with l3.billing_agent.override(model=tm):
        result = l3.billing_agent.run_sync("fix my duplicate charge", deps=l3._sample_deps())

    assert isinstance(result.output, l3.BillingResolution)
    registered = {t.name for t in tm.last_model_request_parameters.function_tools}
    assert registered == EXPECTED_TOOLS

    called = {
        p.tool_name
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if type(p).__name__ == "ToolCallPart"
    }
    assert called >= EXPECTED_TOOLS


def test_scripted_refund_flow_reads_deps_and_completes() -> None:
    """A realistic sequence: check charges, issue refund, then final output."""
    seen_charges: dict[str, bool] = {}

    def script(messages: list, info: AgentInfo) -> ModelResponse:
        # Count how many tool results we've already received to decide the step.
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        step = len(returns)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name="get_recent_charges", args={})])
        if step == 1:
            # We should have just seen the charges list (proves deps were read).
            seen_charges["ok"] = "Monthly subscription" in returns[-1].content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="issue_refund",
                        args={"amount": 49.99, "reason": "duplicate charge on Feb 1"},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "action_taken": "Refunded duplicate Feb 1 charge",
                        "refund_amount": 49.99,
                        "follow_up_needed": False,
                    },
                )
            ]
        )

    with l3.billing_agent.override(model=FunctionModel(script)):
        result = l3.billing_agent.run_sync(
            "I was charged twice on Feb 1st.", deps=l3._sample_deps()
        )

    assert seen_charges.get("ok") is True  # tool actually read the injected DB
    assert result.output.refund_amount == 49.99
    assert result.output.follow_up_needed is False


def test_sample_deps_contain_duplicate_charge() -> None:
    deps = l3._sample_deps()
    feb = [c for c in deps.db["charges"] if c["date"] == "2025-02-01"]
    assert len(feb) == 2  # the duplicate the agent must catch
