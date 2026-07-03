"""Deterministic tests for Level 4 — Agent Harness (filesystem + billing API).

TestModel (with tool calls suppressed) verifies the runtime is wired and the
output schema holds; a scripted FunctionModel drives a real investigation over
the knowledge base — discover, read, verify, refund, report — with no network.
"""

from __future__ import annotations

import l4_agent_harness as l4
from pydantic_ai import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

EXPECTED_TOOLS = {
    "list_files",
    "read_file",
    "search_files",
    "check_payment_gateway",
    "issue_refund",
}


def test_runtime_tools_registered_and_output_schema() -> None:
    # call_tools=[] avoids TestModel calling read_file with a bogus path
    # (which would correctly raise ModelRetry); we just want the wiring + schema.
    tm = TestModel(call_tools=[])
    with l4.harness_agent.override(model=tm):
        result = l4.harness_agent.run_sync(
            "investigate", deps=l4.HarnessDeps(root=l4.KNOWLEDGE_DIR)
        )
    assert isinstance(result.output, l4.HarnessOutput)
    assert isinstance(result.output.customer_email, l4.CustomerEmail)
    registered = {t.name for t in tm.last_model_request_parameters.function_tools}
    assert registered == EXPECTED_TOOLS


def test_scripted_investigation_reads_real_knowledge_base() -> None:
    facts: dict[str, bool] = {}

    def script(messages: list, info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        step = len(returns)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_files", args={})])
        if step == 1:
            facts["listed_customer"] = "customers/cust_12345.md" in returns[-1].content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_file",
                        args={"path": "customers/cust_12345.md"},
                    )
                ]
            )
        if step == 2:
            facts["read_profile"] = "Sarah Johnson" in returns[-1].content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="check_payment_gateway",
                        args={"transaction_date": "2025-02-01", "amount": 49.99},
                    )
                ]
            )
        if step == 3:
            facts["verified_gateway"] = "Refund eligible: YES" in returns[-1].content
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="issue_refund",
                        args={
                            "amount": 49.99,
                            "reason": "duplicate charge",
                            "customer_id": "cust_12345",
                        },
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "action_taken": "Refunded duplicate Feb charge",
                        "refund_amount": 49.99,
                        "policy_compliant": True,
                        "customer_email": {
                            "subject": "Your Refund Has Been Processed",
                            "body": "Hi Sarah, we've refunded the duplicate charge.",
                        },
                    },
                )
            ]
        )

    with l4.harness_agent.override(model=FunctionModel(script)):
        result = l4.harness_agent.run_sync(
            "Investigate the duplicate charge for cust_12345 and refund if valid.",
            deps=l4.HarnessDeps(root=l4.KNOWLEDGE_DIR),
        )

    assert facts == {
        "listed_customer": True,
        "read_profile": True,
        "verified_gateway": True,
    }
    assert result.output.refund_amount == 49.99
    assert result.output.policy_compliant is True
    assert "Sarah" in result.output.customer_email.body


def test_read_file_tool_raises_on_escape_via_agent() -> None:
    """The sandbox guard fires even when the model requests a traversal path."""

    def escape(messages: list, info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        retries = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if type(p).__name__ == "RetryPromptPart"
        ]
        # First: attempt an escape. After the retry prompt: produce valid output.
        if not returns and not retries:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": "../config.py"})]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "action_taken": "denied",
                        "refund_amount": 0.0,
                        "policy_compliant": False,
                        "customer_email": {"subject": "n/a", "body": "n/a"},
                    },
                )
            ]
        )

    with l4.harness_agent.override(model=FunctionModel(escape)):
        result = l4.harness_agent.run_sync(
            "read ../config.py", deps=l4.HarnessDeps(root=l4.KNOWLEDGE_DIR)
        )
    # The traversal was rejected (surfaced as a retry) and the run still finished.
    retry_prompts = [
        p
        for m in result.all_messages()
        if isinstance(m, ModelRequest)
        for p in m.parts
        if type(p).__name__ == "RetryPromptPart"
    ]
    assert any("outside the knowledge base" in str(p.content) for p in retry_prompts)
