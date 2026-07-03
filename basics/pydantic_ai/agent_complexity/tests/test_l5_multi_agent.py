"""Deterministic tests for Level 5 — Multi-Agent Orchestration (delegation).

Verifies the orchestrator delegates to each specialist (research -> draft ->
review) and that delegation actually invokes the sub-agents, all without Ollama.
Every agent involved is overridden so no live inference occurs.
"""

from __future__ import annotations

from contextlib import ExitStack

import l5_multi_agent as l5
from pydantic_ai import ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


def _deps() -> l5.CaseDeps:
    return l5.CaseDeps(root=l5.KNOWLEDGE_DIR)


def test_orchestrator_exposes_delegation_tools() -> None:
    tm = TestModel(call_tools=[])
    with l5.orchestrator.override(model=tm):
        l5.orchestrator.run_sync("resolve", deps=_deps())
    registered = {t.name for t in tm.last_model_request_parameters.function_tools}
    assert registered == {"research", "draft_response", "review_compliance"}


def test_specialists_have_their_own_toolsets() -> None:
    for agent, expected in [
        (l5.researcher, {"list_files", "read_file", "check_payment_gateway"}),
        (l5.drafter, {"list_files", "read_file"}),
        (l5.compliance, {"list_files", "read_file"}),
    ]:
        tm = TestModel(call_tools=[])
        with agent.override(model=tm):
            agent.run_sync("x", deps=_deps())
        registered = {t.name for t in tm.last_model_request_parameters.function_tools}
        assert registered == expected


def test_orchestrator_delegates_to_every_specialist() -> None:
    """Scripted orchestrator calls each delegate; sub-agents return canned text."""
    calls: list[str] = []

    def orchestrate(messages: list, info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        step = len(returns)
        if step == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="research", args={"question": "investigate cust_12345"})]
            )
        if step == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="draft_response", args={"findings": returns[-1].content})]
            )
        if step == 2:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="review_compliance", args={"proposal": returns[-1].content})]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "research_summary": "duplicate confirmed",
                        "duplicate_confirmed": True,
                        "refund_amount": 49.99,
                        "compliance_approved": True,
                        "final_action": "refund issued",
                        "customer_email": {
                            "subject": "Refund processed",
                            "body": "Hi Sarah, done.",
                        },
                    },
                )
            ]
        )

    # Specialists have output_type=str, so a plain TextPart ends their run.
    def specialist_str(tag: str) -> FunctionModel:
        def respond(messages: list, info: AgentInfo) -> ModelResponse:
            calls.append(tag)
            return ModelResponse(parts=[TextPart(content=f"{tag} findings")])

        return FunctionModel(respond)

    with ExitStack() as stack:
        stack.enter_context(l5.orchestrator.override(model=FunctionModel(orchestrate)))
        stack.enter_context(l5.researcher.override(model=specialist_str("researcher")))
        stack.enter_context(l5.drafter.override(model=specialist_str("drafter")))
        stack.enter_context(l5.compliance.override(model=specialist_str("compliance")))
        result = l5.orchestrator.run_sync("resolve the case", deps=_deps())

    # All three specialists were actually invoked, in order.
    assert calls == ["researcher", "drafter", "compliance"]
    assert result.output.compliance_approved is True
    assert result.output.refund_amount == 49.99

    # The orchestrator's history shows the three delegation tool calls.
    delegated = [
        p.tool_name
        for m in result.all_messages()
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart) and p.tool_name in {"research", "draft_response", "review_compliance"}
    ]
    assert delegated == ["research", "draft_response", "review_compliance"]


def test_delegation_shares_usage_totals() -> None:
    """usage=ctx.usage means sub-agent calls roll into the orchestrator total."""

    def orchestrate(messages: list, info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(tool_name="research", args={"question": "q"})])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "research_summary": "s",
                        "duplicate_confirmed": True,
                        "refund_amount": 49.99,
                        "compliance_approved": True,
                        "final_action": "done",
                        "customer_email": {"subject": "s", "body": "b"},
                    },
                )
            ]
        )

    with ExitStack() as stack:
        stack.enter_context(l5.orchestrator.override(model=FunctionModel(orchestrate)))
        stack.enter_context(l5.researcher.override(model=TestModel(call_tools=[])))
        stack.enter_context(l5.drafter.override(model=TestModel(call_tools=[])))
        stack.enter_context(l5.compliance.override(model=TestModel(call_tools=[])))
        result = l5.orchestrator.run_sync("resolve", deps=_deps())

    # More than one model request happened (orchestrator + delegated researcher).
    assert result.usage.requests >= 2
