"""Deterministic tests for Level 1 — Augmented LLM (single call, structured out).

Uses ``TestModel`` for auto-valid output and ``FunctionModel`` to pin exact
field values. No Ollama required.
"""

from __future__ import annotations

import l1_augmented_llm as l1
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


def test_output_is_structured_and_no_tools_registered() -> None:
    """Level 1 is a single call: structured output, zero function tools."""
    tm = TestModel()
    with l1.agent.override(model=tm):
        result = l1.agent.run_sync("some ticket")
    assert isinstance(result.output, l1.TicketClassification)
    # The defining trait of Level 1: no tools, so exactly one model round-trip.
    assert tm.last_model_request_parameters.function_tools == []


def test_classify_returns_pinned_fields() -> None:
    """FunctionModel lets us assert the exact structured result is surfaced."""

    def respond(messages: list, info: AgentInfo) -> ModelResponse:
        output_tool = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=output_tool,
                    args={
                        "category": "billing",
                        "priority": "high",
                        "summary": "Duplicate subscription charge #12345",
                        "can_auto_resolve": True,
                    },
                )
            ]
        )

    with l1.agent.override(model=FunctionModel(respond)):
        result = l1.classify("I was charged twice, order #12345")

    assert result.category == "billing"
    assert result.priority == "high"
    assert result.can_auto_resolve is True
    assert "12345" in result.summary
