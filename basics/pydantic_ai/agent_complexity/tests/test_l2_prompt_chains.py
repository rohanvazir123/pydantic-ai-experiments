"""Deterministic tests for Level 2 — Prompt Chains & Routing.

The point of Level 2 is that *code* controls the flow. These tests pin the
classifier's category with a ``FunctionModel`` and assert the correct handler
runs — proving routing is deterministic, not model-decided.
"""

from __future__ import annotations

from collections.abc import Callable

import l2_prompt_chains as l2
from l2_prompt_chains import Category
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _fixed_output(**args: object) -> FunctionModel:
    """A FunctionModel that always emits the given structured output args."""

    def respond(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)]
        )

    return FunctionModel(respond)


def _classify_as(category: Category) -> FunctionModel:
    return _fixed_output(category=category.value, confidence=0.95)


def _route(category: Category, response: str, escalate: bool = False) -> str:
    """Run process_ticket with the classifier pinned to ``category``."""
    with (
        l2.classifier.override(model=_classify_as(category)),
        l2.billing_handler.override(
            model=_fixed_output(response=f"BILLING:{response}", escalate=escalate)
        ),
        l2.technical_handler.override(
            model=_fixed_output(response=f"TECH:{response}", escalate=escalate)
        ),
        l2.general_handler.override(
            model=_fixed_output(response=f"GENERAL:{response}", escalate=escalate)
        ),
    ):
        return l2.process_ticket("some ticket").response


def test_billing_ticket_routes_to_billing_handler() -> None:
    assert _route(Category.BILLING, "ok").startswith("BILLING:")


def test_technical_ticket_routes_to_technical_handler() -> None:
    assert _route(Category.TECHNICAL, "ok").startswith("TECH:")


def test_general_ticket_routes_to_general_handler() -> None:
    assert _route(Category.GENERAL, "ok").startswith("GENERAL:")


def test_every_category_has_a_handler() -> None:
    """Routing table must be exhaustive over the Category enum."""
    assert set(l2.HANDLERS) == set(Category)


def test_escalation_flag_is_preserved() -> None:
    with (
        l2.classifier.override(model=_classify_as(Category.BILLING)),
        l2.billing_handler.override(
            model=_fixed_output(response="needs manager", escalate=True)
        ),
    ):
        resolution = l2.process_ticket("refund $500 please")
    assert resolution.escalate is True


def test_helpers_are_callable() -> None:
    # Guards against accidental signature drift in the module under test.
    assert isinstance(l2.process_ticket, Callable)
