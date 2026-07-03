"""
Level 2: Prompt Chains & Routing — Deterministic DAGs
=====================================================

Multiple LLM calls orchestrated through fixed paths. Each step validates its
output before passing to the next. No model decides control flow — the *code*
does. This is a directed acyclic graph (DAG) of single-purpose agents.

    ticket --> classify --> route --> handler --> validate --> resolution
                              |                       |
                              +-- billing            +-- escalate to human
                              +-- technical
                              +-- general

When to use: the workflow has known stages and you want each stage small,
testable, and independently swappable. Routing is a code `dict` lookup, not an
agent decision, so it is deterministic and cheap to reason about.

Run:
    python l2_prompt_chains.py
"""

from __future__ import annotations

from enum import StrEnum

import observability
from config import get_model
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput

# --- Models ---


class Category(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"


class TicketClassification(BaseModel):
    category: Category
    confidence: float


class Resolution(BaseModel):
    response: str
    escalate: bool


# --- Agents (each is a single focused LLM call) ---
#
# Tiered LLMs: classification is the canonical cheap task, so the classifier
# *requests* the `nano` tier; drafting a resolution needs more capability, so the
# handlers request `small`. This is the core cost lever — don't pay for a big
# model on a job a tiny one does fine. NOTE: config pins these to `large` by
# default (weak local small models); set AGENT_STRICT_TIERS=1 to honor them on
# capable models. See README "Model tiers".
CLASSIFIER_TIER = "nano"  # classification -> cheapest tier
HANDLER_TIER = "small"  # response generation -> standard tier

_classifier_model = get_model(CLASSIFIER_TIER)
_handler_model = get_model(HANDLER_TIER)

# retries let local models self-correct slightly-off structured output.
# The nano tier is the weakest, so give the classifier a couple more attempts.
_RETRIES = 3
_NANO_RETRIES = 5

classifier = Agent(
    _classifier_model,
    output_type=ToolOutput(TicketClassification),  # tool output = robust on local models
    retries=_NANO_RETRIES,
    system_prompt=(
        "Classify the customer ticket into exactly one category: "
        "billing, technical, or general. Report your confidence 0-1. Be precise."
    ),
)

billing_handler = Agent(
    _handler_model,
    output_type=ToolOutput(Resolution),
    retries=_RETRIES,
    system_prompt=(
        "You handle billing issues. Generate a resolution. "
        "Set escalate=true if a refund over $100 is needed."
    ),
)

technical_handler = Agent(
    _handler_model,
    output_type=ToolOutput(Resolution),
    retries=_RETRIES,
    system_prompt=(
        "You handle technical issues. Generate a resolution. "
        "Set escalate=true if the issue requires engineering intervention."
    ),
)

general_handler = Agent(
    _handler_model,
    output_type=ToolOutput(Resolution),
    retries=_RETRIES,
    system_prompt="You handle general inquiries. Be helpful and concise.",
)


# --- DAG: classify -> route -> handle -> validate ---

HANDLERS: dict[Category, Agent[None, Resolution]] = {
    Category.BILLING: billing_handler,
    Category.TECHNICAL: technical_handler,
    Category.GENERAL: general_handler,
}


def process_ticket(ticket: str) -> Resolution:
    """Route a ticket through classify -> handle. Control flow is pure code."""
    classification = classifier.run_sync(ticket).output
    print(
        f"Classified as: {classification.category.value} "
        f"({classification.confidence:.0%})"
    )

    # The *code* picks the handler — the model never decides where to route.
    handler = HANDLERS[classification.category]
    resolution = handler.run_sync(ticket).output

    if resolution.escalate:
        print("Escalating to human agent")

    return resolution


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    ticket = (
        "I was charged twice for my subscription last month. "
        "Order ID: #12345. The duplicate charge was $49.99."
    )
    resolution = process_ticket(ticket)
    print(f"\nResponse: {resolution.response}")


if __name__ == "__main__":
    main()
