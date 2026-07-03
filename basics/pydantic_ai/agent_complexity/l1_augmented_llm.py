"""
Level 1: Augmented LLM — Single API Call
========================================

One model call with the right context (system prompt + structured output).
No loops, no tools, no autonomy. This handles more than most people think:
classification, extraction, summarization, rewriting, routing decisions.

    Input --> [system prompt + schema] --> LLM --> structured output

When to use: the answer is *derivable from the input alone*. If you find
yourself wanting the model to look something up or take an action, you've
outgrown Level 1 — go to Level 2 (routing) or Level 3 (tools).

Run:
    python l1_augmented_llm.py
"""

from __future__ import annotations

import observability
from config import get_model
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput


class TicketClassification(BaseModel):
    """Structured result of classifying a support ticket."""

    category: str  # billing | technical | general
    priority: str  # low | medium | high
    summary: str
    can_auto_resolve: bool


# Tiered LLMs: pure classification is a natural cheap-tier job. We *request* the
# `small` tier; by default config pins it to `large` for reliability on weak local
# models (set AGENT_STRICT_TIERS=1 to honor it). See README "Model tiers".
CLASSIFIER_TIER = "small"

# A single agent with structured output. No tools => exactly one model call.
agent = Agent(
    get_model(CLASSIFIER_TIER),
    # ToolOutput forces the model to return the result via a tool call rather than
    # free-form JSON text. Chatty local models otherwise sometimes wrap valid JSON
    # in commentary, which breaks prompted-JSON parsing; tool output is far more
    # robust. retries let Pydantic AI feed validation errors back to self-correct.
    output_type=ToolOutput(TicketClassification),
    retries=3,
    system_prompt=(
        "You are a customer support classifier. "
        "Classify incoming tickets by category (billing, technical, general), "
        "priority (low, medium, high), a one-line summary, and whether they can "
        "be auto-resolved. Respond only with the structured result."
    ),
)


def classify(ticket: str) -> TicketClassification:
    """Classify a single ticket in one model call."""
    return agent.run_sync(ticket).output


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    result = classify(
        "I was charged twice for my subscription last month. "
        "Order ID: #12345. Please refund the duplicate charge."
    )
    print(result)


if __name__ == "__main__":
    main()
