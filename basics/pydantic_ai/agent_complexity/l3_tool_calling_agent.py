"""
Level 3: Tool-Calling Agent — Scoped Autonomy
=============================================

The agent decides *which* tools to call and *in what order*, but only within a
fixed set of well-defined capabilities. This is where real autonomy starts: the
model runs a loop (call tool -> observe result -> decide next step) until it can
produce the final structured answer.

    task --> agent <--> get_customer_balance
                   <--> get_recent_charges
                   <--> check_refund_policy
                   <--> issue_refund
             agent --> BillingResolution

When to use: the task needs a *few specific, trusted actions* (look something
up, call an API, do a calculation) and you want the model to sequence them. The
tool set is the guardrail — the agent can only do what you gave it.

Run:
    python l3_tool_calling_agent.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import observability
from config import get_model
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ToolOutput
from utils import print_agent_trace

# --- Dependencies (injected per-run; the tools read from here) ---


@dataclass
class CustomerDeps:
    customer_id: str
    db: dict[str, Any]  # simplified — in production, a real DB client


# --- Output ---


class BillingResolution(BaseModel):
    action_taken: str
    refund_amount: float | None
    follow_up_needed: bool


# --- Agent ---


AGENT_TIER = "large"  # tool-calling reliability -> top tier

billing_agent = Agent(
    get_model(AGENT_TIER),
    deps_type=CustomerDeps,
    output_type=ToolOutput(BillingResolution),
    retries=3,  # local models sometimes need a retry to nail tool args / output
    system_prompt=(
        "You are a billing support agent. Use the available tools to look up "
        "customer data, check policies, and resolve billing issues. "
        "Always verify the charge with the tools before issuing a refund. "
        "When done, return the structured resolution."
    ),
)


# --- Tools (the agent decides when and how to use these) ---


@billing_agent.tool
async def get_customer_balance(ctx: RunContext[CustomerDeps]) -> str:
    """Return the customer's current account balance."""
    balance = ctx.deps.db.get("balance", 0)
    return f"Current balance: ${balance:.2f}"


@billing_agent.tool
async def get_recent_charges(ctx: RunContext[CustomerDeps]) -> str:
    """List the customer's recent charges so duplicates can be spotted."""
    charges = ctx.deps.db.get("charges", [])
    return "\n".join(
        f"- ${c['amount']:.2f} on {c['date']}: {c['description']}" for c in charges
    )


@billing_agent.tool
async def check_refund_policy(
    ctx: RunContext[CustomerDeps], charge_description: str
) -> str:
    """Return the refund policy relevant to a given charge description."""
    return (
        f"Policy for '{charge_description}': "
        "Duplicate charges are eligible for automatic refund within 30 days. "
        "Refunds over $100 require manager approval."
    )


@billing_agent.tool
async def issue_refund(
    ctx: RunContext[CustomerDeps], amount: float, reason: str
) -> str:
    """Issue a refund of the given amount for the given reason."""
    return f"Refund of ${amount:.2f} issued successfully. Reason: {reason}"


# --- Run ---


async def resolve(task: str, deps: CustomerDeps) -> Any:
    """Run the billing agent to resolution and return the full run result."""
    return await billing_agent.run(task, deps=deps)


def _sample_deps() -> CustomerDeps:
    return CustomerDeps(
        customer_id="cust_12345",
        db={
            "balance": 149.97,
            "charges": [
                {
                    "amount": 49.99,
                    "date": "2025-02-01",
                    "description": "Monthly subscription",
                },
                {
                    "amount": 49.99,
                    "date": "2025-02-01",
                    "description": "Monthly subscription",
                },
                {
                    "amount": 49.99,
                    "date": "2025-01-01",
                    "description": "Monthly subscription",
                },
            ],
        },
    )


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    result = asyncio.run(
        resolve(
            "I was charged twice on Feb 1st for my subscription. Please fix this.",
            _sample_deps(),
        )
    )
    print_agent_trace(result)
    print(f"\nAction: {result.output.action_taken}")
    print(f"Refund: ${result.output.refund_amount}")
    print(f"Follow-up needed: {result.output.follow_up_needed}")


if __name__ == "__main__":
    main()
