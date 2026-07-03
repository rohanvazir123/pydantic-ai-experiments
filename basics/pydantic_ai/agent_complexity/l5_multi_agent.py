"""
Level 5: Multi-Agent Orchestration — Delegated Autonomy
=======================================================

An orchestrator decomposes the task and delegates to specialized agents, each
with its own prompt, tools, and (optionally) its own model. The cookbook builds
this with Claude Agent SDK *subagents* (isolated context windows). Pydantic AI
uses the other common architecture the cookbook README calls out —
**agent delegation**: agents wired together in code, one calling the next as a
tool, sharing dependencies and a single usage tally.

    request --> orchestrator
                  |-- research()          --> researcher   (fs + payment gateway)
                  |-- draft_response()     --> drafter      (fs: templates)
                  |-- review_compliance()  --> compliance   (fs: policies)
                  --> OrchestratorOutput

Each specialist is a full agent. The orchestrator keeps control: it calls them
in order (research -> draft -> review), threading findings between them, and
`usage=ctx.usage` rolls every sub-call into one usage total.

When to use: the task needs *parallel domain expertise* — distinct roles with
different instructions/tools/models — and a coordinator to synthesize. This is
the most capable and the most expensive and least deterministic level; reach for
it only when a single tool-calling agent genuinely can't hold all the roles.

Run:
    python l5_multi_agent.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kb_tools
import observability
from config import get_model
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool, ToolOutput

KNOWLEDGE_DIR = (Path(__file__).parent / "knowledge").resolve()


# --- Shared dependencies (the knowledge-base sandbox root) ---


@dataclass
class CaseDeps:
    root: Path


# --- Sandboxed filesystem tools, reused by the specialist agents ---
# Thin wrappers over the pure helpers in kb_tools (unit-tested there).


async def _list_files(ctx: RunContext[CaseDeps], glob: str = "**/*.md") -> str:
    """List files in the knowledge base matching a glob."""
    return kb_tools.list_files_text(ctx.deps.root, glob)


async def _read_file(ctx: RunContext[CaseDeps], path: str) -> str:
    """Read a single file from the knowledge base by relative path."""
    return kb_tools.read_file_text(ctx.deps.root, path)


async def _check_payment_gateway(
    ctx: RunContext[CaseDeps], transaction_date: str, amount: float
) -> str:
    """Verify a transaction's status and refund eligibility with the processor."""
    return kb_tools.payment_gateway_text(transaction_date, amount)


# --- Specialist agents (each its own prompt + tool set + model tier) ---
#
# Tiered LLMs by role — put the big model where reasoning and tool-calling
# reliability matter (the orchestrator's coordination/synthesis and the
# researcher's multi-tool investigation), and use the cheap `small` tier for the
# text-shaped roles (drafting an email, checking a draft against policy). This is
# the multi-agent cost lever: a few large-tier calls + several small-tier calls
# beats running every agent on the big model.

# Requested tiers per role (pinned to `large` by default; AGENT_STRICT_TIERS=1
# honors them on capable models — see README "Model tiers").
ORCHESTRATOR_TIER = "large"  # coordination + structured synthesis
RESEARCHER_TIER = "large"  # multi-tool investigation (needs reliability)
DRAFTER_TIER = "small"  # drafting an email (text generation)
COMPLIANCE_TIER = "small"  # checking a draft against policy (text)

# retries let local models self-correct slightly-off structured output / tool args.
_RETRIES = 3

researcher = Agent(
    get_model(RESEARCHER_TIER),
    deps_type=CaseDeps,
    output_type=str,
    retries=_RETRIES,
    system_prompt=(
        "You are a billing research specialist. Investigate thoroughly: read the "
        "customer file to understand their history, check the payment gateway to "
        "verify the transaction, and read the refund policy to determine "
        "eligibility. Return a clear, factual summary of your findings."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
        Tool(_check_payment_gateway, name="check_payment_gateway"),
    ],
)

drafter = Agent(
    get_model(DRAFTER_TIER),
    deps_type=CaseDeps,
    output_type=str,
    retries=_RETRIES,
    system_prompt=(
        "You are a customer communications specialist. Draft a professional, "
        "empathetic response based on the research findings you receive. Use the "
        "response templates in the knowledge base as a starting point, then "
        "personalize for the customer. Return the drafted email text."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
    ],
)

compliance = Agent(
    get_model(COMPLIANCE_TIER),
    deps_type=CaseDeps,
    output_type=str,
    retries=_RETRIES,
    system_prompt=(
        "You are a compliance reviewer. Verify that the proposed refund action "
        "and customer response follow company policy. Check the escalation matrix "
        "and refund policy. State clearly whether you APPROVE or flag issues."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
    ],
)


# --- Orchestrator output ---


class CustomerEmail(BaseModel):
    subject: str
    body: str


class OrchestratorOutput(BaseModel):
    research_summary: str
    duplicate_confirmed: bool
    refund_amount: float
    compliance_approved: bool
    final_action: str
    customer_email: CustomerEmail


# --- Orchestrator (delegates to specialists via tools) ---


orchestrator = Agent(
    get_model(ORCHESTRATOR_TIER),
    deps_type=CaseDeps,
    output_type=ToolOutput(OrchestratorOutput),
    retries=_RETRIES,
    system_prompt=(
        "You are a senior case manager. Resolve the customer issue by delegating "
        "to your specialist team using the tools, in this order:\n"
        "  1. research(question)          — investigate billing data + gateway\n"
        "  2. draft_response(findings)    — write the customer-facing email\n"
        "  3. review_compliance(proposal) — confirm policy compliance\n"
        "Pass findings between steps. After all three report back, synthesize the "
        "final structured decision including the approved customer email."
    ),
)


@orchestrator.tool
async def research(ctx: RunContext[CaseDeps], question: str) -> str:
    """Delegate investigation to the research specialist."""
    result = await researcher.run(question, deps=ctx.deps, usage=ctx.usage)
    return result.output


@orchestrator.tool
async def draft_response(ctx: RunContext[CaseDeps], findings: str) -> str:
    """Delegate drafting the customer email to the communications specialist."""
    result = await drafter.run(
        f"Draft a customer response based on these findings:\n{findings}",
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return result.output


@orchestrator.tool
async def review_compliance(ctx: RunContext[CaseDeps], proposal: str) -> str:
    """Delegate a policy-compliance review to the compliance specialist."""
    result = await compliance.run(
        f"Review this proposed action and response for compliance:\n{proposal}",
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return result.output


# --- Run ---


async def run_orchestrator(task: str, root: Path = KNOWLEDGE_DIR) -> Any:
    """Run the orchestrator (which delegates to the specialist team)."""
    return await orchestrator.run(task, deps=CaseDeps(root=root))


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    result = asyncio.run(
        run_orchestrator(
            "Customer cust_12345 reports a duplicate charge on their February "
            "bill. Delegate to your team: have the researcher investigate, the "
            "drafter prepare a response, and compliance review the final action "
            "before we send it."
        )
    )
    print(f"\n--- Done | usage: {result.usage} ---")
    print("\nStructured output:")
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
