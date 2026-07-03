"""
Level 4: Agent Harness — Full Runtime Access
============================================

Instead of hand-picking a few tools (Level 3), you give the agent a *runtime*:
the ability to explore a filesystem and call external APIs, then reason,
execute, observe, and iterate on its own. This is the shape of coding agents
like Claude Code or Cursor — read, grep, act, repeat.

The upstream cookbook builds this on the Claude Agent SDK (Bash / Read / Grep +
MCP). That SDK is Anthropic-only, so this local port recreates the same *shape*
with Pydantic AI tools running entirely against Ollama:

    Filesystem runtime (scoped to knowledge/):
        list_files(glob)   — discover what exists
        read_file(path)    — read policies / customer files / templates
        search_files(term) — grep across the knowledge base
    External billing API:
        check_payment_gateway(...)  — verify a transaction
        issue_refund(...)           — process a refund

    task --> agent <--> {filesystem runtime + billing API} --> HarnessOutput

The agent is *not* told which files to read. It discovers the knowledge base,
figures out what's relevant, verifies via the API, and drafts a response —
autonomously. File access is sandboxed to `knowledge/` (path traversal is
rejected), which is the Level 4 guardrail: broad capability, bounded blast area.

When to use: the task needs open-ended *exploration and reasoning* over a body
of files/systems and you can't enumerate the exact steps in advance.

Run:
    python l4_agent_harness.py
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
from pydantic_ai import Agent, RunContext, ToolOutput
from utils import print_agent_trace

KNOWLEDGE_DIR = (Path(__file__).parent / "knowledge").resolve()


# --- Dependencies: the sandbox root the filesystem tools operate within ---


@dataclass
class HarnessDeps:
    root: Path


# --- Output ---


class CustomerEmail(BaseModel):
    subject: str
    body: str


class HarnessOutput(BaseModel):
    action_taken: str
    refund_amount: float
    policy_compliant: bool
    customer_email: CustomerEmail


class Triage(BaseModel):
    """Cheap, fast first-pass classification of an incoming case."""

    category: str  # billing | technical | account | other
    urgency: str  # low | medium | high
    focus: str  # one-line hint on what the harness should investigate first


# --- Agents (tiered) ---
#
# Tiered LLMs: a cheap `nano` triage agent does a fast first pass (category /
# urgency / where to look), then the expensive `large` harness does the actual
# open-ended investigation. In production this is how you keep an autonomous
# harness affordable — a tiny model gates and focuses the big one, so the big
# model spends its (many, expensive) turns on reasoning, not on triage.
# (Requested tiers are pinned to `large` by default; AGENT_STRICT_TIERS=1 honors
# them on capable models. See README "Model tiers".)
TRIAGE_TIER = "nano"  # fast first pass -> cheapest tier
HARNESS_TIER = "large"  # open-ended tool-using reasoning -> top tier

triage_agent = Agent(
    get_model(TRIAGE_TIER),
    output_type=ToolOutput(Triage),
    retries=5,  # nano is the weakest tier; give it room to self-correct
    system_prompt=(
        "You are a triage classifier. In one quick pass, classify the support "
        "case by category (billing, technical, account, other) and urgency "
        "(low, medium, high), and give a one-line focus hint for the analyst. "
        "Do not solve it — just triage."
    ),
)

harness_agent = Agent(
    get_model(HARNESS_TIER),
    deps_type=HarnessDeps,
    output_type=ToolOutput(HarnessOutput),
    retries=3,  # exploration + structured output on local models benefits from retries
    system_prompt=(
        "You are a senior support analyst with a filesystem runtime and an "
        "external billing API.\n\n"
        "The knowledge base contains:\n"
        "  policies/   — refund policy, escalation matrix, subscription management\n"
        "  customers/  — customer profiles with transaction history\n"
        "  templates/  — response templates\n\n"
        "Investigate before acting:\n"
        "  1. Use list_files to see what exists.\n"
        "  2. read_file / search_files to gather the customer's history and policy.\n"
        "  3. check_payment_gateway to VERIFY the transaction before any refund.\n"
        "  4. issue_refund only when policy allows it.\n"
        "  5. Draft a personalized customer email from the relevant template.\n"
        "Think step by step about what information you need before each action."
    ),
)


# --- Filesystem runtime (sandboxed to deps.root) ---
# The tools are thin wrappers around the pure helpers in kb_tools, so the same
# logic is unit-tested there without needing a model or RunContext.


@harness_agent.tool
async def list_files(ctx: RunContext[HarnessDeps], glob: str = "**/*.md") -> str:
    """List files in the knowledge base matching a glob (default: all .md files)."""
    return kb_tools.list_files_text(ctx.deps.root, glob)


@harness_agent.tool
async def read_file(ctx: RunContext[HarnessDeps], path: str) -> str:
    """Read a single file from the knowledge base by its relative path."""
    return kb_tools.read_file_text(ctx.deps.root, path)


@harness_agent.tool
async def search_files(ctx: RunContext[HarnessDeps], term: str) -> str:
    """Grep the knowledge base for a term; returns matching path:line snippets."""
    return kb_tools.search_files_text(ctx.deps.root, term)


# --- External billing API (simulated) ---


@harness_agent.tool
async def check_payment_gateway(
    ctx: RunContext[HarnessDeps], transaction_date: str, amount: float
) -> str:
    """Verify a transaction's status and refund eligibility with the processor."""
    return kb_tools.payment_gateway_text(transaction_date, amount)


@harness_agent.tool
async def issue_refund(
    ctx: RunContext[HarnessDeps], amount: float, reason: str, customer_id: str
) -> str:
    """Process a refund through the payment gateway."""
    return kb_tools.refund_text(amount, reason, customer_id)


# --- Run ---


async def run_harness(task: str, root: Path = KNOWLEDGE_DIR) -> Any:
    """Tiered run: cheap nano triage first, then the large harness investigates.

    The triage result is folded into the harness prompt as a focus hint. The
    harness still explores freely — triage just points it at the right area
    without spending large-tier turns to get there.
    """
    triage = (await triage_agent.run(task)).output
    print(
        f"[triage · nano] category={triage.category} "
        f"urgency={triage.urgency} focus={triage.focus}"
    )
    hinted_task = (
        f"{task}\n\n"
        f"[Triage hint from fast pre-classifier] category={triage.category}, "
        f"urgency={triage.urgency}, focus: {triage.focus}"
    )
    return await harness_agent.run(hinted_task, deps=HarnessDeps(root=root))


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    result = asyncio.run(
        run_harness(
            "Customer cust_12345 reports a duplicate charge on their February bill. "
            "Investigate using the knowledge base, determine the right action per "
            "policy, and draft a personalized response using the appropriate template."
        )
    )
    print_agent_trace(result)
    print("\nStructured output:")
    print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
