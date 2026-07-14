"""
Level 5: Multi-Agent Orchestration — a Reliable Orchestrator-Workers Workflow
============================================================================

Levels 3-4 let *one* model drive. The naive way to "go multi-agent" is to hand a
big model a bag of sub-agent tools and let it decide the order — but then the
*orchestration itself* is a non-deterministic LLM guess: no guaranteed step
order, no place to hang a timeout, no retry policy, no feedback loop when a step
fails review. That is not orchestration; it is hope.

This level makes the **orchestrator** a first-class, code-defined object in the
**orchestrator-workers** shape. A single ``Orchestrator`` (a plain async class —
no graph framework; the control flow is simple enough that a ``while`` loop is
the right tool) owns everything smart: it holds the state, calls each dumb
specialist in turn, saves the result, and decides what happens next. The LLMs are
demoted to what they're good at (the *contents* of each step); every reliability
and control-flow concern lives in the orchestrator.

    Orchestrator.run()  ── owns state, retries, routing ──┐
        │  researcher  → save findings                    │
        │  drafter     → save draft                       │  loops until
        │  compliance  → save verdict                     │  resolved /
        │  approved?   → resolve                          │  escalated
        └  rejected & within budget → redraft ────────────┘
           rejected & over budget   → escalate → resolve

What makes it orchestration "worth the name":

  * **A real coordinator** — ``Orchestrator.run()`` is a single, deterministic
    decision point. It reads state and picks the next step (or finishes). It is
    *code, not a model*: deterministic routing is precisely the reliability win
    over letting a big LLM "decide" the next tool call.
  * **The orchestrator owns state** — it is the sole holder and writer of
    ``CaseState``. The dumb specialists never see or touch it; results flow back
    to the orchestrator, which saves them.
  * **A real feedback loop** — compliance returns a *structured verdict*. On
    rejection the orchestrator redrafts with the reviewer's issues, bounded to
    ``max_redrafts`` before it escalates.
  * **Typed I/O** — the workflow takes a ``CaseInput`` and returns a
    ``CaseResolution``; each step hands off a Pydantic model (``ResearchFindings``,
    ``CustomerEmail``, ``ComplianceVerdict``), so the final result is *assembled
    deterministically from state* — not re-synthesized by a model that might
    contradict the steps it just ran.
  * **Orchestrator-owned reliability** — every specialist call goes through
    :func:`reliable_run`: a per-request **timeout** (via ``ModelSettings``) plus
    tenacity-driven **retries** on transient model/transport errors. The agents
    themselves are dumb (no ``retries=``); retry policy lives in one place.
  * **Shared usage** — a single ``RunUsage`` threads through every call so cost
    rolls up across the whole workflow.

When to use: a multi-step process with *branching* and *quality gates*. Reach for
``pydantic_graph`` only when the control flow is a genuine multi-node state
machine; a linear pipeline with one bounded loop, like this, is cleaner as plain
async code.

Run:
    python l5_multi_agent.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import kb_tools
import observability
from config import get_model
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool, ToolOutput
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult

KNOWLEDGE_DIR = (Path(__file__).parent / "knowledge").resolve()


# ============================================================================
# Reliability layer — the thing the naive "orchestrator with tools" design lacks
# ============================================================================
#
# Every model call in the workflow goes through reliable_run, which wraps the
# bare `agent.run()` in two guarantees:
#
#   1. A per-request TIMEOUT via ModelSettings(timeout=…). The HTTP client gives
#      up on a hung model / stuck socket instead of wedging the workflow, raising
#      a timeout error that (2) then retries.
#   2. Bounded RETRIES via tenacity's @retry decorator: on a *transient* failure
#      (timeout, model API error, transport error) it retries up to
#      _RETRY_ATTEMPTS times, waiting _RETRY_WAIT_SECONDS between tries, then
#      reraises the last exception (`reraise=True`).
#
# This is deliberately NOT the same as Pydantic AI's `retries=` on an Agent.
# That retries when the *model* returns malformed output/tool args and should
# self-correct (a ModelRetry). reliable_run retries when the *infrastructure*
# fails — a different failure class self-correction can't fix. Here the agents
# are dumb (no `retries=`), so this is the workflow's single retry authority.

# Exceptions worth retrying: the request never got a good answer for a transient
# reason (the model's servers, or the wire). httpx.TimeoutException is a subclass
# of httpx.TransportError, so a ModelSettings timeout is covered here too. A
# ModelRetry is deliberately NOT retryable — agent.run's own `retries=` owns it.
RETRYABLE: tuple[type[BaseException], ...] = (
    TimeoutError,  # asyncio-level timeout
    ModelAPIError,  # 5xx / rate-limit / connection wrapped by Pydantic AI
    httpx.TransportError,  # connect/read/timeout errors from the HTTP client
)

_RETRY_ATTEMPTS = 3  # total tries per model call
_RETRY_WAIT_SECONDS = 1  # fixed pause between retries


@dataclass(frozen=True)
class RetryPolicy:
    """Per-step reliability budget, carried on the orchestrator's deps.

    Attributes:
        timeout: Per-request ceiling handed to ModelSettings(timeout=…), in
            seconds. The HTTP client aborts a call that exceeds it; tenacity then
            retries the aborted call.
        max_redrafts: How many times compliance may bounce the draft before the
            workflow escalates instead of looping forever.
    """

    timeout: float = 60.0
    max_redrafts: int = 2


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_fixed(_RETRY_WAIT_SECONDS),
    reraise=True,  # reraise the last transient error if all attempts fail
)
async def reliable_run[T](
    agent: Agent[CaseDeps, T],
    prompt: str,
    *,
    deps: CaseDeps,
    usage: RunUsage,
) -> AgentRunResult[T]:
    """Run ``agent`` with a per-request timeout, retrying transient failures.

    The timeout is enforced by the HTTP client via ``ModelSettings(timeout=…)``;
    tenacity's ``@retry`` decorator retries the call on any :data:`RETRYABLE`
    error, up to ``_RETRY_ATTEMPTS`` tries with a fixed pause, then reraises. A
    non-retryable error (e.g. a bug in a tool) propagates immediately — we only
    paper over *transient* faults.
    """
    return await agent.run(
        prompt,
        deps=deps,
        usage=usage,
        model_settings=ModelSettings(timeout=deps.retry.timeout),
    )


# ============================================================================
# Shared dependencies + sandboxed filesystem tools (reused by the specialists)
# ============================================================================


@dataclass
class CaseDeps:
    """Injected into every specialist: the sandbox root + the retry policy."""

    root: Path
    retry: RetryPolicy = field(default_factory=RetryPolicy)


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


# ============================================================================
# Typed I/O — the workflow's input and every hand-off between nodes is a model
# ============================================================================


class CaseInput(BaseModel):
    """Structured input to the whole workflow (replaces a raw task string)."""

    customer_id: str = Field(description="Customer identifier, e.g. 'cust_12345'.")
    issue: str = Field(description="The customer's reported problem, in prose.")

    def to_brief(self) -> str:
        """Render the input as the researcher's opening instruction."""
        return (
            f"Customer {self.customer_id} reports: {self.issue}\n"
            "Investigate: read the customer file and refund policy, and verify the "
            "transaction with the payment gateway. Report structured findings."
        )


class ResearchFindings(BaseModel):
    """Structured result of the investigation step."""

    summary: str = Field(description="Factual summary of what was found.")
    duplicate_confirmed: bool = Field(description="Is the duplicate charge real?")
    refund_eligible: bool = Field(description="Does policy allow a refund?")
    refund_amount: float = Field(description="Amount to refund (0 if none).")


class CustomerEmail(BaseModel):
    subject: str
    body: str


class ComplianceVerdict(BaseModel):
    """The compliance gate's structured decision — the graph branches on this."""

    approved: bool = Field(description="True only if the action follows policy.")
    issues: str = Field(
        default="",
        description="If not approved, the specific problems the drafter must fix.",
    )


class CaseResolution(BaseModel):
    """Final output, assembled deterministically from state (no synthesis LLM)."""

    research_summary: str
    duplicate_confirmed: bool
    refund_amount: float
    compliance_approved: bool
    redrafts: int
    escalated: bool
    final_action: str
    customer_email: CustomerEmail


# ============================================================================
# Specialist agents (each its own prompt + tools + tier + STRUCTURED output)
# ============================================================================
#
# Tiered LLMs by role: the researcher does multi-tool investigation and needs the
# reliable `large` tier; drafting and policy-checking are text-shaped and run on
# the cheap `small` tier. Requested tiers are pinned to `large` by default;
# AGENT_STRICT_TIERS=1 honors them on capable models (see README "Model tiers").
RESEARCHER_TIER = "large"  # multi-tool investigation
DRAFTER_TIER = "small"  # writing an email
COMPLIANCE_TIER = "small"  # checking a draft against policy

# The specialists are deliberately "dumb": no per-agent `retries=`, no retry logic
# of their own. Each is a single-shot call that does one job. ALL retry policy
# lives in the orchestration layer (`reliable_run`), so there is exactly one place
# that owns reliability — the orchestrator, not the workers. This is the
# orchestrator-pattern rule: smart coordinator, dumb workers.

researcher = Agent(
    get_model(RESEARCHER_TIER),
    name="researcher",
    deps_type=CaseDeps,
    output_type=ToolOutput(ResearchFindings),
    system_prompt=(
        "You are a billing research specialist. Investigate thoroughly: read the "
        "customer file to understand their history, check the payment gateway to "
        "verify the transaction, and read the refund policy to determine "
        "eligibility. Report structured findings: a factual summary, whether the "
        "duplicate charge is confirmed, whether a refund is policy-eligible, and "
        "the refund amount (0 if none)."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
        Tool(_check_payment_gateway, name="check_payment_gateway"),
    ],
)

drafter = Agent(
    get_model(DRAFTER_TIER),
    name="drafter",
    deps_type=CaseDeps,
    output_type=ToolOutput(CustomerEmail),
    system_prompt=(
        "You are a customer communications specialist. Draft a professional, "
        "empathetic email based on the research findings you receive. Use the "
        "response templates in the knowledge base as a starting point, then "
        "personalize for the customer. If you are given reviewer feedback from a "
        "previous draft, you MUST address every point raised. Return the email "
        "subject and body."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
    ],
)

compliance = Agent(
    get_model(COMPLIANCE_TIER),
    name="compliance",
    deps_type=CaseDeps,
    output_type=ToolOutput(ComplianceVerdict),
    system_prompt=(
        "You are a compliance reviewer. Verify that the proposed refund action "
        "and customer email follow company policy. Check the escalation matrix "
        "and refund policy. Approve only if everything is compliant; otherwise "
        "set approved=false and list the specific issues the drafter must fix."
    ),
    tools=[
        Tool(_list_files, name="list_files"),
        Tool(_read_file, name="read_file"),
    ],
)


# ============================================================================
# The workflow — a plain async Orchestrator (no graph; control flow this simple
# doesn't need one). The Orchestrator embodies the orchestrator pattern's
# division of labor:
#
#   * OWNS THE STATE — it is the sole holder and writer of CaseState (findings,
#     draft, verdict, redrafts, …). The specialist agents never see or touch it.
#   * HANDLES RETRIES — every specialist call goes through reliable_run, the
#     orchestration layer's retry wrapper. The agents have no retries of their own.
#   * ROUTES — its run() loop inspects state, calls the next dumb specialist,
#     SAVES the result, and repeats until the case resolves or escalates.
#
# The specialists (researcher/drafter/compliance) are dumb workers: prompt in,
# structured result out — no state, no retries, no knowledge of one another.
# ============================================================================


@dataclass
class CaseState:
    """Workflow state — owned and mutated exclusively by the Orchestrator."""

    case: CaseInput
    usage: RunUsage = field(default_factory=RunUsage)
    findings: ResearchFindings | None = None
    draft: CustomerEmail | None = None
    verdict: ComplianceVerdict | None = None
    feedback: str = ""
    redrafts: int = 0
    escalated: bool = False


@dataclass
class Orchestrator:
    """The coordinator — owns state, retries, and routing. Just a class + a loop.

    ``run()`` drives the case to completion: on each pass it runs the next missing
    step (calling a dumb specialist via :func:`reliable_run` and *saving* the
    result into the state it owns), and once a verdict is in it finishes,
    redrafts, or escalates. The loop terminates by construction: every pass either
    fills a state field or returns, and the only backward step (redraft) is
    bounded by ``max_redrafts``.
    """

    state: CaseState
    deps: CaseDeps

    async def run(self) -> CaseResolution:
        s = self.state
        while True:
            # --- Run the pipeline in order; save each result into state. ---
            # (.output is inlined per call so each specialist's distinct output
            # type is inferred independently — a shared `result` var would pin it.)
            if s.findings is None:
                s.findings = (
                    await reliable_run(
                        researcher, s.case.to_brief(), deps=self.deps, usage=s.usage
                    )
                ).output
                print(
                    f"[orchestrator] saved findings: "
                    f"duplicate={s.findings.duplicate_confirmed} "
                    f"eligible={s.findings.refund_eligible} "
                    f"amount=${s.findings.refund_amount:.2f}"
                )
                continue

            if s.draft is None:
                s.draft = (
                    await reliable_run(
                        drafter, self._draft_prompt(), deps=self.deps, usage=s.usage
                    )
                ).output
                print(f"[orchestrator] saved draft: subject={s.draft.subject!r}")
                continue

            if s.verdict is None:
                s.verdict = (
                    await reliable_run(
                        compliance, self._review_prompt(), deps=self.deps, usage=s.usage
                    )
                ).output
                print(f"[orchestrator] saved verdict: approved={s.verdict.approved}")
                continue

            # --- Verdict is in — decide the outcome. ---
            if s.verdict.approved:
                return self._resolve()

            if s.redrafts >= self.deps.retry.max_redrafts:
                s.escalated = True
                print(
                    f"[orchestrator] still rejected after {s.redrafts} redraft(s); "
                    "escalating to a human."
                )
                return self._resolve()

            # Rejected within budget: record feedback, clear rejected work, loop.
            s.redrafts += 1
            s.feedback = s.verdict.issues
            s.draft = None
            s.verdict = None
            print(f"[orchestrator] review rejected; redraft #{s.redrafts}")

    # --- Prompt builders + deterministic assembly (pure; no model, no I/O). ---

    def _draft_prompt(self) -> str:
        s = self.state
        assert s.findings is not None
        prompt = (
            "Draft a customer response based on these findings:\n"
            f"{s.findings.model_dump_json(indent=2)}"
        )
        if s.feedback:
            prompt += (
                "\n\nA previous draft was REJECTED by compliance. Fix these "
                f"issues in the new draft:\n{s.feedback}"
            )
        return prompt

    def _review_prompt(self) -> str:
        s = self.state
        assert s.findings is not None and s.draft is not None
        return (
            f"Findings:\n{s.findings.model_dump_json(indent=2)}\n\n"
            f"Proposed email:\n{s.draft.model_dump_json(indent=2)}"
        )

    def _resolve(self) -> CaseResolution:
        s = self.state
        assert (
            s.findings is not None and s.draft is not None and s.verdict is not None
        )
        if s.escalated:
            final_action = "Escalated to human review — compliance not satisfied."
        elif s.verdict.approved and s.findings.refund_eligible:
            final_action = (
                f"Refund of ${s.findings.refund_amount:.2f} approved and email sent."
            )
        else:
            final_action = "No refund issued; response sent per policy."
        return CaseResolution(
            research_summary=s.findings.summary,
            duplicate_confirmed=s.findings.duplicate_confirmed,
            refund_amount=s.findings.refund_amount if s.verdict.approved else 0.0,
            compliance_approved=s.verdict.approved,
            redrafts=s.redrafts,
            escalated=s.escalated,
            final_action=final_action,
            customer_email=s.draft,
        )


# ============================================================================
# Run
# ============================================================================


async def run_case(
    case: CaseInput,
    root: Path = KNOWLEDGE_DIR,
    policy: RetryPolicy | None = None,
) -> CaseResolution:
    """Execute the case-resolution workflow and return the final resolution."""
    state = CaseState(case=case)
    deps = CaseDeps(root=root, retry=policy or RetryPolicy())
    return await Orchestrator(state=state, deps=deps).run()


def main() -> None:
    observability.enable_logfire()  # no-op unless AGENT_LOGFIRE=1
    state = CaseState(
        case=CaseInput(
            customer_id="cust_12345",
            issue="A duplicate charge on their February bill.",
        )
    )
    deps = CaseDeps(root=KNOWLEDGE_DIR)
    result = asyncio.run(Orchestrator(state=state, deps=deps).run())

    print(f"\n--- Done | usage: {state.usage} | redrafts: {state.redrafts} ---")
    print("\nStructured output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
