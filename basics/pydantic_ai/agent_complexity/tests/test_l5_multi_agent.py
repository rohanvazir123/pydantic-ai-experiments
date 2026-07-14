"""Deterministic tests for Level 5 — an orchestrator-workers workflow.

Level 5 is a plain async `Orchestrator` (no graph): it owns the state, handles
retries (via tenacity in `reliable_run`), and routes, while dumb specialist agents
(research/draft/compliance) do one job each. These tests verify the coordination,
the redraft/escalation policy, and the retry layer without Ollama — every model is
overridden, so no live inference occurs.

Coverage:
  * specialists still expose their own sandboxed toolsets
  * the orchestrator runs research → draft → compliance → resolve in order
  * a compliance rejection loops back to a redraft, bounded by ``max_redrafts``
  * beyond the budget the workflow escalates instead of looping forever
  * :func:`reliable_run` retries transient failures (tenacity) and reraises
  * usage accumulates across every specialist call
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import l5_multi_agent as l5
import pytest
from pydantic_ai import Agent, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from tenacity import wait_fixed

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _instant_retries() -> Iterator[None]:
    """Zero the tenacity between-retry wait so the retry tests don't sleep.

    The @retry decorator bakes wait_fixed(_RETRY_WAIT_SECONDS) at import time, so
    we mutate the decorator's live retry controller rather than the constant.
    """
    original = l5.reliable_run.retry.wait
    l5.reliable_run.retry.wait = wait_fixed(0)
    yield
    l5.reliable_run.retry.wait = original


def _input() -> l5.CaseInput:
    return l5.CaseInput(customer_id="cust_12345", issue="A duplicate charge.")


def _deps(**kw) -> l5.CaseDeps:
    return l5.CaseDeps(root=l5.KNOWLEDGE_DIR, **kw)


def _structured(**fields) -> FunctionModel:
    """A model that immediately emits the agent's structured-output tool call."""

    def respond(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=fields)]
        )

    return FunctionModel(respond)


# --- Static wiring -----------------------------------------------------------


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


# --- Happy path: deterministic order, deterministic assembly -----------------


@pytest.mark.asyncio
async def test_workflow_runs_in_order_and_assembles_from_state() -> None:
    findings = _structured(
        summary="Duplicate charge confirmed on Feb bill.",
        duplicate_confirmed=True,
        refund_eligible=True,
        refund_amount=49.99,
    )
    email = _structured(subject="Refund on the way", body="Hi Sarah, resolved.")
    approve = _structured(approved=True, issues="")

    with ExitStack() as stack:
        stack.enter_context(l5.researcher.override(model=findings))
        stack.enter_context(l5.drafter.override(model=email))
        stack.enter_context(l5.compliance.override(model=approve))
        result = await l5.run_case(_input())

    assert result.duplicate_confirmed is True
    assert result.refund_amount == 49.99
    assert result.compliance_approved is True
    assert result.redrafts == 0
    assert result.escalated is False
    assert result.customer_email.subject == "Refund on the way"
    assert "Refund of $49.99" in result.final_action


# --- The feedback loop: compliance rejection routes back to a redraft ---------


@pytest.mark.asyncio
async def test_compliance_rejection_loops_back_to_redraft_then_approves() -> None:
    draft_calls = {"n": 0}

    def draft_respond(messages: list, info: AgentInfo) -> ModelResponse:
        draft_calls["n"] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"subject": f"draft-{draft_calls['n']}", "body": "..."},
                )
            ]
        )

    review_calls = {"n": 0}

    def review_respond(messages: list, info: AgentInfo) -> ModelResponse:
        review_calls["n"] += 1
        approved = review_calls["n"] >= 2  # reject once, then approve
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "approved": approved,
                        "issues": "" if approved else "Tone too curt.",
                    },
                )
            ]
        )

    findings = _structured(
        summary="s", duplicate_confirmed=True, refund_eligible=True, refund_amount=10.0
    )

    with ExitStack() as stack:
        stack.enter_context(l5.researcher.override(model=findings))
        stack.enter_context(l5.drafter.override(model=FunctionModel(draft_respond)))
        stack.enter_context(l5.compliance.override(model=FunctionModel(review_respond)))
        result = await l5.run_case(_input())

    assert draft_calls["n"] == 2  # drafted, rejected, redrafted
    assert review_calls["n"] == 2  # reviewed twice
    assert result.redrafts == 1
    assert result.escalated is False
    assert result.compliance_approved is True


@pytest.mark.asyncio
async def test_persistent_rejection_escalates_within_budget() -> None:
    findings = _structured(
        summary="s", duplicate_confirmed=True, refund_eligible=True, refund_amount=10.0
    )
    reject = _structured(approved=False, issues="Never compliant.")
    policy = l5.RetryPolicy(max_redrafts=2)

    draft_calls = {"n": 0}

    def draft_respond(messages: list, info: AgentInfo) -> ModelResponse:
        draft_calls["n"] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"subject": "s", "body": "b"},
                )
            ]
        )

    with ExitStack() as stack:
        stack.enter_context(l5.researcher.override(model=findings))
        stack.enter_context(l5.drafter.override(model=FunctionModel(draft_respond)))
        stack.enter_context(l5.compliance.override(model=reject))
        result = await l5.run_case(_input(), policy=policy)

    assert result.escalated is True
    assert result.redrafts == 2  # bounded — did not loop forever
    assert draft_calls["n"] == 3  # initial + 2 redrafts
    assert result.compliance_approved is False
    assert result.refund_amount == 0.0  # no refund when not approved
    assert "Escalated" in result.final_action


# --- Reliability layer: tenacity retries (orchestrator owns retries) ---------


def _flaky(fail_times: int, exc: BaseException) -> tuple[FunctionModel, dict]:
    calls = {"n": 0}

    async def respond(messages: list, info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(respond), calls


@pytest.mark.asyncio
async def test_reliable_run_retries_transient_failure_then_succeeds() -> None:
    agent: Agent[l5.CaseDeps, str] = Agent(
        "test", deps_type=l5.CaseDeps, output_type=str
    )
    model, calls = _flaky(2, ModelAPIError("x", "boom"))

    with agent.override(model=model):
        result = await l5.reliable_run(agent, "go", deps=_deps(), usage=RunUsage())

    assert result.output == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third (stop_after_attempt=3)


@pytest.mark.asyncio
async def test_reliable_run_reraises_after_exhausting_attempts() -> None:
    agent: Agent[l5.CaseDeps, str] = Agent(
        "test", deps_type=l5.CaseDeps, output_type=str
    )
    model, calls = _flaky(99, ModelAPIError("x", "boom"))

    with agent.override(model=model), pytest.raises(ModelAPIError):
        await l5.reliable_run(agent, "go", deps=_deps(), usage=RunUsage())

    assert calls["n"] == 3  # exactly _RETRY_ATTEMPTS tries, then reraise


@pytest.mark.asyncio
async def test_reliable_run_retries_timeout_errors() -> None:
    # Without asyncio.wait_for, a timeout is just another retryable exception
    # (a ModelSettings timeout surfaces as httpx.TimeoutException ⊂ TransportError).
    agent: Agent[l5.CaseDeps, str] = Agent(
        "test", deps_type=l5.CaseDeps, output_type=str
    )
    model, calls = _flaky(99, TimeoutError("timed out"))

    with agent.override(model=model), pytest.raises(TimeoutError):
        await l5.reliable_run(agent, "go", deps=_deps(), usage=RunUsage())

    assert calls["n"] == 3  # timeout is retryable, then reraised


# --- Shared usage tally ------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_accumulates_across_specialist_calls() -> None:
    findings = _structured(
        summary="s", duplicate_confirmed=True, refund_eligible=True, refund_amount=10.0
    )
    email = _structured(subject="s", body="b")
    approve = _structured(approved=True, issues="")

    state = l5.CaseState(case=_input())
    with ExitStack() as stack:
        stack.enter_context(l5.researcher.override(model=findings))
        stack.enter_context(l5.drafter.override(model=email))
        stack.enter_context(l5.compliance.override(model=approve))
        await l5.Orchestrator(state=state, deps=_deps()).run()

    # research + draft + compliance each made at least one request.
    assert state.usage.requests >= 3
