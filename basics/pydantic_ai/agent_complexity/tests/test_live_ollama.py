"""Live integration + latency tests against a real Ollama daemon.

SKIPPED by default (slow, non-deterministic). Run explicitly:

    RUN_OLLAMA=1 pytest tests/test_live_ollama.py -v -s
    pytest --run-ollama -v -s

Each test:
  * asserts a *well-typed, plausible* result (not exact strings — small local
    models vary run to run), and
  * is wrapped in the `latency` fixture, so the session prints a per-level
    latency table (p50/p95/p99) and writes it to `.sample_runs/latency_tests.txt`.

For meaningful percentiles, repeat the run — e.g. with `pytest-repeat`:

    RUN_OLLAMA=1 pytest tests/test_live_ollama.py -p no:cacheprovider --count=10

or use `benchmark.py`, which is purpose-built for percentile measurement.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

import config
import l1_augmented_llm as l1
import l2_prompt_chains as l2
import l3_tool_calling_agent as l3
import l4_agent_harness as l4
import l5_multi_agent as l5
import pytest

pytestmark = pytest.mark.ollama


def _ollama_up() -> bool:
    try:
        base = config.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(autouse=True)
def _require_ollama() -> None:
    if not _ollama_up():
        pytest.skip(f"Ollama not reachable at {config.OLLAMA_BASE_URL}")


@pytest.mark.level("L1 augmented-llm")
def test_level1_live(latency: None) -> None:
    result = l1.classify(
        "I was charged twice for my subscription. Order #12345. Refund please."
    )
    assert isinstance(result, l1.TicketClassification)
    assert result.category and result.priority
    assert isinstance(result.can_auto_resolve, bool)


@pytest.mark.level("L2 prompt-chains")
def test_level2_live(latency: None) -> None:
    resolution = l2.process_ticket(
        "I was charged twice for my subscription. Order #12345. The duplicate was $49.99."
    )
    assert isinstance(resolution, l2.Resolution)
    assert resolution.response
    assert isinstance(resolution.escalate, bool)


@pytest.mark.level("L3 tool-calling")
def test_level3_live(latency: None) -> None:
    result = asyncio.run(
        l3.resolve(
            "I was charged twice on Feb 1st for my subscription. Please fix this.",
            l3._sample_deps(),
        )
    )
    assert isinstance(result.output, l3.BillingResolution)
    tool_calls = [
        p
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if type(p).__name__ == "ToolCallPart"
    ]
    assert tool_calls, "expected the agent to call at least one tool"


@pytest.mark.level("L4 agent-harness")
def test_level4_live(latency: None) -> None:
    result = asyncio.run(
        l4.run_harness(
            "Customer cust_12345 reports a duplicate charge on their February bill. "
            "Investigate using the knowledge base and refund per policy."
        )
    )
    assert isinstance(result.output, l4.HarnessOutput)
    assert isinstance(result.output.customer_email, l4.CustomerEmail)
    # The harness should have actually read from the knowledge base.
    tool_names = {
        p.tool_name
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if type(p).__name__ == "ToolCallPart"
    }
    assert tool_names, "expected the harness to use its runtime tools"


@pytest.mark.level("L5 multi-agent")
def test_level5_live(latency: None) -> None:
    result = asyncio.run(
        l5.run_orchestrator(
            "Customer cust_12345 reports a duplicate charge on their February bill. "
            "Have the researcher investigate, the drafter prepare a response, and "
            "compliance review before we send it."
        )
    )
    assert isinstance(result.output, l5.OrchestratorOutput)
    assert isinstance(result.output.customer_email, l5.CustomerEmail)
    # Delegation means several model requests happened (orchestrator + specialists).
    assert result.usage.requests >= 2
