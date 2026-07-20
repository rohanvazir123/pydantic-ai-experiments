# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Unit tests for knowledge.agent.intent_classifier.

No live LLM or services required — the pydantic-ai agent is mocked at the
run() boundary. Tests cover all five intents, timeout fallback, and the
invariant that k_multiplier / include_graph are always enforced from the
lookup table regardless of model output.
"""

import asyncio
from unittest import mock

import pytest

from knowledge.agent.intent_classifier import (
    IntentResult,
    _INTENT_PARAMS,
    classify_intent,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _agent_returning(intent: str) -> mock.AsyncMock:
    """Return a mock agent whose run() resolves to an IntentResult for *intent*."""
    ir = IntentResult(
        intent=intent,          # type: ignore[arg-type]
        k_multiplier=99.9,      # deliberately wrong — should be overridden by table
        include_graph=True,     # deliberately wrong for non-relational intents
        reasoning="test",
    )
    mock_result = mock.MagicMock()
    mock_result.output = ir

    agent = mock.AsyncMock()
    agent.run = mock.AsyncMock(return_value=mock_result)
    return agent


def _patch_agent(agent: mock.AsyncMock):
    """Patch the module-level _agent singleton and reset it after the test."""
    return mock.patch("knowledge.agent.intent_classifier._agent", agent)


# ── intent → correct k_multiplier and include_graph ──────────────────────────

class TestIntentParams:
    @pytest.mark.asyncio
    async def test_factual_k1_no_graph(self) -> None:
        with _patch_agent(_agent_returning("factual")):
            result = await classify_intent("What is the PTO policy?")
        assert result.intent == "factual"
        assert result.k_multiplier == 1.0
        assert result.include_graph is False

    @pytest.mark.asyncio
    async def test_procedural_k15_no_graph(self) -> None:
        with _patch_agent(_agent_returning("procedural")):
            result = await classify_intent("How do I submit an expense report?")
        assert result.intent == "procedural"
        assert result.k_multiplier == 1.5
        assert result.include_graph is False

    @pytest.mark.asyncio
    async def test_relational_k15_with_graph(self) -> None:
        with _patch_agent(_agent_returning("relational")):
            result = await classify_intent("Who reports to the CTO?")
        assert result.intent == "relational"
        assert result.k_multiplier == 1.5
        assert result.include_graph is True

    @pytest.mark.asyncio
    async def test_comparison_k2_no_graph(self) -> None:
        with _patch_agent(_agent_returning("comparison")):
            result = await classify_intent("How do Q3 and Q4 revenue compare?")
        assert result.intent == "comparison"
        assert result.k_multiplier == 2.0
        assert result.include_graph is False

    @pytest.mark.asyncio
    async def test_summarization_k2_no_graph(self) -> None:
        with _patch_agent(_agent_returning("summarization")):
            result = await classify_intent("Summarize the company overview.")
        assert result.intent == "summarization"
        assert result.k_multiplier == 2.0
        assert result.include_graph is False


# ── lookup-table enforcement ──────────────────────────────────────────────────

class TestLookupTableEnforcement:
    """Model output for k_multiplier and include_graph is ignored; table wins."""

    @pytest.mark.asyncio
    async def test_model_k_multiplier_overridden(self) -> None:
        # Agent returns 99.9 — table should replace with 1.0 for factual
        with _patch_agent(_agent_returning("factual")):
            result = await classify_intent("Anything")
        assert result.k_multiplier == _INTENT_PARAMS["factual"]["k_multiplier"]

    @pytest.mark.asyncio
    async def test_model_include_graph_overridden_for_factual(self) -> None:
        # Agent returns include_graph=True — table should override to False
        with _patch_agent(_agent_returning("factual")):
            result = await classify_intent("Anything")
        assert result.include_graph is False


# ── fallback on timeout ───────────────────────────────────────────────────────

class TestFallbacks:
    @pytest.mark.asyncio
    async def test_timeout_returns_factual(self) -> None:
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(side_effect=asyncio.TimeoutError)

        with _patch_agent(agent):
            result = await classify_intent("some query")

        assert result.intent == "factual"
        assert result.k_multiplier == 1.0
        assert result.include_graph is False
        assert "fallback" in result.reasoning

    @pytest.mark.asyncio
    async def test_model_error_returns_factual(self) -> None:
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(side_effect=RuntimeError("model unavailable"))

        with _patch_agent(agent):
            result = await classify_intent("some query")

        assert result.intent == "factual"
        assert "fallback" in result.reasoning

    @pytest.mark.asyncio
    async def test_fallback_never_raises(self) -> None:
        """classify_intent must never propagate exceptions."""
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(side_effect=Exception("unexpected"))

        with _patch_agent(agent):
            result = await classify_intent("some query")

        assert result is not None


# ── IntentResult schema ───────────────────────────────────────────────────────

class TestIntentResultModel:
    def test_all_intents_in_params_table(self) -> None:
        for intent in ("factual", "comparison", "summarization", "procedural", "relational"):
            assert intent in _INTENT_PARAMS, f"{intent!r} missing from _INTENT_PARAMS"

    def test_valid_intent_constructs(self) -> None:
        ir = IntentResult(
            intent="relational", k_multiplier=1.5,
            include_graph=True, reasoning="test",
        )
        assert ir.intent == "relational"

    def test_invalid_intent_raises(self) -> None:
        with pytest.raises(Exception):
            IntentResult(
                intent="unknown",   # type: ignore[arg-type]
                k_multiplier=1.0, include_graph=False, reasoning="",
            )
