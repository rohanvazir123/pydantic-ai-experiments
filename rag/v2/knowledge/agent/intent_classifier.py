# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Intent classifier — determines query intent before retrieval.

Runs after validation (V1-V6), before the cache check and hybrid search.
Uses the nano model (qwen2.5:0.5b) with a 2s timeout; falls back to
'factual' on timeout or error so the pipeline is never blocked.

IntentResult drives two retrieval parameters:
  k_multiplier  — scales the default judge_k chunk count
                  comparison / summarization fetch 2× to cover both sides
  include_graph — enables the Apache AGE leg for relational queries

Intent types:
  factual       single fact or definition lookup
  comparison    explicitly comparing two or more entities / periods
  summarization requesting an overview or summary of a topic
  procedural    how-to, steps, instructions, processes
  relational    relationships, org structure, entity connections
"""

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge.agent.prompts import INTENT_CLASSIFIER_PROMPT
from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

_TIMEOUT_S = 2.0

# k_multiplier and include_graph defaults per intent
_INTENT_PARAMS: dict[str, dict[str, Any]] = {
    "factual":       {"k_multiplier": 1.0, "include_graph": False},
    "comparison":    {"k_multiplier": 2.0, "include_graph": False},
    "summarization": {"k_multiplier": 2.0, "include_graph": False},
    "procedural":    {"k_multiplier": 1.5, "include_graph": False},
    "relational":    {"k_multiplier": 1.5, "include_graph": True},
}

_FALLBACK = "factual"


class IntentResult(BaseModel):
    intent:        Literal["factual", "comparison", "summarization", "procedural", "relational"]
    k_multiplier:  float  # multiply settings.judge_k before retrieval
    include_graph: bool   # enable AGE Cypher leg in hybrid search
    reasoning:     str    # one sentence; logged, never shown to user


def _fallback(reason: str) -> IntentResult:
    params = _INTENT_PARAMS[_FALLBACK]
    logger.warning("Intent classifier fallback (%s) → %s", reason, _FALLBACK)
    return IntentResult(
        intent="factual",
        reasoning=f"fallback: {reason}",
        k_multiplier=params["k_multiplier"],
        include_graph=params["include_graph"],
    )


_agent: Any | None = None


def _get_agent(settings: Settings) -> Any:
    global _agent
    if _agent is None:
        provider = OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        model = OpenAIChatModel(settings.model_tier_nano, provider=provider)
        ms: dict[str, Any] = {}
        if settings.llm_provider == "ollama":
            ms = {"extra_body": {"num_ctx": 2048}}
        _agent = Agent(  # type: ignore[call-overload]
            model,
            instructions=INTENT_CLASSIFIER_PROMPT,
            output_type=IntentResult,
            model_settings=ms,
        )
    return _agent


async def classify_intent(
    query: str,
    settings: Settings | None = None,
) -> IntentResult:
    """Classify query intent. Returns fallback on timeout or model error."""
    _settings = settings or load_settings()

    try:
        agent = _get_agent(_settings)
        result = await asyncio.wait_for(agent.run(query), timeout=_TIMEOUT_S)
        ir: IntentResult = result.output
        # Enforce k_multiplier and include_graph from the lookup table
        # (prevents the model from returning arbitrary floats)
        params = _INTENT_PARAMS.get(ir.intent, _INTENT_PARAMS[_FALLBACK])
        ir.k_multiplier  = params["k_multiplier"]
        ir.include_graph = params["include_graph"]
        logger.info(
            "Intent: %s (k×%.1f graph=%s) — %s",
            ir.intent, ir.k_multiplier, ir.include_graph, ir.reasoning,
        )
        return ir
    except TimeoutError:
        return _fallback("timeout")
    except Exception as exc:
        return _fallback(str(exc))
