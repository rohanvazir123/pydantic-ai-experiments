"""Query complexity classifier — intended to select the cheapest LLM tier.

INTENT (not implemented):
  Run the nano model (qwen2.5:0.5b) on every incoming query to classify its
  complexity, then route to the appropriate tier:
    simple   → nano  (qwen2.5:0.5b)   single-fact lookups
    moderate → small (llama3.2:3b)    synthesis across sources
    complex  → large (llama3.1:70b)   multi-hop reasoning, graph traversal
  Goal: pay a ~50ms nano call to avoid paying for a 70b call on simple queries.
  Makes economic sense when routing between cheap local models and expensive
  cloud APIs (e.g. GPT-4o). Less useful in a fully-local stack.

NOT CALLED: this module is not imported or invoked anywhere in the pipeline.
  The pipeline uses the model_tier field from the request body directly (default
  "small"). Wire in route() from pipeline.py if tier selection is needed.
"""

import asyncio
import logging
from typing import Any, Literal, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge.agent.prompts import ROUTER_SYSTEM_PROMPT
from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    complexity:               Literal["simple", "moderate", "complex"]
    requires_graph:           bool
    requires_multipass:       bool
    estimated_context_tokens: int
    rejected:                 bool        = False
    rejection_reason:         str | None  = None


def _get_nano_model(settings: Settings) -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.model_tier_nano, provider=provider)


_router_agent: Agent | None = None


def _get_router_agent(settings: Settings) -> Any:
    global _router_agent
    if _router_agent is None:
        model = _get_nano_model(settings)
        _ms: dict = {}
        if settings.llm_provider == "ollama":
            _ms = {"extra_body": {"num_ctx": 2048}}
        _router_agent = Agent(  # type: ignore[call-overload]
            model,
            instructions=ROUTER_SYSTEM_PROMPT,
            output_type=RoutingDecision,
            model_settings=_ms,
        )
    return _router_agent


async def route(
    query: str,
    settings: Settings | None = None,
) -> RoutingDecision:
    """Run the nano model router. Falls back to 'small' on timeout."""
    _settings = settings or load_settings()

    if not _settings.model_routing_enabled:
        return RoutingDecision(
            complexity="moderate",
            requires_graph=False,
            requires_multipass=False,
            estimated_context_tokens=1500,
        )

    agent = _get_router_agent(_settings)
    try:
        result = await asyncio.wait_for(
            agent.run(query),
            timeout=_settings.model_routing_timeout_s,
        )
        return cast("RoutingDecision", result.output)
    except (TimeoutError, Exception) as exc:
        logger.warning("Model router failed (%s) — defaulting to small", exc)
        return RoutingDecision(
            complexity="moderate",
            requires_graph=False,
            requires_multipass=False,
            estimated_context_tokens=1500,
        )
