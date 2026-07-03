"""
Shared Ollama model configuration for the agent-complexity examples.

Every example builds its agents from :func:`get_model`, so switching model or
endpoint is a one-line change (or an env var) rather than an edit in five files.

Pydantic AI can resolve ``"ollama:<model>"`` strings directly, but only if the
``OLLAMA_BASE_URL`` env var is set. We build the model explicitly via
:class:`OllamaProvider` instead so the examples run out of the box against a
local Ollama daemon with no environment setup required.

Model tiers mirror the RAG v2 convention (nano / small / large):

    nano  -> qwen2.5:0.5b   routing, classification (cheap, fast)
    small -> llama3.2:3b    standard responses
    large -> qwen2.5:14b    reliable tool-calling + structured output

Override any of these without touching code:

    export OLLAMA_BASE_URL=http://localhost:11434/v1
    export AGENT_NANO_MODEL=qwen2.5:0.5b
    export AGENT_SMALL_MODEL=llama3.2:3b
    export AGENT_LARGE_MODEL=qwen2.5:14b
    export AGENT_COMPLEXITY_TIER=large   # default tier for the examples
"""

from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# temperature=0 makes structured output and tool-argument generation far more
# reliable on small local models (the examples want correctness, not creativity).
TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))

# Named tiers -> concrete Ollama model tags. Override via env.
MODEL_TIERS: dict[str, str] = {
    "nano": os.getenv("AGENT_NANO_MODEL", "qwen2.5:0.5b"),
    "small": os.getenv("AGENT_SMALL_MODEL", "llama3.2:3b"),
    "large": os.getenv("AGENT_LARGE_MODEL", "qwen2.5:14b"),
}

# Default tier for the examples. `large` is slower but far more reliable at
# structured output and tool calling with local models — which is exactly what
# levels 3-5 exercise. Drop to `small` for faster (flakier) demos.
DEFAULT_TIER: str = os.getenv("AGENT_COMPLEXITY_TIER", "large")

# Tiering policy. The examples assign a *semantic* tier per role (e.g. the L2
# classifier asks for "nano"), but on this local box the small tiers
# (llama3.2:3b, qwen2.5:0.5b) are NOT reliable at tool/structured output — only
# qwen2.5:14b is. So by default we PIN every agent to `PINNED_TIER` for reliable
# runs, while keeping the per-role tiers one env var away:
#
#   AGENT_STRICT_TIERS=1   -> honor each agent's requested tier (needs capable
#                             models: a hosted provider, or better local models)
#   AGENT_PINNED_TIER=small -> pin everything to a different tier instead
#
# On a hosted provider where small models tool-call fine, set AGENT_STRICT_TIERS=1
# and the per-role tiering below becomes a real cost saver.
STRICT_TIERS: bool = os.getenv("AGENT_STRICT_TIERS", "0") == "1"
PINNED_TIER: str = os.getenv("AGENT_PINNED_TIER", DEFAULT_TIER)


def resolve_model_name(tier: str) -> str:
    """Return the concrete Ollama tag for a tier name (or pass through a raw tag)."""
    return MODEL_TIERS.get(tier, tier)


def effective_tier(requested: str) -> str:
    """Resolve a requested role tier to the tier actually used.

    Honors ``requested`` only when ``AGENT_STRICT_TIERS=1``; otherwise pins to
    ``PINNED_TIER`` (default ``large``) so the examples run reliably on weak
    local small models. This is the one place tiering policy lives.
    """
    return requested if STRICT_TIERS else PINNED_TIER


def get_model(tier: str = DEFAULT_TIER) -> OpenAIChatModel:
    """Build an Ollama-backed model for the given (requested) role tier.

    The requested tier is passed through :func:`effective_tier`, so by default
    every agent pins to ``PINNED_TIER`` regardless of what it asked for (see the
    tiering-policy note above). Constructing the model does no network I/O, so
    importing an example module (and overriding it with ``TestModel`` in the test
    suite) never touches Ollama.

    Args:
        tier: The role's *semantic* tier — ``"nano"``, ``"small"``, ``"large"``,
            or a raw Ollama tag such as ``"qwen2.5:14b"``.
    """
    return OpenAIChatModel(
        resolve_model_name(effective_tier(tier)),
        provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
        settings=ModelSettings(temperature=TEMPERATURE),
    )
