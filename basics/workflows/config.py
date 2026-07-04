"""
Shared Ollama model configuration for the workflows examples.

Mirrors basics/pydantic_ai/agent_complexity/config.py — same env vars,
same model tiers, same get_model() interface.

Override via env:
    OLLAMA_BASE_URL=http://localhost:11434/v1
    AGENT_LARGE_MODEL=qwen2.5:14b
"""
from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))

MODEL_TIERS: dict[str, str] = {
    "nano":  os.getenv("AGENT_NANO_MODEL",  "qwen2.5:0.5b"),
    "small": os.getenv("AGENT_SMALL_MODEL", "llama3.2:3b"),
    "large": os.getenv("AGENT_LARGE_MODEL", "qwen2.5:14b"),
}

DEFAULT_TIER: str = os.getenv("AGENT_COMPLEXITY_TIER", "large")


def get_model(tier: str = DEFAULT_TIER) -> OpenAIChatModel:
    """Return an Ollama-backed model for the given tier."""
    model_name = MODEL_TIERS.get(tier, tier)
    return OpenAIChatModel(
        model_name,
        provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
        settings=ModelSettings(temperature=TEMPERATURE),
    )
