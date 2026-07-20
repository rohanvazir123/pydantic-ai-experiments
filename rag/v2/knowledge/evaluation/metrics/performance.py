# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Performance metrics — latency spans, token counts, cost estimation."""

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 0.00025,  "output": 0.00125},
    "claude-sonnet-4-6": {"input": 0.003,    "output": 0.015},
    "claude-opus-4-8":   {"input": 0.015,    "output": 0.075},
    # Local Ollama models cost $0 — omit from table (default 0.0)
}


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for one LLM call. Returns 0.0 for local models."""
    pricing = COST_PER_1K_TOKENS.get(model_id)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000
