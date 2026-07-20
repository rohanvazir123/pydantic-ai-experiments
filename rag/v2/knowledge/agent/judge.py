# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""LLM-as-judge — Layer 3 gate in the confidence-aware pipeline.

Evaluates the generated answer against retrieved context WITHOUT seeing
chunk_id metadata (prevents being fooled by well-formatted hallucinations).

Model tier: nano by default. Escalates to small if nano's own confidence < 0.5
(prevents incorrect abstentions on genuinely ambiguous queries).
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge.agent.prompts import JUDGE_SYSTEM_PROMPT
from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

_ESCALATION_THRESHOLD = 0.5  # if nano confidence < this, escalate to small


class JudgeResult(BaseModel):
    verdict:    Literal["supported", "partial", "unsupported"]
    confidence: float        # 0.0-1.0; judge's confidence in its own verdict
    reasoning:  str          # one sentence; logged, never shown to user


def _make_judge_agent(model_id: str, settings: Settings) -> Any:
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    model = OpenAIChatModel(model_id, provider=provider)
    _ms: dict = {}
    if settings.llm_provider == "ollama":
        _ms = {"extra_body": {"num_ctx": 4096}}
    return Agent(  # type: ignore[call-overload]
        model,
        instructions=JUDGE_SYSTEM_PROMPT,
        output_type=JudgeResult,
        model_settings=_ms,
    )


_nano_judge:  Any | None = None
_small_judge: Any | None = None


def _get_nano_judge(settings: Settings) -> Any:
    global _nano_judge
    if _nano_judge is None:
        _nano_judge = _make_judge_agent(settings.model_tier_nano, settings)
    return _nano_judge


def _get_small_judge(settings: Settings) -> Any:
    global _small_judge
    if _small_judge is None:
        _small_judge = _make_judge_agent(settings.model_tier_small, settings)
    return _small_judge


def _build_judge_prompt(query: str, context: str, answer: str) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"SOURCE PASSAGES:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}"
    )


async def judge(
    query: str,
    context: str,     # retrieved passages WITHOUT chunk_id metadata
    answer: str,
    settings: Settings | None = None,
) -> JudgeResult:
    """Run the LLM judge. Escalates nano→small if nano confidence < threshold."""
    _settings = settings or load_settings()
    prompt    = _build_judge_prompt(query, context, answer)

    # Nano pass
    nano_agent = _get_nano_judge(_settings)
    try:
        result = await nano_agent.run(prompt)
        jr: JudgeResult = result.output

        if jr.confidence >= _ESCALATION_THRESHOLD:
            logger.debug("Judge: %s (confidence=%.2f, model=nano)", jr.verdict, jr.confidence)
            return jr

        # Escalate to small
        logger.info(
            "Judge escalating nano→small (nano confidence=%.2f < %.2f)",
            jr.confidence, _ESCALATION_THRESHOLD,
        )
    except Exception as exc:
        logger.warning("Nano judge failed (%s) — escalating to small", exc)

    small_agent = _get_small_judge(_settings)
    try:
        result = await small_agent.run(prompt)
        jr = result.output
        logger.debug("Judge (small): %s (confidence=%.2f)", jr.verdict, jr.confidence)
        return jr
    except Exception as exc:
        logger.error("Small judge failed: %s — pessimistic abstain", exc)
        return JudgeResult(
            verdict="unsupported",
            confidence=0.0,
            reasoning=f"Judge model failed: {exc}",
        )
