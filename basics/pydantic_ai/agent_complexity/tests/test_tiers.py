"""Deterministic tests for the tiered-LLM design (no model, no network).

Two things are pinned here:

1. **Intent** — each role *requests* an appropriate tier (cheap tiers for
   classification / text, `large` for reasoning + tool-calling). A careless edit
   can't silently promote a nano job to the big model or demote a reasoning job.
2. **Policy** — by default (``AGENT_STRICT_TIERS`` unset) every agent *resolves*
   to the pinned tier (`large`) so the examples run reliably on weak local small
   models. ``effective_tier`` is the single switch that makes tiering real on
   capable models.
"""

from __future__ import annotations

import config
import l1_augmented_llm as l1
import l2_prompt_chains as l2
import l3_tool_calling_agent as l3
import l4_agent_harness as l4
import l5_multi_agent as l5
from config import MODEL_TIERS

LARGE = MODEL_TIERS["large"]


def _model_name(agent: object) -> str:
    return agent.model.model_name  # type: ignore[attr-defined]


# --- Intent: each role requests the right tier ---


def test_role_tier_intents() -> None:
    assert l1.CLASSIFIER_TIER == "small"  # pure classification -> cheap
    assert l2.CLASSIFIER_TIER == "nano"  # routing -> cheapest
    assert l2.HANDLER_TIER == "small"  # response generation -> standard
    assert l3.AGENT_TIER == "large"  # tool calling -> top tier
    assert l4.TRIAGE_TIER == "nano"  # fast triage -> cheapest
    assert l4.HARNESS_TIER == "large"  # open-ended reasoning -> top tier
    # L5's orchestrator is now a deterministic code node (no model/tier).
    assert l5.RESEARCHER_TIER == "large"
    assert l5.DRAFTER_TIER == "small"  # drafting text -> standard
    assert l5.COMPLIANCE_TIER == "small"  # checking text -> standard


# --- Policy: default pins everything to `large` for reliable local runs ---


def test_default_policy_pins_large() -> None:
    assert config.STRICT_TIERS is False  # default
    for agent in (
        l1.agent,
        l2.classifier,
        l2.billing_handler,
        l3.billing_agent,
        l4.triage_agent,
        l4.harness_agent,
        l5.researcher,
        l5.drafter,
        l5.compliance,
    ):
        assert _model_name(agent) == LARGE


def test_effective_tier_switch() -> None:
    # Pinned by default; honors the request only under strict mode.
    assert config.effective_tier("nano") == config.PINNED_TIER
    assert config.PINNED_TIER == "large"


def test_tiers_are_distinct() -> None:
    # The scheme is only meaningful if the tier names map to different models.
    assert len(set(MODEL_TIERS.values())) == 3
