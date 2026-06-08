"""HookContext — carries full request state through the hook chain.

Passed to every hook at every lifecycle point. Hooks may mutate it
(e.g. inject user memories into system_prompt_prefix) or raise HookAbort
to short-circuit the pipeline with a custom response.
"""

from dataclasses import dataclass, field
from typing import Any


class HookAbort(Exception):
    """Raised by a hook to abort the pipeline and return a custom response.

    Attributes:
        response: the dict to return to the caller instead of running the pipeline.
        status_code: HTTP status code (default 200; use 4xx for policy rejections).
    """

    def __init__(self, response: dict[str, Any], status_code: int = 200) -> None:
        self.response    = response
        self.status_code = status_code
        super().__init__(f"HookAbort: status={status_code}")


@dataclass
class HookContext:
    """Mutable request context passed through the hook chain.

    Fields are populated progressively as the pipeline advances:
    - validation stage: query, corpus_ids, user_id, tenant_id, request_id
    - routing stage:    routing_decision
    - retrieval stage:  retrieved_chunks, aggregate_confidence
    - LLM stage:        llm_response, generation_result
    - error stage:      error
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id:    str       = ""
    user_id:       str       = ""
    tenant_id:     str       = ""
    session_id:    str       = ""

    # ── Request ───────────────────────────────────────────────────────────────
    query:         str       = ""
    corpus_ids:    list[str] = field(default_factory=list)
    model_tier:    str       = "small"

    # ── Prompt prefix (injected by PRE_RETRIEVE hook for user memories) ───────
    system_prompt_prefix: str = ""

    # ── Routing ───────────────────────────────────────────────────────────────
    routing_decision: Any | None = None    # RoutingDecision | None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks:      list[Any] = field(default_factory=list)  # list[SearchResult]
    aggregate_confidence:  float     = 0.0

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_response:      Any | None = None   # GenerationResult | None
    generation_result: Any | None = None

    # ── Error / abstention ────────────────────────────────────────────────────
    error:              Exception | None = None
    abstention_layer:   int | None       = None
    abstention_reason:  str | None       = None

    # ── Arbitrary metadata for hook-to-hook communication ────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)
