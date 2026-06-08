"""V1–V6 validation chain.

All checks run before any DB query or LLM call. Cheap first, expensive last.
Rejection returns a structured ValidationError immediately — no further stages run.

V1  Schema          Pydantic model — handled by FastAPI (not this module)
V2  Length guard    len(query) > MAX_QUERY_CHARS → reject
V3  Language detect optional; skip when allowed_languages = ["*"]
V4  Injection guard regex + (future: embedding-sim against known patterns)
V5  Content policy  nano model → ContentPolicyResult; on_topic | off_topic | inappropriate
V6  RBAC check      JWT roles vs CorpusConfig.allowed_roles; runs before any DB I/O
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from knowledge.config.settings import Settings, load_settings
from knowledge.hooks.context import HookContext
from knowledge.hooks.registry import HookPoint, registry

logger = logging.getLogger(__name__)

# Common injection patterns (expand as needed)
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+", re.IGNORECASE),
    re.compile(r"(system|assistant)\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"###\s*(Human|Assistant|System)\s*:", re.IGNORECASE),
]


@dataclass
class ValidationError:
    code:        str
    message:     str
    status_code: int
    details:     dict[str, Any] | None = None


@dataclass
class ContentPolicyResult:
    verdict:    Literal["on_topic", "off_topic", "inappropriate"]
    confidence: float
    reason:     str | None = None


async def _v2_length_guard(query: str, settings: Settings) -> ValidationError | None:
    if len(query) > settings.max_query_chars:
        return ValidationError(
            code="QUERY_TOO_LONG",
            message=f"Query exceeds maximum length of {settings.max_query_chars} characters.",
            status_code=422,
            details={"length": len(query), "max": settings.max_query_chars},
        )
    return None


async def _v4_injection_guard(query: str) -> ValidationError | None:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return ValidationError(
                code="PROMPT_INJECTION_DETECTED",
                message="Query was rejected by the security filter.",
                status_code=422,
            )
    return None


async def _v5_content_policy(
    query: str,
    corpus_allowed_topics: list[str],
    settings: Settings,
    llm_client: Any | None = None,
) -> ValidationError | None:
    """Nano model content policy check.

    Returns None (pass) or ValidationError (reject).
    Skipped entirely when the corpus has no allowed_topics configured.
    """
    if not corpus_allowed_topics:
        return None

    topics_str = ", ".join(corpus_allowed_topics) or "general knowledge"
    prompt = (
        f"Corpus topics: {topics_str}\n\n"
        f"Query: {query}\n\n"
        "Classify as:\n"
        "  on_topic      — relevant to the corpus topics\n"
        "  off_topic     — coherent but unrelated\n"
        "  inappropriate — harmful, abusive, or policy-violating\n\n"
        "If uncertain between on_topic and off_topic, choose on_topic.\n"
        'Respond with JSON: {"verdict": "...", "confidence": 0.0, "reason": "..."}'
    )

    if llm_client is None:
        # Stub: skip V5 when no client is configured (local dev without nano model)
        return None

    try:
        import json
        response = await llm_client.complete(prompt)
        data = json.loads(response)
        result = ContentPolicyResult(**data)

        if result.verdict == "inappropriate":
            logger.warning("Content policy: inappropriate query (confidence=%.2f)", result.confidence)
            return ValidationError(
                code="CONTENT_POLICY_VIOLATION",
                message="Query was rejected by content policy.",
                status_code=400,
                details={"verdict": result.verdict},
            )
        if result.verdict == "off_topic":
            logger.info("Content policy: off_topic query (confidence=%.2f)", result.confidence)
            return ValidationError(
                code="QUERY_OFF_TOPIC",
                message="Query is not relevant to this knowledge base.",
                status_code=422,
                details={"verdict": result.verdict},
            )
    except Exception as exc:
        logger.warning("V5 content policy check failed (skipping): %s", exc)

    return None


async def _v6_rbac_check(
    corpus_ids: list[str],
    user_roles: list[str],
    corpus_registry: Any | None,
) -> ValidationError | None:
    """JWT role check against CorpusConfig.allowed_roles."""
    if corpus_registry is None:
        return None

    for corpus_id in corpus_ids:
        corpus = corpus_registry.get(corpus_id)
        if corpus is None:
            return ValidationError(
                code="CORPUS_NOT_FOUND",
                message=f"Corpus '{corpus_id}' does not exist.",
                status_code=404,
            )
        allowed = set(corpus.allowed_roles)
        if not allowed.intersection(user_roles):
            return ValidationError(
                code="CORPUS_ACCESS_DENIED",
                message=f"Insufficient role to access corpus '{corpus_id}'.",
                status_code=403,
                details={"corpus_id": corpus_id, "required": list(allowed)},
            )
    return None


class ValidationPipeline:
    """Runs V2–V6 in order. Returns None (all pass) or the first ValidationError."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: Any | None = None,
        corpus_registry: Any | None = None,
    ) -> None:
        self._settings        = settings or load_settings()
        self._llm_client      = llm_client
        self._corpus_registry = corpus_registry

    async def validate(
        self,
        ctx: HookContext,
        corpus_allowed_topics: list[str] | None = None,
        user_roles: list[str] | None = None,
    ) -> ValidationError | None:
        """Run validation chain. Fires PRE_VALIDATE and POST_VALIDATE/ON_VALIDATION_FAIL hooks."""

        await registry.fire(HookPoint.PRE_VALIDATE, ctx)

        # V2 — length
        err = await _v2_length_guard(ctx.query, self._settings)
        if err:
            ctx.abstention_reason = err.code
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return err

        # V4 — injection
        err = await _v4_injection_guard(ctx.query)
        if err:
            ctx.abstention_reason = err.code
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return err

        # V5 — content policy (nano model; skipped if no client or no topics)
        err = await _v5_content_policy(
            ctx.query,
            corpus_allowed_topics or [],
            self._settings,
            self._llm_client,
        )
        if err:
            ctx.abstention_reason = err.code
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return err

        # V6 — RBAC
        err = await _v6_rbac_check(ctx.corpus_ids, user_roles or [], self._corpus_registry)
        if err:
            ctx.abstention_reason = err.code
            await registry.fire(HookPoint.ON_VALIDATION_FAIL, ctx)
            return err

        await registry.fire(HookPoint.POST_VALIDATE, ctx)
        return None
