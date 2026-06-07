"""
Production-Ready Meeting Transcript Multi-Agent Pipeline
=========================================================

Four-agent async pipeline that extracts insights and validated action items
from raw meeting transcripts.

  1. PreProcessing  — clean and label transcript turns (deterministic, no LLM)
  2. Extraction     — insights (sentiment, pain points, competitors)
  3. Commitments    — action items with strict Literal guardrails
  4. Validation     — critic that drops hallucinated or withdrawn items

Stages 2+3 are submitted concurrently via asyncio.gather.  Note that local
Ollama serialises GPU requests, so on a single-GPU machine the gather produces
no wall-clock speedup — effective total is the sum of all LLM call durations.

Production features
-------------------
  Strict guardrails   Literal types on constrained fields → JSON Schema enum; model
                      cannot hallucinate invalid values (automatic retries on violation)
  Checkpointing       Each stage writes JSON to .pipeline_checkpoints/<id>/; on restart
                      completed stages are loaded and skipped (state restore)
  Audit log           Per-meeting JSONL with stage, timestamps, token counts, latencies
  Safety checks       Input size limits, speaker count cap, content length guard,
                      Pydantic field validators on output models
  Memory              ~/.meeting_pipeline/history.json tracks all processed meetings
  Structured logging  Python logging throughout; --debug for verbose output
  Retries             3x per stage; Pydantic AI sends validation errors back to LLM

Latency breakdown (qwen2.5:14b on local Ollama, ~50 tok/s, single GPU)
------------------------------------------------------------------------
  Stage           LLM calls   Approx tokens in   Approx time
  ─────────────────────────────────────────────────────────────
  preprocessing   0           —                  ~0s   (deterministic)
  extraction      1           ~2 k               ~40s  (single-pass; full transcript in ctx)
  commitments     1           ~2 k               ~40s  (same transcript, action items only)
  validation      1           ~0.5 k             ~15s  (items only; transcript not resent)
  ─────────────────────────────────────────────────────────────
  Total (serial)  3                              ~95s  (single GPU; gather adds no speedup)

  Previous run 2 was 176s because extraction made 2 LLM calls (the
  search_transcript tool triggered a round trip: search → result → record pass).
  Removing search_transcript and trimming validation context cut ~80s.

  To get true parallelism: run two Ollama instances on separate GPUs and point
  extraction / commitments at different base URLs via OLLAMA_BASE_URL_2.

Run
---
  python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py
  python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py \\
      --meeting-id 01KQ0C1280EDA4E70AAD7C35
  python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --force
  python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --dry-run
  python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --debug
"""

import argparse
import asyncio
import contextvars
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correlation ID — one UUID per pipeline run, propagated via contextvars.
#
# contextvars copy cleanly across asyncio.gather tasks, so parallel stages
# (extraction + commitments) each carry the same run_id but their own stage.
# ---------------------------------------------------------------------------

_run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="--------")
_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="init")


class _CorrelationFormatter(logging.Formatter):
    """Formatter that injects run_id and stage from contextvars before rendering.

    Using the formatter (not a logger-level Filter) ensures the fields are
    available for ALL records — including those from third-party libraries that
    propagate directly to root-logger handlers, bypassing logger.handle() checks.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.run_id = _run_id_var.get()  # type: ignore[attr-defined]
        record.stage = _stage_var.get()    # type: ignore[attr-defined]
        return super().format(record)


# ---------------------------------------------------------------------------
# Note: pydantic_ai ≥ 1.30 removed the Hooks/capabilities API.
# Per-LLM-call and per-tool timing is now done at the stage level inside
# _run_tool_stage (time.perf_counter around agent.run).
#
# For production observability wire in OpenTelemetry via:
#   Agent(..., instrument=True)   # emits spans for each model request + tool call
# and configure an OTLP exporter (Langfuse, Grafana, etc.) via env vars.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(".pipeline_checkpoints")
HISTORY_FILE = Path.home() / ".meeting_pipeline" / "history.json"

MAX_TRANSCRIPT_TURNS = 500
MAX_SPEAKERS = 20
MAX_TRANSCRIPT_CHARS = 150_000
# Per-agent context cap: ~12k tokens at 4 chars/token; prevents context-window overruns
# on local models with smaller effective context windows.
MAX_AGENT_CONTEXT_CHARS = 50_000

# ---------------------------------------------------------------------------
# Strict guardrails via Literal types
#
# Pydantic AI converts Literal into a JSON Schema "enum" constraint and
# sends it to the LLM.  The model is physically restricted to valid values.
# If it hallucinates an unsupported string, Pydantic validation fails and
# Pydantic AI automatically sends the error back for self-correction.
# ---------------------------------------------------------------------------

SentimentPolarity = Literal["positive", "negative", "neutral", "mixed"]
ValidationVerdict = Literal["valid", "invalid"]

# ---------------------------------------------------------------------------
# Input schemas (type-safe from JSON load onwards)
# ---------------------------------------------------------------------------


class TranscriptEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sentence: str
    speaker_name: str
    time: float
    end_time: float = Field(default=0.0, alias="endTime")
    sentiment_type: str | None = Field(default=None, alias="sentimentType")
    speaker_id: int | None = None
    average_confidence: float | None = Field(default=None, alias="averageConfidence")
    index: int = 0


class MeetingInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str = Field(alias="meetingId")
    title: str
    organizer_email: str | None = Field(default=None, alias="organizerEmail")
    start_time: str | None = Field(default=None, alias="startTime")
    end_time: str | None = Field(default=None, alias="endTime")
    duration: float | None = None


class PipelineInput(BaseModel):
    meeting_info: MeetingInfo
    transcript: list[TranscriptEntry]

    @field_validator("transcript")
    @classmethod
    def non_empty_transcript(cls, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        if not entries:
            raise ValueError("transcript must not be empty")
        return entries


# ---------------------------------------------------------------------------
# Output data models
# ---------------------------------------------------------------------------


class CleanTranscript(BaseModel):
    meeting_title: str
    participants: list[str]
    turns: list[str]  # formatted as "MM:SS Speaker: sentence"

    @field_validator("participants")
    @classmethod
    def non_empty_participants(cls, names: list[str]) -> list[str]:
        cleaned = [n.strip() for n in names if n.strip()]
        if not cleaned:
            raise ValueError("participants must not be empty")
        return cleaned


class SentimentShift(BaseModel):
    speaker: str
    shift: str
    polarity: SentimentPolarity | None = Field(
        default=None,
        description="Sentiment direction: positive, negative, neutral, or mixed",
    )


class Insight(BaseModel):
    sentiment_shifts: list[SentimentShift]
    pain_points: list[str]
    competitor_mentions: list[str]


class ActionItem(BaseModel):
    owner: str
    action: str
    deadline: str


class CommitmentsOutput(BaseModel):
    action_items: list[ActionItem]


class ValidatedActionItem(BaseModel):
    owner: str
    action: str
    deadline: str
    verdict: ValidationVerdict = Field(
        description="'valid' if clearly agreed upon, 'invalid' if rejected or hypothetical",
    )
    reason: str


class ValidationResult(BaseModel):
    items: list[ValidatedActionItem]


# ---------------------------------------------------------------------------
# Tool-call agent state (mutable, passed via deps_type)
#
# Each agent accumulates results into its state through individual tool calls
# rather than producing a single complex JSON output.  This is more reliable
# with local models and keeps each tool signature small and validated.
#
# _tool_calls tracks the budget to prevent infinite loops.
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS_PER_STAGE: int = int(os.getenv("MAX_TOOL_CALLS", "30"))


@dataclass
class ExtractionState:
    sentiment_shifts: list[SentimentShift] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    competitor_mentions: list[str] = field(default_factory=list)
    _calls: int = field(default=0, repr=False)

    def _check_budget(self, tool: str) -> None:
        self._calls += 1
        if self._calls > MAX_TOOL_CALLS_PER_STAGE:
            raise RuntimeError(
                f"[{tool}] tool-call budget exceeded ({MAX_TOOL_CALLS_PER_STAGE})"
            )


@dataclass
class CommitmentsState:
    action_items: list[ActionItem] = field(default_factory=list)
    _calls: int = field(default=0, repr=False)

    def _check_budget(self, tool: str) -> None:
        self._calls += 1
        if self._calls > MAX_TOOL_CALLS_PER_STAGE:
            raise RuntimeError(
                f"[{tool}] tool-call budget exceeded ({MAX_TOOL_CALLS_PER_STAGE})"
            )


@dataclass
class ValidationState:
    action_items: list[ActionItem]
    validated: list[ValidatedActionItem] = field(default_factory=list)
    _validated_indices: set[int] = field(default_factory=set, repr=False)
    _calls: int = field(default=0, repr=False)

    def _check_budget(self, tool: str) -> None:
        self._calls += 1
        if self._calls > MAX_TOOL_CALLS_PER_STAGE:
            raise RuntimeError(
                f"[{tool}] tool-call budget exceeded ({MAX_TOOL_CALLS_PER_STAGE})"
            )


class PipelineOutput(BaseModel):
    meeting_title: str
    participants: list[str]
    insights: Insight
    action_items: list[ValidatedActionItem]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    run_id: str
    stage: str
    meeting_id: str
    started_at: str
    completed_at: str
    duration_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: Literal["success", "error", "skipped"]
    error: str | None = None


def _write_audit(meeting_id: str, entry: AuditEntry) -> None:
    path = CHECKPOINT_DIR / meeting_id / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.debug("Audit: stage=%s status=%s duration=%.2fs", entry.stage, entry.status, entry.duration_s)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def _ckpt_path(meeting_id: str, stage: str) -> Path:
    return CHECKPOINT_DIR / meeting_id / f"{stage}.json"


def save_checkpoint(meeting_id: str, stage: str, data: BaseModel) -> None:
    path = _ckpt_path(meeting_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data.model_dump_json(indent=2))
    logger.debug("Checkpoint saved: %s", path)


def load_checkpoint(meeting_id: str, stage: str, model_cls: type[T]) -> T | None:
    path = _ckpt_path(meeting_id, stage)
    if path.exists():
        logger.info("[%s] Resuming from checkpoint", stage)
        return model_cls.model_validate_json(path.read_text())
    return None


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------


def validate_input(pipeline_input: PipelineInput) -> None:
    """Safety checks on a typed PipelineInput — raises ValueError on any violation."""
    entries = pipeline_input.transcript
    mid = pipeline_input.meeting_info.meeting_id

    if len(entries) > MAX_TRANSCRIPT_TURNS:
        raise ValueError(f"[{mid}] Too many turns: {len(entries)} (limit {MAX_TRANSCRIPT_TURNS})")

    speakers = {e.speaker_name.strip() for e in entries if e.speaker_name.strip()}
    if not speakers:
        raise ValueError(f"[{mid}] No speaker names found in transcript")
    if len(speakers) > MAX_SPEAKERS:
        raise ValueError(f"[{mid}] Too many speakers: {len(speakers)} (limit {MAX_SPEAKERS})")

    total_chars = sum(len(e.sentence) for e in entries)
    if total_chars > MAX_TRANSCRIPT_CHARS:
        raise ValueError(
            f"[{mid}] Content too large: {total_chars} chars (limit {MAX_TRANSCRIPT_CHARS})"
        )
    logger.debug(
        "Safety check passed: %d turns, %d speakers, %d chars",
        len(entries), len(speakers), total_chars,
    )


def cap_context(text: str, max_chars: int = MAX_AGENT_CONTEXT_CHARS) -> str:
    """Truncate text to max_chars before passing to an agent.

    Prevents context-window overruns on local models. A sentinel marker is
    appended so the LLM knows the transcript was cut.
    """
    if len(text) <= max_chars:
        return text
    logger.warning("Context capped: %d → %d chars (%.0f%% dropped)",
                   len(text), max_chars, 100 * (1 - max_chars / len(text)))
    return text[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED — content beyond this point omitted]"


def detect_hallucinations(output: PipelineOutput) -> list[str]:
    """Post-hoc checks for common hallucination patterns.

    Returns a list of warning strings; empty means no issues detected.
    Does NOT raise — callers decide whether warnings are fatal.
    """
    warnings: list[str] = []
    participant_lower = {p.lower() for p in output.participants}

    for item in output.action_items:
        if item.verdict != "valid":
            continue
        # Owner must be a known participant (allow "team" / "engineering" as role names)
        owner_lower = item.owner.lower()
        is_known = any(owner_lower in p or p in owner_lower for p in participant_lower)
        is_role = any(w in owner_lower for w in ("team", "engineering", "support", "product", "ops"))
        if not is_known and not is_role:
            warnings.append(
                f"Action item owner '{item.owner}' does not match any known participant"
            )
        # Action text should be substantive
        if len(item.action.split()) < 4:
            warnings.append(
                f"Suspiciously short action (likely hallucinated): '{item.action}'"
            )
        # Deadline should not be empty string
        if not item.deadline.strip():
            warnings.append(f"Empty deadline for action owned by '{item.owner}'")

    # Competitor mentions should be non-empty strings
    for c in output.insights.competitor_mentions:
        if not c.strip():
            warnings.append("Empty competitor mention detected — possible hallucination")

    if warnings:
        logger.warning("Hallucination warnings (%d):\n  %s", len(warnings), "\n  ".join(warnings))

    return warnings


# ---------------------------------------------------------------------------
# Memory: per-user meeting history
# ---------------------------------------------------------------------------


def record_history(meeting_id: str, title: str, valid_count: int) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history: dict = {}
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            logger.warning("History file corrupted; resetting")
    history[meeting_id] = {
        "title": title,
        "processed_at": datetime.now(UTC).isoformat(),
        "valid_action_items": valid_count,
    }
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    logger.info("History updated: %s (%s)", meeting_id, HISTORY_FILE)


def get_history(meeting_id: str) -> dict | None:
    if not HISTORY_FILE.exists():
        return None
    try:
        return json.loads(HISTORY_FILE.read_text()).get(meeting_id)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Stage 1: Deterministic pre-processor (no LLM)
#
# Speaker names and timestamps are already resolved in TranscriptEntry.
# Formatting "MM:SS Speaker: sentence" is a pure transformation — using an LLM
# here added ~120s per run and introduced a language-switching failure mode.
# ---------------------------------------------------------------------------

_ENGLISH_JSON_GUARD = (
    "\nIMPORTANT: Respond ONLY in English. "
    "Your entire output MUST be valid JSON matching the schema. "
    "Do NOT include any text, preamble, or explanation before or after the JSON."
)


def preprocess_transcript(pipeline_input: PipelineInput) -> CleanTranscript:
    """Format transcript entries into clean labelled turns — deterministic, no LLM."""

    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    turns = [
        f"{_fmt(e.time)} {e.speaker_name}: {e.sentence}"
        for e in pipeline_input.transcript
    ]
    participants = sorted(
        {e.speaker_name.strip() for e in pipeline_input.transcript if e.speaker_name.strip()}
    )
    return CleanTranscript(
        meeting_title=pipeline_input.meeting_info.title,
        participants=participants,
        turns=turns,
    )


# ---------------------------------------------------------------------------
# Agents (stages 2–4)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agent 2 — Extraction  (tool-call pattern)
# ---------------------------------------------------------------------------

extraction_agent: Agent[ExtractionState, str] = Agent(
    "ollama:qwen2.5:14b",
    deps_type=ExtractionState,
    instructions=(
        "You are a meeting analyst. Read the transcript in the user message and extract all insights.\n\n"
        "Workflow:\n"
        "  1. Call record_sentiment_shift for every speaker sentiment change you find\n"
        "  2. Call record_pain_point for every customer or team problem\n"
        "  3. Call record_competitor for every competitor mention\n"
        "  4. When done, reply with a one-line English summary.\n\n"
        "polarity argument must be one of: positive, negative, neutral, mixed\n"
        "Call each record tool once per finding. Do not call the same tool with identical args twice."
        + _ENGLISH_JSON_GUARD
    ),
)


@extraction_agent.tool
def record_sentiment_shift(
    ctx: RunContext[ExtractionState],
    speaker: str,
    shift: str,
    polarity: SentimentPolarity,
) -> str:
    """Record a speaker sentiment shift found in the transcript. Skip if already recorded."""
    ctx.deps._check_budget("record_sentiment_shift")
    key = (speaker.lower(), shift[:80].lower())
    if any((s.speaker.lower(), s.shift[:80].lower()) == key for s in ctx.deps.sentiment_shifts):
        return f"Already recorded: [{polarity}] {speaker}"
    ctx.deps.sentiment_shifts.append(SentimentShift(speaker=speaker, shift=shift, polarity=polarity))
    return f"Recorded [{polarity}] {speaker}: {shift[:60]}"


@extraction_agent.tool
def record_pain_point(ctx: RunContext[ExtractionState], description: str) -> str:
    """Record a customer or team pain point. Skip if already recorded."""
    ctx.deps._check_budget("record_pain_point")
    if any(p[:80].lower() == description[:80].lower() for p in ctx.deps.pain_points):
        return f"Already recorded: {description[:40]}"
    ctx.deps.pain_points.append(description)
    return f"Recorded pain point: {description[:60]}"


@extraction_agent.tool
def record_competitor(ctx: RunContext[ExtractionState], name: str) -> str:
    """Record a competitor product or company. Skip if already recorded."""
    ctx.deps._check_budget("record_competitor")
    if any(c.lower() == name.lower() for c in ctx.deps.competitor_mentions):
        return f"Already recorded: {name}"
    ctx.deps.competitor_mentions.append(name)
    return f"Recorded competitor: {name}"


# ---------------------------------------------------------------------------
# Agent 3 — Commitments  (tool-call pattern)
# ---------------------------------------------------------------------------

commitments_agent: Agent[CommitmentsState, str] = Agent(
    "ollama:qwen2.5:14b",
    deps_type=CommitmentsState,
    instructions=(
        "Extract every explicit and implicit action item from the transcript using record_action_item.\n"
        "Look for conditional verbs (will, should, need to) and timeline markers (by Friday, tomorrow).\n"
        "Call record_action_item once per action item. "
        "Use 'Unspecified' for deadline when no date is stated.\n"
        "When done, reply 'Done' in English."
        + _ENGLISH_JSON_GUARD
    ),
)


@commitments_agent.tool
def record_action_item(
    ctx: RunContext[CommitmentsState],
    owner: str,
    action: str,
    deadline: str,
) -> str:
    """Record one action item. Skip if identical owner + action is already recorded."""
    ctx.deps._check_budget("record_action_item")
    key = (owner.lower(), action[:80].lower())
    if any((a.owner.lower(), a.action[:80].lower()) == key for a in ctx.deps.action_items):
        return f"Already recorded: [{owner}] {action[:40]}"
    ctx.deps.action_items.append(ActionItem(owner=owner, action=action, deadline=deadline))
    return f"Recorded [{deadline}] {owner}: {action[:60]}"


# ---------------------------------------------------------------------------
# Agent 4 — Validation  (tool-call pattern)
# ---------------------------------------------------------------------------

validation_agent: Agent[ValidationState, str] = Agent(
    "ollama:qwen2.5:14b",
    deps_type=ValidationState,
    instructions=(
        "You are a validation critic. For each numbered action item, call validate_action_item.\n"
        "verdict must be 'valid' if clearly agreed upon, 'invalid' if rejected or hypothetical.\n"
        "Call validate_action_item exactly once per item. When done, reply 'Done' in English."
        + _ENGLISH_JSON_GUARD
    ),
)


@validation_agent.instructions
def _inject_items_for_validation(ctx: RunContext[ValidationState]) -> str:
    items = "\n".join(
        f"[{i+1}] Owner: {item.owner} | Action: {item.action} | Deadline: {item.deadline}"
        for i, item in enumerate(ctx.deps.action_items)
    )
    return f"ACTION ITEMS TO VALIDATE:\n{items}"


@validation_agent.tool
def validate_action_item(
    ctx: RunContext[ValidationState],
    item_index: int,
    verdict: ValidationVerdict,
    reason: str,
) -> str:
    """Validate action item by its 1-based index. Each index can only be validated once."""
    ctx.deps._check_budget("validate_action_item")
    idx = item_index - 1
    if not (0 <= idx < len(ctx.deps.action_items)):
        return f"Error: index {item_index} out of range (valid: 1–{len(ctx.deps.action_items)})"
    if idx in ctx.deps._validated_indices:
        return f"Already validated item {item_index}"
    ctx.deps._validated_indices.add(idx)
    item = ctx.deps.action_items[idx]
    ctx.deps.validated.append(ValidatedActionItem(
        owner=item.owner, action=item.action, deadline=item.deadline,
        verdict=verdict, reason=reason,
    ))
    return f"Validated [{verdict}] {item.owner}: {item.action[:50]}"


# ---------------------------------------------------------------------------
# Stage runner: checkpointing + audit
# ---------------------------------------------------------------------------


STAGE_TIMEOUT_S: int = int(os.getenv("STAGE_TIMEOUT_S", "900"))


async def _run_tool_stage(
    stage: str,
    meeting_id: str,
    agent: Agent,
    prompt: str,
    deps: Any,
    result_builder: Callable[[], T],
    checkpoint_cls: type[T],
    force: bool,
) -> T:
    """Run a tool-call-based agent stage with checkpointing and audit logging.

    The agent accumulates findings into `deps` via tool calls.  `result_builder`
    constructs the checkpoint-able Pydantic model from the populated state after
    the agent run completes.
    """
    _stage_var.set(stage)
    run_id = _run_id_var.get()
    now_iso = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()

    if not force:
        cached = load_checkpoint(meeting_id, stage, checkpoint_cls)
        if cached is not None:
            logger.info("stage=SKIP   checkpoint restored")
            _write_audit(meeting_id, AuditEntry(
                run_id=run_id, stage=stage, meeting_id=meeting_id,
                started_at=now_iso, completed_at=now_iso,
                duration_s=0.0, status="skipped",
            ))
            return cached

    logger.info("stage=START")
    try:
        result = await asyncio.wait_for(
            agent.run(prompt, deps=deps, retries=3),
            timeout=STAGE_TIMEOUT_S,
        )
        output = result_builder()
        dur = round(time.perf_counter() - t0, 3)
        usage = result.usage
        save_checkpoint(meeting_id, stage, output)
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            status="success",
        ))
        logger.info(
            "stage=DONE   duration=%.2fs in=%d out=%d tool_calls=%d",
            dur, usage.input_tokens, usage.output_tokens, usage.tool_calls,
        )
        return output

    except asyncio.TimeoutError:
        dur = round(time.perf_counter() - t0, 3)
        msg = f"Stage '{stage}' timed out after {STAGE_TIMEOUT_S}s"
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso, completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur, status="error", error=msg,
        ))
        logger.error("stage=TIMEOUT duration=%.2fs", dur)
        raise

    except Exception as exc:
        dur = round(time.perf_counter() - t0, 3)
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso, completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur, status="error", error=str(exc),
        ))
        logger.error("stage=ERROR  duration=%.2fs error=%s", dur, exc)
        raise

async def _run_stage(
    stage: str,
    meeting_id: str,
    coro,
    checkpoint_cls: type[T],
    force: bool,
) -> T:
    """Await a pipeline stage with checkpointing, audit logging, and a hard timeout.

    Sets the `stage` contextvar so all log lines (including hook-level LLM logs)
    carry the stage label automatically.
    """
    _stage_var.set(stage)
    run_id = _run_id_var.get()
    now_iso = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()

    if not force:
        cached = load_checkpoint(meeting_id, stage, checkpoint_cls)
        if cached is not None:
            coro.close()  # prevent "coroutine never awaited" ResourceWarning
            logger.info("stage=SKIP   checkpoint restored")
            _write_audit(meeting_id, AuditEntry(
                run_id=run_id, stage=stage, meeting_id=meeting_id,
                started_at=now_iso, completed_at=now_iso,
                duration_s=0.0, status="skipped",
            ))
            return cached

    logger.info("stage=START")
    try:
        result = await asyncio.wait_for(coro, timeout=STAGE_TIMEOUT_S)
        dur = round(time.perf_counter() - t0, 3)
        usage = result.usage
        save_checkpoint(meeting_id, stage, result.output)
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            status="success",
        ))
        logger.info(
            "stage=DONE   duration=%.2fs in_tokens=%d out_tokens=%d",
            dur, usage.input_tokens, usage.output_tokens,
        )
        return result.output

    except asyncio.TimeoutError:
        dur = round(time.perf_counter() - t0, 3)
        msg = f"Stage '{stage}' timed out after {STAGE_TIMEOUT_S}s"
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur, status="error", error=msg,
        ))
        logger.error("stage=TIMEOUT duration=%.2fs limit=%ds", dur, STAGE_TIMEOUT_S)
        raise

    except Exception as exc:
        dur = round(time.perf_counter() - t0, 3)
        _write_audit(meeting_id, AuditEntry(
            run_id=run_id, stage=stage, meeting_id=meeting_id,
            started_at=now_iso,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=dur, status="error", error=str(exc),
        ))
        logger.error("stage=ERROR  duration=%.2fs error=%s", dur, exc)
        raise


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    meeting_id: str,
    dataset_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> PipelineOutput:
    meeting_dir = dataset_dir / meeting_id

    # --- Correlation ID for this run ---
    run_id = uuid.uuid4().hex[:8]
    _run_id_var.set(run_id)
    _stage_var.set("pipeline")

    logger.info("pipeline=START run_id=%s meeting_id=%s", run_id, meeting_id)

    # --- Load + validate input with typed Pydantic schemas ---
    pipeline_input = PipelineInput(
        meeting_info=MeetingInfo.model_validate(
            json.loads((meeting_dir / "meeting-info.json").read_text())
        ),
        transcript=[
            TranscriptEntry.model_validate(e)
            for e in json.loads((meeting_dir / "transcript.json").read_text())["data"]
        ],
    )
    validate_input(pipeline_input)

    entries = pipeline_input.transcript
    info = pipeline_input.meeting_info

    raw_text = "\n".join(
        f"[{e.time:.1f}s] {e.speaker_name}: {e.sentence}"
        for e in entries
    )

    prior = get_history(meeting_id)
    print(f"\nMeeting : {info.title}")
    print(f"Turns   : {len(entries)} | Chars: {len(raw_text)}")
    if prior and not force:
        print(f"Note    : previously processed {prior['processed_at']}")
    if dry_run:
        print("--dry-run: safety checks passed.\n")
        raise SystemExit(0)
    print()

    # Stage 1 — deterministic Python formatting (no LLM)
    print("[1/4] Pre-processing (deterministic)...")
    _stage_var.set("preprocessing")
    clean = preprocess_transcript(pipeline_input)
    logger.info(
        "stage=DONE   turns=%d participants=%s",
        len(clean.turns), clean.participants,
    )
    transcript_text = cap_context("\n".join(clean.turns))

    # Stages 2 + 3 submitted concurrently (tool-call pattern)
    # Note: local Ollama serialises GPU requests, so on a single GPU there is no
    # wall-clock speedup — both tasks queue behind each other.
    print("[2+3/4] Extracting insights and commitments (concurrent submission)...")
    extraction_state = ExtractionState()
    commitments_state = CommitmentsState()

    insight, commitments = await asyncio.gather(
        _run_tool_stage(
            "extraction", meeting_id,
            extraction_agent,
            transcript_text,
            extraction_state,
            lambda: Insight(
                sentiment_shifts=extraction_state.sentiment_shifts,
                pain_points=extraction_state.pain_points,
                competitor_mentions=extraction_state.competitor_mentions,
            ),
            Insight, force,
        ),
        _run_tool_stage(
            "commitments", meeting_id,
            commitments_agent,
            transcript_text,
            commitments_state,
            lambda: CommitmentsOutput(action_items=commitments_state.action_items),
            CommitmentsOutput, force,
        ),
    )

    # Stage 4: Validation (tool-call pattern)
    print("[4/4] Validating action items (tool calls)...")
    validation_state = ValidationState(action_items=commitments.action_items)
    items_prompt = "\n".join(
        f"[{i+1}] Owner: {item.owner} | Action: {item.action} | Deadline: {item.deadline}"
        for i, item in enumerate(commitments.action_items)
    )
    validated: ValidationResult = await _run_tool_stage(
        "validation", meeting_id,
        validation_agent,
        f"Validate each action item by calling validate_action_item:\n{items_prompt}",
        validation_state,
        lambda: ValidationResult(items=validation_state.validated),
        ValidationResult, force,
    )

    print("[4/4] Done.\n")

    output = PipelineOutput(
        meeting_title=clean.meeting_title,
        participants=clean.participants,
        insights=insight,
        action_items=validated.items,
    )

    # Post-hoc hallucination checks (non-fatal — logged as warnings)
    hw = detect_hallucinations(output)

    valid_count = sum(1 for a in validated.items if a.verdict == "valid")
    record_history(meeting_id, info.title, valid_count)

    _stage_var.set("pipeline")
    logger.info(
        "pipeline=DONE  run_id=%s valid_actions=%d invalid_actions=%d hallucination_warnings=%d",
        run_id, valid_count, len(validated.items) - valid_count, len(hw),
    )
    return output


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(output: PipelineOutput) -> None:
    print("=" * 60)
    print(f"MEETING REPORT: {output.meeting_title}")
    print("=" * 60)
    print(f"Participants: {', '.join(output.participants)}\n")

    print("--- INSIGHTS ---")

    if output.insights.sentiment_shifts:
        print("Sentiment Shifts:")
        for s in output.insights.sentiment_shifts:
            tag = f"[{s.polarity}] " if s.polarity else ""
            print(f"  • {tag}{s.speaker}: {s.shift}")

    if output.insights.pain_points:
        print("\nPain Points:")
        for p in output.insights.pain_points:
            print(f"  • {p}")

    if output.insights.competitor_mentions:
        print("\nCompetitor Mentions:")
        for c in output.insights.competitor_mentions:
            print(f"  • {c}")

    print("\n--- ACTION ITEMS ---")
    valid_items = [a for a in output.action_items if a.verdict == "valid"]
    rejected_items = [a for a in output.action_items if a.verdict == "invalid"]

    for item in valid_items:
        print(f"  ✓ [{item.deadline}] {item.owner}: {item.action}")

    if rejected_items:
        print("\nRejected (hallucinated or withdrawn):")
        for item in rejected_items:
            print(f"  ✗ {item.owner}: {item.action}")
            print(f"    → {item.reason}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting transcript multi-agent pipeline")
    parser.add_argument(
        "--meeting-id",
        default="01KQ03B0303900521BB089CA",
        help="Meeting ID (subdirectory of dataset/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoints and rerun all stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input only; do not call any LLM",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    fmt = _CorrelationFormatter(
        fmt="%(asctime)s %(levelname)-8s [%(run_id)s][%(stage)-14s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    dataset_dir = Path(__file__).parent / "dataset"
    output = await run_pipeline(
        args.meeting_id,
        dataset_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
    print_report(output)


if __name__ == "__main__":
    asyncio.run(main())
