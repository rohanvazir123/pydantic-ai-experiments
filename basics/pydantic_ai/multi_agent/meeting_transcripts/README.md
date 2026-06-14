# Meeting Transcript Multi-Agent Pipeline

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Evolution — What Changed and Why](#2-pipeline-evolution--what-changed-and-why)
3. [Agentic AI System Design Patterns](#3-agentic-ai-system-design-patterns)
   - 3.1 [Sequential + Parallel Pipeline](#31-sequential--parallel-pipeline)
   - 3.2 [Deterministic Pre-Processing (no LLM)](#32-deterministic-pre-processing-no-llm)
   - 3.3 [Tool-Call-Based Extraction](#33-tool-call-based-extraction)
   - 3.4 [Literal Guardrails on Tool Parameters](#34-literal-guardrails-on-tool-parameters)
   - 3.5 [Mutable State via `deps_type`](#35-mutable-state-via-deps_type)
   - 3.6 [Idempotent Checkpointing with State Restore](#36-idempotent-checkpointing-with-state-restore)
   - 3.7 [Critic / Validation Agent](#37-critic--validation-agent)
   - 3.8 [Tool-Call Budget Guard (Anti-Infinite-Loop)](#38-tool-call-budget-guard-anti-infinite-loop)
4. [Architecture](#4-architecture)
   - 4.1 [System Overview](#41-system-overview)
   - 4.2 [Data Flow Diagram](#42-data-flow-diagram)
   - 4.3 [Call Graph](#43-call-graph)
   - 4.4 [Separation of Concerns](#44-separation-of-concerns)
5. [Agents and Tools](#5-agents-and-tools)
   - 5.1 [Stage 1 — Pre-Processing (Python, no LLM)](#51-stage-1--pre-processing-python-no-llm)
   - 5.2 [Stage 2 — Extraction Agent](#52-stage-2--extraction-agent)
   - 5.3 [Stage 3 — Commitments Agent](#53-stage-3--commitments-agent)
   - 5.4 [Stage 4 — Validation Agent](#54-stage-4--validation-agent)
6. [Observability](#6-observability)
   - 6.1 [Correlation IDs](#61-correlation-ids)
   - 6.2 [Production Hooks](#62-production-hooks)
   - 6.3 [Sample Trace](#63-sample-trace)
7. [Data Models](#7-data-models)
   - 7.1 [Input Schemas](#71-input-schemas)
   - 7.2 [Agent State Schemas (tool-call pattern)](#72-agent-state-schemas-tool-call-pattern)
   - 7.3 [Output Schemas](#73-output-schemas)
   - 7.4 [PostgreSQL Schema](#74-postgresql-schema)
8. [Production Features](#8-production-features)
9. [Performance](#9-performance)
   - 9.1 [Measured Results](#91-measured-results)
   - 9.2 [Parallelism Analysis](#92-parallelism-analysis)
   - 9.3 [Model Comparison](#93-model-comparison)
10. [Test Metrics](#10-test-metrics)
11. [Configuration](#11-configuration)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [Automated Ingestion](#13-automated-ingestion)
14. [Known Failure Modes and Mitigations](#14-known-failure-modes-and-mitigations)

---

## 1. Overview

An asynchronous **3-stage multi-agent pipeline** built with **Pydantic AI** that processes
raw meeting transcripts into structured, validated insights and action items.

```
Raw JSON transcript
        │
  [1] preprocess_transcript()  — deterministic Python, instant (no LLM)
        │  CleanTranscript
        │
    ┌───┴─────────────────────────────┐  asyncio.gather (parallel)
    ▼                                 ▼
  [2] Extraction Agent              [3] Commitments Agent
      tool calls: search,               tool calls: record_action_item
      record_sentiment_shift,           CommitmentsOutput accumulated
      record_pain_point,                in CommitmentsState
      record_competitor
      Insight accumulated
      in ExtractionState
    └───────────────┬─────────────────┘
                    │
              [4] Validation Agent
                  tool calls: validate_action_item
                  ValidationResult accumulated
                  in ValidationState
                    │
              PipelineOutput
              (PostgreSQL via ingestion.py)
```

**Why multi-agent?** A single prompt over a full transcript consistently misses
subtle agreements, hallucinates vague action items, and produces undifferentiated output.
Specialised agents — each with a narrow tool set and a focused role — improve accuracy,
make failures easy to diagnose, and allow independent scaling.

**Why tool calls instead of structured output?** With local models (Ollama), asking the
model to produce a complete JSON object in one shot frequently fails — the model prepends
non-English text or generates malformed JSON, exhausting retries. Tool calls break the
problem into atomic, individually validated decisions: one tool call per finding. This is
more reliable at the cost of ~N+1 LLM iterations instead of 1.

---

## 2. Pipeline Evolution — What Changed and Why

| Version | Preprocessing | Extraction | Commitments | Validation | Key finding |
|---------|--------------|------------|-------------|-----------|------------|
| v1 | LLM agent | `output_type=Insight` | `output_type=CommitmentsOutput` | `output_type=ValidationResult` | Works but fragile — model produces Chinese/Thai prefix, exhausts retries |
| v2 (current) | Python function | Tool calls → `ExtractionState` | Tool calls → `CommitmentsState` | Tool calls → `ValidationState` | Reliable; each tool call is individually validated; preprocessing saves 120s |

### Why remove the preprocessing LLM?

The raw `transcript.json` already has:
- `speaker_name`: fully resolved (not "Speaker 1")
- `time`: float seconds from start
- `sentence`: the spoken text

Formatting `MM:SS Speaker: sentence` is a deterministic string operation.
The LLM added ~120s, introduced language-switching failures (model responding in Thai/Chinese),
and used 2,000–2,500 tokens per run — all for a task Python does in microseconds.

### Why tool calls?

| Dimension | Structured output (`output_type=Model`) | Tool calls |
|-----------|-----------------------------------------|------------|
| Reliability | One complex JSON object → single point of failure | One simple tool call per finding |
| Guardrails | Enum on output field (model must get the whole object right) | Enum on tool parameter (validated individually) |
| LLM iterations | 1 (or N retries on failure) | N+1 (one per finding, one to end) |
| Deduplication | Built into output schema | Enforced in tool handler |
| Debuggability | Opaque — one blob | Auditable — every tool call in the log |
| Local model compat | Fragile on 14B models | Reliable — simpler schema per call |

---

## 3. Agentic AI System Design Patterns

### 3.1 Sequential + Parallel Pipeline

Stages with hard data dependencies run sequentially. Stages that are independent run
concurrently via `asyncio.gather`.

```
preprocess (instant)
    │
    ├── extract (80s) ──┐  asyncio.gather
    └── commit  (61s) ──┤
                        │
                   validate (27s)
```

`asyncio.gather` is the right primitive here because:
- The two tasks are CPU-independent (both block on network I/O to Ollama)
- `contextvars` copy cleanly across tasks — each carries the same `run_id` but its own `stage`
- Failure of one task does NOT cancel the other (unlike `asyncio.wait` with `FIRST_EXCEPTION`)

**Measured saving**: 61s (the commitments stage) runs for free inside the extraction window.

---

### 3.2 Deterministic Pre-Processing (no LLM)

If a transformation is a pure function of the input data, do not call an LLM.

```python
def preprocess_transcript(pipeline_input: PipelineInput) -> CleanTranscript:
    def _fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    turns = [f"{_fmt(e.time)} {e.speaker_name}: {e.sentence}" for e in pipeline_input.transcript]
    participants = sorted({e.speaker_name.strip() for e in pipeline_input.transcript if e.speaker_name.strip()})
    return CleanTranscript(meeting_title=pipeline_input.meeting_info.title,
                           participants=participants, turns=turns)
```

**Savings**: 120s per run, ~2,200 tokens, zero language-switching risk.

**Rule**: Before assigning any task to an LLM, ask — *is this deterministic given the inputs?*
If yes, use Python.

---

### 3.3 Tool-Call-Based Extraction

Instead of `output_type=ComplexModel` (one shot, all-or-nothing), agents accumulate
results through individual tool calls into a mutable `@dataclass` state object.

```python
# Agent produces text output; findings accumulate in deps
extraction_agent: Agent[ExtractionState, str] = Agent("ollama:qwen2.5:14b", ...)

@extraction_agent.tool
def record_sentiment_shift(ctx: RunContext[ExtractionState],
                           speaker: str, shift: str,
                           polarity: SentimentPolarity) -> str:
    ctx.deps.sentiment_shifts.append(SentimentShift(...))
    return f"Recorded [{polarity}] {speaker}"

# After the run, build the checkpoint model from accumulated state:
insight = Insight(
    sentiment_shifts=extraction_state.sentiment_shifts,
    pain_points=extraction_state.pain_points,
    competitor_mentions=extraction_state.competitor_mentions,
)
```

Each tool call produces a log entry (name + latency), making the extraction reasoning
fully auditable without reading the LLM's chain-of-thought.

---

### 3.4 Literal Guardrails on Tool Parameters

`Literal` types on tool parameters are validated **per call** by Pydantic AI.
If the model passes `polarity="mixed-negative"`, Pydantic raises `ValidationError`
immediately for that call — the model retries just that one tool call, not the entire output.

```python
SentimentPolarity = Literal["positive", "negative", "neutral", "mixed"]
ValidationVerdict = Literal["valid", "invalid"]

@extraction_agent.tool
def record_sentiment_shift(ctx, speaker: str, shift: str,
                           polarity: SentimentPolarity) -> str: ...

@validation_agent.tool
def validate_action_item(ctx, item_index: int,
                         verdict: ValidationVerdict, reason: str) -> str: ...
```

This is more granular than `output_type` guardrails: the invalid parameter is rejected
atomically while all previously recorded findings in the state are preserved.

---

### 3.5 Mutable State via `deps_type`

A `@dataclass` is passed as `deps` and mutated in-place by tool calls during the agent run.

```python
@dataclass
class ExtractionState:
    transcript_lines: list[str]
    sentiment_shifts: list[SentimentShift] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    competitor_mentions: list[str] = field(default_factory=list)
    _calls: int = field(default=0, repr=False)  # budget counter
```

`deps` is not serialised or sent to the LLM — it is purely runtime context for tool
handlers. The agent sees only the tool schemas and the `RunContext` wrapper.

---

### 3.6 Idempotent Checkpointing with State Restore

Every stage writes its Pydantic output model to `.pipeline_checkpoints/<id>/<stage>.json`
on success. On restart, the checkpoint is loaded and the stage is skipped.

```
run 1 (extraction fails)          run 2 (--force not set)
  preprocess  ✓ → instant         preprocess  ✓ → instant
  extraction  ✓ → extraction.json extraction  ⏭ → loads extraction.json
  commitments ✗ → no file         commitments ✓ → commitments.json
                                  validation  ✓ → validation.json
```

Key detail: when a checkpoint is restored, the **coroutine is closed** (`coro.close()`)
to prevent Python's `ResourceWarning: coroutine never awaited`.

---

### 3.7 Critic / Validation Agent

A dedicated critic validates each action item individually via `validate_action_item` tool
calls. This prevents the classic failure where items *discussed* but *rejected* end up in
the output.

```
Transcript: "We could deploy tonight, but actually let's wait for QA sign-off."
Commitments agent records: owner=Raj, action="Deploy to production tonight"
Validation agent calls: validate_action_item(1, verdict="invalid",
    reason="Explicitly deferred pending QA sign-off")
```

The validation agent validates each item sequentially in a loop, one tool call per item.
A `_validated_indices: set[int]` in `ValidationState` prevents double-validation.

---

### 3.8 Tool-Call Budget Guard (Anti-Infinite-Loop)

Each state object tracks tool call count and raises `RuntimeError` if the budget is exceeded.

```python
def _check_budget(self, tool: str) -> None:
    self._calls += 1
    if self._calls > MAX_TOOL_CALLS_PER_STAGE:
        raise RuntimeError(f"[{tool}] tool-call budget exceeded ({MAX_TOOL_CALLS_PER_STAGE})")
```

This is called at the start of every tool handler. The `tool_execute_error` hook catches
it, logs it, and re-raises — causing the stage to fail with an auditable error rather
than looping indefinitely. `MAX_TOOL_CALLS_PER_STAGE` defaults to 30 (env-configurable).

Combined with `STAGE_TIMEOUT_S` (hard wall-clock timeout via `asyncio.wait_for`), there
are two independent guards against infinite loops.

---

## 4. Architecture

### 4.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Input Layer (read-only)                                      │
│  dataset/<meeting_id>/transcript.json  ──► PipelineInput     │
│  dataset/<meeting_id>/meeting-info.json ──► MeetingInfo      │
└──────────────────────────────┬───────────────────────────────┘
                               │ validate_input()
┌──────────────────────────────▼───────────────────────────────┐
│  Stage 1 — preprocess_transcript() [Python, 0s]              │
│  → CleanTranscript (turns[], participants[])                  │
└──────────────────────────────┬───────────────────────────────┘
                               │ asyncio.gather
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
┌─────────────────────────────┐  ┌────────────────────────────┐
│  Stage 2 — Extraction Agent │  │ Stage 3 — Commitments Agent│
│  tools: search_transcript   │  │ tools: record_action_item  │
│          record_sentiment_  │  │ state: CommitmentsState     │
│          shift/pain_point/  │  │ → CommitmentsOutput        │
│          competitor         │  └─────────────────────┬──────┘
│  state: ExtractionState     │                        │
│  → Insight                  │                        │
└──────────────┬──────────────┘                        │
               └──────────────────┬────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — Validation Agent                                  │
│  tools: validate_action_item  state: ValidationState         │
│  → ValidationResult (verdict: "valid"|"invalid" per item)    │
└─────────────────────────────┬───────────────────────────────┘
                              │ detect_hallucinations()
┌─────────────────────────────▼───────────────────────────────┐
│  Storage Layer — ingestion.py                                 │
│  PostgreSQL: meetings / meeting_insights / action_items       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  Query Layer — query.py                                       │
│  Stakeholder SQL (Head of Eng / Product / Customer Support)   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow Diagram

```
transcript.json ──► [TranscriptEntry, ...] ──► CleanTranscript
                                                      │
                         ┌────────────────────────────┘
                         │  transcript_text  transcript_lines
                         │
           ┌─────────────┴──────────────────────────────┐
           ▼                                             ▼
  ExtractionState(lines)                  CommitmentsState()
  extraction_agent.run(text, deps)        commitments_agent.run(text, deps)
     │ tool: search_transcript               │ tool: record_action_item ×N
     │ tool: record_sentiment_shift ×M       ▼
     │ tool: record_pain_point ×M       CommitmentsOutput
     │ tool: record_competitor ×M            │
     ▼                                       │
  Insight ─────────────────────────────────  │
  (built from ExtractionState)               │
                                             │
           ┌──────────────────────────────── ┘
           ▼
  ValidationState(transcript, action_items)
  validation_agent.run(items_prompt, deps)
     │ tool: validate_action_item × len(items)
     ▼
  ValidationResult
     │
     ▼
  PipelineOutput
  detect_hallucinations() ─► warnings[]
  record_history()
```

### 4.3 Call Graph

```
main()
└── run_pipeline(meeting_id, dataset_dir, force=False, dry_run=False)
    ├── PipelineInput.model_validate(...)          [typed JSON load]
    ├── validate_input(pipeline_input)             [safety checks]
    ├── get_history(meeting_id)                    [memory read]
    │
    ├── preprocess_transcript(pipeline_input)      [Python, 0s, no LLM]
    │   └── CleanTranscript
    │
    ├── asyncio.gather(
    │   ├── _run_tool_stage("extraction", ...)
    │   │   ├── load_checkpoint(...)               [cache read — skip if hit]
    │   │   ├── extraction_agent.run(text, deps=ExtractionState)
    │   │   │   ├── search_transcript(keyword)  ◄──── [tool, 0.001s]
    │   │   │   ├── record_sentiment_shift(...)  ◄─── [tool, 0.001s]
    │   │   │   ├── record_pain_point(...)       ◄─── [tool, 0.001s]
    │   │   │   └── record_competitor(...)       ◄─── [tool, 0.001s]
    │   │   │       qwen2.5:14b × 4 LLM calls, ~80s total
    │   │   ├── Insight built from ExtractionState
    │   │   ├── save_checkpoint("extraction", Insight)
    │   │   └── _write_audit(...)
    │   │
    │   └── _run_tool_stage("commitments", ...)
    │       ├── load_checkpoint(...)
    │       ├── commitments_agent.run(text, deps=CommitmentsState)
    │       │   └── record_action_item(owner, action, deadline) ×N  ◄── [tool]
    │       │       qwen2.5:14b × 2 LLM calls, ~61s total
    │       ├── CommitmentsOutput built from CommitmentsState
    │       ├── save_checkpoint("commitments", CommitmentsOutput)
    │       └── _write_audit(...)
    │   )
    │
    ├── _run_tool_stage("validation", ...)
    │   ├── load_checkpoint(...)
    │   ├── validation_agent.run(items_prompt, deps=ValidationState)
    │   │   └── validate_action_item(index, verdict, reason) ×N  ◄── [tool]
    │   │       qwen2.5:14b × 3 LLM calls, ~27s total
    │   ├── ValidationResult built from ValidationState
    │   ├── save_checkpoint("validation", ValidationResult)
    │   └── _write_audit(...)
    │
    ├── detect_hallucinations(output)              [post-hoc checks]
    └── record_history(...)                        [memory write]
```

### 4.4 Separation of Concerns

| File | Responsibility | Rule |
|------|---------------|------|
| `pipeline.py` | Agent workflow, tool handlers, checkpointing, tracing | Never imports storage |
| `ingestion.py` | `PipelineOutput` → PostgreSQL | Never imports agents |
| `query.py` | Typed SQL read layer for stakeholder questions | Never imports agents |
| `watcher.py` | File-system event-driven + polling ingestion trigger | Composes pipeline + ingestion |
| `tests/` | Edge-case tests (32 passing) | No LLM calls; uses `OLLAMA_BASE_URL` stub |

---

## 5. Agents and Tools

### 5.1 Stage 1 — Pre-Processing (Python, no LLM)

**Why no LLM**: speaker names are already resolved in `transcript.json`. Formatting
`MM:SS Speaker: sentence` is a deterministic string operation — no reasoning required.

```python
def preprocess_transcript(pipeline_input: PipelineInput) -> CleanTranscript:
    turns = [f"{_fmt(e.time)} {e.speaker_name}: {e.sentence}"
             for e in pipeline_input.transcript]
    participants = sorted({e.speaker_name.strip() for e in pipeline_input.transcript
                           if e.speaker_name.strip()})
    return CleanTranscript(meeting_title=pipeline_input.meeting_info.title,
                           participants=participants, turns=turns)
```

**Output**: `CleanTranscript` — structured, validated (Pydantic field validator rejects
empty participant lists).

---

### 5.2 Stage 2 — Extraction Agent

**Model**: `ollama:qwen2.5:14b`  
**State**: `ExtractionState` (transcript lines + accumulated lists)  
**Output**: `Insight` built from state after agent completes

**Tools**:

| Tool | Parameters | Guard |
|------|-----------|-------|
| `search_transcript` | `keyword: str` | Budget check; returns ≤10 matching lines |
| `record_sentiment_shift` | `speaker, shift, polarity: SentimentPolarity` | Dedup by (speaker, shift[:80]); budget check |
| `record_pain_point` | `description: str` | Dedup by description[:80]; budget check |
| `record_competitor` | `name: str` | Case-insensitive dedup; budget check |

**System instructions** (abbreviated):
```
You are a meeting analyst. Extract all insights using the record tools.
1. Use search_transcript to find relevant segments
2. Call record_sentiment_shift / record_pain_point / record_competitor per finding
3. When done, reply with a one-line English summary.
polarity must be one of: positive, negative, neutral, mixed
IMPORTANT: Respond ONLY in English. Do NOT include any text before or after the JSON.
```

---

### 5.3 Stage 3 — Commitments Agent

**Model**: `ollama:qwen2.5:14b`  
**State**: `CommitmentsState` (action items list)  
**Output**: `CommitmentsOutput` built from state after agent completes  
**Runs**: in parallel with Stage 2

**Tools**:

| Tool | Parameters | Guard |
|------|-----------|-------|
| `record_action_item` | `owner, action, deadline: str` | Dedup by (owner, action[:80]); budget check |

**System instructions** (abbreviated):
```
Extract every explicit and implicit action item using record_action_item.
Look for conditional verbs (will, should, need to) and timeline markers.
Use 'Unspecified' for deadline when no date is stated.
When done, reply 'Done' in English.
```

---

### 5.4 Stage 4 — Validation Agent

**Model**: `ollama:qwen2.5:14b`  
**State**: `ValidationState` (transcript + items + validated list + index set)  
**Output**: `ValidationResult` built from state after agent completes

**Tools**:

| Tool | Parameters | Guard |
|------|-----------|-------|
| `validate_action_item` | `item_index: int, verdict: ValidationVerdict, reason: str` | Index range check; `_validated_indices` set prevents double-validation |

**Dynamic instructions** (injected via `@validation_agent.instructions`):
```
TRANSCRIPT:
<full clean transcript, capped at 50k chars>

ACTION ITEMS TO VALIDATE:
[1] Owner: Tyler | Action: ... | Deadline: ...
[2] Owner: Chris | Action: ... | Deadline: ...
```

**User prompt** also lists items:
```
Validate each action item by calling validate_action_item:
[1] Owner: Tyler Washington | Action: ... | Deadline: Unspecified
```

---

## 6. Observability

### 6.1 Correlation IDs

Every pipeline run generates `run_id = uuid4().hex[:8]` stored in a `ContextVar`.

`asyncio.gather` creates tasks by copying the current context — both parallel stages
automatically inherit the same `run_id` with their own `stage` value.

A `_CorrelationFormatter` (not a `logging.Filter`) injects `run_id` and `stage` into
every `LogRecord` before formatting. Using the Formatter (not a Logger-level Filter)
ensures third-party library logs (e.g. `httpx`) are also tagged — Logger-level filters
are bypassed by `callHandlers()` propagation.

```
20:12:43 INFO [f7ac3a9b][pipeline      ] pipeline=START run_id=f7ac3a9b
20:12:43 INFO [f7ac3a9b][preprocessing ] stage=DONE   turns=40 participants=[...]
20:12:43 INFO [f7ac3a9b][extraction    ] stage=START
20:12:43 INFO [f7ac3a9b][commitments   ] stage=START     ← same run_id, parallel stage
20:12:58 INFO [f7ac3a9b][extraction    ] llm_call=DONE   latency=14.87s attempt=1
20:12:58 INFO [f7ac3a9b][extraction    ] tool=search_transcript    event=DONE latency=0.001s
```

### 6.2 Production Hooks

All 8 hooks are wired on a single global `_tracing_hooks` instance attached to every agent
via `capabilities=[_tracing_hooks]`.

| Hook | Trigger | Action | External TODO |
|------|---------|--------|--------------|
| `before_model_request` | Each LLM call starts | Record `t0` in ContextVar | — |
| `after_model_request` | Each LLM call succeeds | Log latency | Prometheus histogram |
| `model_request_error` | LLM API error | Log + re-raise | Prometheus counter, PagerDuty if repeated |
| `output_validate_error` | Schema validation fails → retry | Log attempt + error | Sentry if rate > threshold |
| `before_tool_execute` | Tool is about to run | Record tool `t0` | — |
| `after_tool_execute` | Tool returns successfully | Log tool name + latency | Prometheus histogram |
| `tool_execute_error` | Tool raises exception | Log + re-raise | Dead-letter queue |
| `run_error` | Agent run fails terminally | Log CRITICAL + re-raise | PagerDuty / Slack alert |

### 6.3 Sample Trace

Full trace from `run_id=f7ac3a9b` (Weekly Engineering Standup, tool-call pattern):

```
20:12:43 [f7ac3a9b][pipeline      ] pipeline=START run_id=f7ac3a9b meeting_id=01KQ0CAE7F064EC93F0540CA
20:12:43 [f7ac3a9b][preprocessing ] stage=DONE   turns=40 participants=['Chris Lee', 'Mike Romano', 'Tom Bradley', 'Tyler Washington']
20:12:43 [f7ac3a9b][extraction    ] stage=START
20:12:43 [f7ac3a9b][commitments   ] stage=START
20:12:58 [f7ac3a9b][extraction    ] llm_call=DONE   latency=14.87s attempt=1
20:12:58 [f7ac3a9b][extraction    ] tool=search_transcript    event=DONE  latency=0.001s
20:13:14 [f7ac3a9b][commitments   ] llm_call=DONE   latency=30.79s attempt=1
20:13:14 [f7ac3a9b][commitments   ] tool=record_action_item   event=DONE  latency=0.001s  ← 3 items
20:13:14 [f7ac3a9b][commitments   ] tool=record_action_item   event=DONE  latency=0.001s
20:13:14 [f7ac3a9b][commitments   ] tool=record_action_item   event=DONE  latency=0.001s
20:13:33 [f7ac3a9b][extraction    ] llm_call=DONE   latency=35.37s attempt=2
20:13:33 [f7ac3a9b][extraction    ] tool=record_sentiment_shift event=DONE latency=0.001s
20:13:33 [f7ac3a9b][extraction    ] tool=record_pain_point    event=DONE  latency=0.001s
20:13:44 [f7ac3a9b][commitments   ] llm_call=DONE   latency=30.11s attempt=2
20:13:44 [f7ac3a9b][commitments   ] stage=DONE   duration=61.39s in=4075 out=140 tool_calls=3
20:14:00 [f7ac3a9b][extraction    ] llm_call=DONE   latency=27.10s attempt=3
20:14:00 [f7ac3a9b][extraction    ] tool=record_competitor    event=DONE  latency=0.000s
20:14:04 [f7ac3a9b][extraction    ] stage=DONE   duration=80.58s in=9633 out=329 tool_calls=5
20:14:04 [f7ac3a9b][validation    ] stage=START
20:14:24 [f7ac3a9b][validation    ] llm_call=DONE   latency=20.23s attempt=1
20:14:24 [f7ac3a9b][validation    ] tool=validate_action_item event=DONE  latency=0.002s
20:14:29 [f7ac3a9b][validation    ] llm_call=DONE   latency=5.56s attempt=2
20:14:29 [f7ac3a9b][validation    ] tool=validate_action_item event=DONE  latency=0.001s
20:14:30 [f7ac3a9b][validation    ] stage=DONE   duration=26.68s in=6956 out=288 tool_calls=6
20:14:30 [f7ac3a9b][pipeline      ] pipeline=DONE  run_id=f7ac3a9b valid_actions=2 invalid_actions=1 hallucination_warnings=0
```

---

## 7. Data Models

### 7.1 Input Schemas

Pydantic models enforce type safety from JSON load to agent input. `ConfigDict(populate_by_name=True)`
allows both alias (camelCase JSON) and Python field name access.

```python
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
    # field_validator: transcript must not be empty
```

### 7.2 Agent State Schemas (tool-call pattern)

Mutable `@dataclass` objects passed via `deps_type`. Never serialised to LLM — only
accessible inside tool handlers via `ctx.deps`.

```python
@dataclass
class ExtractionState:
    transcript_lines: list[str]          # for search_transcript tool
    sentiment_shifts: list[SentimentShift] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    competitor_mentions: list[str] = field(default_factory=list)
    _calls: int = field(default=0, repr=False)  # budget counter

@dataclass
class CommitmentsState:
    action_items: list[ActionItem] = field(default_factory=list)
    _calls: int = field(default=0, repr=False)

@dataclass
class ValidationState:
    transcript: str
    action_items: list[ActionItem]
    validated: list[ValidatedActionItem] = field(default_factory=list)
    _validated_indices: set[int] = field(default_factory=set, repr=False)  # dedup
    _calls: int = field(default=0, repr=False)
```

### 7.3 Output Schemas

```
PipelineOutput
├── meeting_title: str
├── participants: list[str]           # Pydantic validator: non-empty, stripped
├── insights: Insight
│   ├── sentiment_shifts: list[SentimentShift]
│   │   ├── speaker: str
│   │   ├── shift: str
│   │   └── polarity: "positive"|"negative"|"neutral"|"mixed"|None
│   ├── pain_points: list[str]
│   └── competitor_mentions: list[str]
└── action_items: list[ValidatedActionItem]
    ├── owner: str
    ├── action: str
    ├── deadline: str
    ├── verdict: "valid"|"invalid"    ← Literal, validated per tool call
    └── reason: str
```

### 7.4 PostgreSQL Schema

See `DATASTORE.md` for full DDL, indexes, and entity diagram. Summary:

| Table | Rows | Key columns |
|-------|------|------------|
| `meetings` | 1 per run | `id (PK)`, `title`, `participants TEXT[]`, `meeting_date` |
| `meeting_insights` | N per meeting | `insight_type CHECK`, `speaker`, `content`, `polarity` |
| `action_items` | N per meeting | `owner`, `action`, `deadline`, `verdict CHECK('valid','invalid')` |

---

## 8. Production Features

| Feature | How | Config / location |
|---------|-----|------------------|
| Deterministic preprocessing | Pure Python `preprocess_transcript()` | No config — always instant |
| Strict Literal guardrails | `Literal` on tool params → JSON Schema enum | `SentimentPolarity`, `ValidationVerdict` |
| Tool-call dedup | Similarity check before appending to state | In each `record_*` / `validate_*` tool |
| Tool-call budget | `_check_budget()` in every tool → raises at `MAX_TOOL_CALLS` | `MAX_TOOL_CALLS=30` env var |
| Stage timeout | `asyncio.wait_for(coro, timeout=STAGE_TIMEOUT_S)` | `STAGE_TIMEOUT_S=900` env var |
| Correlation ID | `uuid4().hex[:8]` per run, propagated via `ContextVar` | Automatic; in every log line |
| 8 production hooks | `Hooks` capability on all agents | `_tracing_hooks` (see §6.2) |
| Per-LLM-call latency | `before/after_model_request` hooks | Every LLM call logged |
| Per-tool-call latency | `before/after_tool_execute` hooks | Every tool call logged |
| Checkpointing | JSON per stage in `.pipeline_checkpoints/<id>/` | `CHECKPOINT_DIR` env var |
| Audit log | JSONL with run_id, stage, tokens, duration | Next to checkpoints: `audit.jsonl` |
| Context cap | `cap_context()` at 50k chars before each agent | `MAX_AGENT_CONTEXT_CHARS=50000` |
| Hallucination detection | `detect_hallucinations()` post-run | Unknown owner, short action, empty fields |
| Safety checks | Size limits on turns, speakers, chars | `validate_input()` |
| Memory | `~/.meeting_pipeline/history.json` | `HISTORY_FILE` env var |
| Typed input | `PipelineInput` / `TranscriptEntry` / `MeetingInfo` | `pipeline.py` |
| English-only guard | Appended to every agent's instructions | `_ENGLISH_JSON_GUARD` constant |
| CLI flags | `--force`, `--dry-run`, `--debug` | `main()` / `argparse` |

---

## 9. Performance

### 9.1 Measured Results

All runs on local hardware with `qwen2.5:14b` via Ollama.

**Run: Weekly Engineering Standup** (`run_id=f7ac3a9b`, tool-call pattern)

| Stage | Mode | Wall-clock | LLM calls | Tool calls | Input tokens | Output tokens |
|-------|------|-----------|-----------|-----------|-------------|--------------|
| preprocessing | Python | **0s** | 0 | 0 | 0 | 0 |
| extraction | parallel | 80.6s | 4 | 5 | 9,633 | 329 |
| commitments | parallel | 61.4s | 2 | 3 | 4,075 | 140 |
| validation | sequential | 26.7s | 3 | 6 | 6,956 | 288 |
| **Total wall-clock** | | **~108s** | **9** | **14** | **20,664** | **757** |

> Extraction and commitments run in parallel; total for that phase = max(80.6, 61.4) = 80.6s.

**Run: Detect Outage Remediation** (`run_id=603a50fd`, from checkpoint)

| Stage | Wall-clock | Notes |
|-------|-----------|-------|
| All stages | 0s | Loaded from `.pipeline_checkpoints/` |

### 9.2 Parallelism Analysis

```
Sequential:  preprocess(0) + extract(80) + commit(61) + validate(27) = 168s
Parallel:    preprocess(0) + max(extract, commit)(80) + validate(27) = 107s
Saving:      61s (36%)
```

The saving grows as more independent extraction dimensions are added. Each new parallel
agent that completes within the extraction window adds 0s to wall-clock.

### 9.3 Model Comparison

The bottleneck is GPU inference, not orchestration. Switching models changes everything:

| Model | Per-call latency | Total pipeline | Cost/run | Reliability |
|-------|-----------------|----------------|---------|------------|
| `qwen2.5:14b` local | 15–120s | ~108s | Free | Moderate (language switching risk) |
| `llama3.1:8b` local | 5–40s | ~40s | Free | Similar |
| `openai:gpt-4o-mini` | 1–3s | **~10–15s** | ~$0.002 | High |
| `anthropic:claude-haiku-4-5` | 1–4s | **~10–15s** | ~$0.002 | High |

To switch model, change one string per agent definition:
```python
extraction_agent = Agent("openai:gpt-4o-mini", ...)   # or "anthropic:claude-haiku-4-5"
```

---

## 10. Test Metrics

### Extraction Quality Targets

| Metric | Description | Target |
|--------|-------------|--------|
| **Action Item Precision** | `valid` items that were genuinely agreed upon | ≥ 0.85 |
| **Action Item Recall** | Gold-standard items captured | ≥ 0.80 |
| **Hallucination Rate** | Items marked `invalid` by validator | < 0.15 |
| **Pain Point Coverage** | Gold-labelled pain points extracted | ≥ 0.75 |
| **Sentiment Accuracy** | Correct polarity label | ≥ 0.80 |

### Pipeline Health Targets

| Metric | Description | Target |
|--------|-------------|--------|
| **Stage Success Rate** | All stages complete without error | ≥ 0.95 |
| **Checkpoint Restore Rate** | Retried runs that resume correctly | 1.0 |
| **Tool-Call Dedup Rate** | `Already recorded` returns / total calls | measurable |
| **Mean Latency (local)** | Wall-clock, qwen2.5:14b | ~108s |
| **Mean Token Cost** | Total tokens per run | ~21k |

### Edge-Case Test Suite (`tests/test_pipeline_failures.py`)

32 tests, all passing. Key scenarios:

| Test class | Scenario |
|-----------|---------|
| `TestValidateInput` | Empty transcript, too many turns, no speakers, too many speakers, content too large |
| `TestLiteralGuardrails` | Hallucinated `verdict="maybe"`, wrong `polarity` value, valid values accepted |
| `TestCapContext` | Short text unchanged, long text truncated with marker, custom limit |
| `TestDetectHallucinations` | Unknown owner, short action, empty competitor, empty deadline |
| `TestSingleSpeaker` | Passes validation; documents weakness (one possible owner) |
| `TestMultilingualTranscript` | Unicode, RTL text — passes schema validation |
| `TestContradictoryItems` | `verdict='invalid'` unit test; mixed valid/invalid list |
| `TestMalformedInput` | Missing required fields, wrong types, extra fields ignored |
| `TestAllItemsInvalid` | Empty valid list handled correctly; hallucination detector no-ops |
| `TestPromptInjection` | Injection stored as-is; `cap_context` catches overlong sentences |

---

## 11. Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | — | Ollama OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` |
| `STAGE_TIMEOUT_S` | `900` | Hard per-stage timeout (seconds). Local 14B model needs ≥ 900s for retries |
| `MAX_TOOL_CALLS` | `30` | Tool-call budget per stage. Exceeding raises `RuntimeError` (anti-loop) |
| `MAX_TRANSCRIPT_TURNS` | `500` | Input safety cap: max transcript turns |
| `MAX_SPEAKERS` | `20` | Input safety cap: max distinct speakers |
| `MAX_TRANSCRIPT_CHARS` | `150000` | Input safety cap: total content chars |
| `MAX_AGENT_CONTEXT_CHARS` | `50000` | Context cap before passing to each agent (~12k tokens) |
| `CHECKPOINT_DIR` | `.pipeline_checkpoints/` | Checkpoint root directory |
| `HISTORY_FILE` | `~/.meeting_pipeline/history.json` | Cross-session run history |
| `INGEST_POLL_INTERVAL` | `60` | `watcher.py` poll interval in seconds |
| `DATABASE_URL` | — | asyncpg DSN for PostgreSQL ingestion via `ingestion.py` |

---

## 12. Running the Pipeline

```bash
# Required: set Ollama endpoint
export OLLAMA_BASE_URL=http://localhost:11434/v1

# Default meeting
python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py

# Specific meeting
python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py \
    --meeting-id 01KQ0CAE7F064EC93F0540CA

# Force re-run all stages (bypass checkpoints)
python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --force

# Validate input only — no LLM calls
python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --dry-run

# Verbose logging (DEBUG level — shows llm_call=START, tool args, etc.)
python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --debug

# Tune timeouts for slow hardware
STAGE_TIMEOUT_S=1200 MAX_TOOL_CALLS=50 \
    python basics/pydantic_ai/multi_agent/meeting_transcripts/pipeline.py --force
```

### Checkpoint layout

```
.pipeline_checkpoints/
└── 01KQ0CAE7F064EC93F0540CA/
    ├── extraction.json    ← Insight (Pydantic JSON)
    ├── commitments.json   ← CommitmentsOutput
    ├── validation.json    ← ValidationResult
    └── audit.jsonl        ← one JSON line per stage: run_id, stage, duration_s, tokens, status
```

Note: `preprocessing.json` no longer written — preprocessing is instant Python.

### Run tests

```bash
./pydantic-ai/bin/python -m pytest \
    basics/pydantic_ai/multi_agent/meeting_transcripts/tests/ -v
# 32 passed, 0 failed
```

---

## 13. Automated Ingestion

`watcher.py` combines two strategies:

### Event-driven (watchfiles)
```python
async for changes in awatch(str(dataset_dir)):
    new_meetings = {p.parent.name for _, p in changes
                    if p.name == "transcript.json"}
    for mid in new_meetings:
        if not is_processed(mid):
            await process_meeting(mid, dataset_dir)
```
Fires immediately on new files if `watchfiles` is installed.

### Polling fallback
```python
async def poll_loop(dataset_dir):
    while True:
        for mid in discover_unprocessed(dataset_dir):
            await process_meeting(mid, dataset_dir)
        await asyncio.sleep(POLL_INTERVAL)  # default 60s
```
Always running as a safety net. Both run concurrently via `asyncio.gather`.

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1 \
INGEST_POLL_INTERVAL=30 \
DATABASE_URL=postgresql://user:pass@host/dbname \
    python basics/pydantic_ai/multi_agent/meeting_transcripts/watcher.py
```

---

## 14. Known Failure Modes and Mitigations

| Failure | Root cause | Current mitigation | Remaining risk |
|---------|-----------|-------------------|---------------|
| Model produces non-English prefix | qwen2.5 language switching | `_ENGLISH_JSON_GUARD` in all agent instructions | Low — tool calls with simple schemas are less affected |
| Tool-call budget exceeded | Model loops on same findings | `_check_budget()` raises at `MAX_TOOL_CALLS` | Stage fails; checkpoint not written; next run retries |
| Stage timeout | Slow local GPU + retries | `STAGE_TIMEOUT_S=900` env var | Increase to 1800s for very slow hardware |
| Duplicate findings | Model re-records same insight on each iteration | Deduplication in every `record_*` tool | Tested; handled |
| Invalid Literal on tool param | Model hallucinates enum value | Pydantic validates per-call; Pydantic AI retries | Resolved atomically without losing other findings |
| Checkpoint corruption | Crash during write | Atomic write (TODO: write to `.tmp` then rename) | Low probability on modern filesystems |
| Empty validation output | Model doesn't call `validate_action_item` | `ValidationState.validated` remains `[]`; pipeline succeeds but `valid_actions=0` | Alert via `pipeline=DONE valid_actions=0` log |
| Long transcript context overflow | Transcript > 50k chars | `cap_context()` truncates with marker | May miss late transcript content |
| Cloud API rate limits | High meeting volume | Not applicable to local Ollama | Add retry with backoff when switching to cloud |
