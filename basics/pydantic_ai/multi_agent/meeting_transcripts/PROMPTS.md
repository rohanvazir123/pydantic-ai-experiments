# Agent Prompts Reference

All system prompts and dynamic instructions used in the four-agent pipeline,
with rationale for each design choice.

## Table of Contents

1. [Overview](#1-overview)
2. [Agent 1 — PreProcessing](#2-agent-1--preprocessing)
3. [Agent 2 — Extraction](#3-agent-2--extraction)
4. [Agent 3 — Commitments](#4-agent-3--commitments)
5. [Agent 4 — Validation](#5-agent-4--validation)
6. [Dynamic Instructions Pattern](#6-dynamic-instructions-pattern)
7. [Prompt Design Principles](#7-prompt-design-principles)

---

## 1. Overview

Each agent has:
- A **static system prompt** (`instructions=`) — defines the agent's role and output rules
- Zero or one **dynamic instruction** (`@agent.instructions` decorator) — injects runtime data
- Zero or one **few-shot example block** — grounds local models on expected output shape

Prompt text is kept minimal and verb-focused. Local models (qwen2.5, llama) respond
better to concise, imperative instructions than to long prose descriptions.

---

## 2. Agent 1 — PreProcessing

### Role
Format raw JSON turns into a clean, consistently labelled transcript.

### Static Instructions

```
You are a meeting transcript formatter.
Given raw transcript turns, format each into 'MM:SS Speaker: sentence'.
Collect all unique speaker names into participants.
Use the provided meeting title as meeting_title.
```

### User Prompt Template

```
Meeting title: {meeting_info.title}

Raw transcript:
[7.4s] Megan Lawson: Alright, I think we're all on — Raj, Brian, can you both hear me okay?
[13.4s] Raj Kapoor: Yeah, I'm here. Audio's good.
...
```

### Output Schema

```python
class CleanTranscript(BaseModel):
    meeting_title: str
    participants: list[str]   # validator: non-empty, stripped
    turns: list[str]          # "MM:SS Speaker: sentence"
```

### Design Notes
- **No few-shot examples** — the format instruction is concrete enough that examples add noise.
- Context is capped to `MAX_AGENT_CONTEXT_CHARS` (50k chars) before being passed in.
- `participants` has a Pydantic `field_validator` that strips whitespace and rejects empty lists.

---

## 3. Agent 2 — Extraction

### Role
Read the clean transcript and extract structured insights: sentiment shifts, pain
points, and competitor mentions.

### Static Instructions

```
You are an expert meeting analyst. Extract structured insights from transcripts.

EXAMPLES
Segment: 'Pricing is frustrating but the dashboard is what we needed.'
  → sentiment_shift: speaker='Customer',
    shift='Frustrated with pricing → excited about dashboard', polarity='mixed'

Segment: 'Integration takes too long — customers are churning.'
  → pain_point: 'Integration process too slow, causing churn'

Segment: 'Competitors like Acme Corp have this natively.'
  → competitor_mention: 'Acme Corp'

Use search_transcript to look up specific terms before extracting.
If polarity is set, it MUST be one of: positive, negative, neutral, mixed
```

### User Prompt Template

```
{transcript_text}   ← clean formatted turns from Agent 1, capped at 50k chars
```

### Tool: `search_transcript`

```python
def search_transcript(ctx: RunContext[ExtractionDeps], keyword: str) -> str:
    """Search the meeting transcript for lines containing keyword.
    Returns up to 10 matching lines."""
```

The model calls this before committing to an insight, grounding it in actual
transcript text rather than hallucinating from context.

### Output Schema

```python
class SentimentShift(BaseModel):
    speaker: str
    shift: str
    polarity: Literal["positive","negative","neutral","mixed"] | None

class Insight(BaseModel):
    sentiment_shifts: list[SentimentShift]
    pain_points: list[str]
    competitor_mentions: list[str]
```

### Design Notes
- **Few-shot examples** are essential here — without them, local models produce
  generic or incorrectly structured insight objects.
- `polarity` uses `Literal` (JSON Schema enum constraint). `| None` allows the model
  to omit it when uncertain rather than hallucinating an invalid value.
- The `search_transcript` tool call appears in the audit log with keyword + latency,
  making the extraction reasoning transparent.

---

## 4. Agent 3 — Commitments

### Role
Identify every explicit and implicit action item in the transcript.

### Static Instructions

```
You extract explicit and implicit action items from meeting transcripts.
Look for conditional verbs (will, should, can, need to) and timeline markers
(by Friday, tomorrow, next week, by end of day).
For each action item:
  - owner: the specific person or party responsible
  - action: a verb-centric, measurable task
  - deadline: the stated date or time frame, or 'Unspecified'
```

### User Prompt Template

```
{transcript_text}   ← same capped clean transcript as Agent 2
```

### Output Schema

```python
class ActionItem(BaseModel):
    owner: str
    action: str
    deadline: str   # explicit date/phrase, or "Unspecified"

class CommitmentsOutput(BaseModel):
    action_items: list[ActionItem]
```

### Design Notes
- Runs **in parallel** with Agent 2 (`asyncio.gather`) — no dependency between them.
- **No tool** — the agent reads the full transcript directly. A search tool is less
  useful here because commitments are distributed throughout, not concentrated in
  specific segments.
- `deadline` is intentionally a free-form `str` (not a date type) because LLMs
  express deadlines in natural language ("by end of day", "next Wednesday").

---

## 5. Agent 4 — Validation

### Role
Cross-reference extracted action items against the source transcript. Mark each
as `valid` (agreed upon) or `invalid` (rejected, withdrawn, or hypothetical).

### Static Instructions (static part)

```
You are a validation critic. Cross-reference extracted action items against
the original meeting transcript.
Set verdict='invalid' if the item was explicitly rejected, abandoned, or only
hypothetically discussed.
Set verdict='valid' if participants clearly agreed on the item.
verdict MUST be exactly 'valid' or 'invalid'. Always provide a one-sentence reason.
```

### Dynamic Instructions (runtime-injected via `@validation_agent.instructions`)

```
TRANSCRIPT:
{full clean transcript text, capped at 50k chars}

ACTION ITEMS TO VALIDATE:
  [1] Owner: Megan Lawson | Action: Draft updated communication... | Deadline: Within the hour
  [2] Owner: Raj Kapoor   | Action: Send evening status update... | Deadline: Tonight
  ...
```

The transcript and item list are injected at call time, not baked into the static
prompt. This keeps the agent definition clean and reusable across meetings.

### Output Schema

```python
class ValidatedActionItem(BaseModel):
    owner: str
    action: str
    deadline: str
    verdict: Literal["valid", "invalid"]   # strict enum — hallucination impossible
    reason: str

class ValidationResult(BaseModel):
    items: list[ValidatedActionItem]
```

### Design Notes
- `verdict: Literal["valid","invalid"]` is the hardest guardrail in the pipeline.
  If the model outputs `"maybe"` or `"unclear"`, Pydantic raises `ValidationError`
  immediately and Pydantic AI sends the error back to the model to self-correct
  (up to `retries=3`).
- The `run=FAILED` hook fires if all retries are exhausted, enabling alerting.
- **Known failure mode**: if the transcript context is in a non-English language
  or the model defaults to its training language, it may produce non-JSON output
  that exhausts retries. Mitigation: add `"Respond only in English JSON."` to
  the static instructions if this is observed in production.

---

## 6. Dynamic Instructions Pattern

Pydantic AI supports two instruction injection mechanisms:

### Static (at agent definition time)
```python
agent = Agent("ollama:qwen2.5:14b", instructions="You are a ...")
```

### Dynamic (at run time, via decorator)
```python
@agent.instructions
def inject_context(ctx: RunContext[MyDeps]) -> str:
    return f"TRANSCRIPT:\n{ctx.deps.transcript}"
```

Dynamic instructions are appended to the static instructions for every run.
They allow injecting large, run-specific data (transcripts, action item lists)
without polluting the agent definition.

**Rule**: Never put meeting-specific or run-specific data in the static instructions.
Always use the `@agent.instructions` decorator or the user prompt for that.

---

## 7. Prompt Design Principles

| Principle | Why |
|-----------|-----|
| **Imperative verbs** | "Extract", "Format", "Validate" — not "You might want to..." |
| **Few-shot before abstract** | Local models need examples; abstract rules alone fail |
| **Literal types over free-form** | JSON Schema enum constraint prevents hallucination |
| **Minimal static prompt** | Dynamic instructions carry runtime data; static carries role |
| **CAPS for hard constraints** | `verdict MUST be exactly...` catches the model's attention |
| **`| None` for optional Literal** | Better than forcing a wrong value — omit vs. hallucinate |
| **Explicit retry count** | `retries=3` on every `agent.run()` — never use default |
| **Context cap before call** | `cap_context()` at 50k chars prevents silent truncation by the model |
