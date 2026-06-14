# Meeting Transcript Pipeline — TODO

## Table of Contents

1. [Completed](#completed)
2. [Documentation (initial)](#documentation-create-before-further-implementation)
3. [Code Changes](#code-changes)
4. [Testing](#testing)
5. [Automated Ingestion](#automated-ingestion)
6. [Observability & Production Hooks](#observability--production-hooks)
7. [Guardrails & Stability](#guardrails--stability)
8. [Latency](#latency)
9. [Documentation (outstanding)](#documentation)

---

## Completed
- [x] Build 4-agent async pipeline (PreProcessing → Extraction → Commitments → Validation)
- [x] Use local Ollama model (qwen2.5:14b / llama3.1:8b switchable)
- [x] Run pipeline end-to-end on dataset
- [x] Strict guardrails: `Literal` / `Enum` types on constrained output fields
- [x] Production features: checkpointing, state restore, audit log (JSONL), memory (history.json), search_transcript tool, `--force` / `--dry-run` / `--debug` flags, safety input validation, structured logging, `retries=3`
- [x] Fix `result.usage()` → `result.usage` deprecation

## Documentation (create before further implementation)
- [x] `README.md` with full TOC:
  - [x] Overview
  - [x] Agentic AI system design pattern (clearly documented)
  - [x] Architecture diagram (text-based)
  - [x] Call graph (stage-by-stage with data types)
  - [x] Prompts used per agent
  - [x] Production features table
  - [x] Test metrics (Precision, Recall, Hallucination Rate, etc.)
  - [x] Running the pipeline
  - [x] Configuration reference
- [x] `test_questions.md` with TOC:
  - [x] Head of Engineering questions + SQL queries against PostgreSQL tables
  - [x] Product questions + SQL queries
  - [x] Customer Support questions + SQL queries
  - [x] PostgreSQL schema for stored insights / action items

## Code Changes
- [x] Pydantic input schemas: `TranscriptEntry`, `MeetingInfo`, `PipelineInput` — type-safe from JSON load onwards
- [x] Always use `await agent.run()` (async default) — no `run_sync()`
- [x] Context size guardrails: `cap_context()` caps at `MAX_AGENT_CONTEXT_CHARS` (50k chars); appends truncation marker
- [x] Hallucination catching: `detect_hallucinations()` — unknown owner, short action text, empty competitor, empty deadline
- [x] Separate data ingestion from query — no mixed concerns:
  - [x] `ingestion.py`  — pipeline run → persist to PostgreSQL (idempotent upsert)
  - [x] `query.py`      — typed async read layer for all 15 stakeholder questions
  - [x] `pipeline.py`   — agent workflow only (no storage imports)
- [x] PostgreSQL schema: `meetings`, `meeting_insights`, `action_items` tables (DDL in `ingestion.py`)

## Testing
- [x] Test cases that intentionally break the pipeline (32 tests, 32 passed):
  - [x] Empty transcript — `ValidationError` from `PipelineInput` validator
  - [x] Single-speaker transcript — passes validation; weakness documented in test
  - [x] Transcript > MAX_TRANSCRIPT_TURNS — `ValueError`
  - [x] All speakers have empty names — `ValueError`
  - [x] Contradictory action items — `verdict='invalid'` unit test
  - [x] Transcript with no action items — empty valid list, no crash
  - [x] Non-ASCII / multilingual / RTL transcript — passes schema validation
  - [x] Malformed JSON input — missing fields, wrong types, missing meetingId
  - [x] LLM hallucinates wrong Literal value — `ValidationError` at schema level
  - [x] Validation agent marks all items invalid — empty valid list handled correctly
  - [x] Context overflow / prompt injection via long sentence — `cap_context` catches it

## Automated Ingestion
- [x] Event-driven job: `watcher.py` uses `watchfiles` (inotify/FSEvents) when available
- [x] Polling fallback: `poll_loop()` checks every `INGEST_POLL_INTERVAL` seconds (default 60s)
- [x] Skip already-ingested meetings via `get_history()` / `is_processed()`
- [x] Configurable poll interval via `INGEST_POLL_INTERVAL` env var
- [x] Optional PostgreSQL write after each run via `DATABASE_URL` env var

## Observability & Production Hooks
- [x] Correlation ID (`run_id`) via `contextvars` — propagates cleanly through `asyncio.gather`
- [x] `_CorrelationFormatter` injects `run_id` + `stage` into every log line
- [x] `before_model_request` / `after_model_request` hooks — per-LLM-call latency
- [x] `model_request_error` hook — LLM API errors (TODO stub: Prometheus counter)
- [x] `output_validate_error` hook — schema validation failures + retry logging (TODO: Sentry)
- [x] `before_tool_execute` / `after_tool_execute` hooks — per-tool latency
- [x] `tool_execute_error` hook — tool failures (TODO stub: dead-letter queue)
- [x] `run_error` hook — terminal failures (TODO stub: PagerDuty / Slack)
- [x] `STAGE_TIMEOUT_S` env var (default 900s) — hard per-stage timeout via `asyncio.wait_for`
- [x] `PROMPTS.md` — all agent prompts documented with rationale
- [x] `DATASTORE.md` — PostgreSQL schema, file stores, entity diagram, key queries

## Guardrails & Stability
- [ ] **Prevent infinite tool-call loops**: set `model_settings={"max_tokens": N}` cap per agent;
      add `prepare_tools` hook that injects a tool-call-budget counter, raising if exceeded
- [ ] **Add `"Respond in English JSON only"` to all agent instructions** to prevent
      language-switching failures (qwen2.5 produces Thai/Chinese prefix on some transcripts)
- [ ] **Replace PreProcessing agent with pure-Python formatter** — speaker names are already
      resolved in `transcript.json`; formatting `[time] Speaker: sentence` is deterministic
      and needs no LLM (saves ~120s per run, eliminates one failure mode)
- [ ] **Add `max_retries` env var** (default 3) so operators can tune retry budget without
      code changes

## Latency

**Baseline (run 2, qwen2.5:14b, single GPU):**

| Stage | Status | Wall-clock | LLM calls | Input tokens | Output tokens |
|---|---|---|---|---|---|
| preprocessing | deterministic | 0.0s | 0 | 0 | 0 |
| extraction | done | 90.89s | 2 | 4,296 | 125 |
| commitments | done (parallel) | 42.84s | 1 | 1,968 | 106 |
| validation | done | 43.02s | 1 | 2,095 | 191 |
| **Total wall-clock** | | **~176s** | **4** | **8,359** | **422** |

**Root cause of 176s:** `asyncio.gather` submits extraction and commitments concurrently but local Ollama serialises GPU requests — effective total is the sum of all LLM call durations, not the max. Extraction's second LLM call was caused by the `search_transcript` tool forcing a round trip (search → result → record pass) despite the full transcript already being in context.

**Fixes applied (2026-06-06):**
- Removed `search_transcript` tool — extraction drops from 2 LLM calls to 1 (~45s saved)
- Removed full transcript from `_inject_items_for_validation` — validation input tokens cut by ~1–2k (~10s saved)
- Removed `ValidationState.transcript` (unused after above)
- Fixed stage print labels (`[2/3]`/`[3/3]` → `[2+3/4]`/`[4/4]`)
- Fixed `pydantic_ai.capabilities.hooks.Hooks` import (removed in pydantic-ai ≥ 1.30); replaced with stage-level timing and OTel note
- Fixed stale test imports (`ExtractionDeps`/`ValidationDeps` → `ExtractionState`/`ValidationState`)

**Projected after fixes (~103s, still on single GPU):**

| Stage | LLM calls | Approx time |
|---|---|---|
| preprocessing | 0 | ~0s |
| extraction | 1 | ~45s |
| commitments | 1 | ~43s |
| validation | 1 | ~15s |
| **Total (serial on 1 GPU)** | **3** | **~103s** |

**Outstanding remedies to get below 100s:**

- [ ] **Downsize to `qwen2.5:7b`** — ~2× throughput on same GPU, cuts total to ~50s. Biggest single win; quality for structured tool-calling is comparable.
- [ ] **Merge extraction + commitments into one agent** — eliminates one LLM round trip (~43s). Both agents receive the same transcript; a single `combined_agent` with all 4 record tools runs in one pass.
- [ ] **Skip validation for small meetings** — if `len(action_items) <= 3`, bypass validation stage entirely. Saves ~15s for the majority of short standups.
- [ ] **Two Ollama instances for true parallelism** — point extraction and commitments at separate Ollama processes (different ports) so `asyncio.gather` gives real speedup: `max(extraction, commitments) + validation` ≈ 25 + 12 = ~37s.
- [ ] **Tighten `MAX_AGENT_CONTEXT_CHARS`** — current 50k chars (~12k tokens); reducing to 15k chars (~3.5k tokens) halves prefill time without quality loss for typical meeting lengths.

**Realistic combined target:** ~25s (single GPU, qwen2.5:7b, merged extraction+commitments, validation gated on item count).

---

## Documentation
- [ ] Update `README.md` production features table with new hooks and timeout config
- [ ] Update `PROMPTS.md` to note English-only instruction addition
