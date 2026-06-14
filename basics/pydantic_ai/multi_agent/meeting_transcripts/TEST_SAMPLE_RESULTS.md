# Test Sample Results

End-to-end pipeline run results. Each run uses `qwen2.5:14b` via Ollama on local hardware.
Performance note at the bottom explains the latency profile.

---

## Run 1 — Detect Outage Remediation (01KQ03B0303900521BB089CA)

| Field | Value |
|-------|-------|
| **run_id** | `603a50fd` |
| **Meeting** | Detect Outage - Remediation Plan Review |
| **Participants** | Megan Lawson, Raj Kapoor, Brian Cho |
| **Model** | qwen2.5:14b (Ollama local) |
| **Mode** | From checkpoint (prior run) |

### Stage Latency (from audit.jsonl)

| Stage | Status | Duration | LLM Calls | Input Tokens | Output Tokens |
|-------|--------|----------|-----------|-------------|--------------|
| preprocessing | skipped | 0s | 0 | — | — |
| extraction | skipped | 0s | — | — | — |
| commitments | skipped | 0s | — | — | — |
| validation | skipped | 0s | — | — | — |
| **Total** | ✓ | **~0s** | **0** | — | — |

### Output

**Pain Points**
- Timeline pressures during day-3 of outage window

**Sentiment Shifts**
- `[negative]` Brian Cho: Frustrated with timeline delays
- `[mixed]` Raj Kapoor: Acknowledging need for conservative timelines → defending against credibility loss
- `[neutral]` Raj Kapoor: Explaining benefits and necessity of circuit breaker architecture

**Action Items (valid)**

| Owner | Action | Deadline |
|-------|--------|---------|
| Brian Cho | Update customer tickets with revised timeline | — |
| Megan Lawson | Draft updated customer communication acknowledging delay, explaining phased rollout, providing concrete timeline | Within the hour |

**Hallucination warnings**: 0

---

## Run 2 — Weekly Engineering Standup (01KQ0CAE7F064EC93F0540CA)

| Field | Value |
|-------|-------|
| **run_id** | `44fc4300` |
| **Meeting** | Weekly Engineering Standup |
| **Participants** | Chris Lee, Mike Romano, Tom Bradley, Tyler Washington |
| **Model** | qwen2.5:14b (Ollama local) |
| **Mode** | `--force` (full fresh run) |

### Stage Latency

| Stage | Status | Duration | LLM Calls | Input Tokens | Output Tokens |
|-------|--------|----------|-----------|-------------|--------------|
| preprocessing | ✓ deterministic | **0.0s** | 0 | 0 | 0 |
| extraction | ✓ | 90.89s | 2 | 4,296 | 125 |
| commitments | ✓ (parallel) | 42.84s | 1 | 1,968 | 106 |
| validation | ✓ | 43.02s | 1 | 2,095 | 191 |
| **Total wall-clock** | ✓ | **~176s** | **4** | **8,359** | **422** |

> **Parallelism saving**: extraction + commitments ran concurrently.
> Wall-clock = max(90.89, 42.84) + 43.02 = **133.91s** vs sequential 176.75s → **25% faster**.

### Trace (from log)

```
19:59:25 [44fc4300][pipeline      ] pipeline=START run_id=44fc4300
19:59:25 [44fc4300][preprocessing ] stage=DONE   turns=40 participants=[Chris Lee, Mike Romano, Tom Bradley, Tyler Washington]
19:59:25 [44fc4300][extraction    ] stage=START
19:59:25 [44fc4300][commitments   ] stage=START
20:00:07 [44fc4300][commitments   ] llm_call=DONE   latency=42.22s attempt=1
20:00:08 [44fc4300][commitments   ] stage=DONE   duration=42.84s in_tokens=1968 out_tokens=106
20:00:26 [44fc4300][extraction    ] llm_call=DONE   latency=61.06s attempt=1
20:00:56 [44fc4300][extraction    ] llm_call=DONE   latency=29.82s attempt=2
20:00:56 [44fc4300][extraction    ] stage=DONE   duration=90.89s in_tokens=4296 out_tokens=125
20:00:56 [44fc4300][validation    ] stage=START
20:01:39 [44fc4300][validation    ] llm_call=DONE   latency=43.02s attempt=1
20:01:39 [44fc4300][validation    ] stage=DONE   duration=43.02s in_tokens=2095 out_tokens=191
20:01:39 [44fc4300][pipeline      ] pipeline=DONE  valid_actions=2 invalid_actions=0 hallucination_warnings=0
```

### Output

**Pain Points**
- There's some technical uncertainty we're actively working down

**Action Items (valid)**

| Owner | Action | Deadline |
|-------|--------|---------|
| Tyler Washington | Carve out time to start investigating the Kafka piece for Detect reliability and document findings for Mike Romano | This week |
| Tom Bradley | Communicate with product that end of March is still the target for Detect reliability but there is technical uncertainty being actively addressed | Unspecified |

**Hallucination warnings**: 0

---

## Performance Analysis

### Why is the pipeline slow?

The bottleneck is **local LLM inference**, not architecture or orchestration.

| Metric | This run (qwen2.5:14b local) | Cloud API (gpt-4o-mini) | Cloud API (claude-haiku-4-5) |
|--------|------------------------------|------------------------|------------------------------|
| Per-call latency | 40–120s | 1–3s | 1–4s |
| Total pipeline | ~2–3 min | ~10–15s | ~10–15s |
| Cost | Free (local GPU) | ~$0.001–0.003 | ~$0.001–0.003 |

### What parallelism buys

Stages 2 (extraction) and 3 (commitments) run concurrently via `asyncio.gather`.

```
Sequential:  preprocess(0) → extract(90) → commit(43) → validate(43) = 176s
Parallel:    preprocess(0) → max(extract,commit)(90) → validate(43) = 133s
Savings:     43s (25%)
```

Parallelism scales as you add more independent extraction dimensions — each new parallel
agent adds ~0s to wall-clock if it finishes before the slowest existing stage.

### To make it production-fast

| Option | Expected speedup | Change required |
|--------|-----------------|----------------|
| Switch to `llama3.1:8b` | 2–3× faster (fewer params) | Change model string |
| Use `openai:gpt-4o-mini` | 40–80× faster | Set `OPENAI_API_KEY` |
| Use `anthropic:claude-haiku-4-5` | 40–80× faster | Set `ANTHROPIC_API_KEY` |
| Add a GPU to local Ollama | 3–10× faster (hardware) | — |
| Tool-call pattern | Same speed, higher reliability | Refactoring underway |

### Tool calls vs structured output

The current structured-output approach (`output_type=Model`) requires the model to produce
a complete, valid JSON object in one shot. With local models this sometimes fails (language
switching, malformed JSON prefix) causing retries.

The **tool-call pattern** (being implemented) has the model make N simple validated calls:
- Each tool has a flat, simple signature — less likely to fail
- Pydantic validates each parameter independently
- The model makes one decision at a time
- Trade-off: N+1 LLM calls instead of 1, so slightly slower in the happy path
  but eliminates the retry cost on failures

For a 14B local model, tool calls are **more reliable** and roughly the same speed overall.
For a cloud API where each call costs ~$0.001 and takes 1–3s, they are both fast enough.
