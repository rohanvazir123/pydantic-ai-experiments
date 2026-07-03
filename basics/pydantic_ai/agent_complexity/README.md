# The 5 Levels of Agent Complexity (Pydantic AI + local Ollama)

Five worked examples of the same customer-support problem solved at five
increasing levels of agent complexity — from a single model call to a
multi-agent orchestrator. Every example runs **entirely locally on Ollama**
(no API keys), is fully typed, and ships with a deterministic test suite.

Adapted from the Anthropic cookbook's *agent-complexity* series. The cookbook's
Levels 4–5 use the Anthropic-only Claude Agent SDK; here they are re-implemented
with native Pydantic AI patterns so the whole ladder runs on a local model.

## Table of Contents

- [Why this exists](#why-this-exists)
- [The five levels](#the-five-levels)
- [Choosing a level: use cases & trade-offs](#choosing-a-level-use-cases--trade-offs)
  - [Decision guide](#decision-guide)
  - [Level-by-level: when to use, when not to](#level-by-level-when-to-use-when-not-to)
- [Cost & latency at a glance (L1→L5)](#cost--latency-at-a-glance-l1l5)
- [Level 5 deep-dive: multi-agent system design](#level-5-deep-dive-multi-agent-system-design)
  - [Cost & latency: what's realistic (p50/p95/p99)](#cost--latency-whats-realistic-p50p95p99)
  - [Why multi-agent is slow *despite* parallelism](#why-multi-agent-is-slow-despite-parallelism)
  - [Resiliency & self-healing](#resiliency--self-healing)
  - [Other system-design trade-offs](#other-system-design-trade-offs)
  - [When NOT to go multi-agent](#when-not-to-go-multi-agent)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Model tiers (tiered LLMs)](#model-tiers-tiered-llms)
- [Running the examples](#running-the-examples)
- [Debuggability & observability (L1→L5)](#debuggability--observability-l1l5)
  - [Viewing Pydantic AI Logfire traces](#viewing-pydantic-ai-logfire-traces)
- [Testing](#testing)
- [Test report](#test-report)
- [Project structure](#project-structure)
- [Adapting from the Claude Agent SDK to Pydantic AI](#adapting-from-the-claude-agent-sdk-to-pydantic-ai)
- [Notes on local models](#notes-on-local-models)

## Why this exists

The hardest decision in agent engineering is *how much agent* a task needs. Too
little and it can't do the job; too much and you pay for latency, cost, and
non-determinism you didn't need. These five runnable examples make the
trade-offs concrete on one consistent problem: **resolving a duplicate-charge
support ticket for customer `cust_12345`.**

> The routing decision isn't about severity — it's about what the task *needs*.
> Use the simplest level that gets the job done.

## The five levels

| Level | Pattern | File | What the model controls |
|------:|---------|------|-------------------------|
| 1 | Augmented LLM | [`l1_augmented_llm.py`](l1_augmented_llm.py) | Nothing — one shot, structured output |
| 2 | Prompt Chains & Routing | [`l2_prompt_chains.py`](l2_prompt_chains.py) | Content of each step (code controls flow) |
| 3 | Tool-Calling Agent | [`l3_tool_calling_agent.py`](l3_tool_calling_agent.py) | Which tools to call, and when |
| 4 | Agent Harness | [`l4_agent_harness.py`](l4_agent_harness.py) | Open-ended exploration over a runtime |
| 5 | Multi-Agent Orchestration | [`l5_multi_agent.py`](l5_multi_agent.py) | Decomposition + delegation to specialists |

```
Level 1   input ──▶ [system prompt + schema] ──▶ LLM ──▶ structured output

Level 2   ticket ──▶ classify ──▶ route ──▶ handler ──▶ validate ──▶ done
                       (code picks the branch, not the model)

Level 3   task ──▶ agent ⇄ {balance, charges, policy, refund} ──▶ resolution
                   (model sequences a fixed tool set)

Level 4   task ──▶ agent ⇄ {list/read/grep files + billing API} ──▶ report
                   (model explores a runtime autonomously)

Level 5   request ──▶ orchestrator ─┬▶ researcher  (fs + gateway)
                                    ├▶ drafter     (fs: templates)
                                    └▶ compliance  (fs: policies)  ──▶ decision
```

## Choosing a level: use cases & trade-offs

| | L1 Augmented | L2 Chains | L3 Tool-Calling | L4 Harness | L5 Multi-Agent |
|---|---|---|---|---|---|
| **Cost** | $ | $ | $$ | $$$ | $$$$ |
| **Latency** | ~1 call | N fixed calls | a few calls | many calls | many × agents |
| **Reliability** | Deterministic\* | High | High | Medium | Lower |
| **Control flow** | none | code | model (bounded) | model (open) | orchestrator |
| **Best when** | answer is in the input | known stages | needs a few actions | needs exploration | needs parallel expertise |

\* *Deterministic in shape (always one call, one schema); the model's content
still varies unless you pin `temperature=0`, which these examples do.*

### Decision guide

```
Is the answer derivable from the input alone (classify / extract / rewrite)?
│
├─ YES ─────────────────────────────────────────────▶ Level 1 (Augmented LLM)
│
└─ NO ─ Does it follow known, fixed stages you can hard-code?
        │
        ├─ YES ─────────────────────────────────────▶ Level 2 (Prompt Chains)
        │
        └─ NO ─ Can a *small, fixed* set of tools do it?
                │
                ├─ YES ─────────────────────────────▶ Level 3 (Tool-Calling)
                │
                └─ NO ─ Does it need open-ended exploration
                        over files / systems you can't script?
                        │
                        ├─ YES ─ one perspective enough? ─▶ Level 4 (Harness)
                        │
                        └─ needs distinct expert roles
                          working in concert? ───────────▶ Level 5 (Multi-Agent)
```

**Golden rule: start at the lowest level that works and only climb when it
demonstrably can't do the job.** Each climb multiplies cost, latency, and the
number of ways a run can go wrong.

### Level-by-level: when to use, when not to

**Level 1 — Augmented LLM.** One model call with a system prompt and a
structured (`BaseModel`) output.
- ✅ Use for: classification, extraction, summarization, rewriting, sentiment,
  a routing *decision*, "turn this text into this schema."
- ❌ Avoid when: the model needs to look something up or take an action. If you
  catch yourself wanting a tool, go to Level 3.
- Real example: triage every inbound ticket into `{category, priority, summary,
  can_auto_resolve}` to feed a queue.

**Level 2 — Prompt Chains & Routing.** Several single-purpose agents wired by
*code* into a DAG. The model decides the *content* of each step; your code
decides *which step runs next*.
- ✅ Use for: pipelines with known stages (classify → handle → validate),
  where you want each stage small, independently testable, and swappable; where
  routing must be auditable and deterministic.
- ❌ Avoid when: you can't predict the stages, or the "routing" genuinely needs
  the model to choose tools dynamically (that's Level 3).
- Why not just one big prompt? Small focused prompts are more reliable and far
  easier to test — see `tests/test_l2_prompt_chains.py`, which pins routing.

**Level 3 — Tool-Calling Agent.** One agent, a fixed set of well-defined tools,
and dependency injection. The agent loops: call a tool → observe → decide →
finish with structured output.
- ✅ Use for: tasks needing a *few specific, trusted actions* (look up a record,
  call an API, run a calculation) sequenced by the model. The tool set is the
  guardrail.
- ❌ Avoid when: you need dozens of ad-hoc capabilities or free-form file
  exploration (Level 4), or the actions are actually a fixed sequence (Level 2
  is cheaper and more predictable).
- Real example: the billing agent verifies charges and issues a refund.

**Level 4 — Agent Harness.** Give the agent a *runtime*: a sandboxed filesystem
(list / read / grep) plus external APIs, and let it explore, reason, act, and
iterate. This is the shape of coding agents (Claude Code, Cursor).
- ✅ Use for: open-ended investigation over a body of files/systems where you
  *can't enumerate the steps in advance* — "figure out what's relevant, then
  act."
- ❌ Avoid when: a handful of tools already covers it (Level 3 is faster,
  cheaper, more predictable). Broad access = broader blast radius; note the
  path-traversal guard that keeps file access inside `knowledge/`.
- Real example: read the customer file + policy, verify via the gateway, refund,
  and draft a personalized reply — without being told which files to open.

**Level 5 — Multi-Agent Orchestration.** An orchestrator decomposes the task and
delegates to specialist agents, each with its own prompt, tools, and model.
- ✅ Use for: tasks needing *parallel domain expertise* — distinct roles
  (research, drafting, compliance) with different instructions/tools — plus a
  coordinator to synthesize.
- ❌ Avoid when: a single tool-calling agent can hold all the roles. This is the
  most capable **and** the most expensive and least deterministic level; every
  added agent adds latency and failure modes.
- Real example: researcher investigates → drafter writes the reply → compliance
  reviews → orchestrator issues the final decision.

## Cost & latency at a glance (L1→L5)

Consolidated **latency and token-cost** estimates for one end-to-end run of each
level on the *same* duplicate-charge task. Numbers are **estimates** — latency
is dominated by your hardware/model and token cost by prompt sizes — but the
*shape* (roughly 1→50× from L1 to L5) is the durable takeaway.

Grounding: the local latencies are observed on this repo (`qwen2.5:14b`, single
GPU, `temperature=0`); the L5 token count is measured
(`RunUsage(input=6291, output=1985, requests=11)` for one case); everything else
is estimated from call counts and typical prompt sizes.

| Level | Model calls (typical) | Tokens/run (in + out, est) | Local latency¹ | Prod latency² | Prod $/run³ | Cost ×L1 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **L1** Augmented LLM | 1 | ~160 + ~40 ≈ **0.2k** | ~1–3s | ~0.5–1.5s | ~$0.001 | 1× |
| **L2** Prompt Chains | 2 (fixed) | ~400 + ~120 ≈ **0.5k** | ~5–10s | ~1–3s | ~$0.003 | ~3× |
| **L3** Tool-Calling | 3–7 | ~2.5k + ~0.4k ≈ **3k** | ~10–25s | ~3–8s | ~$0.014 | ~10× |
| **L4** Agent Harness | 6–14 | ~7k + ~0.8k ≈ **8k** | ~30–70s | ~15–40s | ~$0.033 | ~30× |
| **L5** Multi-Agent | 10–20+ | ~6k + ~2k ≈ **8–12k** | ~90–180s | ~25–60s | ~$0.049 | ~40–50× |

¹ Local = `qwen2.5:14b` on one GPU via Ollama; wall-clock, `temperature=0`.
Regenerate with `python benchmark.py` (see [`LATENCY.md`](LATENCY.md)).
² Prod = a hosted frontier model, well-tuned; p50-ish. Faster per call than local
14B, but the *call count* still sets the floor.
³ Prod $/run at an illustrative blended **$3 / 1M input, $15 / 1M output**
(swap in your model's pricing). Local marginal cost ≈ **$0** (self-hosted; you pay
GPU time/electricity, not per token).

**Why token cost grows super-linearly with level:** every agent turn re-sends the
*entire* conversation so far. A 10-turn agentic run doesn't cost 10× a single
call — the context accrues (tool results, file contents, prior reasoning), so
later turns are the most expensive. Multi-agent (L5) compounds this: the
orchestrator *and* each specialist each carry their own growing context. Tokens —
not code — are the cost, and they scale with sequential calls. Percentile
breakdowns (p50/p95/p99) and the reasons the tail blows out are in the
[Level 5 deep-dive](#level-5-deep-dive-multi-agent-system-design).

## Level 5 deep-dive: multi-agent system design

Level 5 is where the interesting engineering lives — and where most teams
over-invest. This section is the "when you really do need it, here's what you're
signing up for" guide. Measured latency numbers for this repo are in
[`LATENCY.md`](LATENCY.md) (regenerate with `python benchmark.py`).

### Cost & latency: what's realistic (p50/p95/p99)

Two very different regimes. Use the shape, not the absolute numbers — your model,
hardware, and prompt sizes dominate.

**Local single-GPU Ollama (this repo, `qwen2.5:14b`)** — illustrative, measured:

| Level | Sequential model calls | p50 | p95 | p99 |
|-------|:---:|:---:|:---:|:---:|
| L1 Augmented | 1 | ~1–3s | ~4s | ~6s |
| L2 Chains | 2 (fixed) | ~4–8s | ~12s | ~15s |
| L3 Tool-Calling | 3–7 (model-driven) | ~10–25s | ~40s | ~60s |
| L4 Harness | 6–14 | ~30–70s | ~110s | ~150s |
| L5 Multi-Agent | 10–20+ across agents | ~90–180s | ~240s | ~300s+ |

**Hosted frontier model (production estimate, well-tuned):**

| Level | p50 | p95 | p99 | Cost multiple vs L1 |
|-------|:---:|:---:|:---:|:---:|
| L1 Augmented | ~0.5–1.5s | ~3s | ~5s | 1× |
| L3 Tool-Calling | ~3–8s | ~15s | ~25s | 3–6× |
| L4 Harness | ~15–40s | ~60s | ~90s | 10–20× |
| L5 Multi-Agent | ~25–60s | ~90s | ~150s+ | 20–50× |

**How many tokens is a "3-turn" multi-agent call?** A 3-*delegation* run
(research → draft → review) is **not** 3 model calls — each delegate runs its own
tool loop, so it expands to ~10–20 model requests, and *every* request re-sends
the accumulated context. Measured here (toy case, tiny KB): ~8k tokens
(`input=6291, output=1985, requests=11`). In production with real system prompts +
retrieved context + longer tool outputs, expect **~30k–70k** (modest) to
**150k–300k+** (heavy) tokens per case. So yes — tens of thousands is the floor,
not the ceiling. The lever is *smaller per-call context* (trim history, don't
re-inject whole documents each turn, isolate sub-agent contexts), not "fewer
turns."

Why the **tail (p95/p99) blows out** faster than p50 as you climb:

- **Multiplicative reliability.** If one model call succeeds 97% of the time, a
  20-call L5 run completes clean only ~0.97²⁰ ≈ 54% of the time. The rest incur
  a **retry** (validation error, malformed tool args, refusal), and each retry is
  a full extra model round-trip. Retries land in the tail.
- **Slowest-link gating.** An orchestrator that waits on 3 sub-agents is as slow
  as the slowest one on each step; p99 of the whole is driven by the p99 of *any*
  sub-agent (tail amplification).
- **Unbounded-ish loops.** Agentic levels loop until "done." A confused run can
  wander for many extra turns before `max_turns` cuts it off — a fat right tail.
- **Cost tracks calls, not levels.** L5 re-sends context to the orchestrator *and*
  every sub-agent; token spend (and $) is roughly the sum over all agents' turns.

### Why multi-agent is slow *despite* parallelism

The intuition "N sub-agents ⇒ N× faster" almost never holds. Reasons:

1. **The critical path is sequential.** Our pipeline is
   `research → draft → review`: the drafter needs the researcher's findings; the
   reviewer needs the draft. Data dependencies serialize the stages — you can't
   parallelize step *k+1* until step *k* returns. Amdahl's law caps the win at
   the fraction that is genuinely independent.
2. **Each sub-agent is itself a multi-call loop.** The researcher alone may make
   4–6 sequential model calls (list → read → verify → summarize). "One
   delegation" ≠ "one model call" — it's a nested loop. Total latency ≈ Σ (each
   agent's internal turns) along the critical path, plus the orchestrator's own
   turns to decide each delegation.
3. **Local GPU serialization.** A single GPU runs one forward pass at a time.
   `asyncio.gather()` over sub-agents yields *concurrency* (overlapped waiting)
   but not *parallelism* — Ollama queues the requests and runs them back-to-back,
   so wall-clock ≈ the **sum**, not the max. (This repo's meeting-transcripts
   notes hit exactly this: gather'd calls summed instead of maxing.) You only get
   real speedup with independent workers on *separate* model endpoints/GPUs.
4. **Orchestrator overhead.** The coordinator spends its own turns reasoning
   about *who* to call and *how* to merge results. In our L5 run that was ~11
   total requests / ~9 tool calls for one case — several of them pure
   coordination, not domain work.
5. **Context growth.** As findings accumulate, prompts get longer, so later calls
   have higher time-to-first-token and more tokens to generate.

**Where parallelism *does* pay off:** genuinely independent sub-tasks — e.g. fan
out "summarize each of these 20 documents," or run research across three
unrelated domains at once — *and* the workers sit on different endpoints (hosted
API concurrency, or multiple Ollama processes/GPUs). Restructure dependent
pipelines into independent fan-out where you can; that is the only reliable lever.

### Resiliency & self-healing

More moving parts ⇒ more failure modes. A production L5 system needs most of the
following (this teaching example implements the first two):

- **Bounded retries with backoff.** Pydantic AI's `retries=` feeds validation /
  tool errors back so the model self-corrects (used here). Add exponential
  backoff + jitter for *transient* external failures (429/503/timeouts). Cap
  attempts — infinite retries turn a blip into an outage.
- **Idempotency for side effects.** `issue_refund` must not double-refund when a
  step is retried. Attach an idempotency key (e.g. `ticket_id + charge_id`) so
  repeats are no-ops. This is the single most important safety property once
  agents take real actions.
- **Circuit breakers** per external dependency (payment gateway, CRM, a specific
  sub-agent/model). After N consecutive failures, trip open and fail fast for a
  cooldown instead of piling latency onto a dead dependency; probe half-open to
  recover. (See `rag/v2/knowledge/bus` for a real breaker in this repo.)
- **Timeouts & budgets at every layer.** Per-tool timeout, per-agent `max_turns`,
  and a global token/cost/wall-clock budget for the whole case. Without a global
  budget, one pathological run can dwarf a thousand normal ones.
- **Checkpointing / durable execution** for long-running work. Persist state after
  each completed stage so a crash resumes from the last checkpoint instead of
  re-paying for research + drafting. Pydantic AI integrates with **Temporal**
  (`TemporalAgent`), **DBOS**, and **Prefect** for exactly this — each agent step
  becomes a durable, replayable activity. Essential when a case spans minutes and
  touches paid side effects you don't want to repeat.
- **Graceful degradation / fallbacks.** On sub-agent failure: retry once → fall
  back to a smaller/faster model → **drop to a lower complexity level** (e.g. hand
  the case to the L3 tool-calling agent) → escalate to a human. Never let one
  specialist's failure hard-fail the whole case if a safe partial result exists.
- **Partial-failure policy in the orchestrator.** Decide explicitly per stage:
  required vs optional. If `compliance` is down, do you block (safe) or queue for
  async review (available)? Make it a policy, not an accident.
- **Human-in-the-loop gates.** High-risk actions (refund > $100, per our
  escalation matrix) should require approval. Pydantic AI supports tool approval /
  deferred tools for this.
- **Observability is non-negotiable.** With 5 agents and 20 calls, "it was slow /
  wrong" is undebuggable without traces. Instrument with Logfire
  (`logfire.instrument_pydantic_ai()`) to see per-agent, per-tool spans, token
  counts, and where the tail time went.

### Other system-design trade-offs

- **Determinism vs capability.** Lower levels are auditable and repeatable; L5
  trades that away for flexibility. If a regulator or a test needs reproducible
  behavior, push logic *down* into code (L2) rather than up into more agents.
- **Debuggability & blast radius.** Each added agent/tool widens what can go
  wrong and what it can touch. Note the filesystem sandbox in L4/L5 — broad
  capability demands hard boundaries.
- **Prompt/version coupling.** Five prompts that must stay mutually consistent is
  five things to regression-test on every model upgrade. Version and eval them.
- **Shared vs isolated context.** Pydantic AI delegation shares deps and a single
  usage tally (cheaper, coupled); SDK-style subagents get isolated context
  windows (cleaner, more tokens). Pick per how much cross-talk the roles need.
- **Cost attribution & quotas.** Track spend per tenant/case; a runaway multi-
  agent loop is a budget incident. (See `rag/v2/knowledge/billing`.)

### When NOT to go multi-agent

Reach for L5 **only** when a single tool-calling agent genuinely can't hold the
roles — distinct expertise, conflicting instructions, or independent parallel
work. If the "roles" are just steps, that's Level 2. If they're just tools,
that's Level 3. Most "multi-agent" designs are a tool-calling agent with a good
prompt — cheaper, faster, and far easier to keep correct.

## Quick start

```bash
# 0. Prerequisites: Ollama running, with the models pulled
ollama serve                       # in another terminal
ollama pull qwen2.5:14b            # default (reliable tool calling)
ollama pull llama3.2:3b            # optional: faster "small" tier
ollama pull qwen2.5:0.5b           # optional: "nano" tier

# 1. From this directory
cd basics/pydantic_ai/agent_complexity

# 2. Run the deterministic tests (no Ollama needed, ~2s)
python -m pytest -q

# 3. Run an example against your local model
python l1_augmented_llm.py
python l3_tool_calling_agent.py
```

Dependencies (`pydantic-ai`, `pytest`) are already in the repo's `.venv`; use
`../../../.venv/bin/python` if that venv isn't otherwise activated.

## Configuration

All model configuration lives in [`config.py`](config.py). Override anything via
environment variables — no code edits:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `AGENT_COMPLEXITY_TIER` | `large` | Default tier for the examples |
| `AGENT_NANO_MODEL` | `qwen2.5:0.5b` | routing / classification tier |
| `AGENT_SMALL_MODEL` | `llama3.2:3b` | standard responses tier |
| `AGENT_LARGE_MODEL` | `qwen2.5:14b` | reliable tool-calling / reasoning tier |
| `AGENT_TEMPERATURE` | `0.0` | low temp = reliable structured output |

```bash
# e.g. run the examples on the faster (flakier) small tier
AGENT_COMPLEXITY_TIER=small python l2_prompt_chains.py
```

## Model tiers (tiered LLMs)

A production agent system routes each step to the *cheapest model that can do it*
— the single biggest cost/latency lever. This repo bakes that in with three
tiers (`config.py`):

| Tier | Default model | Intended for |
|------|---------------|--------------|
| `nano` | `qwen2.5:0.5b` | routing, classification, triage |
| `small` | `llama3.2:3b` | standard text generation |
| `large` | `qwen2.5:14b` | reasoning + reliable tool/structured output |

Each agent **requests** a semantic tier (visible as a `*_TIER` constant at the
top of each example), and the levels are wired tier-aware:

| Level | Role → requested tier |
|-------|----------------------|
| L1 | classifier → `small` |
| L2 | classifier → `nano`, handlers → `small` |
| L3 | tool agent → `large` |
| L4 | triage → `nano`, harness → `large` |
| L5 | orchestrator/researcher → `large`, drafter/compliance → `small` |

### The local reality: why the examples pin `large`

Measured on this box (environment healthy — `large` scored 3/3 in the same
session):

| Tier | Tool/structured output |
|------|------------------------|
| `large` (qwen2.5:14b) | ✅ reliable (3/3) |
| `small` (llama3.2:3b) | ❌ 0/3 (also refused a one-word classify) |
| `nano` (qwen2.5:0.5b) | ❌ 0/2 |

Small local models simply aren't reliable at tool calling / structured output.
So by default **every agent pins to `large`** (`effective_tier` in `config.py`)
and the examples run reliably. The per-role tiering above is one env var away:

```bash
AGENT_STRICT_TIERS=1 python l5_multi_agent.py   # honor per-role tiers
AGENT_PINNED_TIER=small python l2_prompt_chains.py  # pin everything to small instead
```

On a **hosted provider** (where small models tool-call fine) or with more capable
local models, flip `AGENT_STRICT_TIERS=1` and the tiering becomes a real cost
saver — a few `large` calls plus several `small`/`nano` calls instead of `large`
everywhere. The deterministic test suite pins the *intent* (each role's requested
tier) and the *policy* (default pins `large`) so neither can drift silently.

## Running the examples

Each file is runnable (`python lN_*.py`) and importable (the test suite imports
the agents and overrides the model). Levels 3–5 print a step-by-step **agent
trace** so you can see exactly which tools were called and in what order.

Sample captured output from a real local run is in
[`TEST_OUTPUT.md`](TEST_OUTPUT.md).

## Debuggability & observability (L1→L5)

As you climb the ladder, a run goes from *one* model call to *twenty* across
several agents — and "it was slow / wrong" becomes impossible to debug by
eyeballing the final output. Each level here ships with a way to see inside.

**Built-in, per level (no setup):**

| Level | What you can inspect | How |
|-------|----------------------|-----|
| L1 | the structured result; validation retries on failure | `print(result)`; `capture_run_messages()` |
| L2 | the routing decision (which handler ran) + escalation | it prints `Classified as: …`; assert in `tests/test_l2_*` |
| L3 | the full tool-call → tool-return → final sequence | `print_agent_trace()` (`utils.py`) |
| L4 | which files it read, gateway checks, refund, + triage tier | `print_agent_trace()`; the `[triage · nano]` line |
| L5 | delegation order, per-agent work, aggregated usage | delegation tool calls in `result.all_messages()`; `result.usage` |

Three tools do most of the work:

- **`print_agent_trace(result)`** (`utils.py`) — walks `result.all_messages()`
  and prints every tool call, tool return, and the final text in order. This is
  your first stop for "what did the agent actually do?" (L3–L5).
- **`result.usage`** — `RunUsage(input_tokens, output_tokens, requests,
  tool_calls)`. For L5, delegation rolls sub-agent usage into one total via
  `usage=ctx.usage`, so you see the true cost of a whole case.
- **`capture_run_messages()`** — wrap a run to get the exact request/response
  history that led to a failure (e.g. the "Exceeded max output retries" you hit
  when a local model wraps JSON in prose). Best for debugging one bad run.

**Performance observability:** the live suite times every level (the `latency`
fixture) and `benchmark.py` reports p50/p95/p99 — see [Testing](#testing) and
[`LATENCY.md`](LATENCY.md).

### Viewing Pydantic AI Logfire traces

For the full picture — a span tree of every agent run, model request, and tool
call, with token counts and timings — use **Logfire**. Pydantic AI has
first-class Logfire support; this repo wires it in behind one env var
(`observability.py`, a no-op unless enabled, so tests/default runs are
unaffected).

**Where you actually see the traces — two places:**

**1. Your terminal (default, no signup).** With `AGENT_LOGFIRE=1`, spans print to
**stdout** right where you ran the command — nothing leaves your machine:

```bash
AGENT_LOGFIRE=1 python l3_tool_calling_agent.py
```
```
[observability] Logfire tracing enabled.
15:03:31.240 billing_agent run
15:03:31.243   chat qwen2.5:14b
15:03:35.867   running tool: get_recent_charges
15:03:40.814   running tool: check_refund_policy
15:03:40.815   running tool: issue_refund
15:03:40.818   chat qwen2.5:14b
```

**2. The Logfire web app (rich UI, browse/filter/expand).** A one-time login,
then traces stream to your project dashboard in the browser:

```bash
uv run logfire auth        # one-time: opens a browser, writes local creds
AGENT_LOGFIRE=1 python l5_multi_agent.py
```
On startup Logfire prints your project URL, e.g.
`https://logfire.pydantic.dev/<you>/<project>` — **open that link** to see the
nested span tree, click any span to expand it, and read per-span **token counts
and latency**. (Alternatively set `LOGFIRE_TOKEN=...` instead of `logfire auth`,
e.g. in CI.) Without a token it stays console-only.

What the web UI gives you that the console doesn't: **per-span token counts and
latency**, so you can see *which sub-agent* and *which tool call* dominated a
slow/expensive L5 run — the tail-latency and token-amplification effects from the
[Level 5 deep-dive](#level-5-deep-dive-multi-agent-system-design). For L5 the tree
nests: `orchestrator ▸ research ▸ (researcher ▸ list_files / read_file /
check_payment_gateway) ▸ draft_response ▸ review_compliance ▸ final_result`.

Under the hood `enable_logfire()` calls:

```python
import logfire
logfire.configure(send_to_logfire="if-token-present")  # console-only unless you auth
logfire.instrument_pydantic_ai()      # agent/model/tool spans
logfire.instrument_httpx(capture_all=True)  # exact payloads to/from Ollama
```

## Testing

Two layers, by design:

1. **Deterministic suite (default, always green, no network).** Uses Pydantic
   AI's `TestModel` and `FunctionModel` with `agent.override(...)` to test the
   *wiring and logic* — tools registered, routing correct, delegation happens,
   schemas hold, the path-traversal guard fires — with zero model calls.

   ```bash
   python -m pytest -q            # 36 tests, ~2s
   ```

2. **Live Ollama suite (opt-in, slow, non-deterministic).** Actually hits the
   local model and asserts *well-typed, plausible* results.

   ```bash
   RUN_OLLAMA=1 python -m pytest tests/test_live_ollama.py -v
   # or
   python -m pytest --run-ollama -v
   ```

Why the split? Small local models are non-deterministic — great for a demo,
unfit as a CI gate. The deterministic suite is the gate; the live suite is proof
the examples really run on Ollama. See [`tests/README.md`](tests/README.md).

### Latency measurement

The live tests are each wrapped in a `latency` fixture: every level is timed and
a per-level table (p50/p95/p99) is printed at the end of the run and saved to
`.sample_runs/latency_tests.txt`.

```bash
RUN_OLLAMA=1 python -m pytest tests/test_live_ollama.py -v -s   # times all 5 levels
```

For real percentiles, use the dedicated benchmark, which runs each level many
times and writes [`LATENCY.md`](LATENCY.md):

```bash
python benchmark.py                 # tiered default run counts
python benchmark.py --runs 10       # 10 runs per level (slow but tighter numbers)
python benchmark.py --levels 1,3    # subset
```

See the [Level 5 deep-dive](#level-5-deep-dive-multi-agent-system-design) for how
to read these numbers and why latency climbs with complexity.

## Test report

Latest run on this machine (Pydantic AI 1.107.0, Python 3.13, Ollama
`qwen2.5:14b`, `temperature=0`). Full captured output — including per-level agent
traces — is in [`TEST_OUTPUT.md`](TEST_OUTPUT.md).

**Static checks**

| Check | Command | Result |
|-------|---------|:------:|
| Lint | `ruff check .` | ✅ All checks passed |
| Types | `mypy *.py` | ✅ no issues (10 source files) |

**Deterministic suite** (no model, no network — the CI gate):

```
$ python -m pytest -q
36 passed, 5 skipped, 1 warning in ~2s     # 41 collected; 5 skipped = live tests
```

| Test file | Tests | Covers |
|-----------|:-----:|--------|
| `test_config_and_kb_tools.py` | 18 | config tiers + sandboxed fs/billing runtime + path-traversal guard |
| `test_l1_augmented_llm.py` | 2 | single-call structured output, no tools |
| `test_l2_prompt_chains.py` | 6 | deterministic routing to the correct handler |
| `test_l3_tool_calling_agent.py` | 3 | tool wiring + scripted refund flow |
| `test_l4_agent_harness.py` | 3 | runtime tools + real KB investigation + sandbox guard |
| `test_l5_multi_agent.py` | 4 | orchestrator delegates to every specialist; shared usage |
| **Total** | **36** | + 5 live tests (skipped by default) |

**Live Ollama suite** (all five levels, real local model) — ✅ 5/5 pass on an idle
GPU. Observed end-to-end latency (single sample per level, `qwen2.5:14b`):

| Level | Observed latency | Model calls | Notes |
|-------|:---:|:---:|-------|
| L1 Augmented | ~1–3s | 1 | deterministic at temp 0 |
| L2 Chains | ~4–8s | 2 (fixed) | classify + handle |
| L3 Tool-Calling | ~10–25s | 3–7 | model sequences tools |
| L4 Harness | ~30–70s | 6–14 | explores the knowledge base |
| L5 Multi-Agent | ~90–180s | 10–20+ | 11 reqs / 9 tool calls in one run |

Percentile breakdowns are in [`LATENCY.md`](LATENCY.md) (`python benchmark.py`);
the [Level 5 deep-dive](#level-5-deep-dive-multi-agent-system-design) explains why
latency and cost climb with complexity.

## Project structure

```
agent_complexity/
├── README.md                     # you are here
├── TEST_OUTPUT.md                # captured test + live-run output
├── LATENCY.md                    # per-level latency benchmark (generated)
├── config.py                     # Ollama model tiers + temperature (one place)
├── utils.py                      # print_agent_trace()
├── benchmark.py                  # per-level latency benchmark (p50/p95/p99)
├── kb_tools.py                   # sandboxed fs + fake billing API (shared by L4/L5)
├── l1_augmented_llm.py           # Level 1
├── l2_prompt_chains.py           # Level 2
├── l3_tool_calling_agent.py      # Level 3
├── l4_agent_harness.py           # Level 4
├── l5_multi_agent.py             # Level 5
├── pytest.ini                    # standalone test config
├── knowledge/                    # sample corpus the agents read (L4/L5)
│   ├── customers/                # customer profiles
│   ├── policies/                 # refund policy, escalation matrix, subscriptions
│   └── templates/                # response templates
└── tests/                        # deterministic + live test suites
```

## Adapting from the Claude Agent SDK to Pydantic AI

The upstream cookbook builds Levels 4–5 on the Claude Agent SDK, which is
Anthropic-only. This port keeps the *shape* of each level but swaps the
mechanism so everything runs on a local model:

| Cookbook (Claude Agent SDK) | Here (Pydantic AI + Ollama) |
|-----------------------------|------------------------------|
| `Agent('anthropic:claude-…')` | `get_model()` → `OpenAIChatModel` + `OllamaProvider` |
| SDK built-in `Read`/`Glob`/`Grep` | `@agent.tool` wrappers over `kb_tools` (sandboxed) |
| `create_sdk_mcp_server(...)` tools | plain `@agent.tool` functions |
| Subagents via `AgentDefinition` + `Task` | **agent delegation**: orchestrator tools call sub-`Agent`s with `usage=ctx.usage` |
| `output_format={json_schema}` | `output_type=SomeBaseModel` |

The cookbook README itself names this alternative: *"Passed-down agents (e.g.
PydanticAI, LangGraph): you wire agents together in code — passing outputs from
one to the next, sharing dependencies, or nesting agent calls."* Level 5 is
exactly that.

## Notes on local models

- **Structured output can be flaky on small models.** Two mitigations are baked
  in: `temperature=0` (in `config.py`) and `retries=3` on the agents, which
  feeds the validation error back so the model self-corrects.
- **Bigger = more reliable for tools.** `qwen2.5:14b` (the `large` default)
  handles tool calling and structured output much more reliably than
  `llama3.2:3b`. Drop to `small` for speed when you don't need the reliability.
- **This flakiness is itself a lesson.** The higher the level, the more model
  calls must each succeed — which is why reliability drops as you climb the
  ladder, and why "use the simplest level that works" is the whole point.
