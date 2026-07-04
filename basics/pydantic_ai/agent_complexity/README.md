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
  - [Autonomy scale](#autonomy-scale)
  - [Decision guide](#decision-guide)
  - [Level-by-level: when to use, when not to](#level-by-level-when-to-use-when-not-to)
- [Cost & latency at a glance (L1→L5)](#cost--latency-at-a-glance-l1l5)
  - [What a real ticket costs in dollars](#what-a-real-ticket-costs-in-dollars)
- [Level 5 deep-dive: multi-agent system design](#level-5-deep-dive-multi-agent-system-design)
  - [Cost & latency: what's realistic (p50/p95/p99)](#cost--latency-whats-realistic-p50p95p99)
  - [Why multi-agent is slow *despite* parallelism](#why-multi-agent-is-slow-despite-parallelism)
  - [Resiliency & self-healing](#resiliency--self-healing)
  - [Other system-design trade-offs](#other-system-design-trade-offs)
  - [When NOT to go multi-agent](#when-not-to-go-multi-agent)
- [Why multi-agent systems fail in production](#why-multi-agent-systems-fail-in-production)
  - [Reliability & error propagation](#reliability--error-propagation)
  - [Context & memory](#context--memory)
  - [Cost & resource runaway](#cost--resource-runaway)
  - [Observability & debugging](#observability--debugging)
  - [Engineering & design failures](#engineering--design-failures)
- [Mitigating production failures](#mitigating-production-failures)
  - [Make every side effect idempotent](#make-every-side-effect-idempotent)
  - [Bound everything](#bound-everything)
  - [Validate at every boundary](#validate-at-every-boundary)
  - [Design context deliberately](#design-context-deliberately)
  - [Progressive tool disclosure](#progressive-tool-disclosure)
    - [1. Phase-gated tool sets](#1-phase-gated-tool-sets)
    - [2. Dynamic tool injection via `prepare`](#2-dynamic-tool-injection-via-prepare)
    - [3. Sub-agent specialization](#3-sub-agent-specialization)
  - [Instrument before you ship](#instrument-before-you-ship)
  - [Build the degraded path first](#build-the-degraded-path-first)
  - [Keep humans in the loop for high-risk actions](#keep-humans-in-the-loop-for-high-risk-actions)
  - [Test failure modes, not just the happy path](#test-failure-modes-not-just-the-happy-path)
  - [The one-slide summary](#the-one-slide-summary)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Model tiers (tiered LLMs)](#model-tiers-tiered-llms)
  - [Models & hardware per level (L1→L5)](#models--hardware-per-level-l1l5)
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

The levels differ on two axes that matter most in practice:

1. **Who controls flow** — code (predictable, auditable) vs model (flexible, non-deterministic)
2. **Who controls sequencing** — a fixed DAG in your code, or a model that decides what to call next

Note: *agent count is not the distinguishing axis.* L2 already uses multiple agents —
a classifier agent, one or more handler agents, a validator. What makes it L2 is that
**your code decides which agent runs next**, not a model. The jump to L5 is not "add
more agents" — it is "let a model orchestrate other models."

| | L1 Augmented | L2 Chains | L3 Tool-Calling | L4 Harness | L5 Multi-Agent |
|---|---|---|---|---|---|
| **Agents involved** | 1 | Multiple | 1 | 1 | Multiple |
| **Who controls flow** | Code (1 call, done) | Code (fixed DAG) | Model (bounded loop) | Model (open loop) | Orchestrator model |
| **What model controls** | Content only | Content of each step | Which tools to call & when | Which files/APIs to explore & how | Which specialists to delegate to & how to synthesize |
| **Steps known in advance** | Yes — exactly 1 | Yes — by you | No — model decides | No — model explores | No — model decomposes |
| **Tools per agent** | None | None (agents are the units) | Fixed, bounded set | Broad, open-ended set | Scoped per specialist |
| **Autonomy** | None | None | Partial | High | Full |
| **Cost** | $ | $ | $$ | $$$ | $$$$ |
| **Latency** | 1 call | N fixed calls | 3–7 calls | 6–14 calls | 10–20+ calls across agents |
| **Reliability** | Deterministic\* | High | High | Medium | Lower |
| **Best when** | Answer is in the input | Stages are known and auditable | A bounded set of actions covers it | Open-ended exploration needed | Distinct expert roles must work in parallel |

\* *Deterministic in shape (always one call, one schema); the model's content
still varies unless you pin `temperature=0`, which these examples do.*

### Autonomy scale

| Level | Autonomy | What the model decides on its own |
|-------|----------|-----------------------------------|
| L1 | None | Nothing — one call, code drives the whole interaction |
| L2 | None | Content of each step only; code decides which step runs |
| L3 | Partial | Which tools to call and in what order, within a fixed bounded set |
| L4 | High | Which files/APIs to open, what to look for, when to stop — open-ended exploration |
| L5 | **Full** | How to decompose the task, which specialists to delegate to, how to synthesize their results — a model orchestrating models |

L5 is the only fully autonomous level: the orchestrator makes meta-decisions (who
should do this? is the result good enough? do I need another delegation?) that are
themselves model calls. L4 is highly autonomous within a single agent's scope; L5
extends that autonomy to the coordination layer. More autonomy = more capability and
more ways to go wrong — which is why the golden rule is to stay at the lowest level
that actually needs it.

### Decision guide

The real question at each fork is not "how many agents?" but "who should be in
charge of this decision?"

```
Does the answer come entirely from the input — no lookups, no actions?
│
├─ YES ──────────────────────────────────────────────────────▶ L1 Augmented LLM
│         One call. Model controls content only. You control everything else.
│
└─ NO ─ Are the stages fixed and enumerable in advance?
        Can you write: classify → route → handle → validate in code?
        │
        ├─ YES ──────────────────────────────────────────────▶ L2 Prompt Chains
        │         Multiple single-purpose agents. Code controls which runs next.
        │         Model controls the content of each stage. Fully auditable.
        │         (The classifier, each handler, and the validator are all
        │         separate agents — L2 is already multi-agent. The difference
        │         from L5 is that *your code* is the orchestrator.)
        │
        └─ NO ─ Can a small, fixed set of tools cover all the actions needed?
                Does the model need to decide *which* tools to call and in what order,
                but the tool set itself won't change run-to-run?
                │
                ├─ YES ──────────────────────────────────────▶ L3 Tool-Calling Agent
                │         One agent, bounded tool set. Model controls sequencing.
                │         Code controls what the tools can do. The tool set is the guardrail.
                │
                └─ NO ─ Does it need open-ended exploration — reading files,
                        grepping logs, calling APIs the model discovers at runtime?
                        You cannot enumerate the steps in advance.
                        │
                        ├─ YES ─ Can one agent hold all the expertise?  ──▶ L4 Harness
                        │         One agent with a broad runtime. Model explores freely.
                        │         This is the shape of coding agents (Claude Code, Cursor).
                        │
                        └─ NO ─ Do the sub-tasks need genuinely distinct
                                instructions, tool sets, or conflicting constraints
                                that can't coexist in one system prompt?
                                Do they benefit from running in parallel?
                                │
                                └─ YES ─────────────────────────────────▶ L5 Multi-Agent
                                          An orchestrator model delegates to specialists,
                                          each with their own prompt + tools. The model
                                          controls decomposition and synthesis.
```

**Golden rule: start at the lowest level that works and only climb when it
demonstrably can't do the job.** Each climb multiplies cost, latency, and the
number of ways a run can go wrong. If the "distinct roles" in your L5 design are
really just steps, that's L2. If they're just tools, that's L3.

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

### What a real ticket costs in dollars

The table above uses this repo's *toy* token counts (tiny KB, terse prompts). A
**production** multi-agent ticket — real system prompts + retrieved context +
longer tool outputs — runs **~100–150k tokens** per resolution (see the
[Level 5 deep-dive](#cost--latency-whats-realistic-p50p95p99)). That split is
heavily input-weighted (~85% input, because each of the ~12–20 calls re-sends the
accumulated context), i.e. ~85–130k input + ~15–22k output. In dollars:

| Pricing tier (in / out per 1M) | 100k tokens | 150k tokens |
|---|--:|--:|
| Cheap / small (~$0.30 / $1.20) | ~$0.04 | ~$0.07 |
| Mid-tier (~$3 / $15) | ~$0.48 | ~$0.72 |
| Frontier / all-large (~$10 / $40) | ~$1.45 | ~$2.15 |
| Local Ollama (this repo) | ~$0 | ~$0 |

Formula: `cost = input_tok × in_price + output_tok × out_price`. So budget
**~$0.50–$0.75 per ticket at mid-tier**, ranging ~$0.05 (cheap/tiered) to ~$2
(all-frontier). At scale that's ~**$500–750 per 1,000 tickets** mid-tier.

**Two levers cut it 5–10×:**

- **Prompt caching.** ~85% of the tokens are the *same context re-sent every
  turn*; cached input reads bill at ~10% of full price, dropping a mid-tier
  ticket from ~$0.60 to **~$0.20–0.35**. Biggest single win for multi-agent.
- **Tiering** (see [Model tiers](#model-tiers-tiered-llms)). Route the many cheap
  sub-calls (draft, check, triage) to a small model, reserve the large model for
  reasoning/tools — **½ to ⅓** of all-large cost.

**Is 100–150k/ticket reasonable?** Per token, yes — it's just ~15 context-heavy
calls; it's large *for one query* only because "3 delegations" fans out into many
re-sending calls. Economically, easily: a human support agent runs ~$5–12 per
handled ticket, so even the $2 all-frontier case is 3–6× cheaper, and the
tiered+cached ~$0.25 case is ~20–50× cheaper. **The catch:** that math only holds
if the run actually *resolves* (or safely deflects) the ticket — a 150k-token run
that then escalates to a human is pure overhead. That's why the golden rule is
"use the lowest level that works" and keep human-in-the-loop gates for high-risk
actions.

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

## Why multi-agent systems fail in production

This is the section that follows naturally from "when NOT to go multi-agent." You
decided to go multi-agent anyway — here is what bites teams in production, and why.

### Reliability & error propagation

**Multiplicative failure is the default math.** If each model call succeeds 97% of
the time, a 20-call L5 run completes without a single error only ~54% of the time
(0.97²⁰). Every extra agent or tool added multiplies the failure surface. Most teams
build multi-agent systems assuming near-100% per-call reliability without accounting
for the compound. The fix: retry budgets, graceful degradation, and explicit failure
policies per stage — not optimism.

**Retries without idempotency cause double side effects.** When a tool call fails
mid-run (network blip, timeout, validation error) and the agent retries, it may
re-execute an action that already completed. A refund already issued gets issued
again; a message already sent gets sent twice. Idempotency keys (ticket ID + charge
ID, request fingerprint) on every side-effectful tool call are not optional. This is
the single most dangerous failure mode because it is silent — the agent "succeeds"
and nothing looks wrong until the customer complains.

**Validation errors cascade.** One sub-agent returning a malformed response (wrong
JSON, missing field, unexpected schema) breaks every downstream agent that depends on
it. Without per-agent output validation and an explicit partial-failure policy
(required vs. optional stages), one specialist's bad output kills the whole run. The
orchestrator's error message is typically unhelpful ("expected field X") while the
real root cause is three levels deeper.

**Tool argument hallucination.** Agents — especially smaller models — occasionally
call real tools with plausible-looking but wrong arguments: a customer ID that
doesn't exist, a charge amount that doesn't match the record, a file path outside the
sandbox. Without argument validation at the tool boundary (not just the schema),
these calls silently fail or silently succeed with wrong data. Check inputs at the
tool, not just at the model.

**Unbounded loops exhaust budgets.** An agent that hasn't quite solved the task keeps
looping. Without `max_turns`, a wall-clock budget, and a cost ceiling, one
pathological run can consume tokens equivalent to a thousand normal ones. The first
time you see a $40 single-case bill is usually the last time you forget to add
`max_turns`.

### Context & memory

**Context window overflow mid-run.** Agents accumulate context: system prompt, tool
results, prior model turns, sub-agent findings. A run that starts at 5k tokens can
grow to 50k–150k before it finishes. When the context hits the model's limit, the
run dies mid-task — after you've already paid for everything up to that point.
Mitigations: sliding history windows, trimming tool outputs to summaries after
reading, keeping sub-agent contexts isolated. Most teams discover this at p95, not
p50.

**Context poisoning / instruction drift.** An early tool result (a long, confusingly
formatted file; a sub-agent's verbose summary) overwrites the agent's attention on
the actual task. Later decisions silently diverge. This is especially bad with small
models: the system prompt that worked for a 5-turn run breaks at 15 turns because the
model's "attention" on the instructions fades as context grows. Test long runs, not
just the happy path.

**Shared context causes cross-contamination.** When sub-agents share a context
(delegation via `usage=ctx.usage` passes the shared dependency pool), one agent's
tool results pollute another's reasoning. The researcher's raw findings appear in the
drafter's context even when the drafter only needs the summary. Isolate sub-agent
contexts when roles genuinely conflict; pay the extra token cost for clean
separation.

**Isolated context causes information loss.** The mirror problem: isolating agents
too aggressively means the orchestrator must explicitly re-pack every piece of
context into each delegation call. A researcher finding that is critical for
compliance gets lost because the handoff prompt was too terse. Multi-agent systems
need an explicit information-passing contract at every delegation boundary.

**No persistent memory across runs.** Most agent systems are stateless between runs.
A ticket that spans multiple sessions (customer follows up a day later) restarts from
zero: the agent re-reads every file, re-calls every API, re-derives conclusions the
previous run already reached. Without explicit checkpointing or a memory store, every
retry of a long case starts cold.

### Cost & resource runaway

**Token cost scales with call count, not complexity.** Every model call re-sends the
full accumulated context. A 15-turn run does not cost 15× a single call — the context
grows, so later calls are the most expensive. A 3-delegation L5 run expands to ~10–20
model requests by the time each sub-agent runs its own loop. Teams budget for "3
agents" and get billed for 20 calls worth of tokens.

**Tail runs dwarf median runs.** The median run costs $0.50. The p99 run — where one
sub-agent looped 8 extra times, another timed out and retried, and the orchestrator
re-delegated twice — costs $8. Billing alarms and per-case token budgets are
non-negotiable; the tail, not the median, determines your actual spend at scale.

**Prompt caching misses on dynamic content.** Prompt caching can drop input token
cost to ~10% — but only if the cached prefix is stable across calls. If your system
prompt includes timestamps, request IDs, or dynamic context that changes every call,
the cache never hits. Structure prompts so the stable prefix (instructions, persona,
policy) comes first and dynamic content (the current ticket, tool results) comes
last.

**Runaway fan-out.** An orchestrator that spawns sub-agents in a loop (e.g.,
"research each of these N documents") with no concurrency cap can fan out to N
simultaneous agents. On a hosted API that's N× the per-call cost hitting at once;
internally it can exhaust connection pools or trigger rate limits. Cap fan-out
explicitly.

### Observability & debugging

**"It gave a wrong answer" is undebuggable without traces.** With 20 model calls
across 5 agents, eyeballing the final output tells you nothing about where the
reasoning went wrong. Which sub-agent produced the bad finding? Which tool returned
unexpected data? Which turn caused context drift? Without per-agent, per-tool span
traces (e.g., Logfire), the answer is "re-run it and hope" — which is not a
debugging strategy.

**Flaky failures are hard to reproduce.** Non-determinism at temperature > 0 means
the failure you saw at 14:37 may not reproduce at 14:42. Add structured logging (not
just print statements) to every tool call, capture full `all_messages()` on failure,
and log the random seed or full prompt if you need reproducibility. Without this, you
spend hours chasing a ghost.

**Errors surface at the wrong layer.** The orchestrator raises `ValidationError:
expected field 'refund_id'`. The actual bug: the researcher passed a customer ID that
didn't exist, the payment gateway returned an empty response, the drafter hallucinated
a `refund_id` field that was never populated. Three layers removed from the root
cause, and the error message points to none of them. Instrument at every boundary, not
just the top level.

**Silent successes hide wrong behavior.** The run completed. The structured output
validated. The customer got a reply. And the refund was for $0 because the agent
misread the charge amount. Structural validity and semantic correctness are different
things. Evaluation — either LLM-as-judge or golden-set regression — is necessary
once you care about output quality, not just output shape.

### Engineering & design failures

**Over-engineering is the most common failure.** The team builds a 5-agent
orchestration system for a task that a single tool-calling agent with a good prompt
would handle reliably at 10× lower cost and latency. Every agent added is a new
failure mode, a new prompt to maintain, and a new thing to debug. Most "multi-agent"
designs in production are a Level 3 solution wearing a Level 5 costume. Start at the
lowest level that works.

**Prompt/version coupling across agents.** Five agent prompts that must stay mutually
consistent — the orchestrator's delegation language must match each specialist's
expected input format; the compliance agent's policy references must match the
researcher's output schema. A model upgrade (new version, different tokenizer) can
silently break one prompt while leaving the others intact, and the failure only
appears in production edge cases. Version all prompts together; regression-test them
as a set.

**No human-in-the-loop for high-risk actions.** The agent issues a $500 refund
autonomously because the escalation threshold was set to "$200" in a config file that
nobody updated after a policy change. High-risk actions — above a dollar threshold,
affecting more than N records, touching billing or auth — need an approval gate. Build
the gate into the tool, not into the hope that the model reads the policy correctly
every time.

**Determinism traded away without accounting for it.** Lower-level systems (L1/L2)
are auditable and repeatable. Multi-agent systems are neither, by default. If a
regulator, auditor, or test suite needs a reproducible decision trace, "the LLM
decided" is not a sufficient answer. Push auditable logic (routing rules, thresholds,
required fields) into code rather than prompts. What the model controls should be
limited to what genuinely requires language understanding.

**No fallback when a sub-agent is unavailable.** The compliance agent's model
endpoint is down. The orchestrator raises an unhandled exception and the whole ticket
fails. A $0 cost defensive fallback — queue the case for async human review, drop to
a simpler level (L3), or proceed with a conservative default — often exists; it just
wasn't designed in. Treat sub-agent availability the same way you treat external API
availability: assume it will go down, plan the degraded path.

**Runaway abstraction.** Teams add a "meta-orchestrator" to coordinate the
orchestrators, a "router agent" to decide which orchestration pipeline to use, and a
"result validator agent" to check the orchestrator's output. Each layer is a new
prompt, new failure mode, and new token cost. Complexity compounds. If you find
yourself adding agents to manage agents, the design has escaped its justification —
stop and ask what the actual task requires.

## Mitigating production failures

Each mitigation maps directly to a failure class above. The pattern: **make the
common case cheap and the failure case safe.** None of these are novel; they are the
same reliability engineering principles that apply to distributed systems — applied
to a system where one of the nodes is a language model.

### Make every side effect idempotent

Attach an idempotency key to every tool that writes, charges, sends, or deletes.
Derive it deterministically from inputs the agent already has (ticket ID + action
type + charge ID), not from a random UUID generated inside the tool. On retry, the
tool checks the key, sees the action already completed, and returns the original
result without re-executing. This is the single highest-leverage mitigation for
multi-agent systems that take real actions.

```python
@agent.tool
async def issue_refund(ctx: RunContext[Deps], charge_id: str, amount: float) -> str:
    key = f"{ctx.deps.ticket_id}:refund:{charge_id}"
    if await ctx.deps.gateway.already_processed(key):
        return f"Refund already issued for {charge_id} (idempotent)"
    return await ctx.deps.gateway.refund(charge_id, amount, idempotency_key=key)
```

### Bound everything

Set explicit limits at every layer. Without them, one bad run consumes the resources
of a thousand normal ones.

| Layer | What to bound | How |
|-------|--------------|-----|
| Per agent | tool-calling loop | `Agent(retries=3, max_turns=15)` |
| Per tool | execution time | `asyncio.wait_for(tool_fn(), timeout=10)` |
| Per case | total token spend | check `ctx.usage.total_tokens` inside a tool and raise if over budget |
| Per case | wall-clock time | outer `asyncio.wait_for` wrapping the orchestrator run |
| Per tenant/day | aggregate cost | billing quota checked before each run starts |

Retries should use exponential backoff with jitter for transient failures (429, 503,
network timeouts) and a hard cap on total attempts. Infinite retries turn a blip into
an outage.

### Validate at every boundary

Validation has three distinct layers — do all three:

1. **Schema validation** (Pydantic, automatic): the model's structured output matches
   the declared `output_type`. Pydantic AI handles this and retries on failure.

2. **Semantic validation** (your code): the values make sense given the context —
   the customer ID actually exists, the refund amount is ≤ the original charge, the
   file path is inside the sandbox. Do this inside the tool, not inside the prompt.

3. **Output evaluation** (LLM-as-judge or golden set): the final answer is *correct*,
   not just structurally valid. A `refund_amount=0.0` passes schema validation but is
   wrong. Run a lightweight judge agent on a sample, or maintain a golden-set eval
   that runs on every model upgrade.

Don't rely on prompts alone to enforce any of these — models drift, are upgraded, and
get confused by long contexts. Put invariants in code.

### Design context deliberately

Context mismanagement is the subtlest and most common failure class. A few concrete
rules:

**Trim tool outputs to summaries after reading.** If the researcher reads a 20-page
policy document, inject a 3-sentence summary into the orchestrator's context — not
the full text. Full documents re-sent every turn are the main driver of context
window overflow and late-turn instruction drift.

**Put the stable prefix first.** Instructions, persona, and policy go at the top of
the system prompt (so prompt caching covers them). Dynamic content — the current
ticket, prior tool results — goes at the bottom. Mixing them defeats caching and
makes the stable instructions harder for the model to attend to.

**Isolate contexts when roles conflict.** If two sub-agents have contradictory
instructions (the drafter is told "be empathetic and verbose"; the compliance agent
is told "be terse and legal"), they should not share a context. Isolation costs
tokens but prevents one agent's framing from polluting the other's reasoning.

**Checkpoint long runs.** Persist state after each completed sub-agent stage (to a
DB, Redis, or a durable workflow engine like Temporal). A crash at stage 3 of 5
should resume from stage 3, not restart from zero — paying again for research and
drafting that already succeeded.

### Progressive tool disclosure

Loading every tool into every agent turn is the default — and a context bloat trap.
Each tool definition (name, description, parameter schema) consumes tokens in the
system prompt on *every* model call. An agent with 20 registered tools spends those
tokens whether it ever calls 18 of them or not, and the long tool list competes with
your instructions for the model's attention.

**Progressive disclosure** flips this: start with the minimal tool set, and surface
additional tools only when the agent reaches a stage that needs them.

Three practical patterns in Pydantic AI:

#### 1. Phase-gated tool sets

Run the agent in explicit phases, each with its own restricted tool set. When phase
1 finishes, re-enter the agent with the phase-2 tools added. The model never sees
write tools during the read phase:

```python
from pydantic_ai import Agent

# Phase 1: investigation only — no write tools in scope
investigator = Agent(model, tools=[get_customer, get_charges, check_policy])
finding = await investigator.run(ticket)

# Phase 2: action tools unlocked, scoped to what investigation found
resolver = Agent(model, tools=[issue_refund, send_email])
result = await resolver.run(finding.output)
```

#### 2. Dynamic tool injection via `prepare`

Pydantic AI's `prepare` parameter on `@agent.tool` lets you conditionally include or
exclude a tool on each call based on current context — for example, only offering
`issue_refund` if the investigation phase has already set a `can_refund` flag in
deps:

```python
async def only_if_approved(ctx: RunContext[Deps], tool_def: ToolDefinition):
    return tool_def if ctx.deps.can_refund else None

@agent.tool(prepare=only_if_approved)
async def issue_refund(ctx: RunContext[Deps], charge_id: str, amount: float) -> str:
    ...
```

The tool is invisible to the model — zero tokens — until the condition is met.

#### 3. Sub-agent specialization

In a multi-agent system, keep each sub-agent's tool set to exactly what its role
needs. The researcher gets file + gateway tools; the drafter gets template tools only;
the compliance agent gets policy-lookup tools only. No agent sees tools outside its
domain. This is the natural expression of progressive disclosure at the orchestration
layer: tools are disclosed to the agent whose *role* requires them, not broadcast to
all agents.

**Why this matters at scale.** A 20-tool agent system prompt might spend 800–1500
tokens on tool definitions per call. At 15 calls per L5 run that's 12k–22k tokens of
pure overhead — before any context or reasoning. With phase-gating or `prepare`
filtering, a run that only ever uses 4 tools pays for 4 tools.

The pattern also reduces hallucinated tool calls: a model that can only see 3 tools
relevant to the current phase has a much smaller space of wrong choices than one
staring at 20.

### Instrument before you ship

Without traces, a failed multi-agent run is a black box. Wire observability in during
development, not after the first production incident.

**Minimum viable instrumentation:**
- One structured log line per tool call: agent name, tool name, arguments, result
  summary, latency, token delta.
- `result.all_messages()` captured and stored on any run that ends in error or
  unexpected output.
- `result.usage` totals (input tokens, output tokens, request count) logged per case,
  not just in aggregate.

**Logfire (first-class Pydantic AI support):** `logfire.instrument_pydantic_ai()`
gives you a span tree of every agent run, model request, and tool call with timings
and token counts — including nested sub-agent spans. Enable it with `AGENT_LOGFIRE=1`
(see [Debuggability & observability](#debuggability--observability-l1l5)). The web UI
lets you see exactly which sub-agent and which tool call dominated a slow or expensive
run.

**Alert on tail behavior, not just errors.** A run that completes but takes 5× the
median cost is a canary for a prompt regression or a model that started looping.
Alert on p95 latency and p95 token spend, not only on exceptions.

### Build the degraded path first

Every sub-agent and external dependency will go down. Design the failure path before
the happy path.

**Degradation ladder** (implement in this order):
1. **Retry once** with backoff on the same sub-agent/model.
2. **Fall back to a simpler level** — if the L5 compliance agent is unavailable, run
   the compliance check as a direct L3 tool call against a policy file.
3. **Proceed with a conservative default** — if compliance cannot be reached and the
   refund is under the auto-approve threshold, approve it and flag for async audit.
4. **Escalate to a human** — if none of the above applies, queue the ticket for a
   human agent rather than failing the whole case.

Never let one specialist's outage hard-fail the entire orchestration if a safe partial
result exists. Make the fallback policy explicit in the orchestrator's system prompt
and in code — not in a comment.

**Circuit breakers per dependency:** after N consecutive failures, trip open and fail
fast for a cooldown window instead of letting callers pile up latency on a dead
endpoint. Half-open state probes recovery. See `rag/v2/knowledge/bus` for a
production implementation.

### Keep humans in the loop for high-risk actions

Autonomy should be proportional to confidence and reversibility. Define thresholds in
code:

```python
HUMAN_APPROVAL_THRESHOLD_USD = 100.0

@agent.tool
async def issue_refund(ctx: RunContext[Deps], charge_id: str, amount: float) -> str:
    if amount > HUMAN_APPROVAL_THRESHOLD_USD:
        await ctx.deps.queue_for_human_review(charge_id, amount)
        return f"Refund of ${amount} queued for human approval (exceeds auto-approve threshold)"
    ...
```

Pydantic AI supports deferred / approval-gated tools for this pattern. The gate
belongs in the tool implementation, not in the prompt — prompts drift and are
upgraded; code doesn't change unless you change it.

Keep the threshold in config (not hard-coded), and make it per-action-type: a $200
refund may be fine to auto-approve; a $200 account deletion is not.

### Test failure modes, not just the happy path

The deterministic test suite in this repo uses `TestModel` and `FunctionModel` to
test the *wiring* — routing, delegation, tool registration, schema correctness. Extend
it to cover failure paths:

| Failure to test | How |
|----------------|-----|
| Sub-agent returns malformed output | `FunctionModel` that returns a bad schema |
| Tool raises an exception | tool that raises `ValueError` or `httpx.TimeoutException` |
| Idempotency key prevents double execution | call the tool twice, assert single side effect |
| Refund above threshold queues for review | mock the gateway, assert the human-review queue was called |
| Context overflow gracefully degrades | inject a very long tool result, assert the agent still completes |
| `max_turns` cutoff produces a safe output | configure a very low `max_turns`, assert a fallback result |

Test the degraded path with the same rigour as the success path. Failures in
production are always in the paths you didn't test.

### The one-slide summary

| Failure class | Mitigation |
|---|---|
| Multiplicative reliability failure | Retry budgets + per-call validation |
| Double side effects on retry | Idempotency keys on every write |
| Unbounded loops | `max_turns` + token budget + wall-clock timeout |
| Context window overflow | Trim outputs to summaries; checkpoint long runs |
| Instruction drift in long runs | Stable prefix first; test at p95 turn counts |
| Context bloat from tool definitions | Progressive disclosure — phase-gated or `prepare`-filtered tool sets |
| Undetectable wrong answers | LLM-as-judge eval + golden-set regression |
| Undebuggable failures | Per-tool structured logging + Logfire traces |
| Sub-agent outage kills the run | Degradation ladder: retry → fallback level → human |
| High-risk autonomous actions | Approval gates in tool code, not in prompts |
| Runaway cost at the tail | Per-case token/cost ceiling + billing alerts on p95 |
| Prompt coupling across agents | Version prompts together; eval as a set on model upgrades |
| Over-engineering | Start at L1; only climb when the lower level demonstrably fails |

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

### Models & hardware per level (L1→L5)

**Which models each level's roles need** (local = Ollama; hosted = any Pydantic
AI / OpenAI-compatible provider):

| Level | Role(s) → *ideal* tier | Local (Ollama) | Hosted |
|-------|------------------------|----------------|--------|
| **L1** Augmented | classifier → small | `qwen2.5:14b`† | cheap+fast — Claude Haiku / mini |
| **L2** Chains | classifier → nano; handlers → small | `qwen2.5:14b`† | classifier cheap; handlers mid (Sonnet-class) |
| **L3** Tool-Calling | agent → large | `qwen2.5:14b` | frontier (tool calls) — Sonnet / GPT-class |
| **L4** Harness | triage → nano; harness → large | triage `qwen2.5:14b`†; harness `qwen2.5:14b` | triage cheap; harness frontier |
| **L5** Multi-Agent | orch/researcher → large; drafter/compliance → small | orch/researcher `qwen2.5:14b`; drafter/compliance `qwen2.5:7b`†† (else `14b`) | orch/researcher frontier; drafter/compliance cheap+fast |

† The *ideal* cheap tier isn't reliable locally: `llama3.2:3b` and `qwen2.5:0.5b`
failed tool/structured output here (see the table above). So the pinned default
uses **`qwen2.5:14b` for every role at every level**.
†† A capable ~7B (e.g. `qwen2.5:7b`) is the smallest we'd trust for the text-only
roles; below that, pin `large`.

**The key local takeaway:** because the small tiers don't work, the pinned config
runs **all five levels on one `qwen2.5:14b`** — so the **hardware floor is the same
for L1–L5**. Climbing the ladder costs more *time and energy* (more sequential
model calls), **not more memory**. You only need a *second* resident model when you
turn on real tiering (`AGENT_STRICT_TIERS=1`) for L2 / L4 / L5 with a capable small
model; L1 and L3 use a single tier and never need two.

**Local hardware estimates** (via Ollama):

| Setup (applies to) | GPU / accelerator | VRAM | System RAM | CPU | Disk |
|--------------------|-------------------|:----:|:----------:|:---:|:----:|
| **Pinned large — L1–L5 default** (one 14B) | 1 GPU, or Apple Silicon | **~10–12 GB** (16 comfortable) | 16 GB min / **32 GB rec** | 4+ cores | ~9 GB |
| **Strict tiering — L2/L4/L5** (14B + 7B resident) | 1 big GPU, or Apple Silicon | **~16–20 GB** (24 comfortable) | **32 GB** | 4+ cores | ~14 GB |
| **CPU-only — any level** (no GPU) | — | — | **32 GB+** | 8+ cores (AVX2) | ~9 GB |

Notes:
- **What changes up the ladder:** not the memory floor but the run — L1 is one
  call (~seconds), L5 is 10–20 sequential calls (~minutes locally). Same 14B, same
  VRAM; more wall-clock and energy per case.
- **Concrete GPUs:** RTX 3060 12 GB / 4070 → pin-large (tight); RTX 3090 / 4090
  24 GB → tiering with headroom. Apple Silicon **M1–M4 with 16 GB** unified memory
  runs `qwen2.5:14b` (~10–11 GB used); **32–36 GB+** for two resident models.
  *(This repo was developed on Apple Silicon.)*
- **Unified memory (Apple Silicon):** VRAM = system RAM. 16 GB works for
  pin-large (all levels); 32 GB+ for real tiering (two models loaded at once).
- **A single GPU serializes** and, if two models don't both fit in VRAM, Ollama
  **swaps** them per call — adding seconds each switch. For tiering to actually
  help, keep both models resident (24 GB+ VRAM / 32 GB+ unified) or run two Ollama
  instances / GPUs. See the [Level 5 deep-dive](#why-multi-agent-is-slow-despite-parallelism).
- **CPU-only** runs a 14B but slowly (~a few tok/s → L4/L5 cases take many
  minutes). Fine for a one-off test, not interactive use.

**Hosted (no local hardware):** set `AGENT_STRICT_TIERS=1` and point each tier at
a hosted model — the `large` roles (L3 agent, L4 harness, L5 orchestrator/
researcher) on a frontier reasoning model, the cheap roles (L1/L2 classifier, L4
triage, L5 drafter/compliance) on a fast small model. Hosted small models
tool-call reliably, so tiering genuinely saves cost there. Latest **Claude**
models (Sonnet for the reasoning roles, Haiku for the cheap roles) are a strong
default; any OpenAI-compatible provider works via the model string.

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
