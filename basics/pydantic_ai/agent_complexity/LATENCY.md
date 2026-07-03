# Latency benchmark

## Table of Contents

- [Setup](#setup)
- [Results](#results)
- [Reading these numbers](#reading-these-numbers)

## Setup

- Model: `qwen2.5:14b` via local Ollama, `temperature=0`
- Machine: local single-GPU (your hardware will differ)
- Latency = wall-clock for one full end-to-end run of the level.

These are representative **healthy single-run observations** plus the one
measured L5 token count; percentile columns show the observed sample (so
p50=p95=p99 at n=1). Regenerate proper percentiles on an **idle** GPU with:

```bash
python benchmark.py                 # tiered defaults
python benchmark.py --runs 10       # tighter percentiles (slow)
```

> Note: running the benchmark while another Ollama job is in flight — or
> immediately after killing one — degrades results (the 9GB model reloads and
> early calls hang or return empty). Run it on a quiet GPU.

## Results

| Level | model calls | tokens/run (in+out) | latency (obs) | notes |
|-------|:---:|:---:|:---:|-------|
| L1 Augmented LLM | 1 | ~0.2k | ~2s | deterministic at temp 0 |
| L2 Prompt Chains | 2 | ~0.5k | ~6s | classify + handle |
| L3 Tool-Calling | 3–7 | ~3k | ~15s | model sequences tools |
| L4 Agent Harness | 6–14 | ~8k | ~50s | explores the knowledge base |
| L5 Multi-Agent | 10–20+ | ~8–12k (measured 6291+1985) | ~140s | 11 reqs / 9 tool calls in one run |

## Reading these numbers

- Percentiles from a handful of samples are **indicative**; raise the run
  count for tighter numbers.
- Latency scales with the number of *sequential* model calls a level makes,
  not the amount of code. That is why Level 5 is slowest even though each
  sub-agent could parallelize — see the Level 5 deep-dive in `README.md`.
- Local single-GPU Ollama **serializes** concurrent model calls, so
  `asyncio.gather` across sub-agents does not reduce wall-clock here.
- Token cost grows super-linearly: every agent turn re-sends the accumulated
  context, so later turns in an agentic run are the most expensive. See the
  "Cost & latency at a glance" table in `README.md`.
