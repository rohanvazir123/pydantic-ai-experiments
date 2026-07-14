"""
Latency benchmark for the five agent-complexity levels (live, against Ollama).

Runs each level end to end N times, measures wall-clock latency, and reports
min / p50 / p95 / p99 / max plus (for the agentic levels) request and tool-call
counts. Writes a Markdown table to LATENCY.md.

    python benchmark.py                      # tiered default run counts
    python benchmark.py --runs 10            # 10 runs for every level
    python benchmark.py --levels 1,2,3       # only these levels
    python benchmark.py --runs 3 --levels 5  # e.g. 3 runs of the multi-agent level

Percentiles from small samples are indicative, not authoritative — bump --runs
for tighter numbers (at the cost of a much longer benchmark). Latency is
dominated by local GPU throughput and how many sequential model calls a level
makes; see README.md ("Notes on local models" and the Level 5 deep-dive).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import l1_augmented_llm as l1
import l2_prompt_chains as l2
import l3_tool_calling_agent as l3
import l4_agent_harness as l4
import l5_multi_agent as l5
from config import DEFAULT_TIER, MODEL_TIERS

if TYPE_CHECKING:
    from collections.abc import Callable

_TICKET = "I was charged twice on Feb 1st for my subscription. Order #12345. Please fix this."
_CASE = (
    "Customer cust_12345 reports a duplicate charge on their February bill. "
    "Investigate using the knowledge base, verify, and resolve per policy."
)


# --- One callable per level; returns (requests, tool_calls) when available ---


def _run_l1() -> tuple[int, int]:
    l1.classify(_TICKET)
    return (1, 0)


def _run_l2() -> tuple[int, int]:
    l2.process_ticket(_TICKET)
    return (2, 0)  # classify + handle (fixed 2 calls)


def _run_l3() -> tuple[int, int]:
    r = asyncio.run(l3.resolve(_TICKET, l3._sample_deps()))
    return (r.usage.requests, r.usage.tool_calls)


def _run_l4() -> tuple[int, int]:
    r = asyncio.run(l4.run_harness(_CASE))
    return (r.usage.requests, r.usage.tool_calls)


def _run_l5() -> tuple[int, int]:
    case = l5.CaseInput(customer_id="cust_12345", issue=_CASE)
    state = l5.CaseState(case=case)
    deps = l5.CaseDeps(root=l5.KNOWLEDGE_DIR)
    asyncio.run(l5.Orchestrator(state=state, deps=deps).run())
    return (state.usage.requests, state.usage.tool_calls)


@dataclass
class Level:
    key: int
    label: str
    run: Callable[[], tuple[int, int]]
    default_runs: int


LEVELS: dict[int, Level] = {
    1: Level(1, "L1 Augmented LLM", _run_l1, 8),
    2: Level(2, "L2 Prompt Chains", _run_l2, 6),
    3: Level(3, "L3 Tool-Calling", _run_l3, 5),
    4: Level(4, "L4 Agent Harness", _run_l4, 3),
    5: Level(5, "L5 Multi-Agent", _run_l5, 2),
}


@dataclass
class Result:
    label: str
    latencies: list[float] = field(default_factory=list)
    requests: list[int] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    errors: int = 0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
    return ordered[k]


def _measure(level: Level, runs: int) -> Result:
    res = Result(label=level.label)
    for i in range(runs):
        start = time.perf_counter()
        try:
            requests, tool_calls = level.run()
            elapsed = time.perf_counter() - start
            res.latencies.append(elapsed)
            res.requests.append(requests)
            res.tool_calls.append(tool_calls)
            print(f"  {level.label}  run {i + 1}/{runs}: {elapsed:6.2f}s  "
                  f"({requests} reqs, {tool_calls} tool calls)")
        except Exception as e:  # a flaky run shouldn't abort the whole benchmark
            res.errors += 1
            print(f"  {level.label}  run {i + 1}/{runs}: ERROR {type(e).__name__}: {str(e)[:80]}")
    return res


def _render(results: list[Result], model: str) -> str:
    lines = [
        "# Latency benchmark",
        "",
        "## Table of Contents",
        "",
        "- [Setup](#setup)",
        "- [Results](#results)",
        "- [Reading these numbers](#reading-these-numbers)",
        "",
        "## Setup",
        "",
        f"- Model: `{model}` (via local Ollama)",
        "- Machine: local single-GPU (your hardware will differ)",
        "- Latency = wall-clock for one full end-to-end run of the level.",
        "",
        "## Results",
        "",
        "| Level | n | p50 | p95 | p99 | max | reqs (p50) | tool calls (p50) | errors |",
        "|-------|--:|----:|----:|----:|----:|-----------:|-----------------:|-------:|",
    ]
    for r in results:
        if not r.latencies:
            lines.append(f"| {r.label} | 0 | — | — | — | — | — | — | {r.errors} |")
            continue
        reqs = int(statistics.median(r.requests)) if r.requests else 0
        tcs = int(statistics.median(r.tool_calls)) if r.tool_calls else 0
        lines.append(
            f"| {r.label} | {len(r.latencies)} | "
            f"{statistics.median(r.latencies):.1f}s | {_pct(r.latencies, 95):.1f}s | "
            f"{_pct(r.latencies, 99):.1f}s | {max(r.latencies):.1f}s | "
            f"{reqs} | {tcs} | {r.errors} |"
        )
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- Percentiles from a handful of samples are **indicative**; raise the run",
        "  count for tighter numbers.",
        "- Latency scales with the number of *sequential* model calls a level makes,",
        "  not the amount of code. That is why Level 5 is slowest even though each",
        "  sub-agent could parallelize — see the Level 5 deep-dive in `README.md`.",
        "- Local single-GPU Ollama **serializes** concurrent model calls, so",
        "  `asyncio.gather` across sub-agents does not reduce wall-clock here.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=None, help="runs per level (overrides tiered defaults)")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5", help="comma-separated level numbers")
    args = parser.parse_args()

    selected = [LEVELS[int(x)] for x in args.levels.split(",") if x.strip()]
    model = MODEL_TIERS.get(DEFAULT_TIER, DEFAULT_TIER)
    print(f"Benchmarking on model '{model}' (tier '{DEFAULT_TIER}')\n")

    results: list[Result] = []
    for level in selected:
        runs = args.runs if args.runs is not None else level.default_runs
        print(f"== {level.label} ({runs} runs) ==")
        results.append(_measure(level, runs))
        print()

    report = _render(results, model)
    out = Path(__file__).parent / "LATENCY.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
