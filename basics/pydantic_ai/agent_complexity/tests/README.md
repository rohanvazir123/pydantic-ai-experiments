# Tests — agent-complexity examples

Test suite for the five agent-complexity examples. Two layers: a fast
deterministic suite that is the CI gate, and an opt-in live suite that proves
the examples run on a real Ollama model.

## Table of Contents

- [Layout](#layout)
- [Running](#running)
- [How the deterministic tests work](#how-the-deterministic-tests-work)
- [The live suite](#the-live-suite)
- [Latency measurement](#latency-measurement)

## Layout

| File | Covers | Needs Ollama? |
|------|--------|:---:|
| `test_config_and_kb_tools.py` | model config + the sandboxed fs / billing runtime (incl. path-traversal guard) | no |
| `test_l1_augmented_llm.py` | single-call structured output, no tools | no |
| `test_l2_prompt_chains.py` | deterministic routing to the correct handler | no |
| `test_l3_tool_calling_agent.py` | tool wiring + a scripted refund flow | no |
| `test_l4_agent_harness.py` | runtime tools + a real KB investigation + sandbox guard | no |
| `test_l5_multi_agent.py` | orchestrator delegates to every specialist; usage is shared | no |
| `test_live_ollama.py` | Levels 1 & 3 against the real local model | **yes** |
| `conftest.py` | adds the example dir to `sys.path`; defines the `ollama` marker/gate | — |

## Running

```bash
cd basics/pydantic_ai/agent_complexity

python -m pytest -q                       # deterministic suite (default)
python -m pytest -v                       # verbose
RUN_OLLAMA=1 python -m pytest -v          # include live tests
python -m pytest --run-ollama -v          # same, via flag
```

The deterministic suite makes **no** model or network calls and runs in ~2s.

## How the deterministic tests work

Local models are non-deterministic, so we never assert on real model output in
the gate. Instead we swap the model with Pydantic AI test doubles via
`agent.override(...)`:

- **`TestModel`** — auto-generates valid output and (by default) calls every
  registered tool once. Perfect for asserting *wiring*: which tools exist, that
  the output schema holds. We use `TestModel(call_tools=[])` where a real tool
  would reject `TestModel`'s synthetic arguments (e.g. `read_file`).
- **`FunctionModel`** — you write the exact response sequence. We use it to
  script realistic flows (classify → route, investigate → refund, delegate →
  synthesize) and assert the *logic* deterministically.

Because the example modules build their agents at import time (with the Ollama
model, which does no network I/O until called), tests import the real agents and
only replace the model — so the wiring under test is the wiring that ships.

## The live suite

`test_live_ollama.py` is marked `@pytest.mark.ollama` and skipped unless
`RUN_OLLAMA=1` or `--run-ollama` is passed. It first pings the Ollama daemon and
skips (not fails) if it is unreachable. Assertions are deliberately loose —
well-typed and plausible, not exact — because small local models vary run to run.
It covers all five levels (L1–L5).

## Latency measurement

Each live test is wrapped in the `latency` fixture (defined in `conftest.py`) and
tagged with `@pytest.mark.level("…")`. The fixture times the test body; at the
end of the session `pytest_terminal_summary` prints a per-level p50/p95/p99 table
and writes it to `.sample_runs/latency_tests.txt`.

```bash
RUN_OLLAMA=1 python -m pytest tests/test_live_ollama.py -v -s
```

A single pass gives one sample per level (so p50 = p95 = p99). For real
percentiles, repeat the run or use the dedicated `../benchmark.py`, which runs
each level multiple times and writes `../LATENCY.md`.

To trace *what happened inside* a run (span tree, tool calls, per-span tokens),
set `AGENT_LOGFIRE=1` — spans print to the terminal, and to the Logfire web app
(`https://logfire.pydantic.dev`) after `uv run logfire auth`. See the main
README's [observability section](../README.md#viewing-pydantic-ai-logfire-traces).
