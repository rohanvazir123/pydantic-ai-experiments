# Pydantic AI — Modes of Operation

Reference for the two files in this directory:

| File | What it shows |
|------|--------------|
| `agent_basics.py` | All three single-agent execution modes + FastAPI SSE streaming |
| `multi_agent.py` | Planner → parallel Executors → Synthesizer pipeline for RAG |

---

## 1. Single-Agent Execution Modes

### `agent.run()` — async, full response

```python
result = await agent.run("What are the payment terms?")
print(result.output)          # final answer (str or structured type)
print(result.usage())         # token counts
result.all_messages()         # full message history for multi-turn
```

**When to use:** API handlers, anywhere you `await` and want the complete answer before continuing.

---

### `agent.run_sync()` — synchronous wrapper

```python
result = agent.run_sync("Summarise clause 4.")
print(result.output)
```

Internally calls `asyncio.run()`. **Do not** use inside an existing async context (FastAPI, Jupyter with a running loop) — it will raise a `RuntimeError`. Use `agent.run()` there instead.

**When to use:** CLI scripts, plain Python files, pytest without `pytest-asyncio`.

---

### `agent.iter()` — node-by-node streaming

```python
async with agent.iter("What is the termination notice period?") as run:
    async for node in run:
        if hasattr(node, "stream"):
            async for event in node.stream(run.ctx):
                if isinstance(event, PartDeltaEvent):
                    print(event.delta.content_delta, end="", flush=True)
```

The agent alternates between two node types:

| Node | Description |
|------|-------------|
| `ModelRequestNode` | LLM is called — yields `PartStartEvent`, `PartDeltaEvent`, `FunctionToolCallEvent` |
| `CallToolsNode` | Registered tools execute — yields `FunctionToolResultEvent` |

**When to use:** streaming to a browser via SSE/WebSocket, live token display in a CLI, observability hooks on individual events.

---

## 2. Multi-Agent Architecture

### The pattern

```
User question
      │
      ▼
  PLANNER          result_type=QueryPlan (structured, typed)
      │
      │  asyncio.gather — all sub-queries in parallel
      ▼
  EXECUTOR × N     one agent per sub-query, independent context windows
      │
      ▼
  SYNTHESIZER      merges sub-answers into a single cited response
```

See `multi_agent.py` for a complete runnable implementation with the RAG knowledge base.

---

## 3. When Multi-Agent Clearly Wins

### 3.1 Parallel retrieval over multiple scopes

**Question:** "Compare termination clauses across the Amazon, Google, and Microsoft contracts."

A single agent makes 3 sequential tool calls. With multi-agent, 3 Executor agents run concurrently — each searches its own document scope. Speedup is proportional to the number of sub-queries.

```
Single agent:   500ms × 3 calls = 1500ms
Multi-agent:    max(500ms, 500ms, 500ms) = 500ms   (3× faster)
```

### 3.2 Context window isolation

Each Executor agent receives only the chunks relevant to its sub-query. A single agent accumulates all chunks in one context window — increasing cost, hallucination risk, and the chance of the model confusing clauses across documents.

```
Single agent context:  [Amazon chunks] + [Google chunks] + [Microsoft chunks]
                       = 3× tokens, mixed signals

Executor 1 context:    [Amazon chunks only]
Executor 2 context:    [Google chunks only]
Executor 3 context:    [Microsoft chunks only]
```

### 3.3 Role specialisation

Different agents can use different models for different roles:

```python
planner_agent    = Agent("openai:gpt-4o", result_type=QueryPlan)   # reasoning
executor_agent   = Agent("ollama:llama3.1:8b", ...)                 # fast, cheap retrieval
synthesizer_agent = Agent("openai:gpt-4o", ...)                     # coherent prose
```

### 3.4 Structured intermediate outputs

The Planner returns a typed `QueryPlan` (Pydantic model), not free text. This eliminates brittle string parsing between agents — if the plan is malformed, Pydantic validation raises an error before any executor runs.

### 3.5 Independent retry and fault isolation

If Executor 2 fails, Executors 1 and 3 still complete. You can retry just the failed sub-query. A single-agent failure aborts the entire run.

---

## 4. When Single Agent is Better

| Scenario | Reason |
|----------|--------|
| Simple factual question with one retrieval | Planner overhead exceeds benefit |
| Latency SLA < 200ms | Multi-agent adds 1–2 extra LLM calls (planner + synthesizer) |
| Tight token budget | Planner and synthesizer each consume tokens |
| Linear reasoning chain where step N depends on step N-1 | Parallel execution doesn't help; dependencies force sequential ordering anyway |
| Prototyping / debugging | Single agent is much easier to trace and step through |

---

## 5. Concurrency and Parallelism

### Single agent

Pydantic AI processes `CallToolsNode` tool calls **sequentially by default**. Even if the LLM requests parallel tool calls in one response, the framework executes them one at a time in `CallToolsNode`.

You can add concurrency inside an individual tool using `asyncio.gather`, but the agent itself is single-threaded across its nodes:

```python
@agent.tool
async def search_multiple(ctx, queries: list[str]) -> str:
    # concurrency lives inside the tool, not in the agent loop
    results = await asyncio.gather(*[retrieve(q) for q in queries])
    return format(results)
```

**Summary:** single-agent parallelism = manual `asyncio.gather` inside tools only.

### Multi-agent

True parallel agent execution via `asyncio.gather` at the orchestration layer:

```python
# All three executor agents run concurrently — each is an independent coroutine
sub_answers = await asyncio.gather(
    executor_agent.run(sub_query_1, deps=deps1),
    executor_agent.run(sub_query_2, deps=deps2),
    executor_agent.run(sub_query_3, deps=deps3),
)
```

Each agent run is a fully independent async coroutine:
- its own LLM connection
- its own context window
- its own tool execution loop
- its own `RunContext` / deps — **no shared mutable state**

This is safe because Pydantic AI agent objects are stateless between `.run()` calls. The same `executor_agent` instance can be called concurrently from multiple coroutines without locking.

**Summary:**

| | Single agent | Multi-agent |
|--|-------------|-------------|
| Parallel tool calls | Manual, inside tools | Native via `asyncio.gather` |
| Parallel LLM calls | Not supported | Yes — each agent has its own |
| Shared state between parallel units | N/A | None (each run is isolated) |
| Retry granularity | Whole run | Individual sub-agent |
| Context window isolation | No | Yes — each agent has its own |
| Setup complexity | Low | Medium (planner + executor + synthesizer) |

---

## 6. Message History Between Agents

Pass context from one agent to the next using `result.all_messages()`:

```python
plan_result = await planner_agent.run(question)

# Synthesizer gets the planner's full conversation as prior context
final_result = await synthesizer_agent.run(
    synthesis_prompt,
    message_history=plan_result.all_messages(),
)
```

`all_messages()` returns the complete `[UserPrompt, ModelResponse, ToolCall, ToolResult, ...]` list in Pydantic AI's message format. Passing it to the next agent gives that agent full visibility into the reasoning that preceded it, without needing to re-run the planner.

---

## 7. Quick Reference

```python
# Typed structured output from planner
planner: Agent[None, QueryPlan] = Agent(model, result_type=QueryPlan)

# Injected dependencies for executor
executor: Agent[ExecutorDeps, str] = Agent(model, deps_type=ExecutorDeps)

# Access deps inside a tool
@executor.tool
async def get_chunks(ctx: RunContext[ExecutorDeps]) -> str:
    return format(ctx.deps.retrieved_chunks)

# Run N agents in parallel — safe, no locking needed
results = await asyncio.gather(*[
    executor.run(sq.query, deps=ExecutorDeps(...))
    for sq in plan.sub_queries
])

# Pass history between agents
result2 = await agent2.run(prompt, message_history=result1.all_messages())
```
