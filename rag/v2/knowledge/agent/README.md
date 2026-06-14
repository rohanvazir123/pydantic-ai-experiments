# knowledge/agent/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Three-Layer Gate](#three-layer-gate)
- [Streaming](#streaming)

---

## What This Is

The Pydantic AI agent and the confidence-aware pipeline that wraps it. Every user query passes through three gates before an answer reaches the client — retrieval confidence, citation verification, and an LLM judge.

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | `ConfidenceAwarePipeline`: orchestrates all 3 gates; returns `RAGResponse` |
| `agent.py` | Pydantic AI `agent` singleton; `RAGState` lazy-init deps; `traced_agent_run()`; 5 tools |
| `judge.py` | `LLMJudge`: nano model verdict (`supported`/`partial`/`unsupported`); escalates to small if nano is uncertain |
| `model_router.py` | `QueryRouter`: nano model → `RoutingDecision` (complexity, requires_graph, model tier) |
| `cost_guard.py` | `check_cost_circuit_breaker()`: Redis INCRBYFLOAT budget check at `PRE_LLM` hook |
| `prompts.py` | System prompt templates |

---

## Three-Layer Gate

Every call to `ConfidenceAwarePipeline.run()` passes through three gates in sequence. A failed gate returns an abstention response — no answer, no citations, no LLM cost (except for the gate check itself):

| Layer | Gate | Abstention status |
|-------|------|-----------------|
| 1 | `sum(confidence for top-K) < retrieval_confidence_threshold` | `abstained_retrieval` |
| 2 | `len(uncited_claims) > 0` in generation output | `abstained_citation` |
| 3 | Judge verdict `unsupported` or `judge_confidence < threshold` | `abstained_judge` |

---

## Streaming

The streaming path (`run_stream()`) bypasses the judge gate — the judge adds ~80ms and is incompatible with token-by-token streaming. Only Layer 1 (retrieval confidence) applies to streaming.

```python
async with agent.run_stream(query, deps=state) as streamed:
    async for delta in streamed.stream_text(delta=True):
        yield f"data: {json.dumps({'delta': delta})}\n\n"
usage = streamed.usage()   # token counts available after stream ends
```

Token counts come from Pydantic AI's built-in `result.usage()` — no manual token counting.
