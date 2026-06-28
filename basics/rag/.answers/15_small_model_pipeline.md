# Multi-Stage Small Model Architecture

Answers to Q61–Q63 from `rag_system_design.md`.

---

## Q61. Where in the RAG pipeline does a small model add value, and what is the latency budget?

A 3B–8B model is fast and cheap enough to insert at multiple micro-stages without meaningful latency impact, provided its output is constrained to a token or two. The three natural insertion points:

### Stage 1 — Pre-retrieval: guardrails and intent routing

Before the query touches the vector DB or cache:

**Safety and jailbreak detection.** Filter prompt injections, toxic content, and out-of-scope system overrides before they reach backend infrastructure. This is cheaper to catch here than to sanitise after retrieval or after generation.

**Intent routing.** Classify whether the query requires retrieval at all:
- "What are our Q3 compliance rules?" → needs retrieval → vector DB path
- "Thank you!" / "Hello" → chit-chat → fast direct response, RAG bypassed entirely

Bypassing retrieval for chit-chat eliminates the embedding call, DB query, and main LLM call entirely. At scale this is a significant cost reduction.

**Latency budget:** 5–20 ms with `max_tokens=2` and a forced True/False output.

### Stage 2 — Post-retrieval: binary chunk relevance filtering

Vector databases return top-k chunks that match semantically but are often noisy — they share keywords with the query without answering it. Passing all 10 chunks to a 70B LLM burns tokens on irrelevant context and degrades generation quality (lost-in-the-middle).

The small model evaluates each chunk individually:

```
Query: "What is our Q3 overtime policy?"
Chunk: "Q3 revenue was $4.2M, up 18% YoY..."
→ Relevant: NO

Chunk: "Overtime beyond 40 hours in Q3 requires manager pre-approval..."
→ Relevant: YES
```

Drop all NO chunks. Pass only the YES chunks to the main LLM. This shrinks token overhead and removes distracting content.

**Latency budget:** ~5 ms per chunk with `max_tokens=1`. For 10 chunks, this is ~50 ms total — less than a single embedding call.

### Stage 3 — Post-generation: output auditing

After the main LLM produces a response, before the user sees it:

**Hallucination / grounding check.** Ask the small model: "Is every claim in this answer explicitly supported by the retrieved passages? YES or NO." If NO, either abstain, flag for human review, or route to a retry.

**PII and data leakage scan.** Did the LLM accidentally surface API keys, internal IDs, email addresses, or personal financial data in its response? A small model scanning the output catches this before it reaches the user.

**Structured output validation.** If the pipeline is expected to return JSON or a specific Markdown schema, use the small model to validate syntax and flag for auto-repair.

**Latency budget:** 10–30 ms with constrained output. Run in parallel with citation extraction to avoid adding to the critical path.

---

## Q62. Binary chunk relevance filter versus cross-encoder reranker — when to use which?

### Cross-encoder reranker

A cross-encoder (e.g., `BAAI/bge-reranker-base`) reads the query and chunk together and outputs a continuous relevance score. You rank by score and take top-k.

- **Adds:** nuanced ranking, distinguishes between "slightly relevant" and "highly relevant"
- **Costs:** 50–200 ms per batch (GPU), must process all chunks before trimming
- **Output:** continuous score → requires a threshold decision

### Small model YES/NO filter

A small LLM asked "Is this chunk relevant to the query? YES or NO":

- **Adds:** aggressive pruning — drops clearly irrelevant chunks before they consume main LLM tokens
- **Costs:** 5 ms per chunk, easily parallelised, no GPU required
- **Output:** binary decision — no threshold tuning needed

### When to use each

| Scenario | Better choice |
|----------|--------------|
| You have a GPU, need ranked output, top-5 from 100 chunks | Cross-encoder reranker |
| You need to prune noise before sending to an expensive cloud API | Small model binary filter |
| Retrieval is clean but ordering matters | Cross-encoder only |
| Retrieval is noisy and token budget is the bottleneck | Binary filter first, then reranker on survivors |
| CPU-only inference, latency-sensitive | Binary filter (reranker is slow on CPU) |

**In practice, compose them:** binary filter drops the obvious noise (40–60% of chunks), then a reranker orders the survivors. This is strictly cheaper than reranking all top-k.

### Keeping latency near zero

Force `max_tokens=1` and accept only "Y" or "N". Use a Pydantic / instructor constraint:

```python
from pydantic import BaseModel
from typing import Literal

class RelevanceVerdict(BaseModel):
    relevant: Literal["Y", "N"]

# With instructor:
verdict = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[
        {"role": "system", "content": "Answer only Y or N."},
        {"role": "user",   "content": f"Query: {query}\nChunk: {chunk[:500]}\nRelevant?"},
    ],
    response_model=RelevanceVerdict,
    max_tokens=1,
)
```

Batch all chunks in parallel with `asyncio.gather`. At 5 ms per chunk, 10 chunks takes ~5–10 ms wall time.

---

## Q63. Post-generation auditing with a small model — failure modes and calibration

### What it does

After the main LLM produces its answer, the small model runs three checks:

1. **Grounding check:** "Is every factual claim in this answer supported by the retrieved passages? YES or NO."
2. **PII scan:** "Does this response contain names, email addresses, API keys, financial data, or internal identifiers that were not in the original query? YES or NO."
3. **Structured output validation:** Parse the output against the expected schema; if it fails, flag for auto-repair.

### Failure modes

**Self-referential optimism bias.** A model from the same family as the generator is biased toward judging outputs from that family as correct. A `llama3.2:3b` model asked to judge a `llama3.1:70b` answer may miss hallucinations it would not itself produce, because they are plausible to the same model family.

*Mitigation:* Use a model from a different architecture family for the judge (e.g., Qwen as judge for Llama generation), or use a model fine-tuned specifically on hallucination detection.

**Context-blind grounding checks.** The small model cannot verify a claim if it lacks the retrieved passages in its context. If the judge only sees the query and answer — not the source chunks — it can only use parametric knowledge to assess grounding, which defeats the purpose.

*Mitigation:* Always pass the retrieved chunks to the judge, not just the answer. Keep chunk input short (first 300 tokens each) to stay within the judge's context window.

**PII false negatives on obfuscated data.** An LLM may leak PII in paraphrased form ("the user's email ending in @gmail.com") or embedded in a JSON value. Simple pattern matching catches obvious leakage; small models catch paraphrased leakage better but still miss adversarial obfuscation.

*Mitigation:* Layer a regex/NER PII scanner before the small model, not instead of it.

**Structured output repair loops.** If the judge flags bad JSON and triggers an auto-repair call back to the main LLM, you've added a full round-trip to the critical path. Repair loops can cascade.

*Mitigation:* Set a repair attempt limit (1–2 retries). On failure, return a safe fallback response rather than looping indefinitely.

### Calibration

The judge's recall (catching real hallucinations) matters more than precision (avoiding false positives that block valid answers). Tune the threshold to accept some false positives (unnecessary abstentions) in exchange for near-zero false negatives (hallucinations that reach users).

Collect a labelled sample of (answer, passages, verdict) triples, evaluate judge precision/recall, and set the threshold based on acceptable abstention rate for your product.

---

## The Full Multi-Stage Architecture

```
              User Query
                 │
                 ▼
STAGE 1 ──► [ Small Model: Guardrails & Intent Routing ]
             │  • Safety / jailbreak filter → block
             │  • Intent: chit-chat? → fast response (skip RAG)
             │  • Intent: needs retrieval? → continue
                 │
                 ▼
        [ Check Cache Layer ] ──► (Hit → return answer)
                 │
                 ▼ (Miss)
        [ Vector DB Retrieval ] ──► Returns Top 10 Chunks
                 │
                 ▼
STAGE 2 ──► [ Small Model: Chunk Relevance Filter ]
             │  • YES/NO per chunk, max_tokens=1
             │  • Drop NO chunks (typically 40–60% pruned)
                 │
                 ▼ (3–4 high-quality chunks remain)
        [ Main RAG LLM Generation ]
                 │
                 ▼
STAGE 3 ──► [ Small Model: Citation & Hallucination Judge ]
             │  • Grounding check → block if unsupported
             │  • PII scan → block if leakage detected
             │  • Schema validation → repair or fallback
                 │
                 ▼
          Final User Answer
```

**Production efficiency rule:** Every small model call must use `max_tokens=1` or `max_tokens=2` with a constrained output format. This keeps each stage to single-digit milliseconds. The total overhead of all three stages is typically under 100 ms — less than one retrieval round-trip.
