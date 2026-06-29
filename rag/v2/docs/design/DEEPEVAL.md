# DeepEval RAG Evaluation — Metrics, Integration, and Results

How DeepEval is wired into this project, what each metric measures, how the LLM judge works, and what the first run told us.

## Table of Contents

- [Why DeepEval](#why-deepeval)
- [The Five RAG Metrics](#the-five-rag-metrics)
  - [Faithfulness](#1-faithfulness)
  - [Answer Relevancy](#2-answer-relevancy)
  - [Contextual Relevancy](#3-contextual-relevancy)
  - [Contextual Precision](#4-contextual-precision)
  - [Contextual recall](#5-contextual-recall)
- [Metric Decision Map](#metric-decision-map)
- [How the LLM Judge Works](#how-the-llm-judge-works)
- [Integration Architecture](#integration-architecture)
- [How to Run](#how-to-run)
- [First Run Results — 2026-06-29](#first-run-results--2026-06-29)
- [Bugs Found During First Run](#bugs-found-during-first-run)
- [Known Limitations](#known-limitations)
- [Thresholds and Pass Criteria](#thresholds-and-pass-criteria)
- [When to Use Which Metric](#when-to-use-which-metric)

---

## Why DeepEval

The existing IR metrics (Hit Rate, MRR, NDCG — in `knowledge/evaluation/metrics/retrieval.py`) only tell you whether the right *chunk* was retrieved. They say nothing about what the LLM did with that chunk. DeepEval fills the generation layer:

| Layer | Question answered | Tools |
|-------|------------------|-------|
| Retrieval | Was the right chunk found? | Hit Rate, MRR, NDCG (`retrieval.py`) |
| Generation | Was the answer faithful? Relevant? | DeepEval (`run_deepeval.py`) |

DeepEval uses an LLM judge to score each metric — the same approach as RAGAS, but with a richer Python API, cleaner metric isolation, and active development.

**Version used:** `deepeval==4.0.7`

---

## The Five RAG Metrics

DeepEval's RAG metric suite requires three inputs per test case:

```python
LLMTestCase(
    input="the user question",
    actual_output="the LLM's answer",
    expected_output="optional ground truth answer",   # only for Precision + Recall
    retrieval_context=["chunk1 text", "chunk2 text", ...],
)
```

### 1. Faithfulness

> **"Does the answer contain only claims that are grounded in the retrieved context?"**

**Formula:**

```
faithfulness = (number of claims in answer supported by context)
               ─────────────────────────────────────────────────
               (total number of claims in answer)
```

**How the judge works:**
1. Judge extracts every factual claim from `actual_output`.
2. For each claim, judge checks whether it can be inferred from `retrieval_context`.
3. Score = fraction of supported claims.

**Range:** 0.0 – 1.0. A score of 1.0 means every claim is grounded. Hallucinations push this toward 0.

**Threshold:** 0.7. Below this the answer is adding facts that aren't in the retrieved chunks.

**What it catches:** Hallucinations, confabulated numbers, facts the model knows from pre-training but that aren't in the corpus.

**Does NOT need `expected_output`.**

---

### 2. Answer Relevancy

> **"Does the answer actually address what was asked?"**

**Formula:**

```
answer_relevancy = mean cosine_similarity(embed(original_question), embed(back_question_i))
                   for i in back_questions_generated_from_answer
```

**How the judge works:**
1. Judge generates N synthetic questions that the answer appears to answer.
2. Each back-question is embedded (using the judge's embedding).
3. Cosine similarity is computed between each back-question and the original `input`.
4. Mean similarity = score.

A non-answer ("I don't know") generates back-questions that don't resemble the original question, scoring near 0. An on-topic answer generates back-questions that closely match.

**Range:** 0.0 – 1.0.

**Threshold:** 0.7.

**What it catches:** Evasive answers, topic drift, refusals when the corpus actually has the answer.

**Does NOT need `expected_output`.**

---

### 3. Contextual Relevancy

> **"Are the retrieved chunks actually about what was asked?"**

**Formula:**

```
contextual_relevancy = (number of statements in context relevant to the input)
                       ─────────────────────────────────────────────────────
                       (total number of statements in context)
```

**How the judge works:**
1. Judge splits each context chunk into individual statements.
2. For each statement, judge decides: relevant to `input` or not?
3. Score = fraction of relevant statements across all chunks.

**Range:** 0.0 – 1.0.

**Threshold:** 0.6. Lower threshold than Faithfulness because retrieval inherently returns some noise.

**What it catches:** Retriever returning off-topic chunks, corpus contamination (our corpus includes a BIS annual report which is unrelated to NeuralFlow HR questions), embedding space collisions.

**Does NOT need `expected_output`.**

---

### 4. Contextual Precision

> **"Are the most useful chunks ranked highest?"**

**Formula:**

```
contextual_precision = weighted_precision over relevant nodes
                     = Σ_k [ (# relevant nodes in top-k / k) × relevance_k ]
                       ─────────────────────────────────────────────────────
                       total relevant nodes
```

Where `relevance_k = 1` if node k is relevant to achieving `expected_output`, else 0.

**How the judge works:**
1. For each chunk in `retrieval_context`, judge decides: does this chunk support the `expected_output`?
2. Score rewards relevant chunks appearing early in the list.

**Range:** 0.0 – 1.0.

**Threshold:** 0.6.

**What it catches:** Retriever ranking good chunks low (a ranking/reranker problem, not just a recall problem).

**REQUIRES `expected_output`.** Only runs for test cases where ground truth is provided.

---

### 5. Contextual Recall

> **"Does the retrieved context cover everything in the expected answer?"**

**Formula:**

```
contextual_recall = (sentences in expected_output attributable to retrieval_context)
                    ─────────────────────────────────────────────────────────────────
                    (total sentences in expected_output)
```

**How the judge works:**
1. Judge splits `expected_output` into individual sentences.
2. For each sentence, judge determines whether it can be attributed to at least one chunk in `retrieval_context`.
3. Score = fraction of expected sentences covered.

**Range:** 0.0 – 1.0.

**Threshold:** 0.6.

**What it catches:** Retriever missing chunks that contain information needed for a complete answer; top-K too small.

**REQUIRES `expected_output`.**

---

## Metric Decision Map

```
Do you have a ground-truth expected answer?
│
├── YES → run all 5 metrics
│         Faithfulness + AnswerRelevancy + ContextualRelevancy
│         + ContextualPrecision + ContextualRecall
│
└── NO  → run 3 metrics
          Faithfulness + AnswerRelevancy + ContextualRelevancy
          (still catches hallucination, relevance, and retrieval quality)
```

For a regression gate, prioritise:
- **Faithfulness** — catches generation regressions
- **ContextualRelevancy** — catches retrieval regressions  
- **AnswerRelevancy** — catches abstention regressions

---

## How the LLM Judge Works

DeepEval's metrics are **LLM-as-judge**. Every score is computed by prompting a judge LLM with a structured prompt and parsing its verdict as JSON.

### Our judge: OllamaJudge

We implement `OllamaJudge(DeepEvalBaseLLM)` in `scripts/run_deepeval.py` to point DeepEval at the local Ollama endpoint:

```python
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._openai = OpenAI(base_url=settings.llm_base_url, api_key="ollama")

    def generate(self, prompt: str, schema=None) -> str | BaseModel:
        kwargs = {"model": model, "messages": [...], "temperature": 0}
        if schema:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._openai.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        if schema:
            return schema.model_validate_json(text)
        return text

    async def a_generate(self, prompt, schema=None):
        return await run_in_executor(self.generate, prompt, schema)
```

### Judge quality vs. generator quality

Both our RAG generator and judge are `llama3.2:3b` (a 3B parameter model). This matters:

- A weak judge under-scores answers that are actually good (false negatives).
- A weak judge over-scores hallucinated answers it can't detect.
- Using the same model as generator and judge creates an optimism bias — the judge tends to consider the output faithful even when it shouldn't.

**To get more reliable scores, use a stronger judge** (`llama3.1:70b` or a cloud model):

```bash
make deepeval-large           # uses llama3.1:70b as judge
```

The generator model is set by `settings.llm_model`. The judge model defaults to the same but can be overridden:

```bash
uv run python scripts/run_deepeval.py --judge llama3.1:70b
```

---

## Integration Architecture

```
scripts/run_deepeval.py
        │
        ├─ 1. initialise stores (PostgresHybridStore, RedisCache, Embedder)
        │
        ├─ 2. for each TEST_CASE:
        │       │
        │       ├─ retriever.retrieve(query, k=top_k)
        │       │       → retrieval_context: list[str]  (full chunk texts)
        │       │
        │       └─ pipeline.run_stream(query)            (streaming path, output_type=str)
        │               → collect delta tokens → actual_output: str
        │
        ├─ 3. build LLMTestCase(input, actual_output, expected_output, retrieval_context)
        │
        ├─ 4. OllamaJudge scores each metric
        │
        └─ 5. write evals/deepeval_results.md (latest)
               write evals/deepeval_YYYY-MM-DD_HHMMSS.md (archive)
```

**Why `run_stream()` not `run()`:**
`pipeline.run()` uses `output_type=GenerationResult` — a nested Pydantic schema requiring the LLM to produce structured JSON with UUID fields. `llama3.2:3b` cannot reliably do this even with `retries=3`. `run_stream()` uses `output_type=str` (plain text) which the model handles fine. The retrieval context is fetched separately via `retriever.retrieve()` so we get full chunk text, not the ≤200-char excerpts in `Citation.excerpt`.

**Why retrieve separately:**
`Citation.excerpt` is capped at 200 characters — too short for DeepEval's statement-level analysis. Calling the retriever directly gives us the full `SearchResult.content` (~500 tokens per chunk).

---

## How to Run

```bash
# Install eval dependency
uv sync --extra eval

# Run with default judge (llama3.2:3b)
make deepeval
# or:
uv run python scripts/run_deepeval.py

# Run with stronger judge
make deepeval-large
# or:
uv run python scripts/run_deepeval.py --judge llama3.1:70b

# Change top-K retrieved
uv run python scripts/run_deepeval.py --top-k 10
```

**Requirements:** PostgreSQL + Redis + Ollama running, corpus seeded (`make seed`).

**Output files:**
```
evals/
  deepeval_results.md              ← overwritten each run (latest)
  deepeval_YYYY-MM-DD_HHMMSS.md   ← timestamped archive (kept)
```

---

## First Run Results — 2026-06-29

**Configuration:** `llama3.2:3b` as both generator and judge, top-K=5, 7 test cases.

### Summary

| Metric | Avg Score | Threshold | Pass Rate | Status |
|--------|-----------|-----------|-----------|--------|
| Faithfulness | 0.801 | 0.7 | 71% (5/7) | ✅ Pass |
| Answer Relevancy | 0.762 | 0.7 | 71% (5/7) | ✅ Pass |
| Contextual Relevancy | 0.618 | 0.6 | 57% (4/7) | ✅ Pass |
| Contextual Precision | 0.500 | 0.6 | 50% (1/2) | ❌ Fail |
| Contextual Recall | 0.500 | 0.6 | 0% (0/2) | ❌ Fail |

### Per-query observations

| Query | Faithfulness | Answer Relevancy | Contextual Relevancy | Notes |
|-------|-------------|-----------------|---------------------|-------|
| What does NeuralFlow AI do? | 1.00 ✅ | 1.00 ✅ | 0.57 ❌ | Retrieval mixes NeuralFlow docs with BIS report chunks |
| PTO and leave policy? | 0.61 ❌ | 0.33 ❌ | 0.21 ❌ | Retriever returning BIS central bank docs instead of team-handbook |
| Q4 business units? | 1.00 ✅ | 0.00 ❌ | 0.79 ✅ | Answer correctly says info not in context; judge penalises non-answer |
| Engineering team tools? | 1.00 ✅ | 1.00 ✅ | 0.85 ✅ | Best result — right chunks, faithful answer |
| Onboarding steps? | 0.00 ❌ | 1.00 ✅ | 0.46 ❌ | Faithfulness=0 is a judge false negative (answer is correct); retrieval pulled BIS docs |
| Company goals? | 1.00 ✅ | 1.00 ✅ | 0.82 ✅ | Good result |
| Performance review? | 1.00 ✅ | 1.00 ✅ | 0.63 ✅ | Good result |

### Key findings from first run

1. **Corpus contamination is the main retrieval problem.** The `documents/` folder contains a `bis_annual_report_2024.pdf` (Bank for International Settlements annual report). Its chunks are semantically similar to some NeuralFlow HR queries (both involve "policy", "risk", "balance"). These appear in the top-5 for PTO, onboarding, and other HR queries — dragging Contextual Relevancy down. **Fix: remove or tag non-NeuralFlow documents in the corpus.**

2. **Faithfulness scores are artificially optimistic with a weak judge.** The `llama3.2:3b` judge gave Faithfulness=1.00 for answers that contain phrases like "I can only answer questions about the knowledge base" alongside factual content — it didn't notice the contradiction. A stronger judge would catch this.

3. **Answer Relevancy=0 for Q4 business units is a judgment call, not a bug.** The pipeline correctly said "the specific information about individual business unit performance is not available in the provided context." DeepEval scored this 0.00 for Answer Relevancy because the back-questions generated from that non-answer don't resemble the original question. This is technically correct — the answer isn't relevant because the corpus doesn't have the answer. The right fix is to remove this question from the eval set or mark it as expected-to-abstain.

4. **Contextual Precision/Recall are unreliable without good expected_output.** Our `expected_output` strings were generic ("Employees receive paid time off…") rather than grounded in actual document text. The judge couldn't find those sentences in the retrieved chunks, driving Recall to 0.5. Fix: write `expected_output` by copying actual sentences from the documents.

---

## Bugs Found During First Run

### 1. PII false positive — `US_DRIVER_LICENSE` at score 0.3

**Symptom:** Query "Which business units performed best in Q4?" was blocked by the PII gate with status `abstained_pii`. "Q4" matched the `US_DRIVER_LICENSE` Presidio recognizer.

**Root cause:** Presidio's `UsLicenseRecognizer` uses patterns that match short alphanumeric tokens. "Q4" scored 0.3 confidence — low, but the scanner had no threshold.

**Fix:** Added `score_threshold=0.7` to `analyzer.analyze()` in `knowledge/validation/pii_scanner.py`. Only high-confidence detections now trigger the gate.

```python
results = await asyncio.to_thread(
    analyzer.analyze,
    text=text,
    language="en",
    entities=_SENSITIVE_ENTITIES,
    score_threshold=0.7,
)
```

### 2. `Exceeded maximum output retries` — structured JSON schema too complex for `llama3.2:3b`

**Symptom:** `pipeline.run()` failed on every query with `Exceeded maximum output retries (1)`.

**Root cause:** `output_type=GenerationResult` requires the model to produce a nested JSON object containing `Citation` records with UUID fields and a `CitationCheck` sub-object. `llama3.2:3b` cannot reliably produce this schema.

**Fix:** The eval script uses `pipeline.run_stream()` (`output_type=str`) to collect the answer, and calls the retriever directly for chunk content. The `retries=3` parameter was also added to the agent for the blocking path.

---

## Known Limitations

### Judge model quality

`llama3.2:3b` is a 3B parameter model used as both generator and judge. It:
- Misses subtle hallucinations (too lenient on Faithfulness)
- Sometimes fails to generate valid JSON for schema-based responses (occasional `ValueError` from `OllamaJudge.generate()`)
- Has optimism bias when scoring its own outputs

Use `--judge llama3.1:70b` for more reliable scores. With the 70B model, expect Faithfulness scores to drop by 0.1–0.2 (it catches more hallucinations).

### Score variance

Each run with a non-deterministic judge produces slightly different scores even for identical inputs. `temperature=0` reduces but doesn't eliminate variance. Average across 3 runs before locking a baseline.

### Contextual Relevancy underestimates for short queries

For one-word or two-word queries, the judge has fewer statements to score in `retrieval_context` and the denominator is small, so a single irrelevant statement has a large impact. This is a known DeepEval limitation.

### Expected output quality affects Precision/Recall heavily

The judge for ContextualPrecision and ContextualRecall splits `expected_output` into sentences and looks for them in the retrieved chunks. If the expected output is paraphrased rather than quoting the document directly, the judge will score low even when the right chunks were retrieved. Always write `expected_output` using phrases taken verbatim from the source documents.

---

## Thresholds and Pass Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Faithfulness | 0.70 | Accept up to 30% unsupported claims from a small model |
| Answer Relevancy | 0.70 | Allow some hedging / partial answers |
| Contextual Relevancy | 0.60 | Hybrid search returns some noise; 60% relevant is acceptable |
| Contextual Precision | 0.60 | Relevant chunks should rank in the top half |
| Contextual Recall | 0.60 | Context should cover at least 60% of expected answer sentences |

These are starting thresholds for `llama3.2:3b`. Raise them to 0.80 when using a 70B judge.

**Regression gate rule (proposed):** if any metric drops more than 0.05 below its baseline across a run of ≥10 queries, flag as a regression.

---

## When to Use Which Metric

| Signal you want | Metric to check |
|----------------|----------------|
| Is the LLM hallucinating? | Faithfulness |
| Is the LLM going off-topic or refusing? | Answer Relevancy |
| Is the retriever returning garbage? | Contextual Relevancy |
| Are the best chunks ranked first? | Contextual Precision |
| Is the retriever missing key information? | Contextual Recall |
| Did a code change break retrieval? | Contextual Relevancy (fastest, no ground truth) |
| Did a prompt change introduce hallucinations? | Faithfulness |
| Is the answer complete? | Contextual Recall (requires expected_output) |

---

## Enum Reference

Frequency guide: **★★★** use on every eval · **★★** use when the feature applies · **★** rarely touched directly

---

### Test Case Params

#### `SingleTurnParams` ★★★
*(import: `from deepeval.test_case import SingleTurnParams`)*
*(note: `LLMTestCaseParams` is a deprecated alias for the same enum)*

Passed to `evaluation_params` in `GEval` and other metrics to tell the judge which fields to look at.

| Value | One-liner |
|-------|-----------|
| `INPUT` | The user's question or prompt |
| `ACTUAL_OUTPUT` | What the LLM produced |
| `EXPECTED_OUTPUT` | Ground-truth answer; required by `GEval` correctness, `ContextualPrecision`, `ContextualRecall` |
| `CONTEXT` | Known ground-truth facts (not retrieved — what the answer *should* draw from) |
| `RETRIEVAL_CONTEXT` | Chunks returned by the retriever; used by Faithfulness and all Contextual metrics |
| `METADATA` | Arbitrary key-value dict attached to the test case |
| `TAGS` | Label list on the test case |
| `TOOLS_CALLED` | Actual tool calls the agent made |
| `EXPECTED_TOOLS` | Tool calls the agent was supposed to make |
| `MCP_SERVERS` | MCP server configs available to the agent |
| `MCP_TOOLS_CALLED` | MCP tool calls made during the run |
| `MCP_RESOURCES_CALLED` | MCP resources fetched during the run |
| `MCP_PROMPTS_CALLED` | MCP prompt templates invoked |

#### `MultiTurnParams` ★★
*(import: `from deepeval.test_case import MultiTurnParams`)*
*(note: `TurnParams` is a deprecated alias)*

Used when building `ConversationalTestCase` for multi-turn / chatbot evals.

| Value | One-liner |
|-------|-----------|
| `ROLE` | Speaker role for a turn (`user` or `assistant`) |
| `CONTENT` | Text of a turn |
| `SCENARIO` | High-level description of the conversation setup |
| `EXPECTED_OUTCOME` | What the conversation should have achieved |
| `CONTEXT` | Background facts available to the chatbot |
| `USER_DESCRIPTION` | Who the simulated user is |
| `CHATBOT_ROLE` | System persona the chatbot is playing |
| `RETRIEVAL_CONTEXT` | Retrieved chunks available across turns |
| `TOOLS_CALLED` | Tool calls across all turns |
| `MCP_TOOLS` | MCP tool calls across all turns |
| `MCP_RESOURCES` | MCP resources accessed across all turns |
| `MCP_PROMPTS` | MCP prompt templates invoked across all turns |
| `METADATA` | Arbitrary metadata for the conversation |
| `TAGS` | Labels for the conversation |

#### `ToolCallParams` ★★
*(import: `from deepeval.test_case import ToolCallParams`)*

Fields inspected when comparing individual `ToolCall` objects in agentic evals.

| Value | One-liner |
|-------|-----------|
| `INPUT_PARAMETERS` | Arguments passed into the tool |
| `OUTPUT` | Value the tool returned |

---

### Synthesizer / Gold Dataset Generation

#### `Evolution` ★★★
*(import: `from deepeval.synthesizer import Evolution`)*

Question mutation strategies passed to `TestsetGenerator`. Controls how questions are rewritten from source context to increase difficulty and diversity.

| Value | One-liner |
|-------|-----------|
| `REASONING` | Rewrites to require multi-step logical inference |
| `MULTICONTEXT` | Rewrites to require synthesising two or more passages |
| `CONCRETIZING` | Replaces vague terms with specific details from context |
| `CONSTRAINED` | Adds an explicit constraint (e.g. "in under 50 words") |
| `COMPARATIVE` | Turns into a comparison between two entities |
| `HYPOTHETICAL` | Adds a counterfactual ("what if…") framing |
| `IN_BREADTH` | Generates a related-but-new question to expand topic coverage |

#### `PromptEvolution` ★★
*(import: `from deepeval.synthesizer import PromptEvolution`)*

Same mutation strategies as `Evolution` but applied to raw prompts rather than QA pairs. All values are identical minus `MULTICONTEXT`.

| Value | One-liner |
|-------|-----------|
| `REASONING` | Multi-step inference rewrite |
| `CONCRETIZING` | Replace vague terms with specifics |
| `CONSTRAINED` | Add an explicit constraint |
| `COMPARATIVE` | Turn into a comparison |
| `HYPOTHETICAL` | Add counterfactual framing |
| `IN_BREADTH` | Generate a related new prompt |

#### `GenerationMethod` ★★
*(import: `from deepeval.cli.generate.utils import GenerationMethod`)*

Controls what source the CLI `deepeval generate` command uses to create goldens.

| Value | One-liner |
|-------|-----------|
| `DOCS` | Generate from document files (e.g. PDFs, Markdown) |
| `CONTEXTS` | Generate from a pre-built list of context strings |
| `SCRATCH` | Generate from scratch with no source material |
| `GOLDENS` | Evolve / mutate an existing set of goldens |

#### `GoldenVariation` ★★
*(import: `from deepeval.cli.generate.utils import GoldenVariation`)*

Whether the CLI generates single-turn or multi-turn test cases.

| Value | One-liner |
|-------|-----------|
| `SINGLE_TURN` | Produce `LLMTestCase` (one question, one answer) |
| `MULTI_TURN` | Produce `ConversationalTestCase` (full conversation thread) |

#### `FileType` ★★
*(import: `from deepeval.cli.generate.utils import FileType`)*

Output format when saving generated goldens to disk.

| Value | One-liner |
|-------|-----------|
| `JSON` | Single JSON array |
| `CSV` | Flat CSV, one row per golden |
| `JSONL` | One JSON object per line (preferred for large datasets) |

---

### Metrics

#### `ScoreType` ★★
*(import: `from deepeval.metrics.summarization.schema import ScoreType`)*

Sub-scores surfaced by `SummarizationMetric`. Useful when you want to pull the two components apart instead of using the blended score.

| Value | One-liner |
|-------|-----------|
| `ALIGNMENT` | Claims in the summary are supported by the source document |
| `COVERAGE` | Key facts from the source document appear in the summary |

#### `NodeType` ★
*(import: `from deepeval.metrics.dag.serialization.types import NodeType`)*

Node types used when building custom `DAGMetric` graphs.

| Value | One-liner |
|-------|-----------|
| `TASK` | A prompt-based reasoning step |
| `BINARY_JUDGEMENT` | Yes/No decision node |
| `NON_BINARY_JUDGEMENT` | Scored decision node (0–1) |
| `VERDICT` | Terminal node that produces the final score |

#### `ChildType` ★
*(import: `from deepeval.metrics.dag.serialization.types import ChildType`)*

Edge types in a `DAGMetric` graph, controlling how child nodes are wired.

| Value | One-liner |
|-------|-----------|
| `NODE` | Another DAG node |
| `GEVAL` | Inline GEval metric as a leaf |
| `METRIC` | Any other DeepEval metric as a leaf |

---

### Tracing & Observability

#### `SpanType` ★★★
*(import: `from deepeval.tracing import SpanType`)*

Span categories emitted by DeepEval's tracing decorators. Used to classify operations in the Confident AI trace view.

| Value | One-liner |
|-------|-----------|
| `AGENT` | Top-level agent orchestration |
| `LLM` | A single LLM call |
| `RETRIEVER` | A retrieval operation |
| `TOOL` | A tool / function call |

#### `TraceSpanStatus` ★★
*(import: `from deepeval.tracing.types import TraceSpanStatus`)*

Outcome of a traced span.

| Value | One-liner |
|-------|-----------|
| `SUCCESS` | Span completed without error |
| `ERRORED` | Span raised an exception |
| `IN_PROGRESS` | Span is still running |

#### `TraceWorkerStatus` ★
*(import: `from deepeval.tracing.types import TraceWorkerStatus`)*

Internal status of the background trace worker that batches and uploads spans.

| Value | One-liner |
|-------|-----------|
| `SUCCESS` | Batch uploaded cleanly |
| `FAILURE` | Upload failed |
| `WARNING` | Upload succeeded with non-fatal issues |

#### `EvalMode` ★
*(import: `from deepeval.tracing.types import EvalMode`)*

Internal flag that tells the trace manager how spans should be routed.

| Value | One-liner |
|-------|-----------|
| `OFF` | Not inside an eval pipeline; traces post to the API as normal |
| `EVALUATE` | Inside `evaluate(...)` — traces route into the test-run pipeline |
| `ITERATOR_SYNC` | Inside synchronous `evals_iterator` |
| `ITERATOR_ASYNC` | Inside async `evals_iterator` |

#### `Environment` ★★
*(import: `from deepeval.tracing.utils import Environment`)*

Deployment environment tag attached to traces in Confident AI.

| Value | One-liner |
|-------|-----------|
| `PRODUCTION` | Live traffic |
| `DEVELOPMENT` | Local dev |
| `STAGING` | Pre-production |
| `TESTING` | Automated test runs |

---

### Provider & Integration Config

#### `ProviderSlug` ★★★
*(import: `from deepeval.constants import ProviderSlug`)*

String slug identifying which LLM provider a `DeepEvalBaseLLM` implementation wraps. Used when registering a custom judge.

| Value | Provider |
|-------|----------|
| `OPENAI` | OpenAI API |
| `AZURE` | Azure OpenAI |
| `ANTHROPIC` | Anthropic API |
| `BEDROCK` | AWS Bedrock |
| `DEEPSEEK` | DeepSeek |
| `GOOGLE` | Google AI |
| `GROK` | xAI Grok |
| `KIMI` | Moonshot Kimi |
| `LITELLM` | LiteLLM proxy |
| `LOCAL` | Any locally-hosted model |
| `OLLAMA` | Ollama (used by our `OllamaJudge`) |
| `OPENROUTER` | OpenRouter |
| `PORTKEY` | Portkey gateway |

#### `Integration` ★★
*(import: `from deepeval.tracing.integrations import Integration`)*

Framework integrations that auto-instrument traces when `deepeval.trace` is active.

| Value | One-liner |
|-------|-----------|
| `LANGCHAIN` | LangChain chains and agents |
| `CREW_AI` | CrewAI multi-agent crews |
| `LLAMA_INDEX` | LlamaIndex query engines |
| `OPENAI_AGENTS` | OpenAI Agents SDK |
| `OPEN_AI` | Raw OpenAI client calls |
| `ANTHROPIC` | Raw Anthropic client calls |
| `PYDANTIC_AI` | PydanticAI agents (used in this project) |
| `GOOGLE_ADK` | Google Agent Development Kit |
| `STRANDS` | Strands agent framework |
| `OTEL` | OpenTelemetry exporter |
| `OPEN_INFERENCE` | OpenInference spans |
| `AGENTCORE` | AWS AgentCore |

#### `Provider` ★★
*(import: `from deepeval.tracing.integrations import Provider`)*

LLM provider enum used in tracing metadata to label which model produced a span.

| Value | One-liner |
|-------|-----------|
| `OPEN_AI` | OpenAI |
| `ANTHROPIC` | Anthropic |
| `GEMINI` | Google Gemini |
| `X_AI` | xAI (Grok) |
| `DEEP_SEEK` | DeepSeek |
| `MISTRAL` | Mistral AI |
| `PERPLEXITY` | Perplexity |
| `BEDROCK` | AWS Bedrock |
| `VERTEX_AI` | Google Vertex AI |
| `AZURE` | Azure OpenAI |
| `OPEN_ROUTER` | OpenRouter |
| `PORTKEY` | Portkey |
| `TRUE_FOUNDRY` | TrueFoundry |
| `MOONSHOT` | Moonshot (Kimi) |

---

### Prompt Management (Confident AI)

These enums are only needed if using Confident AI's hosted prompt management. Not required for self-hosted / local eval.

#### `ReasoningEffort` ★
| Value | One-liner |
|-------|-----------|
| `MINIMAL` | Minimal chain-of-thought in the prompt |
| `LOW` | Light reasoning |
| `MEDIUM` | Balanced reasoning |
| `HIGH` | Thorough chain-of-thought |

#### `Verbosity` ★
| Value | One-liner |
|-------|-----------|
| `LOW` | Terse prompt output |
| `MEDIUM` | Moderate detail |
| `HIGH` | Full verbose output |

#### `ModelProvider` ★
Confident AI's internal provider enum (separate from `ProviderSlug`).

| Value | One-liner |
|-------|-----------|
| `OPEN_AI` | OpenAI |
| `ANTHROPIC` | Anthropic |
| `GEMINI` | Google Gemini |
| `X_AI` | xAI |
| `DEEPSEEK` | DeepSeek |
| `BEDROCK` | AWS Bedrock |
| `OPENROUTER` | OpenRouter |

#### `ToolMode` ★
Controls how strictly tool calls are validated in a prompt run.

| Value | One-liner |
|-------|-----------|
| `ALLOW_ADDITIONAL` | Allow tool calls beyond those declared |
| `NO_ADDITIONAL` | Block undeclared tool calls |
| `STRICT` | All tool calls must match the declared schema exactly |

#### `OutputType` ★
| Value | One-liner |
|-------|-----------|
| `TEXT` | Plain string output |
| `JSON` | Unstructured JSON object |
| `SCHEMA` | JSON validated against a `SchemaDataType` schema |

#### `SchemaDataType` ★
JSON schema primitive types used when `OutputType.SCHEMA` is set.

| Value | One-liner |
|-------|-----------|
| `OBJECT` | JSON object `{}` |
| `ARRAY` | JSON array `[]` |
| `STRING` | String value |
| `FLOAT` | Floating-point number |
| `INTEGER` | Integer number |
| `BOOLEAN` | `true` / `false` |
| `NULL` | `null` |

#### `PromptInterpolationType` ★
Template syntax used when rendering a Confident AI prompt with variables.

| Value | One-liner |
|-------|-----------|
| `MUSTACHE` | `{{variable}}` |
| `MUSTACHE_WITH_SPACE` | `{{ variable }}` |
| `FSTRING` | `{variable}` (Python f-string style) |
| `DOLLAR_BRACKETS` | `${variable}` |
| `JINJA` | Jinja2 `{{ variable }}` with full template logic |

#### `PromptType` ★
| Value | One-liner |
|-------|-----------|
| `TEXT` | Single string prompt |
| `LIST` | List of message dicts (chat format) |

---

### Annotation (Confident AI)

#### `AnnotationType` ★★
*(import: `from deepeval.annotation.api import AnnotationType`)*

Rating UI style used when humans annotate traces in the Confident AI dashboard.

| Value | One-liner |
|-------|-----------|
| `THUMBS_RATING` | Binary thumbs up / thumbs down |
| `FIVE_STAR_RATING` | 1–5 star scale |

---

### Telemetry

#### `Feature` ★
*(import: `from deepeval.telemetry import Feature`)*

Internal feature flags used by DeepEval's opt-in telemetry. No action needed unless you're building on top of the SDK.

| Value | One-liner |
|-------|-----------|
| `REDTEAMING` | Red-teaming / adversarial test generation |
| `SYNTHESIZER` | `TestsetGenerator` usage |
| `EVALUATION` | Standard `evaluate()` runs |
| `COMPONENT_EVALUATION` | Per-component metric evaluation |
| `GUARDRAIL` | Guardrail metric usage |
| `BENCHMARK` | Public benchmark runs |
| `CONVERSATION_SIMULATOR` | Conversational simulation |
| `TRACING_INTEGRATION` | Framework auto-instrumentation |
| `UNKNOWN` | Unclassified usage |

---

### Benchmarks

DeepEval ships task enums for public NLP benchmarks. These are irrelevant to domain RAG eval but listed for completeness.

| Enum | Benchmark | Use case |
|------|-----------|---------|
| `ARCMode` | ARC (AI2 Reasoning Challenge) | Science Q&A |
| `BBQTask` | BBQ | Bias evaluation |
| `BigBenchHardTask` | BIG-Bench Hard | Difficult reasoning |
| `DROPTask` | DROP | Discrete reasoning over paragraphs |
| `EquityMedQATask` | Equity-Med QA | Medical equity |
| `HellaSwagTask` | HellaSwag | Commonsense NLI |
| `HumanEvalTask` | HumanEval | Code generation |
| `LogiQATask` | LogiQA | Logical reasoning |
| `MathQATask` | MathQA | Math word problems |
| `MMLUTask` | MMLU | Multi-subject knowledge |
| `SQuADTask` | SQuAD | Reading comprehension |
| `TruthfulQAMode` / `TruthfulQATask` | TruthfulQA | Truthfulness |
