# RAG v2 — Eval Pipeline Design

How to build, run, and gate on a domain-specific eval pipeline for the NeuralFlow RAG system.

## Table of Contents

- [What We Want to Achieve](#what-we-want-to-achieve)
- [Why Not Use an Off-the-Shelf Benchmark Directly](#why-not-use-an-off-the-shelf-benchmark-directly)
- [Eval Architecture Overview](#eval-architecture-overview)
- [Phase 1 — Gold Dataset Generation (RAGAS)](#phase-1--gold-dataset-generation-ragas)
- [Phase 2 — Retrieval Eval (NDCG, MRR, Hit Rate)](#phase-2--retrieval-eval-ndcg-mrr-hit-rate)
- [Phase 3 — End-to-End Generation Eval](#phase-3--end-to-end-generation-eval)
- [Phase 4 — CI Regression Gate](#phase-4--ci-regression-gate)
- [Metric Definitions](#metric-definitions)
- [Failure Analysis](#failure-analysis)
- [Baselines and Thresholds](#baselines-and-thresholds)
- [File Layout](#file-layout)
- [Implementation Plan](#implementation-plan)

---

## What We Want to Achieve

**Goal:** A repeatable eval pipeline that can answer two questions after any code change:

1. **Retrieval:** Does the right chunk appear in the top-K results for a given query?
2. **Generation:** Given retrieved chunks, does the LLM produce a faithful, relevant answer?

**Non-goals (for now):**
- Evaluating against public benchmarks (BEIR, FinanceBench, SQuAD). We have a domain corpus — synthetic eval from that corpus is more actionable.
- Human annotation pipeline. We use an LLM judge for speed.
- Evaluating the KG / AGE retrieval leg. Out of scope until the graph is seeded.

---

## Why Not Use an Off-the-Shelf Benchmark Directly

| Benchmark | Why it doesn't fit |
|-----------|-------------------|
| BEIR | Document-level relevance judgements, not chunk-level. Measures embedding model + reranker in isolation; doesn't test our hybrid search, caching, or context assembly. |
| FinanceBench | Finance domain, not NeuralFlow HR/tech docs. Questions test SEC filings — irrelevant to our corpus. |
| SQuAD 2.0 | Wikipedia passages provided directly — tests generation, not retrieval. Useful as a sanity check but not a regression gate for our pipeline. |
| HotpotQA | Multi-hop over Wikipedia. Our corpus is single-corpus; multi-hop is not the primary use case yet. |

**Decision:** Generate a domain-specific gold dataset from `rag/v2/documents/` using RAGAS, then run it against the live `/api/v2/chat/stream` endpoint. This directly tests the full pipeline end-to-end against our actual documents.

---

## Eval Architecture Overview

```
rag/v2/documents/          ← source corpus (company docs, PDFs, MD files)
        │
        ▼
[Phase 1] RAGAS TestsetGenerator
        │   └─ LLM: claude-haiku-4-5 (cheap, fast)
        │   └─ Embeddings: nomic-embed-text (local Ollama)
        │   └─ Question types: simple, reasoning, multi_context, conditional
        ▼
evals/gold_dataset.jsonl   ← GoldSample records (question + gold_answer + reference_chunks)
        │
        ├─── [Phase 2] Retrieval eval
        │        POST /api/v2/search  →  top-K chunks
        │        Compare chunk_ids to reference_chunks
        │        Metrics: Hit Rate, MRR, NDCG@5, Precision@5, Recall@5
        │
        └─── [Phase 3] Generation eval
                 POST /api/v2/chat/stream  →  full answer + citations
                 LLM judge: faithfulness, answer relevance (RAGAS metrics)
                 Lexical metrics: BLEU-4, ROUGE-L
                 Metrics: faithfulness, answer_relevance, bleu4, rouge_l, latency_p95
                 │
                 ▼
        evals/results/{timestamp}.jsonl   ← per-question results
        evals/results/baseline.json       ← locked baseline scores
                 │
        [Phase 4] CI regression gate
                 ├─ NDCG@5 < baseline - 0.03  →  FAIL
                 ├─ Hit Rate < baseline - 0.05 →  FAIL
                 ├─ faithfulness < baseline - 0.05 → FAIL
                 └─ all pass  →  GREEN
```

---

## Phase 1 — Gold Dataset Generation (RAGAS)

**Script:** `rag/v2/evals/generate_gold_dataset.py`

### Input

The NeuralFlow corpus already ingested in the default tenant/corpus. Script loads documents from `rag/v2/documents/` using Docling (the same loader the ingestion pipeline uses) so the test chunks match what is actually in the DB.

### RAGAS configuration

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OllamaEmbeddings

generator = TestsetGenerator(
    llm=LangchainLLMWrapper(
        ChatAnthropic(model="claude-haiku-4-5-20251001")  # cheap judge
    ),
    embedding_model=LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text:latest")
    ),
)
testset = generator.generate_with_langchain_docs(
    docs,
    testset_size=100,
    distributions={
        "simple":       0.40,   # direct factual lookups
        "reasoning":    0.25,   # inference from one passage
        "multi_context": 0.25,  # synthesise two passages
        "conditional":  0.10,   # if/then questions
    },
)
```

### Output format

Each row saved to `evals/gold_dataset.jsonl` as a `GoldSample`:

```jsonl
{
  "question": "What is the company's parental leave policy?",
  "gold_answer": "NeuralFlow offers 16 weeks of paid parental leave for all employees.",
  "reference_chunks": ["<chunk_id_1>", "<chunk_id_2>"],
  "reference_context": "... raw passage text ...",
  "evolution_type": "simple",
  "source_doc": "team-handbook.md"
}
```

`reference_chunks` is populated by: after generating the question, retrieve the top-5 chunks from the live pipeline and intersect with the chunks RAGAS used as source context. Any chunk whose content overlaps >80% (token F1) with the RAGAS reference context is considered a gold chunk.

### Human review step (before locking)

After generation, review the output and:
1. Delete questions that are trivially easy or obviously off-topic.
2. Fix any gold answers that are wrong or ambiguous.
3. Mark 5–10 questions as "hard" in metadata for targeted failure analysis.

Target: ~80 questions after review. Do not commit the dataset until reviewed.

---

## Phase 2 — Retrieval Eval (NDCG, MRR, Hit Rate)

**Script:** `rag/v2/evals/run_retrieval_eval.py`

For each `GoldSample`, call `POST /api/v2/search`:

```python
response = httpx.post(
    "http://localhost:7100/api/v2/search",
    json={"query": sample.question, "corpus_id": "default", "k": 5},
    headers={"Authorization": f"Bearer {token}"},
)
returned_chunk_ids = [r["chunk_id"] for r in response.json()["data"]["results"]]
```

Compare `returned_chunk_ids` against `sample.reference_chunks`.

### Why use `/search` not `/chat`

`/search` returns chunk IDs directly. `/chat` returns an answer — to evaluate retrieval separately, we need the raw chunk list before the LLM runs. This also means retrieval eval can run without Ollama (no LLM call).

### Metrics computed

All metrics already implemented in `knowledge/evaluation/metrics/retrieval.py`. The eval script calls them directly:

```python
from knowledge.evaluation.metrics.retrieval import (
    hit_rate, reciprocal_rank, ndcg_at_k,
    precision_at_k, recall_at_k,
)
```

No new metric code needed.

---

## Phase 3 — End-to-End Generation Eval

**Script:** `rag/v2/evals/run_generation_eval.py`

For each `GoldSample`, call `POST /api/v2/chat/stream` and collect the full answer (SSE stream drained to completion).

### Generation metrics

#### Faithfulness (RAGAS)

Does every claim in the generated answer appear in the retrieved context?

```python
# LLM judge (claude-haiku-4-5)
prompt = f"""
Context:
{retrieved_context}

Answer:
{generated_answer}

List every factual claim in the answer. For each claim, state whether it is
directly supported by the context above. Return JSON:
{{"claims": [{{"claim": "...", "supported": true/false}}]}}
"""
faithfulness = sum(c["supported"] for c in claims) / len(claims)
```

Score: 0.0–1.0. A hallucinated answer that adds facts not in context scores low.

#### Answer Relevance (RAGAS)

Does the answer actually address the question? Measured by generating back-questions from the answer and measuring cosine similarity to the original question:

```python
# Generate N back-questions from the answer
back_questions = llm.generate(f"Generate 3 questions that this answer addresses:\n{answer}")
# Embed original question and each back-question
# Score = mean cosine similarity
relevance = mean(cosine(embed(question), embed(bq)) for bq in back_questions)
```

Score: 0.0–1.0. An answer that says "I don't know" scores near 0.

#### Lexical metrics (no LLM needed)

| Metric | What it measures |
|--------|-----------------|
| BLEU-4 | N-gram overlap between generated and gold answer (4-gram precision) |
| ROUGE-L | Longest common subsequence between generated and gold answer |

These are fast and deterministic. They penalise paraphrase, so a low BLEU/ROUGE with high faithfulness/relevance is expected — don't use them as the primary gate.

#### Latency

Record `time_to_first_token` (TTFT) and `time_to_last_token` (TTLT) for each question. Report p50 and p95.

### Output per question

```jsonl
{
  "question": "...",
  "gold_answer": "...",
  "generated_answer": "...",
  "citations": [...],
  "pipeline_status": "answered",
  "faithfulness": 0.91,
  "answer_relevance": 0.88,
  "bleu4": 0.31,
  "rouge_l": 0.54,
  "hit_rate": 1,
  "ndcg_at_5": 0.87,
  "ttft_ms": 312,
  "ttlt_ms": 1840,
  "evolution_type": "simple",
  "cache_hit": null
}
```

---

## Phase 4 — CI Regression Gate

**Script:** `rag/v2/evals/check_regression.py`

Reads the latest result file and `evals/results/baseline.json`. Fails if any metric drops more than its allowed slack from baseline.

### Baseline format

```json
{
  "locked_at": "2026-06-28T00:00:00Z",
  "locked_by": "manual after Phase 1 review",
  "metrics": {
    "hit_rate":         0.82,
    "mrr":              0.74,
    "ndcg_at_5":        0.79,
    "faithfulness":     0.88,
    "answer_relevance": 0.85,
    "bleu4":            0.29,
    "rouge_l":          0.51,
    "latency_p95_ms":   3200
  }
}
```

### Regression thresholds

| Metric | Max allowed drop | Rationale |
|--------|-----------------|-----------|
| `hit_rate` | 0.05 | Core signal — if the right chunk isn't found, nothing else matters |
| `ndcg_at_5` | 0.03 | Ranking quality; tighter than hit_rate |
| `mrr` | 0.04 | First-relevant-result position |
| `faithfulness` | 0.05 | Hallucination guard |
| `answer_relevance` | 0.05 | Relevance guard |
| `bleu4` | 0.07 | High variance lexically, wider slack |
| `rouge_l` | 0.06 | Same |
| `latency_p95_ms` | +500 ms | Speed regression |

### When to run in CI

```yaml
# .github/workflows/ci.yml addition (Phase 4)
- name: Retrieval regression gate
  if: contains(github.event.commits[0].modified, 'knowledge/retrieval') ||
      contains(github.event.commits[0].modified, 'knowledge/store')
  run: uv run python evals/run_retrieval_eval.py --check-regression
```

Run retrieval eval on every PR that touches `knowledge/retrieval/` or `knowledge/store/`. Generation eval is slower (~5 min for 80 questions) and runs nightly only.

---

## Metric Definitions

### Retrieval metrics

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Hit Rate@K** | `1 if any gold chunk in top-K else 0`, averaged | Fraction of questions where at least one gold chunk was retrieved |
| **MRR** | `1/rank_of_first_gold_chunk`, averaged | How high the first relevant result is ranked |
| **NDCG@K** | Normalised Discounted Cumulative Gain | Rewards gold chunks appearing higher in the list; penalises lower positions |
| **Precision@K** | `gold chunks in top-K / K` | Fraction of top-K results that are relevant |
| **Recall@K** | `gold chunks in top-K / total gold chunks` | Fraction of all gold chunks found in top-K |

### Generation metrics

| Metric | Range | Good threshold |
|--------|-------|---------------|
| **Faithfulness** | 0–1 | ≥ 0.85 |
| **Answer Relevance** | 0–1 | ≥ 0.80 |
| **BLEU-4** | 0–1 | ≥ 0.25 (domain text) |
| **ROUGE-L** | 0–1 | ≥ 0.45 |

### The NDCG-to-E2E gap

A high NDCG@5 (e.g. 0.80) does not guarantee high faithfulness. Reasons the gap exists:

- **NDCG is chunk-level; faithfulness is claim-level.** The right chunk was retrieved but the LLM cited a different part of it (or ignored it).
- **Context window overflow.** With 5 chunks × ~500 tokens = ~2500 tokens, plus system prompt and history, total context may exceed the small model's effective window.
- **Abstention suppresses E2E.** If the pipeline abstains (Gate 1 or Gate 3), the answer is empty — which is correct behaviour, not a generation failure. Filter abstentions out of faithfulness calculation.

---

## Failure Analysis

After each eval run, segment failures:

```python
df = pd.DataFrame(results)

# 1. Is the failure retrieval or generation?
retrieval_miss = df[df["hit_rate"] == 0]
generation_miss = df[(df["hit_rate"] == 1) & (df["faithfulness"] < 0.7)]

print(f"Retrieval misses: {len(retrieval_miss)} ({len(retrieval_miss)/len(df):.0%})")
print(f"Generation misses (chunks found, answer bad): {len(generation_miss)} ({len(generation_miss)/len(df):.0%})")

# 2. By question type
print(df.groupby("evolution_type")[["hit_rate","faithfulness"]].mean().round(3))

# 3. Cache hit vs miss
print(df.groupby(df["cache_hit"].notna())[["hit_rate","faithfulness"]].mean())
```

The most actionable split:
- **`hit_rate == 0`** → problem is retrieval. Fix embedding model, chunk size, hybrid search weights, or add more specific documents to the corpus.
- **`hit_rate == 1` + `faithfulness < 0.7`** → problem is generation. Fix prompt, model tier, context assembly, or working memory trimming.

---

## Baselines and Thresholds

Baselines are set manually after the first full eval run passes human review. They are not set speculatively.

**Initial target (before first run):**

| Metric | Aspirational target |
|--------|-------------------|
| Hit Rate@5 | ≥ 0.80 |
| NDCG@5 | ≥ 0.75 |
| Faithfulness | ≥ 0.85 |
| Answer Relevance | ≥ 0.80 |
| p95 latency | ≤ 4 s |

These are not locked baselines — they are aspirational targets for the first run. Lock the actual numbers after the first run.

---

## File Layout

```
rag/v2/
└── evals/
    ├── gold_dataset.jsonl            # generated + reviewed; committed to git
    ├── results/
    │   ├── baseline.json             # locked baseline; committed to git
    │   └── {YYYYMMDD_HHMMSS}.jsonl  # run results; NOT committed (gitignored)
    ├── generate_gold_dataset.py      # Phase 1: RAGAS generation
    ├── run_retrieval_eval.py         # Phase 2: retrieval metrics
    ├── run_generation_eval.py        # Phase 3: generation metrics + LLM judge
    ├── check_regression.py           # Phase 4: compare to baseline
    └── README.md                     # how to run the eval suite
```

`results/*.jsonl` are gitignored — only `baseline.json` and `gold_dataset.jsonl` are tracked.

---

## Implementation Plan

### Phase 1 — Gold Dataset (1–2 days)

1. Install RAGAS + langchain-anthropic in dev deps: `uv add --dev ragas langchain-anthropic langchain-community`
2. Write `evals/generate_gold_dataset.py`:
   - Load all docs from `rag/v2/documents/` via `langchain_community.document_loaders.DirectoryLoader`
   - Run `TestsetGenerator` with claude-haiku-4-5 + nomic-embed-text
   - Save raw output to `evals/gold_dataset_raw.jsonl`
3. Manually review: delete bad questions, fix wrong answers, label hard questions
4. Save final to `evals/gold_dataset.jsonl`
5. Commit dataset

**Blocker:** Anthropic API key must be available in the environment. Ollama must be running for embedding.

### Phase 2 — Retrieval Eval (1 day)

1. Write `evals/run_retrieval_eval.py`
2. Auth: call `POST /api/v2/auth/token` with a test user credential from `.env`
3. For each sample in `gold_dataset.jsonl`, call `POST /api/v2/search` and record returned chunk IDs
4. Compute Hit Rate, MRR, NDCG@5, Precision@5, Recall@5 using existing `knowledge/evaluation/metrics/retrieval.py`
5. Print summary table; save to `evals/results/{timestamp}.jsonl`

**Dependency:** The corpus must be seeded (`make seed`). The API must be running.

### Phase 3 — Generation Eval (1–2 days)

1. Write `evals/run_generation_eval.py`
2. For each sample, call `POST /api/v2/chat/stream`, drain the SSE stream, collect the full answer
3. Implement faithfulness + answer relevance judges using the Anthropic SDK directly (no LangChain dependency)
4. Compute BLEU-4 and ROUGE-L using `nltk` (already in dev deps) and `rouge-score`
5. Record TTFT and TTLT from SSE timing
6. Merge retrieval + generation results per question; save to `evals/results/{timestamp}.jsonl`

**Cost estimate:** 80 questions × 3 judge calls × ~1000 tokens each = ~240K tokens. At claude-haiku-4-5 pricing (~$0.25/1M input): ~$0.06 per full eval run. Negligible.

### Phase 4 — CI Regression Gate (0.5 days)

1. After Phase 2 first run: lock baseline in `evals/results/baseline.json`
2. Write `evals/check_regression.py`: load latest results + baseline, apply thresholds, `sys.exit(1)` on regression
3. Add CI step to `.github/workflows/ci.yml` (retrieval eval on retrieval-touching PRs; generation eval nightly)
4. Update `Makefile`:
   ```makefile
   eval-retrieval:
       uv run python evals/run_retrieval_eval.py
   eval-generation:
       uv run python evals/run_generation_eval.py
   eval-check:
       uv run python evals/check_regression.py
   ```

### Timeline

| Phase | Est. effort | Blocker |
|-------|-------------|---------|
| 1 — Gold dataset generation | 1–2 days | Anthropic API key, Ollama running, corpus seeded |
| 2 — Retrieval eval script | 1 day | API running, corpus seeded |
| 3 — Generation eval + judge | 1–2 days | Phases 1 + 2 done |
| 4 — CI gate | 0.5 days | Phase 2 baseline locked |
| **Total** | **3.5–5.5 days** | |

**Recommended order:** Phase 1 → Phase 2 → lock baseline → Phase 4 (gate on retrieval only) → Phase 3. Don't wait for generation eval to add the CI gate — retrieval regression detection is the most valuable guard.
