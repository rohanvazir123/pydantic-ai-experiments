# RAG v2 — Evaluation Guide

Two complementary approaches: **handcrafted gold rows** for deterministic regression gates, and **DeepEval** for LLM-judged quality metrics.

---

## Handcrafted Gold Rows

Gold rows live in `knowledge/evaluation/data/<corpus_id>.jsonl`. Each line is a `GoldSample`.

### Format

```json
{
  "id": "a1b2c3d4-...",
  "corpus_id": "default",
  "query": "What is the PTO policy?",
  "relevant_doc_sources": ["team-handbook"],
  "ground_truth_answer": "Employees receive 20 days of PTO per year, accrued monthly.",
  "difficulty": "easy",
  "tags": ["factual", "hr"]
}
```

`relevant_doc_sources` — substring-matched against `SearchResult.document_source`. Leave `ground_truth_answer` null when you only need retrieval metrics.

### Sample rows — NeuralFlow corpus

```jsonl
{"id": "00000001-0000-0000-0000-000000000001", "corpus_id": "default", "query": "What does NeuralFlow AI do?", "relevant_doc_sources": ["company-overview"], "ground_truth_answer": "NeuralFlow AI builds enterprise retrieval-augmented generation systems for knowledge-intensive workflows.", "difficulty": "easy", "tags": ["factual", "company"]}
{"id": "00000001-0000-0000-0000-000000000002", "corpus_id": "default", "query": "What is the PTO policy?", "relevant_doc_sources": ["team-handbook"], "ground_truth_answer": "Employees receive 20 days of paid time off per year, accrued at 1.67 days per month.", "difficulty": "easy", "tags": ["factual", "hr"]}
{"id": "00000001-0000-0000-0000-000000000003", "corpus_id": "default", "query": "Which tech stack does the company use?", "relevant_doc_sources": ["company-overview", "team-handbook"], "ground_truth_answer": null, "difficulty": "medium", "tags": ["multi-source"]}
{"id": "00000001-0000-0000-0000-000000000004", "corpus_id": "default", "query": "How does the performance review process work and when does it affect PTO accrual?", "relevant_doc_sources": ["team-handbook"], "ground_truth_answer": null, "difficulty": "hard", "tags": ["multi-hop", "hr"]}
{"id": "00000001-0000-0000-0000-000000000005", "corpus_id": "default", "query": "What is the capital of France?", "relevant_doc_sources": [], "ground_truth_answer": null, "difficulty": "easy", "tags": ["out-of-domain"]}
```

Row 3 has no ground truth — only retrieval metrics run. Row 5 has no relevant sources — tests that the system doesn't hallucinate an answer for an out-of-domain query.

### Retrieval metrics (no LLM needed)

| Metric | What it measures |
|--------|-----------------|
| Hit Rate@k | Any relevant doc in top-k? |
| MRR@k | How early does the first relevant result appear? |
| NDCG@k | Are relevant docs ranked above irrelevant ones? |
| Precision@k | Fraction of top-k that are relevant |
| Recall@k | Fraction of all known-relevant docs found |

Target baselines for the NeuralFlow sample corpus: Hit Rate@5 ≥ 0.85, MRR@5 ≥ 0.70.

---

## DeepEval

DeepEval provides LLM-judged metrics (faithfulness, answer relevance, contextual precision/recall) without needing handwritten ground truth for every row.

### Install

```bash
uv add deepeval
```

### OllamaJudge (local, no API key)

```python
from deepeval.models import OllamaJudge

judge = OllamaJudge(model="llama3.2:3b", base_url="http://localhost:11434")
```

Pass `judge` as the `model` argument to any metric.

### Sample test rows

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

# --- Row 1: factual, ground truth available ---
case_pto = LLMTestCase(
    input="What is the PTO policy?",
    actual_output="Employees get 20 days PTO per year, accrued monthly at 1.67 days.",
    expected_output="Employees receive 20 days of paid time off per year, accrued at 1.67 days per month.",
    retrieval_context=[
        "Section 4.2 — Paid Time Off: All full-time employees receive 20 days PTO per calendar year. "
        "PTO accrues at 1.67 days per month starting from the first day of employment."
    ],
)

# --- Row 2: multi-source, no ground truth ---
case_stack = LLMTestCase(
    input="Which tech stack does the company use?",
    actual_output="NeuralFlow AI uses FastAPI, pgvector, Apache AGE, Redis Streams, and Ollama for local LLM inference.",
    expected_output=None,
    retrieval_context=[
        "Our backend is built on FastAPI with asyncpg and pgvector for vector search.",
        "We use Apache AGE for knowledge graph storage and Redis Streams for async job queuing.",
        "Local LLM inference runs through Ollama; the default model is llama3.2:3b.",
    ],
)

# --- Row 3: out-of-domain — answer should admit ignorance ---
case_ood = LLMTestCase(
    input="What is the capital of France?",
    actual_output="I don't have information about that in the knowledge base.",
    expected_output=None,
    retrieval_context=[],
)
```

### Running metrics

```python
from deepeval import evaluate

faithfulness   = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)
relevancy      = AnswerRelevancyMetric(threshold=0.7, model=judge)
ctx_precision  = ContextualPrecisionMetric(threshold=0.7, model=judge)
ctx_recall     = ContextualRecallMetric(threshold=0.7, model=judge)
ctx_relevancy  = ContextualRelevancyMetric(threshold=0.7, model=judge)

evaluate(
    test_cases=[case_pto, case_stack, case_ood],
    metrics=[faithfulness, relevancy, ctx_precision, ctx_recall, ctx_relevancy],
)
```

### As pytest tests

```python
# tests/eval/test_deepeval_rag.py
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OllamaJudge
from deepeval.test_case import LLMTestCase

judge = OllamaJudge(model="llama3.2:3b", base_url="http://localhost:11434")

@pytest.mark.parametrize("case,metrics", [
    (
        LLMTestCase(
            input="What is the PTO policy?",
            actual_output="Employees get 20 days PTO per year.",
            expected_output="Employees receive 20 days of paid time off per year.",
            retrieval_context=["All employees receive 20 days PTO per calendar year."],
        ),
        [FaithfulnessMetric(threshold=0.7, model=judge),
         AnswerRelevancyMetric(threshold=0.7, model=judge)],
    ),
    (
        LLMTestCase(
            input="What does NeuralFlow AI do?",
            actual_output="NeuralFlow builds RAG systems for enterprise knowledge workflows.",
            expected_output="NeuralFlow AI builds enterprise retrieval-augmented generation systems.",
            retrieval_context=["NeuralFlow AI develops RAG infrastructure for knowledge-intensive enterprise use cases."],
        ),
        [FaithfulnessMetric(threshold=0.7, model=judge),
         AnswerRelevancyMetric(threshold=0.7, model=judge)],
    ),
])
def test_rag_quality(case, metrics):
    assert_test(case, metrics)
```

Run with:

```bash
cd rag/v2
uv run pytest tests/eval/test_deepeval_rag.py -v
```

### Metric quick reference

| Metric | Needs ground truth | What it checks |
|--------|--------------------|----------------|
| `FaithfulnessMetric` | No | Answer claims are supported by retrieved context |
| `AnswerRelevancyMetric` | No | Answer addresses the question asked |
| `ContextualPrecisionMetric` | Yes | Retrieved nodes that are relevant rank above irrelevant ones |
| `ContextualRecallMetric` | Yes | All ground-truth facts appear in retrieved context |
| `ContextualRelevancyMetric` | No | Retrieved context is relevant to the question |
| `GEval` | Yes | Custom criteria you write — catches what the above five miss |

Use faithfulness + answer relevancy for corpora without ground truth. Add contextual precision/recall once you have gold answers. Add `GEval` when the domain has legally or semantically significant qualifiers.

---

## GEval — Custom Correctness Criteria

`GEval` is a general-purpose LLM-as-judge metric. Instead of a fixed scoring rubric, you write the criteria yourself. The judge scores `actual_output` against `expected_output` according to your instructions.

### Why the standard metrics aren't enough

Consider this compliance-domain case:

```python
LLMTestCase(
    input="What does GLBA require?",
    actual_output="GLBA requires financial institutions to protect customer information.",
    expected_output="GLBA requires financial institutions to protect nonpublic personal information.",
    retrieval_context=[
        "GLBA requires financial institutions to protect customers' nonpublic personal information.",
        "The Safeguards Rule requires an information security program.",
    ],
)
```

| Metric | Score | Why |
|--------|-------|-----|
| Faithfulness | ~0.9 PASS | "customer information" is entailed by the context — no contradiction |
| AnswerRelevancy | ~1.0 PASS | Directly answers the question |
| ContextualPrecision | ~0.9 PASS | Most relevant chunk is ranked first |
| ContextualRecall | ~1.0 PASS | Expected output is covered by context |

Every metric passes. But the answer dropped **"nonpublic"** — a legally significant qualifier under GLBA. The Safeguards Rule applies to *nonpublic* personal information, not all customer information. A compliance audit would flag this.

### GEval catches it

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

legal_precision = GEval(
    name="Legal Precision",
    criteria=(
        "The actual output must preserve legally significant qualifiers from the expected output. "
        "Dropping 'nonpublic' from 'nonpublic personal information' is a failure. "
        "Replacing specific regulatory terms with broader informal language is a failure."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge,
)

legal_precision.measure(glba_case)
# Score: ~0.2 FAIL — judge flags the dropped qualifier with a reason
print(legal_precision.reason)
# "The actual output uses 'customer information' where the expected output specifies
#  'nonpublic personal information'. This omits a legally significant qualifier..."
```

### Adding GEval to the runner

In `scripts/run_deepeval.py`, set `geval_criteria` on any test case that needs precision checking. GEval only runs when both `geval_criteria` and `expected_output` are present:

```python
{
    "input": "What is the PTO and leave policy?",
    "expected_output": (
        "Employees receive paid time off that accrues over the year. "
        "The policy covers vacation days, sick leave, and public holidays."
    ),
    "geval_criteria": (
        "The actual output must preserve all specific leave categories from the expected output "
        "(vacation days, sick leave, public holidays). Dropping any category or replacing "
        "specific terms with vague phrases like 'various leave types' is a failure."
    ),
    "tags": ["hr", "benefits"],
},
```

### Writing good criteria

- **Be specific about what counts as a failure**, not just what counts as a pass. The judge reasons from failure cases.
- **Name the exact terms that must be preserved** ("nonpublic", "20 days", "accrues monthly").
- **One concern per criteria string** — if you have multiple independent checks, run multiple `GEval` instances.
- **Avoid tautologies** like "the answer must be correct" — the judge has nothing to reason against.
