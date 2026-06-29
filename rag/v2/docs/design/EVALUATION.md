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

Use faithfulness + answer relevancy for corpora without ground truth. Add contextual precision/recall once you have gold answers.
