# Domain-Specific RAG Benchmarking — Setup and Integration

How to evaluate a production RAG system against real benchmarks, build domain-specific eval sets, and interpret the results.

## Table of Contents

- [Why There Is No Single Spider2-lite for RAG](#why-there-is-no-single-spider2-lite-for-rag)
- [BEIR — Retrieval Component Evaluation](#beir--retrieval-component-evaluation)
- [FinanceBench — End-to-End Financial RAG](#financebench--end-to-end-financial-rag)
- [SQuAD and SQuAD 2.0 — Reading Comprehension Baseline](#squad-and-squad-20--reading-comprehension-baseline)
- [BioASQ — Biomedical Domain](#bioasq--biomedical-domain)
- [QASPER — Scientific Paper RAG](#qasper--scientific-paper-rag)
- [HotpotQA — Multi-Hop Reasoning](#hotpotqa--multi-hop-reasoning)
- [LegalBench and CUAD — Legal Domain](#legalbench-and-cuad--legal-domain)
- [RAGAS Synthetic Test Generation — Build Your Own Benchmark](#ragas-synthetic-test-generation--build-your-own-benchmark)
- [Selecting the Right Benchmark](#selecting-the-right-benchmark)
- [Wiring Your RAG Pipeline into a Benchmark](#wiring-your-rag-pipeline-into-a-benchmark)
- [Segmenting Failure Analysis](#segmenting-failure-analysis)
- [The NDCG-to-End-to-End Gap](#the-ndcg-to-end-to-end-gap)
- [Using Benchmarks as CI Regression Gates](#using-benchmarks-as-ci-regression-gates)

---

## Why There Is No Single Spider2-lite for RAG

Spider2-lite is clean to evaluate because SQL execution is deterministic: your query either produces the same result set as the gold query or it doesn't. RAG answers are open-ended text — two correct answers to the same question may share no words.

This forces RAG evaluation into one of three modes, each with trade-offs:

| Mode | Example | Deterministic? | Cost | Bias |
|------|---------|---------------|------|------|
| Exact/span match | SQuAD extractive answers | Yes | Free | Penalises paraphrase |
| Human annotation | FinanceBench, BioASQ factoid | Yes (numerical/factoid) | High | Gold answers may have multiple valid forms |
| LLM judge | RAGAS faithfulness | No | Medium | Judge shares biases with generator |

The practical answer: use a combination. BEIR for retrieval component evaluation (deterministic), a domain benchmark for end-to-end quality (human-annotated where available), and RAGAS for fast iteration on your own corpus.

---

## BEIR — Retrieval Component Evaluation

**Repo:** https://github.com/beir-cellar/beir  
**What it tests:** Your embedding model and reranker, not the full RAG pipeline.  
**Metric:** NDCG@k, Recall@k, Precision@k, MRR@k

### Setup

```bash
pip install beir
```

### Available datasets (no credentials needed for most)

```python
from beir import util

DATASETS = {
    # Scientific
    "scifact":           "Fact-checking scientific claims (5K docs, 300 queries)",
    "scidocs":           "Scientific document retrieval (25K docs, 1K queries)",
    "nfcorpus":          "Medical information retrieval (3.6K docs, 323 queries)",
    "trec-covid":         "COVID biomedical retrieval (171K docs, 50 queries)",

    # Financial
    "fiqa":              "Financial opinion QA from StackExchange/Reddit (57K docs, 648 queries)",

    # Legal/argumentative
    "arguana":           "Counterargument retrieval (8.7K docs, 1.4K queries)",
    "webis-touche2020":  "Argument retrieval (382K docs, 49 queries)",

    # Fact-checking
    "fever":             "Fact verification (5.4M docs, 6.7K queries)",
    "climate-fever":     "Climate fact-checking (5.4M docs, 1.5K queries)",

    # Open-domain QA
    "nq":                "Natural Questions (2.68M docs, 3.4K queries)",
    "hotpotqa":          "HotpotQA multi-hop (5.23M docs, 7.4K queries)",
    "quora":             "Duplicate question detection (523K docs, 10K queries)",
    "dbpedia-entity":    "Entity retrieval (4.6M docs, 400 queries)",
    "msmarco":           "Web passage retrieval (8.84M docs, 6.98K queries)",
}

# Download a dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
data_path = util.download_and_unzip(url, "/tmp/beir_datasets")
```

### Run evaluation against your embedding model

```python
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

# Load dataset
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
# corpus: {doc_id: {"title": str, "text": str}}
# queries: {query_id: query_text}
# qrels: {query_id: {doc_id: relevance_score}}   relevance is 0 or 1 for most datasets

# Evaluate your embedding model
model = DRES(models.SentenceBERT("your-embedding-model"), batch_size=32)
retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1, 5, 10, 100])

results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(f"NDCG@10:    {ndcg['NDCG@10']:.4f}")
print(f"Recall@100: {recall['Recall@100']:.4f}")
print(f"MRR@10:     {_map['MRR@10']:.4f}")
```

### Evaluating a reranker separately from the embedding model

BEIR's standard evaluation tests the embedding model alone. To measure the reranker's contribution:

```python
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoderReranker

# Stage 1: embedding retrieval (top-100)
dense_model = DRES(models.SentenceBERT("your-embedding-model"), batch_size=32)
dense_retriever = EvaluateRetrieval(dense_model, k_values=[100])
stage1_results = dense_retriever.retrieve(corpus, queries)

# Stage 2: reranker (top-100 → top-10)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=128)
reranked_results = reranker.rerank(corpus, queries, stage1_results, top_k=10)

# Evaluate each stage
ndcg_dense,    *_ = dense_retriever.evaluate(qrels, stage1_results,  [1,5,10])
ndcg_reranked, *_ = dense_retriever.evaluate(qrels, reranked_results, [1,5,10])

print(f"NDCG@10 dense-only: {ndcg_dense['NDCG@10']:.4f}")
print(f"NDCG@10 reranked:   {ndcg_reranked['NDCG@10']:.4f}")
print(f"Reranker gain:      {ndcg_reranked['NDCG@10'] - ndcg_dense['NDCG@10']:+.4f}")
```

### What BEIR misses about your RAG pipeline

BEIR measures whether the right document is retrieved — it does not measure:
- Whether the right **chunk** within the document is retrieved (BEIR operates at document level)
- Whether the LLM generates a faithful answer from the retrieved document
- Whether the context assembly (ordering, deduplication) affects answer quality
- Latency or token cost

Use BEIR for: comparing embedding models, measuring the reranker's lift, tracking retrieval regression in CI. Do not use it as your sole quality metric for a production RAG system.

---

## FinanceBench — End-to-End Financial RAG

**Repo:** https://github.com/patronus-ai/financebench  
**What it tests:** Full pipeline — retrieve from real PDFs, generate a correct numerical/factual answer  
**Scale:** 150 open-source examples (10,231 total, not fully public)

### Data format

```json
{
  "financebench_id": "financebench_id_00001",
  "question": "What is Apple's revenue for fiscal year 2022?",
  "answer": "$394.33 billion",
  "evidence": [
    {
      "evidence_text": "Net sales: $394,328 million for fiscal 2022",
      "page_number": 31,
      "ticker": "AAPL",
      "period_of_report": "FY2022",
      "doc_name": "AAPL_2022_10K",
      "doc_type": "10-K"
    }
  ],
  "justification": "Revenue is Net sales from the Consolidated Statements of Operations"
}
```

### Running your pipeline against FinanceBench

```python
import json
from pathlib import Path
from your_rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    document_store=...,   # pre-indexed SEC filings
    retriever=...,
    generator=...,
)

examples = [json.loads(l) for l in open("financebench_open_source.jsonl")]
results = []

for ex in examples:
    predicted_answer = pipeline.query(ex["question"])
    results.append({
        "id": ex["financebench_id"],
        "question": ex["question"],
        "gold_answer": ex["answer"],
        "predicted": predicted_answer,
        "gold_evidence_page": ex["evidence"][0]["page_number"],
    })

# Save for evaluation
with open("financebench_predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
```

### Scoring

FinanceBench does not ship an automated scorer — human annotators evaluated the original paper. For automated scoring, use an LLM judge:

```python
from openai import OpenAI
client = OpenAI()

def score_answer(question: str, gold: str, predicted: str) -> int:
    """Returns 1 (correct) or 0 (incorrect)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Gold answer: {gold}\n"
                f"Predicted answer: {predicted}\n\n"
                "Is the predicted answer correct? It is correct if it conveys the same "
                "numerical value or fact as the gold answer, allowing for formatting differences "
                "(e.g. '$394.33 billion' and '$394,328 million' are the same). "
                "Reply with only 'correct' or 'incorrect'."
            )
        }]
    )
    return 1 if "correct" in response.choices[0].message.content.lower() else 0

scores = [score_answer(r["question"], r["gold_answer"], r["predicted"]) for r in results]
print(f"Accuracy: {sum(scores)/len(scores):.1%}  ({sum(scores)}/{len(scores)})")
```

### Common failure modes in financial RAG

The original FinanceBench paper found GPT-4 got 81% wrong. The main failure classes:

| Failure | Cause | Fix |
|---------|-------|-----|
| Wrong document retrieved | Many companies + many years — retrieval fetches wrong year's 10-K | Add document metadata filters (ticker, period) to retrieval |
| Right document, wrong table row | Revenue from continuing operations vs total revenue | Richer table metadata; table-aware retrieval |
| Correct value, wrong unit | $394M vs $394B (millions vs billions) | Unit extraction post-processing |
| Calculation required | Question asks for YoY growth, answer requires subtraction | Agentic step: retrieve two values, compute |
| Fiscal year confusion | Apple's fiscal year ends in September — "FY2022" ≠ calendar 2022 | Fiscal calendar metadata in schema |

---

## SQuAD and SQuAD 2.0 — Reading Comprehension Baseline

**URL:** https://rajpurkar.github.io/SQuAD-explorer/

SQuAD is a reading comprehension benchmark where every answer is an **extractive span** from a Wikipedia passage. The model must find the exact text in the passage that answers the question.

### SQuAD 1.1 vs SQuAD 2.0

| Version | Questions | Key difference |
|---------|-----------|---------------|
| SQuAD 1.1 | 100K | Every question has an answer in the passage |
| SQuAD 2.0 | 150K | ~50K questions have no answer in the passage — tests when to abstain |

SQuAD 2.0 is more realistic for RAG: it tests whether your system correctly says "I don't know" when the retrieved context doesn't contain the answer.

### Is there a "lite" version?

There is no official SQuAD-lite. However, common practice is to use a stratified sample:

```python
from datasets import load_dataset

# Full SQuAD 2.0 validation set: 11,873 examples
squad = load_dataset("squad_v2", split="validation")

# Sample a representative 500-example subset
import random
random.seed(42)

# Balance: ~50% answerable, ~50% unanswerable
answerable   = [ex for ex in squad if ex["answers"]["text"]]
unanswerable = [ex for ex in squad if not ex["answers"]["text"]]

lite = (
    random.sample(answerable, 250) +
    random.sample(unanswerable, 250)
)
print(f"SQuAD-lite: {len(lite)} examples ({len(answerable[:250])} answerable, {len(unanswerable[:250])} unanswerable)")
```

### Running your RAG system on SQuAD 2.0

SQuAD provides the passage (context) directly — you can test generation quality without running retrieval, or inject your own retriever to fetch from Wikipedia:

```python
from datasets import load_dataset
from your_rag_pipeline import RAGPipeline

squad = load_dataset("squad_v2", split="validation")
pipeline = RAGPipeline(...)

exact_matches, f1_scores = [], []

for ex in squad.select(range(500)):  # SQuAD-lite
    # Option A: use provided context (tests generation only)
    context = ex["context"]

    # Option B: retrieve from Wikipedia using the question (tests full pipeline)
    # context = pipeline.retrieve(ex["question"])

    predicted = pipeline.generate(question=ex["question"], context=context)
    gold_answers = ex["answers"]["text"]  # list of valid answer strings

    if not gold_answers:
        # Unanswerable — correct if model abstains
        em = 1 if is_abstention(predicted) else 0
        f1 = em
    else:
        em = compute_exact_match(predicted, gold_answers)
        f1 = compute_f1(predicted, gold_answers)

    exact_matches.append(em)
    f1_scores.append(f1)

print(f"Exact Match: {sum(exact_matches)/len(exact_matches):.1%}")
print(f"F1 Score:    {sum(f1_scores)/len(f1_scores):.1%}")
```

### SQuAD metrics

**Exact Match (EM):** 1 if predicted answer string exactly matches any gold answer (after normalisation: lowercase, remove punctuation, remove articles).

**F1:** Token-level overlap between prediction and the best-matching gold answer. Partial credit for getting some tokens right.

```python
import re
from collections import Counter

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def compute_exact_match(pred: str, golds: list[str]) -> int:
    pred_norm = normalize(pred)
    return int(any(pred_norm == normalize(g) for g in golds))

def compute_f1(pred: str, golds: list[str]) -> float:
    pred_tokens = normalize(pred).split()
    best_f1 = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall    = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1
```

### What SQuAD tests vs what it misses

**Tests:** Whether the model can extract the exact answer span from a provided passage. Whether the model correctly abstains when the answer is not present (SQuAD 2.0).

**Misses:** Retrieval quality (passage is provided). Answer synthesis across multiple passages. Open-ended questions where the answer is not an extractable span. Domain-specific terminology.

**When to use SQuAD:** As a reading comprehension sanity check — if your system performs poorly on SQuAD 2.0 with the passage provided directly, the problem is in generation and grounding, not retrieval. Fix generation first.

---

## BioASQ — Biomedical Domain

**URL:** http://bioasq.org  
**What it tests:** Retrieval + answer generation over biomedical literature (PubMed)  
**Question types:** Yes/No, factoid (single entity), list (multiple entities), summary

BioASQ requires registration to download. The factoid and yes/no questions are useful for automated scoring; summary questions require human evaluation.

**When to use:** If your RAG system operates in a medical or life sciences domain. BioASQ questions are genuinely hard — they require multi-hop reasoning over PubMed abstracts and precise biomedical entity knowledge.

---

## QASPER — Scientific Paper RAG

**Dataset:** https://huggingface.co/datasets/allenai/qasper

Q&A over NLP research papers. Each question targets a specific paper; the answer may be extractive (a span from the paper), abstractive (a synthesised answer), or yes/no. Includes evidence annotations — which paragraph(s) the answer comes from.

```python
from datasets import load_dataset
qasper = load_dataset("allenai/qasper", split="validation")

# Each example:
# {
#   "id": "1902.00571",
#   "title": "BERT: Pre-training of Deep Bidirectional Transformers...",
#   "full_text": {...},
#   "qas": [{
#     "question": "What tasks does the model achieve state-of-the-art on?",
#     "answers": [{"answer": {"free_form_answer": "11 NLP tasks", "extractive_spans": [...], ...}}]
#   }]
# }
```

QASPER tests whether your system can retrieve the right section of a long academic paper and generate a grounded answer. It is a good proxy for internal knowledge base RAG (policy documents, technical manuals).

---

## HotpotQA — Multi-Hop Reasoning

**Dataset:** https://huggingface.co/datasets/hotpot_qa

Multi-hop questions requiring connecting facts from two Wikipedia paragraphs. Gold supporting facts are annotated — you can measure whether you retrieved the right passages AND generated the right answer.

```python
from datasets import load_dataset
hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
# "distractor" mode: 10 paragraphs provided, 2 are supporting, 8 are distractors
# Tests whether retrieval/generation can identify the right 2 from the 10

# Each example:
# {
#   "question": "What is the hometown of the director of Crouching Tiger, Hidden Dragon?",
#   "answer": "Beigang, Yunlin County, Taiwan",
#   "supporting_facts": {"title": ["Ang Lee", "Crouching Tiger..."], "sent_id": [0, 2]},
#   "context": {"title": [...], "sentences": [...]},
# }
```

HotpotQA is the best benchmark for testing multi-hop retrieval — whether your system can surface supporting facts across two separate documents and synthesise them.

---

## LegalBench and CUAD — Legal Domain

**LegalBench:** https://huggingface.co/datasets/nguha/legalbench — 162 legal reasoning tasks  
**CUAD:** https://huggingface.co/datasets/theatticusproject/cuad — Contract clause extraction

CUAD (Contract Understanding Atticus Dataset) is particularly useful for legal RAG: 510 contracts, 41 question types (each a type of contract clause), with span annotations. Your RAG system must retrieve the right clause and extract or generate the right answer.

```python
from datasets import load_dataset
cuad = load_dataset("theatticusproject/cuad", split="test")

# Each example:
# {
#   "context": "... full contract text ...",
#   "question": "Does the contract contain a limitation of liability clause?",
#   "answers": {"text": ["LIMITATION OF LIABILITY..."], "answer_start": [4521]}
# }
```

---

## RAGAS Synthetic Test Generation — Build Your Own Benchmark

When no off-the-shelf benchmark covers your corpus, generate domain-specific eval questions from your own documents.

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# Load your documents
loader = DirectoryLoader("./your_corpus/", glob="**/*.pdf")
docs = loader.load()

# Generate test set
generator = TestsetGenerator(
    llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")),
    embedding_model=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
)
testset = generator.generate_with_langchain_docs(
    docs,
    testset_size=200,
)

# Convert to DataFrame
df = testset.to_pandas()
# Columns: question, ground_truth, reference_contexts, evolution_type
# evolution_type: simple | reasoning | multi_context | conditional

df.to_json("domain_eval_set.jsonl", orient="records", lines=True)
```

### Question evolution types

| Type | What it tests |
|------|--------------|
| `simple` | Direct factual lookup from one passage |
| `reasoning` | Requires inference from one passage |
| `multi_context` | Requires synthesising two passages |
| `conditional` | Conditional logic ("if X, then what Y?") |

Aim for a distribution roughly matching your production query types.

### Mitigating synthetic eval bias

Synthetic questions are generated by an LLM and therefore biased toward:
- Questions the LLM can answer well
- Question styles the generation model finds natural
- Passages that are well-written and unambiguous

Mitigations:
1. **Human review:** Have domain experts review and reject synthetic questions that feel artificial or trivially easy. Target: reject 20–30% and replace.
2. **Adversarial augmentation:** Manually add questions that are known hard cases — questions where the answer requires the exact section that is hardest to retrieve, or questions that test boundary conditions.
3. **Production query injection:** After going live, sample real user queries and add them (with human-annotated answers) to the eval set. These are the most realistic questions.

### Cost of test generation

```
200 questions × ~2,000 tokens per generation (question + context + answer) = 400K tokens
At GPT-4o: 400K × $2.50/1M = $1.00 per 200-question eval set generation

Running RAGAS evaluation on 200 questions:
  Faithfulness: 200 × ~500 tokens = 100K tokens input + 50K output
  Answer relevance: 200 × ~300 tokens = 60K tokens
  Total: ~$0.50 per evaluation run

A full generation + evaluation cycle: ~$1.50
```

---

## Selecting the Right Benchmark

```
Your goal → Use this benchmark

Test your embedding model / reranker in isolation
  → BEIR (scifact for science, fiqa for finance, nfcorpus for medical)

Test full pipeline on financial documents
  → FinanceBench (150 free examples)

Test reading comprehension and abstention
  → SQuAD 2.0 or SQuAD-lite (500 sampled examples)

Test multi-hop reasoning
  → HotpotQA

Test on your own domain corpus
  → RAGAS synthetic generation

Test legal contract understanding
  → CUAD

Test scientific paper QA
  → QASPER

Test biomedical domain
  → BioASQ (requires registration)
```

---

## Wiring Your RAG Pipeline into a Benchmark

Generic harness that works with any benchmark:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkExample:
    question:      str
    context:       str | None      # None = pipeline must retrieve
    gold_answer:   str | list[str]
    metadata:      dict

class BenchmarkHarness:

    def __init__(self, pipeline, scorer: Callable, use_provided_context: bool = False):
        self.pipeline = pipeline
        self.scorer   = scorer
        self.use_provided_context = use_provided_context

    def evaluate(self, examples: list[BenchmarkExample]) -> dict:
        results = []
        for ex in examples:
            if self.use_provided_context and ex.context:
                answer = self.pipeline.generate(ex.question, ex.context)
            else:
                answer = self.pipeline.query(ex.question)  # full retrieval + generation

            score = self.scorer(ex.question, ex.gold_answer, answer)
            results.append({
                "question": ex.question,
                "gold":     ex.gold_answer,
                "predicted": answer,
                "score":    score,
                **ex.metadata,
            })

        df = pd.DataFrame(results)
        return {
            "overall_accuracy": df["score"].mean(),
            "by_type": df.groupby("type")["score"].mean().to_dict() if "type" in df else {},
            "n": len(df),
        }
```

---

## Segmenting Failure Analysis

After running any benchmark, segment failures before drawing conclusions:

```python
import pandas as pd

df = pd.DataFrame(results)

print("=== Failure Analysis ===")

# By question type
if "evolution_type" in df.columns:
    print("\nBy question type:")
    print(df.groupby("evolution_type")["score"].agg(["mean", "count"]).round(3))

# By answer type (extractive vs abstractive vs yes/no)
if "answer_type" in df.columns:
    print("\nBy answer type:")
    print(df.groupby("answer_type")["score"].agg(["mean", "count"]).round(3))

# Failures where retrieval was likely wrong (answer not mentioned in retrieved context)
if "retrieved_context" in df.columns:
    df["answer_in_context"] = df.apply(
        lambda r: any(
            str(r["gold"]).lower()[:20] in r["retrieved_context"].lower()
        ), axis=1
    )
    retrieval_miss = df[~df["answer_in_context"]]
    print(f"\nRetrieval misses: {len(retrieval_miss)} ({len(retrieval_miss)/len(df):.1%})")
    print(f"Score when answer IS in context:  {df[df['answer_in_context']]['score'].mean():.3f}")
    print(f"Score when answer NOT in context: {df[~df['answer_in_context']]['score'].mean():.3f}")
```

The last block is the most actionable: if score-when-answer-is-in-context is high (0.85) but score-when-not-in-context is low (0.12), your problem is retrieval. If both are low, your problem is generation.

---

## The NDCG-to-End-to-End Gap

A common confusion: a system scores NDCG@10=0.78 on BEIR but only 62% accuracy on end-to-end answer quality. What explains the gap?

**Reason 1 — BEIR is document-level; RAG is chunk-level.**
NDCG@10 measures whether the right document appears in the top-10 retrieved documents. If the right document is retrieved but the specific chunk containing the answer is not the top chunk passed to the LLM, end-to-end accuracy suffers. BEIR cannot measure this.

**Reason 2 — Retrieval is necessary but not sufficient.**
NDCG=0.78 means in 78% of queries the right document is retrieved. End-to-end accuracy depends on: (a) the right chunk within the document being surfaced, (b) the LLM reading it faithfully, (c) the answer format matching what the scorer expects. Each step has its own loss.

**Reason 3 — BEIR query distribution ≠ your production distribution.**
BEIR scifact has short factual queries. Your production queries may be longer, more complex, or use domain vocabulary not well-represented in the BEIR dataset. A high BEIR NDCG does not transfer if query distributions differ.

**What to prioritise:**
If NDCG is high but E2E is low: problem is downstream of retrieval — chunk selection, context assembly, or generation.
If NDCG is low: fix retrieval first (embedding model, reranker, chunk size, hybrid vs dense).

---

## Using Benchmarks as CI Regression Gates

```bash
# CI script: run BEIR scifact after any retrieval change
pip install beir datasets

python -c "
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

corpus, queries, qrels = GenericDataLoader('scifact').load('test')
model = DRES(models.SentenceBERT('your-embedding-model'), batch_size=32)
retriever = EvaluateRetrieval(model, score_function='cos_sim')
results = retriever.retrieve(corpus, queries)
ndcg, *_ = retriever.evaluate(qrels, results, [10])
score = ndcg['NDCG@10']
baseline = 0.74

if score < baseline - 0.02:
    print(f'REGRESSION: NDCG@10={score:.3f} < {baseline:.3f} - 0.02')
    exit(1)
print(f'PASS: NDCG@10={score:.3f} (baseline {baseline:.3f})')
"
```

**Runtime:** BEIR scifact has 300 queries and 5K documents. Dense retrieval takes 2–5 minutes. Fast enough for a CI gate.

For a SQuAD-lite gate (generation quality):
```bash
python scripts/squad_lite_eval.py \
    --n_examples 200 \
    --baseline_em 0.72 \
    --baseline_f1 0.81
```

Run the BEIR gate on every PR touching the embedding model or reranker. Run the SQuAD-lite gate on every PR touching the prompt or generation model. Update baselines when you intentionally improve the system.
