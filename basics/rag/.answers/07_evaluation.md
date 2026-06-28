# Evaluation — Answers

## Q31. What metrics do you use to evaluate a RAG system?

**Answer:**

RAG evaluation requires measuring both retrieval and generation independently. Conflating them hides whether a problem is in the retrieval layer or the generation layer.

**Retrieval metrics:**

*Context Recall:* Of all the information needed to answer the question, what fraction is present in the retrieved context? Requires ground truth "relevant chunks" labels.

*Context Precision:* Of all the retrieved chunks, what fraction is actually relevant to the question? High context precision means the retrieval is not including noise.

*Hit Rate @ k:* Does the correct chunk appear in the top-k retrieved results? Binary: 1 if yes, 0 if no. Simple and interpretable.

*Mean Reciprocal Rank (MRR):* The reciprocal of the rank position of the first relevant chunk. MRR=1 means the correct chunk was ranked first. MRR=0.5 means it was ranked second. Captures not just whether the right chunk was retrieved but how highly it was ranked.

**Generation metrics:**

*Faithfulness:* What fraction of claims in the generated answer are directly supported by the retrieved context? Measures hallucination. Computed by decomposing the answer into atomic claims and checking each against the context.

*Answer Relevance:* Does the generated answer actually address the question that was asked? A faithful but off-topic answer scores high on faithfulness and low on relevance. Computed by embedding the question and the answer and measuring semantic similarity (or by using an LLM judge).

*Answer Correctness:* If ground-truth answers exist, how well does the generated answer match them? This combines faithfulness and relevance.

**What each metric fails to capture:**

| Metric | What it misses |
|--------|---------------|
| Context Recall | A chunk can be "relevant" but not contain the actual answer |
| Context Precision | A chunk can be "irrelevant" but help the LLM understand context |
| Faithfulness | A faithful answer can be misleadingly incomplete |
| Answer Relevance | An answer can be relevant but factually wrong |
| Answer Correctness | Requires ground truth — unavailable for open-ended questions |

**Token cost of automated evaluation:**

Faithfulness evaluation: ~$0.002/query with a small LLM (claim extraction + verification)
Answer relevance: ~$0.001/query (embedding similarity, no LLM call)
Full RAGAS evaluation: ~$0.01–0.05/query depending on the judge model

At 10,000 queries/day, automated evaluation costs $100–500/day. Sample 5% of production queries for evaluation rather than evaluating 100%.

---

## Q32. How do you build a ground-truth evaluation set for RAG?

**Answer:**

The challenge: correct answers in RAG are often synthesised from multiple sources, not a single extractable string. A ground-truth answer is inherently subjective.

**Construction approach:**

*Step 1 — Seed questions from domain experts:*
Work with domain experts to generate 100–200 questions they would genuinely ask the knowledge base. The expert writes the question AND the ideal answer AND cites the source documents. This gives you: question, ground_truth_answer, ground_truth_chunks.

*Step 2 — Schema-driven coverage:*
For each document section, generate at least: one factual lookup question, one inference question (requires synthesising two facts from the same document), one comparison question (requires two documents). Ensures no document type is unrepresented.

*Step 3 — Adversarial questions:*
Questions designed to fail in specific ways:
- Questions that are out-of-scope (ground truth: "I don't know")
- Questions with answers in unexpected sections (tests retrieval breadth)
- Questions requiring multi-hop reasoning
- Questions with ambiguous terms that exist in multiple documents

*Step 4 — Synthetic QA generation (scales to thousands of examples):*
For each chunk in the corpus, use an LLM to generate questions that the chunk answers:

```python
def generate_qa_pairs(chunk: str, llm) -> list[QAPair]:
    response = llm.complete(
        f"Given this document passage, generate 3 questions that this passage answers. "
        f"For each question, provide the answer drawn strictly from the passage.\n\n"
        f"Passage: {chunk}\n\n"
        f"Format: Q: [question]\nA: [answer]\nEvidence: [exact quote]"
    )
    return parse_qa_pairs(response)
```

*Caveat:* Synthetic QA pairs are biased toward well-structured, informative chunks. Short, contextual chunks (table captions, section headers) generate poor synthetic questions. Human-generated questions must supplement synthetic ones.

**Dataset structure:**

```json
{
  "id": "eval_0042",
  "question": "What is the notice period for an employee resigning?",
  "ground_truth_answer": "Employees must provide a minimum of 2 weeks written notice for positions up to Manager level, and 4 weeks for Director and above.",
  "ground_truth_chunks": ["doc_012_chunk_034", "doc_012_chunk_035"],
  "difficulty": "medium",
  "question_type": "factual_lookup",
  "requires_multi_document": false
}
```

---

## Q33. How do you detect hallucinations automatically at scale?

**Answer:**

**Method 1 — NLI-based entailment checking (fast, cheap):**

Use a Natural Language Inference model to check whether each claim in the generated answer is entailed by the retrieved context.

```python
from transformers import pipeline
nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def check_entailment(premise: str, hypothesis: str) -> float:
    """Returns entailment probability (0-1)."""
    result = nli(f"{premise} [SEP] {hypothesis}")
    return next(r["score"] for r in result if r["label"] == "ENTAILMENT")
```

For each claim (hypothesis) in the answer, check against each retrieved chunk (premise). A claim not entailed by any retrieved chunk is a hallucination candidate.

*Cost:* NLI models run locally (no API cost). Latency: 10–50ms per (claim, chunk) pair on CPU.
*Limitation:* NLI models trained on general data may not reliably handle domain-specific text. Fine-tune on domain examples for better reliability.

**Method 2 — LLM-as-judge (slower, more accurate):**

```python
def llm_faithfulness_check(answer: str, context: str, judge_llm) -> dict:
    response = judge_llm.complete(
        f"Context: {context}\n\n"
        f"Answer: {answer}\n\n"
        f"For each factual claim in the answer, determine if it is supported by the context. "
        f"Return JSON: {{claims: [{{claim: str, supported: bool, evidence: str}}]}}"
    )
    return parse_json(response)
```

*Cost:* One LLM call per answer. With GPT-4o-mini: ~$0.002/answer. Reliable for most domains without fine-tuning.

**Method 3 — SelfCheckGPT (probabilistic hallucination detection):**

Generate the same answer N times at temperature > 0. Measure consistency across generations. Inconsistent facts across multiple generations are likely hallucinations (the model has low confidence in them).

```python
def selfcheck_score(question: str, context: str, llm, n_samples=5) -> float:
    answers = [llm.generate(question, context, temperature=0.7) for _ in range(n_samples)]
    # measure pairwise claim consistency across answers
    # high consistency → high confidence → lower hallucination probability
    return compute_consistency(answers)
```

*Cost:* N generation calls per query (5–10× cost). Only practical for high-stakes queries or offline evaluation, not real-time production use.

**At-scale strategy:**
Run NLI-based checking on 100% of production queries (fast, free, catches gross hallucinations). Run LLM-judge on 5% sampled queries and on any response flagged by NLI checking. Route to human review anything below a faithfulness threshold.

---

## Q34. How do you evaluate retrieval quality separately from generation quality?

**Answer:**

**Why independent evaluation matters:**

If you evaluate only end-to-end answer quality, you cannot tell whether a bad answer is caused by:
- Retrieval failure (correct chunk not retrieved)
- Context assembly failure (correct chunk retrieved but not in the final context)
- Generation failure (correct chunk in context but LLM generated wrong answer)

Each has a different fix: retrieval failure → improve embedding/reranking; assembly failure → improve selection or ordering; generation failure → improve prompt or model.

**Retrieval-only evaluation:**

Build a retrieval evaluation set: for each question, manually identify the correct source chunks (the chunks that, if retrieved, contain sufficient information to answer the question).

Metrics (computed without running the LLM):
- `recall@k`: what fraction of golden chunks appear in the top-k retrieved chunks?
- `precision@k`: what fraction of retrieved chunks are relevant?
- `NDCG@k`: Normalised Discounted Cumulative Gain — measures both presence and rank position of relevant chunks

```python
def evaluate_retrieval(eval_set, retrieval_fn, k=10):
    results = []
    for item in eval_set:
        retrieved_ids = retrieval_fn(item.question, k=k)
        golden_ids = set(item.ground_truth_chunk_ids)
        
        recall = len(set(retrieved_ids) & golden_ids) / len(golden_ids)
        precision = len(set(retrieved_ids) & golden_ids) / k
        results.append({"recall": recall, "precision": precision})
    
    return {
        "recall@k": mean(r["recall"] for r in results),
        "precision@k": mean(r["precision"] for r in results),
    }
```

**Generation-only evaluation (with oracle context):**

Inject the ground-truth chunks directly (bypassing retrieval) and evaluate only the LLM's ability to generate a correct answer from perfect context. If generation with oracle context is poor, the problem is the LLM/prompt, not retrieval.

This tells you the ceiling: the best answer quality achievable if retrieval were perfect. If this ceiling is 95% and your end-to-end performance is 75%, 20pp is being lost to retrieval. If this ceiling is 85%, you have a fundamental generation problem regardless of retrieval quality.

---

## Q35. What is RAGAS and what are its limitations?

**Answer:**

RAGAS (Retrieval Augmented Generation Assessment) is an automated evaluation framework that computes a suite of metrics — faithfulness, answer relevance, context precision, context recall, answer correctness — using an LLM as a judge.

**How it works:**
RAGAS uses an LLM (typically GPT-4 or GPT-4o) to:
1. Decompose the generated answer into atomic claims (for faithfulness)
2. Verify each claim against the retrieved context
3. Generate questions from the answer and check if they match the original question (answer relevance)
4. Check which retrieved chunks are relevant to the question (context precision/recall)

**Specific limitations:**

*Limitation 1 — LLM judge bias:*
RAGAS uses GPT-4 as the judge. If your RAG system also uses GPT-4 for generation, the judge and the system share the same biases. A claim that GPT-4 tends to make will be rated as "supported" by the GPT-4 judge even if it is hallucinated. Use a different model family for judging than for generation.

*Limitation 2 — Faithfulness score gaming:*
A system that generates very short, highly hedged answers ("According to the document, possibly...") scores high on faithfulness but poorly on usefulness. RAGAS does not measure answer completeness — a one-sentence answer scores the same as a comprehensive one if both are faithful.

*Limitation 3 — Answer relevance is a proxy:*
RAGAS measures answer relevance by generating questions from the answer and checking if they resemble the original question. This is an indirect proxy. An answer that directly addresses the question using different vocabulary can score poorly.

*Limitation 4 — Context recall requires ground truth chunks:*
To compute context recall, RAGAS needs to know which chunks are "correct" — this requires a labeled ground-truth dataset. For new corpora without labeled data, context recall cannot be computed.

*Limitation 5 — Cost:*
A full RAGAS evaluation runs 5–10 LLM calls per query. At $0.01 per call: $0.05–0.10 per query evaluated. For 10,000 production queries/day, full RAGAS evaluation costs $500–1,000/day. Only practical on a sampled subset (1–5%).

**What to supplement RAGAS with:**
- NLI-based faithfulness checking at scale (cheap, runs on all queries)
- Human evaluation for subjective quality (answer helpfulness, tone, completeness)
- Domain-specific correctness checks (for factual domains, verify key facts against a ground-truth database)

---

## Q36. How do you evaluate a RAG system when there is no single correct answer?

**Answer:**

Open-ended questions ("Summarise the key risks in the due diligence report", "What are the main arguments made in the consultation responses?") have no single correct answer. Traditional accuracy metrics cannot be applied.

**Evaluation criteria for open-ended answers:**

*Coverage:* Does the answer include the key points that a domain expert would identify? Build a "key facts checklist" for each question and measure what fraction of checklist items appear in the generated answer.

```python
key_facts = [
    "mentions liquidity risk",
    "mentions regulatory exposure in EU market",
    "quantifies the potential fine as $50-200M",
    "notes management's response plan",
]
coverage_score = sum(is_mentioned(answer, fact) for fact in key_facts) / len(key_facts)
```

*Coherence:* Is the answer logically organised and internally consistent? Use an LLM judge: "On a 1–5 scale, how logically organised and internally consistent is this summary?"

*Source faithfulness:* Do the points made in the summary actually appear in the source documents? Faithfulness still applies even for open-ended answers — every claim should be traceable to a retrieved chunk.

*Comparative evaluation (A/B):*
Generate the same answer with two system versions. Present both to a human evaluator (or LLM judge) side by side and ask: "Which answer is more useful, complete, and accurate?" A/B evaluation does not require knowing the "correct" answer — only which is better.

*Reference-free metrics:*
G-Eval (Liu et al. 2023) uses an LLM to score answers on dimensions like relevance, coherence, consistency, and fluency using a step-by-step chain of thought evaluation. Does not require reference answers.

**The honest limitation:**
For truly subjective questions ("What does this consultant recommend?"), quality is inherently interpretive. Even human evaluators will disagree. The best you can do is:
1. Use multiple human evaluators and measure inter-annotator agreement
2. Accept that evaluation quality is bounded by human agreement
3. Use automated metrics as relative comparisons (system A vs system B) rather than absolute quality measures
