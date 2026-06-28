# RAG Metrics Deep Dive

Detailed treatment of six metrics every production RAG system should measure: NDCG, RRF, hybrid search quality, ROUGE, METEOR, and token cost efficiency. Each section covers the formula, a worked example, what the metric captures, and what it misses.

## Table of Contents

- [NDCG — Normalised Discounted Cumulative Gain](#ndcg--normalised-discounted-cumulative-gain)
- [RRF — Reciprocal Rank Fusion](#rrf--reciprocal-rank-fusion)
- [Hybrid Search Quality Measurement](#hybrid-search-quality-measurement)
- [ROUGE](#rouge)
- [METEOR](#meteor)
- [Token Cost as a First-Class Metric](#token-cost-as-a-first-class-metric)
- [When to Use Each Metric](#when-to-use-each-metric)
- [Full Metric Suite — Production Dashboard](#full-metric-suite--production-dashboard)

---

## NDCG — Normalised Discounted Cumulative Gain

### What it measures

NDCG measures the quality of a ranked list of retrieved results. Unlike recall@k (binary: is the correct chunk present?) or MRR (rank of the first correct chunk), NDCG rewards:
- Retrieving relevant chunks at all
- Placing more relevant chunks higher in the ranking
- Graded relevance (a chunk that fully answers the question is better than one that partially does)

It is the standard metric in information retrieval and search engine evaluation, and the most informative single retrieval metric for RAG.

### Formula

**Step 1 — Relevance scores:**
Assign a relevance score to each retrieved chunk (0 = not relevant, 1 = partially relevant, 2 = highly relevant, 3 = exactly answers the question).

```
Relevance grade scale:
  0: Not relevant to the query
  1: Tangentially related, mentions the topic but doesn't answer
  2: Relevant, partially answers the question
  3: Directly answers the question
```

**Step 2 — DCG (Discounted Cumulative Gain):**

```
DCG@k = Σ (from i=1 to k) [ rel_i / log2(i + 1) ]

Where:
  rel_i = relevance score of the chunk at rank i
  log2(i + 1) = discount factor — higher ranks are worth more
```

The logarithmic discount means rank 1 is worth 3× rank 4 and 5× rank 8. Relevant chunks buried at rank 10 contribute much less than the same chunk at rank 1.

**Step 3 — IDCG (Ideal DCG):**
Sort all available relevant chunks by relevance score descending and compute DCG as if they were all retrieved in perfect order. This is the theoretical maximum DCG.

**Step 4 — NDCG:**
```
NDCG@k = DCG@k / IDCG@k
```

NDCG is always between 0 and 1. NDCG=1 means perfect ranking; NDCG=0 means no relevant chunks retrieved.

### Worked Example

Query: "What is the parental leave policy for directors?"

Retrieved chunks at k=5:

| Rank | Chunk content | Relevance score |
|------|---------------|-----------------|
| 1 | "Directors are entitled to 16 weeks paid parental leave..." | 3 |
| 2 | "All employees must comply with the code of conduct..." | 0 |
| 3 | "Parental leave is available to eligible employees..." | 1 |
| 4 | "Senior managers can apply for extended leave under..." | 2 |
| 5 | "Leave requests must be submitted 8 weeks in advance..." | 1 |

```python
import math

relevance_scores = [3, 0, 1, 2, 1]

# DCG@5
dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
# = 3/log2(2) + 0/log2(3) + 1/log2(4) + 2/log2(5) + 1/log2(6)
# = 3/1.0 + 0/1.585 + 1/2.0 + 2/2.322 + 1/2.585
# = 3.0 + 0 + 0.5 + 0.861 + 0.387
# = 4.748

# IDCG@5 (perfect ordering: [3, 2, 1, 1, 0])
ideal_scores = sorted(relevance_scores, reverse=True)  # [3, 2, 1, 1, 0]
idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores))
# = 3/1.0 + 2/1.585 + 1/2.0 + 1/2.322 + 0/2.585
# = 3.0 + 1.262 + 0.5 + 0.431 + 0
# = 5.193

ndcg = dcg / idcg  # = 4.748 / 5.193 = 0.914
```

NDCG@5 = **0.914** — good ranking. The most relevant chunk (score 3) is at rank 1. The irrelevant chunk (score 0) is at rank 2 which is penalised but not catastrophically.

### What NDCG misses

- **Graded relevance requires human labels.** Binary relevance (relevant/not) is much cheaper to collect. NDCG with binary labels degrades to a weighted recall metric.
- **Treats all queries equally.** NDCG on a 10-query evaluation set is dominated by the hardest queries. Weight by query frequency in production.
- **Does not measure generation quality.** A retrieved chunk with relevance=3 may still lead to a hallucinated answer if the LLM ignores it.

### Implementation

```python
def ndcg_at_k(retrieved_ids: list[str], relevant_ids: dict[str, int], k: int) -> float:
    """
    retrieved_ids: ordered list of retrieved chunk IDs
    relevant_ids: {chunk_id: relevance_score} for all known relevant chunks
    """
    dcg = sum(
        relevant_ids.get(chunk_id, 0) / math.log2(rank + 2)
        for rank, chunk_id in enumerate(retrieved_ids[:k])
    )
    
    ideal_scores = sorted(relevant_ids.values(), reverse=True)[:k]
    idcg = sum(
        score / math.log2(rank + 2)
        for rank, score in enumerate(ideal_scores)
    )
    
    return dcg / idcg if idcg > 0 else 0.0

# Typical production targets:
# NDCG@5  > 0.75 : acceptable
# NDCG@5  > 0.85 : good
# NDCG@10 > 0.80 : good for diverse queries
```

---

## RRF — Reciprocal Rank Fusion

### What it measures / does

RRF is not an evaluation metric — it is a **rank aggregation algorithm** used in hybrid retrieval to merge ranked lists from multiple retrieval methods (dense embedding + BM25) into a single unified ranking. It is measured as an output quality improvement, not as an intrinsic score.

However, understanding RRF precisely — including how to evaluate whether it is working — is essential for any hybrid RAG system.

### Formula

```
RRF_score(document d) = Σ_r [ 1 / (k + rank_r(d)) ]

Where:
  r   = each retrieval method (e.g., dense, sparse)
  k   = constant (typically 60) that smooths the effect of top-ranked results
  rank_r(d) = rank position of document d in retrieval method r (1-indexed)
              If d is not retrieved by method r, its contribution is 0
```

The constant k=60 was empirically found to give robust performance across many retrieval tasks. It prevents the fusion from being completely dominated by the rank-1 document from a single retriever.

### Worked Example

Query: "SOC 2 Type II audit report Q3 2023"

Dense retrieval results (top 5):
```
Rank 1: doc_A "Q3 2023 Security Assessment Summary"   → score 1/(60+1) = 0.01639
Rank 2: doc_B "SOC 2 Overview and Framework"           → score 1/(60+2) = 0.01613
Rank 3: doc_C "Q3 2023 SOC 2 Type II Audit Report"    → score 1/(60+3) = 0.01587
Rank 4: doc_D "Annual Security Report 2023"            → score 1/(60+4) = 0.01563
Rank 5: doc_E "Compliance Checklist"                   → score 1/(60+5) = 0.01538
```

BM25 (sparse) results (top 5):
```
Rank 1: doc_C "Q3 2023 SOC 2 Type II Audit Report"    → score 1/(60+1) = 0.01639
Rank 2: doc_F "SOC 2 Type II Certification Q2 2023"   → score 1/(60+2) = 0.01613
Rank 3: doc_A "Q3 2023 Security Assessment Summary"    → score 1/(60+3) = 0.01587
Rank 4: doc_G "Type II Audit Procedures"               → score 1/(60+4) = 0.01563
Rank 5: doc_B "SOC 2 Overview and Framework"           → score 1/(60+5) = 0.01538
```

RRF scores (sum across both retrievers):
```
doc_C: 0.01587 (dense rank 3) + 0.01639 (sparse rank 1) = 0.03226  ← winner
doc_A: 0.01639 (dense rank 1) + 0.01587 (sparse rank 3) = 0.03226  ← tie
doc_B: 0.01613 (dense rank 2) + 0.01538 (sparse rank 5) = 0.03151
doc_F: 0.00000 (not in dense) + 0.01613 (sparse rank 2) = 0.01613
doc_D: 0.01563 (dense rank 4) + 0.00000 (not in sparse) = 0.01563
```

**Result:** doc_C ("Q3 2023 SOC 2 Type II Audit Report") moves from dense rank 3 to RRF rank 1. Dense retrieval ranked it third because it semantically matched "security assessment" content more broadly; BM25 ranked it first because it contains the exact keywords "SOC 2 Type II" and "Q3 2023". RRF correctly surfaces it at rank 1.

### Implementation

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],  # each inner list is a ranked list of doc IDs
    k: int = 60
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF. Returns (doc_id, rrf_score) sorted descending."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Usage: hybrid retrieval
dense_results  = dense_retriever.search(query, k=50)   # top-50 by cosine similarity
sparse_results = bm25_retriever.search(query, k=50)     # top-50 by BM25 score

fused = reciprocal_rank_fusion([
    [doc.id for doc in dense_results],
    [doc.id for doc in sparse_results],
])
final_results = fused[:10]  # top-10 after fusion
```

### Evaluating whether RRF is working

RRF cannot be evaluated intrinsically — you must measure whether hybrid retrieval (using RRF) outperforms single-mode retrieval on your evaluation set:

```python
def evaluate_rrf_benefit(eval_set, dense_retriever, sparse_retriever, k=10):
    dense_only_ndcg  = []
    sparse_only_ndcg = []
    hybrid_rrf_ndcg  = []
    
    for item in eval_set:
        dense_ids  = [d.id for d in dense_retriever.search(item.query, k=k)]
        sparse_ids = [d.id for d in sparse_retriever.search(item.query, k=k)]
        hybrid_ids = [d for d, _ in reciprocal_rank_fusion([dense_ids, sparse_ids])][:k]
        
        dense_only_ndcg.append(ndcg_at_k(dense_ids,  item.relevant_ids, k))
        sparse_only_ndcg.append(ndcg_at_k(sparse_ids, item.relevant_ids, k))
        hybrid_rrf_ndcg.append(ndcg_at_k(hybrid_ids,  item.relevant_ids, k))
    
    print(f"Dense-only  NDCG@{k}: {mean(dense_only_ndcg):.3f}")
    print(f"Sparse-only NDCG@{k}: {mean(sparse_only_ndcg):.3f}")
    print(f"Hybrid RRF  NDCG@{k}: {mean(hybrid_rrf_ndcg):.3f}")

# Expected output for a typical mixed corpus:
# Dense-only  NDCG@10: 0.742
# Sparse-only NDCG@10: 0.681
# Hybrid RRF  NDCG@10: 0.819  ← typically 5-15pp above the better single mode
```

---

## Hybrid Search Quality Measurement

Beyond just comparing NDCG between dense, sparse, and hybrid, measuring hybrid search quality requires understanding **when each retrieval mode wins** and whether RRF is correctly weighting them.

### Query-type breakdown

Not all queries benefit equally from hybrid. Measure NDCG separately per query type:

```python
query_types = {
    "semantic":  [],  # "explain the implications of the new policy"
    "keyword":   [],  # "SOC 2 Type II Q3 2023 audit"
    "mixed":     [],  # "what does the Q3 2023 compliance report say about access controls"
}

for item in eval_set:
    qtype = classify_query_type(item.query)
    ndcg = ndcg_at_k(hybrid_retriever.search(item.query), item.relevant_ids, k=10)
    query_types[qtype].append(ndcg)

# Typical pattern:
# Semantic queries: dense wins (+0.12 NDCG vs sparse), hybrid ≈ dense
# Keyword queries:  sparse wins (+0.18 NDCG vs dense), hybrid ≈ sparse
# Mixed queries:    hybrid wins (+0.08 vs best single mode)
```

### The alpha tuning problem

Standard RRF uses equal weighting across retrievers. But for your specific corpus, one retriever may be consistently stronger. Test weighted RRF:

```python
def weighted_rrf(dense_ids, sparse_ids, alpha=0.5, k=60):
    """alpha=0.7 means 70% weight to dense, 30% to sparse."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank)
    for rank, doc_id in enumerate(sparse_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0) + (1-alpha) / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Grid search alpha from 0.0 to 1.0 on your evaluation set
# Find the alpha that maximises NDCG@10
# Typical result for prose-heavy corpora: alpha=0.65 (dense-dominant)
# For ID/keyword-heavy corpora: alpha=0.35 (sparse-dominant)
```

### Hybrid search precision vs recall trade-off

```
Metric              Dense    Sparse   Hybrid (RRF)
────────────────────────────────────────────────────
Recall@10           0.74     0.68     0.87   ← recall improves most
Precision@5         0.71     0.65     0.74   ← precision improves moderately
NDCG@10             0.742    0.681    0.819
False positive rate 0.29     0.35     0.26
Query latency       45ms     12ms     57ms   ← parallel execution: latency ≈ max(dense, sparse)
Index storage       +0       +40%     +40%   ← BM25 index cost
```

The latency of hybrid search with parallelised dense + sparse retrieval is approximately the maximum of the two, not their sum. If dense takes 45ms and sparse takes 12ms: hybrid takes ~50ms (45ms dense + small merge overhead), not 57ms.

---

## ROUGE

### What it measures

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap between a generated text and one or more reference texts. Originally designed for automatic summarisation evaluation. In RAG, it is used to measure how similar generated answers are to reference answers.

### Variants

**ROUGE-1:** Unigram overlap (individual word match)
```
ROUGE-1 recall    = |generated ∩ reference| / |reference|
ROUGE-1 precision = |generated ∩ reference| / |generated|
ROUGE-1 F1        = 2 × (precision × recall) / (precision + recall)
```

**ROUGE-2:** Bigram overlap (consecutive two-word pairs)
```
ROUGE-2 = same formula but with bigrams instead of unigrams
```

**ROUGE-L:** Longest Common Subsequence (LCS) — captures in-order word matches that are not necessarily consecutive
```
ROUGE-L precision = LCS(generated, reference) / |generated|
ROUGE-L recall    = LCS(generated, reference) / |reference|
ROUGE-L F1        = (1 + β²) × precision × recall / (β² × precision + recall)
                    where β = 1 for equal precision/recall weighting
```

### Worked Example

```
Reference answer: "Employees must submit parental leave requests at least 8 weeks before the expected start date."
Generated answer:  "Leave requests for parental leave must be filed 8 weeks in advance of the planned start."

Tokenise (lowercased, stop words retained):
  Reference tokens: [employees, must, submit, parental, leave, requests, at, least, 8, weeks, before, the, expected, start, date]
  Generated tokens:  [leave, requests, for, parental, leave, must, be, filed, 8, weeks, in, advance, of, the, planned, start]

ROUGE-1:
  Intersection: {leave, requests, parental, must, 8, weeks, the, start} = 8 tokens
  Recall    = 8 / 15 = 0.533
  Precision = 8 / 16 = 0.500
  F1        = 2 × (0.500 × 0.533) / (0.500 + 0.533) = 0.516

ROUGE-2 (bigrams):
  Reference bigrams: {employees must, must submit, submit parental, parental leave, leave requests, ...}
  Generated bigrams: {leave requests, requests for, for parental, parental leave, leave must, ...}
  Intersection: {parental leave, leave requests} = 2 bigrams
  Recall    = 2 / 14 = 0.143
  Precision = 2 / 15 = 0.133
  F1        = 0.138

ROUGE-L:
  LCS: [leave, requests, parental, must, 8, weeks, the, start] = 8 tokens
  F1: same as ROUGE-1 in this case (since the LCS matches the unigram intersection)
```

The answer is semantically correct but scores low on ROUGE-2 because the exact bigrams differ ("submit parental leave requests" vs "filed 8 weeks"). This illustrates ROUGE-2's main weakness: it penalises paraphrase.

### Implementation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_rouge(generated: str, reference: str) -> dict:
    scores = scorer.score(reference, generated)
    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

# use_stemmer=True: "submitting" and "submit" are treated as the same token
# This reduces false penalties from morphological variation
```

### What ROUGE misses for RAG

| Failure mode | Example |
|-------------|---------|
| Paraphrase blindness | "The deadline is 8 weeks" and "8 weeks in advance" score differently despite being equivalent |
| Synonym blindness | "vehicle" vs "car" — no match despite same meaning |
| Rewards verbosity | A generated answer that copies the reference verbatim scores 1.0 regardless of whether it is useful |
| Cannot assess faithfulness | High ROUGE means the answer matches the reference, not that it is grounded in the retrieved context |
| Reference dependency | Requires reference answers — unavailable for open-ended or synthesis queries |

**When to use ROUGE in RAG:**
- Extractive summarisation tasks (summarise this document) — ROUGE-L is appropriate
- FAQ-style Q&A with clean reference answers
- Regression testing: did a prompt change cause the output to diverge from a known-good reference?

**When not to use ROUGE:**
- Open-ended synthesis questions with no single correct phrasing
- As a proxy for faithfulness or factual correctness
- When paraphrase is expected and acceptable

### Typical production targets

```
ROUGE-1 F1 > 0.50 : acceptable for extractive summarisation
ROUGE-2 F1 > 0.20 : acceptable (ROUGE-2 is always much lower)
ROUGE-L F1 > 0.45 : acceptable
```

---

## METEOR

### What it measures

METEOR (Metric for Evaluation of Translation with Explicit ORdering) was designed to address ROUGE's main weakness: insensitivity to paraphrase and synonymy. It incorporates:
1. **Exact match:** same word, same form
2. **Stem match:** "submitting" matches "submit"
3. **Synonym match:** "vehicle" matches "car" (via WordNet)
4. **Paraphrase match:** multi-word synonym pairs

METEOR also applies a **chunk penalty** for fragmented matches — it rewards consecutive word matches over scattered ones.

### Formula

```
Step 1 — Alignment:
  Match words in generated to reference using:
  a) Exact match
  b) Porter stemmer match
  c) WordNet synonym match
  Prefer exact > stem > synonym. Each reference word matches at most one generated word.

Step 2 — Precision and Recall:
  P = matched_words / |generated|
  R = matched_words / |reference|
  F_mean = 10 × P × R / (R + 9P)   ← harmonic mean weighted 9:1 toward recall

Step 3 — Chunk penalty:
  chunks = number of contiguous matched word groups
  penalty = 0.5 × (chunks / matched_words)³

Step 4 — METEOR score:
  METEOR = F_mean × (1 - penalty)
```

The chunk penalty punishes scattered matches. An answer that matches every other word of the reference (many chunks) scores lower than one that matches a contiguous block of the same words (few chunks).

### Worked Example

Using the same example as ROUGE:
```
Reference: "Employees must submit parental leave requests at least 8 weeks before the expected start date."
Generated: "Leave requests for parental leave must be filed 8 weeks in advance of the planned start."
```

**Alignment (exact + stem):**
```
Exact matches:  leave, requests, parental, must, 8, weeks, the*, start
                (*"the" not in generated — skip)
Stem matches:   "filed" ≠ "submit" (different stems), no additional matches
Synonym matches: "planned" ≈ "expected" via WordNet → match

Total matched: leave, requests, parental, must, 8, weeks, start, planned/expected = 8 words
```

**Precision and Recall:**
```
P = 8/16 = 0.500   (8 matched out of 16 generated words)
R = 8/15 = 0.533   (8 matched out of 15 reference words)
F_mean = 10 × 0.500 × 0.533 / (0.533 + 9 × 0.500) = 2.665 / 5.033 = 0.530
```

**Chunk penalty:**
Matched words form 4 chunks: [leave requests], [parental], [must], [8 weeks], [start]
```
penalty = 0.5 × (5/8)³ = 0.5 × 0.244 = 0.122
```

**METEOR:**
```
METEOR = 0.530 × (1 - 0.122) = 0.530 × 0.878 = 0.465
```

METEOR = **0.465** vs ROUGE-1 F1 = **0.516**. Similar score here, but METEOR would score higher relative to ROUGE when synonyms are involved (e.g., "vehicle" vs "car" in the reference).

### Implementation

```python
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('punkt')

def evaluate_meteor(generated: str, reference: str) -> float:
    gen_tokens = word_tokenize(generated.lower())
    ref_tokens = word_tokenize(reference.lower())
    return meteor_score([ref_tokens], gen_tokens)

# Multiple references (use the best match):
def evaluate_meteor_multi_ref(generated: str, references: list[str]) -> float:
    gen_tokens = word_tokenize(generated.lower())
    ref_token_lists = [word_tokenize(r.lower()) for r in references]
    return meteor_score(ref_token_lists, gen_tokens)
```

### ROUGE vs METEOR — when each wins

| Scenario | Better metric |
|----------|--------------|
| Extractive summarisation (copies phrases from source) | ROUGE-L |
| Abstractive summarisation (paraphrases) | METEOR |
| Multiple valid phrasings exist | METEOR (synonym matching) |
| Fast, cheap evaluation on many samples | ROUGE (simpler, no WordNet) |
| Technical domains (domain synonyms not in WordNet) | ROUGE (WordNet covers general English) |
| Multi-language evaluation | ROUGE (METEOR WordNet is English-only) |

### Typical production targets

```
METEOR > 0.40 : acceptable for abstractive summarisation in RAG
METEOR > 0.55 : good
METEOR > 0.70 : excellent (usually means the answer closely matches reference phrasing)
```

### Limitations of METEOR for RAG

- **English-centric:** WordNet synonym matching is primarily English. Poor support for other languages.
- **Domain synonym gaps:** WordNet does not contain domain-specific synonyms (technical, legal, medical). "ARR" and "annual recurring revenue" are not synonymous in WordNet.
- **Reference dependency:** Same as ROUGE — requires reference answers.
- **Does not measure faithfulness:** A METEOR score of 0.8 tells you the answer looks like the reference; it does not tell you whether the answer is grounded in the retrieved context.

---

## Token Cost as a First-Class Metric

Token cost should be tracked alongside quality metrics — not instead of them. A system with NDCG=0.90 that costs $0.50/query is a worse system than one with NDCG=0.87 that costs $0.02/query, all else being equal.

### Core cost metrics

```python
@dataclass
class QueryCostMetrics:
    query_id:          str
    embedding_tokens:  int       # query embedding input tokens
    schema_tokens:     int       # schema context in generation prompt
    few_shot_tokens:   int       # few-shot examples in prompt  
    query_tokens:      int       # the user's query itself
    total_input_tokens: int      # sum of all input components
    output_tokens:     int       # generated SQL or answer
    model:             str       # which model was used
    from_cache:        bool      # was this a cache hit?
    retry_count:       int       # number of retries
    cost_usd:          float     # computed from tokens × price
    
    # Computed efficiency metrics
    @property
    def cost_per_output_token(self) -> float:
        return self.cost_usd / max(self.output_tokens, 1)
    
    @property
    def input_output_ratio(self) -> float:
        return self.total_input_tokens / max(self.output_tokens, 1)
```

### The cost efficiency metric

Cost efficiency combines quality and cost into a single comparable number:

```
cost_efficiency = quality_score / cost_per_query

Example:
  System A: NDCG=0.87, cost=$0.014/query → efficiency = 0.87/0.014 = 62.1
  System B: NDCG=0.85, cost=$0.002/query → efficiency = 0.85/0.002 = 425.0
  System C: NDCG=0.91, cost=$0.025/query → efficiency = 0.91/0.025 = 36.4
```

System B has 7× better cost efficiency than System A despite 2pp lower quality. At scale, this matters.

### Token cost as a regression signal

Track cost per query over time. An unexplained cost increase is often a regression signal:

```
Cost spike patterns and their likely causes:

  +50% input tokens overnight:
    → Schema context expanded unexpectedly (new tables indexed, schema cache invalidated)
    → Few-shot example library grew without pruning
    → Retry rate increased (each retry doubles cost)

  +200% output tokens:
    → Model changed to a more verbose generation style (prompt regression)
    → Query type distribution shifted toward complex multi-join queries
    → Result explanation feature accidentally enabled for all queries

  +500% total cost:
    → Caching layer broke (all queries hitting LLM)
    → Model routing misconfiguration (all queries going to expensive model)
    → Loop in retry logic (queries retrying indefinitely)
```

Alert thresholds:
```python
daily_cost_alerts = {
    "total_cost_150pct_of_7day_avg": True,   # sudden spike
    "avg_input_tokens_120pct_of_baseline": True,  # prompt bloat
    "cache_hit_rate_below_20pct": True,        # cache breakdown
    "retry_rate_above_15pct": True,            # systematic failure
    "p99_cost_5x_p50_cost": True,              # runaway queries
}
```

### Cost breakdown dashboard

```
Daily Cost Report
─────────────────────────────────────────────────────────
Total queries:       12,450
Cache hits:          4,320  (34.7%)
LLM queries:         8,130

Input token breakdown (LLM queries only):
  Schema context:    52%    avg 4,230 tokens/query
  Few-shot examples: 27%    avg 2,190 tokens/query
  System prompt:     16%    avg 1,300 tokens/query
  User query:         5%    avg 410 tokens/query
  TOTAL input avg:           8,130 tokens/query

Output avg:          210 tokens/query

Cost breakdown:
  GPT-4o queries (40%):    3,252 × $0.022 = $71.54
  GPT-4o-mini (60%):       4,878 × $0.0012 = $5.85
  Cached (free):           4,320 × $0 = $0
  TOTAL:                   $77.39/day

Efficiency opportunities:
  Schema compression (-40% schema tokens): saves $15.12/day
  Increase cache hit rate to 50%:          saves $22.30/day
  Route 10% more queries to mini:          saves $6.80/day
  ───────────────────────────────────────────────────────
  Potential monthly savings: $1,329/month with above changes
```

### Quality-adjusted cost

The most useful metric for comparing system versions is quality-adjusted cost — the cost to achieve one unit of quality:

```python
def quality_adjusted_cost(
    quality_score: float,   # NDCG, faithfulness, answer relevance, etc.
    cost_per_query: float,
) -> float:
    """Lower is better. Cost to achieve one point of quality."""
    return cost_per_query / quality_score

# Compare system versions:
versions = {
    "v1.0 (GPT-4o, k=15, 5-shot)":      (0.87, 0.022),   # (quality, cost)
    "v1.1 (GPT-4o, k=8, 3-shot)":       (0.85, 0.014),
    "v1.2 (GPT-4o-mini fine-tuned)":    (0.83, 0.002),
    "v1.3 (routing: 60% mini, 40% 4o)": (0.86, 0.010),
}

for version, (quality, cost) in versions.items():
    qac = quality_adjusted_cost(quality, cost)
    print(f"{version}: quality={quality:.2f}, cost=${cost:.3f}, QAC={qac:.4f}")

# Output:
# v1.0: quality=0.87, cost=$0.022, QAC=0.0253  ← most expensive per quality point
# v1.1: quality=0.85, cost=$0.014, QAC=0.0165
# v1.2: quality=0.83, cost=$0.002, QAC=0.0024  ← most efficient
# v1.3: quality=0.86, cost=$0.010, QAC=0.0116  ← best quality/cost balance
```

---

## When to Use Each Metric

| Metric | Use for | Don't use for |
|--------|---------|--------------|
| **NDCG@k** | Retrieval ranking quality, comparing retrieval methods | Measuring generation quality |
| **RRF score** | Merging dense + sparse results (algorithm, not evaluation metric) | Standalone quality measurement |
| **Hybrid search NDCG delta** | Measuring whether hybrid beats single-mode retrieval | Anything other than retrieval comparison |
| **ROUGE-1/2/L** | Extractive summarisation, regression testing against reference answers | Open-ended synthesis, faithfulness measurement |
| **METEOR** | Abstractive paraphrase-tolerant evaluation | Multi-language, domain-specific jargon |
| **Token cost** | System efficiency, A/B testing, regression detection, budget tracking | Never as the sole quality signal |

---

## Full Metric Suite — Production Dashboard

A complete RAG evaluation dashboard tracks metrics at three layers:

```
RETRIEVAL LAYER
  NDCG@5:            0.83   (target: > 0.80)
  NDCG@10:           0.87   (target: > 0.85)
  Recall@10:         0.91   (target: > 0.88)
  Precision@5:       0.76   (target: > 0.70)
  Hybrid vs dense ∆: +0.09  (confirms hybrid is working)

GENERATION LAYER
  Faithfulness:      0.89   (target: > 0.85)
  Answer relevance:  0.84   (target: > 0.80)
  ROUGE-L F1:        0.52   (reference required; track regression)
  METEOR:            0.48   (reference required; track regression)
  Citation accuracy: 0.91   (% of citations that actually support the claim)

EFFICIENCY LAYER
  Cost per query:    $0.011  (target: < $0.015)
  Cache hit rate:    38.2%   (target: > 30%)
  Quality-adj cost:  0.0131  (lower = better; track week-over-week)
  Retry rate:        4.1%    (target: < 10%)
  Input/output ratio: 42:1   (high ratio → schema compression opportunity)
  Avg input tokens:  4,620   (track for prompt bloat regression)
```

This three-layer view makes it immediately clear where problems are: a faithfulness drop with stable NDCG is a generation problem; a NDCG drop with stable faithfulness is a retrieval problem; a cost spike with stable quality is a caching or routing problem.
