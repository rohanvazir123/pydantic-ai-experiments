# Generation and Hallucination Prevention — Answers

## Q26. How do you prevent the LLM from hallucinating facts not in the retrieved context?

**Answer:**

Hallucination prevention in RAG is a layered problem: no single technique is sufficient.

**Layer 1 — System prompt grounding:**

Instruct the LLM explicitly to use only the provided context:

```
System: You are a knowledge base assistant. Answer the user's question using ONLY the 
        information provided in the context below. If the context does not contain 
        sufficient information to answer the question, say "I don't have enough 
        information in the knowledge base to answer this."

        Do NOT use any knowledge from your training data that is not confirmed by 
        the provided context. Every factual claim in your answer must be directly 
        traceable to a specific passage in the context.

Context:
{retrieved_chunks}
```

*What this still fails to prevent:* The LLM may paraphrase retrieved information in a way that subtly distorts the meaning. It may also "fill in" plausible-sounding details when the context is ambiguous, even with this instruction. System prompt grounding reduces hallucination by ~40–60% but does not eliminate it.

**Layer 2 — Attribution forcing:**

Require the LLM to cite a specific source for every factual claim. A claim without a citation is flagged as potentially hallucinated.

```
System: For every factual statement in your answer, provide a citation in the format 
        [Source: {document_title}, {section}]. If you cannot cite a source for a claim, 
        do not make that claim.
```

A generated answer without citations, or with citations that don't match any retrieved chunk, signals hallucination.

**Layer 3 — Post-generation faithfulness check:**

After generation, run an automated check: does each claim in the answer appear in the retrieved context?

```python
def faithfulness_score(answer: str, context_chunks: list[str], llm) -> float:
    claims = llm.extract_claims(answer)  # decompose answer into atomic facts
    supported = 0
    for claim in claims:
        is_supported = any(
            llm.is_claim_supported(claim, chunk) for chunk in context_chunks
        )
        if is_supported:
            supported += 1
    return supported / len(claims) if claims else 1.0
```

Flag answers with faithfulness score < 0.8 for human review or regeneration.

**Token cost of faithfulness checking:**
One LLM call for claim extraction + N calls for verification (N = number of claims × number of chunks). For a 5-claim answer and 5 chunks, that's 25 additional LLM calls. Use a small, cheap model (GPT-4o-mini, Haiku) for faithfulness checking — it does not require the full capability of the generation model. Cost: ~$0.002/query for faithfulness checking with a small model.

**Layer 4 — Temperature tuning:**
Lower temperature (0.0–0.3) reduces creative generation and keeps the LLM closer to the provided context. For factual RAG (policy Q&A, technical documentation), temperature = 0 is often the right setting. For summaries or synthesis tasks, allow slightly higher temperature (0.3–0.5).

---

## Q27. How do you implement citation and attribution?

**Answer:**

Citations are non-negotiable in any professional RAG deployment. A user should always be able to verify where a claim came from.

**Implementation approach — span-level citation:**

Ask the LLM to embed citations inline, mapped to chunk IDs:

```
System: When answering, cite your sources inline using [^1], [^2] format. 
        Map each citation to the provided source chunks as follows:
        [^1] = Source: Employee Handbook 2024, Section 4.2
        [^2] = Source: HR Policy Update March 2024

Answer: Employees are entitled to 20 days of annual leave per year [^1]. 
        Contract employees are excluded from this benefit [^2].

References:
[^1] Employee Handbook 2024, Section 4.2, Page 34
[^2] HR Policy Update March 2024, Section 1.1
```

**Citation verification:**

For every (claim, citation) pair, verify that the cited chunk actually supports the claim:

```python
def verify_citation(claim: str, cited_chunk: str, llm) -> bool:
    response = llm.complete(
        f"Does this passage support this claim?\n\n"
        f"Claim: {claim}\n\n"
        f"Passage: {cited_chunk}\n\n"
        f"Answer yes or no."
    )
    return response.lower().startswith("yes")
```

**Failure modes of citation systems:**

*Citation hallucination:* The LLM generates a citation that points to a chunk that does not actually support the claim. The citation format is correct but the content is wrong.
Detection: run citation verification for every (claim, citation) pair. A verification failure rate > 10% signals systematic citation hallucination.

*Citation omission:* The LLM makes a claim without any citation. This is harder to detect — you need claim extraction (decompose the answer into atomic facts) and then check each for a citation.

*Over-citation:* The LLM cites the same source for every claim, even when claims come from different sources. Reduces user trust and obscures the actual provenance.

**User-facing citation design:**
Inline citations are best as expandable footnotes in the UI. Clicking `[^1]` expands to show the verbatim passage from the source document with the relevant sentence highlighted. This creates an auditable chain: user question → generated answer → cited passage → source document.

---

## Q28. Insufficient context — how does your system detect and handle it?

**Answer:**

**Detection:**

*Signal 1 — Low retrieval similarity:*
Maximum cosine similarity between query and top-retrieved chunk is < 0.3. The corpus likely does not contain relevant information.

*Signal 2 — Faithfulness score = 1.0 with "I don't know" response:*
If the LLM generates "I don't have enough information to answer this," and the faithfulness score is high (the LLM is faithfully grounding its refusal in the absence of evidence), retrieval has correctly failed and the LLM is correctly declining to answer.

*Signal 3 — LLM-reported uncertainty with low context coverage:*
Ask the LLM to self-report whether the context is sufficient:
```
After answering, rate your confidence that your answer is fully supported by the provided 
context: high / medium / low. If low, explain what information is missing.
```

*Signal 4 — Empty retrieved set:*
No chunks above the similarity threshold were returned. This is an unambiguous retrieval failure.

**Handling:**

*Step 1 — Retry with expanded retrieval:*
Increase k, apply query transformation (HyDE or expansion), and re-retrieve. If the relevant document exists in the corpus but was ranked below k, this catches it.

*Step 2 — Return a scoped answer:*
If the corpus partially covers the question, answer the part that is covered and explicitly state what is missing:
"The knowledge base contains information about the expense approval policy [answer here], but does not include information about the exception process for international expenses."

*Step 3 — Corpus gap logging:*
Log every query that resulted in insufficient context. Cluster these by topic. If 50 queries about "international expense exceptions" all failed retrieval, that is a corpus gap worth filling. Route to the content team to create the missing document.

*Never:* Generate a hallucinated answer to "fill the gap." A confident wrong answer is worse than an honest "I don't know."

---

## Q29. How do you calibrate when to say "I don't know" vs attempt an answer?

**Answer:**

The calibration is a product decision with real consequences in both directions:

- Too aggressive "I don't know" → users don't trust the system and stop using it
- Too lenient "I don't know" → system generates hallucinated answers and users make bad decisions

**The calibration signal:**

Build a confidence model from:

```python
@dataclass
class AnswerConfidence:
    max_retrieval_similarity: float  # 0-1, how close is the best chunk to the query?
    reranker_top_score: float        # 0-1, cross-encoder score for best chunk
    faithfulness_score: float        # 0-1, fraction of claims supported by context
    coverage_score: float           # 0-1, does context address all aspects of the query?
    context_chunk_count: int        # how many relevant chunks were found?

def compute_confidence(c: AnswerConfidence) -> float:
    return (
        0.30 * c.max_retrieval_similarity +
        0.25 * c.reranker_top_score +
        0.25 * c.faithfulness_score +
        0.15 * c.coverage_score +
        0.05 * min(c.context_chunk_count / 5, 1.0)
    )

# Thresholds (calibrate per deployment context):
# > 0.75: answer confidently
# 0.50–0.75: answer with confidence qualifier ("Based on available information...")
# < 0.50: decline or strongly hedge
```

**Domain-specific calibration:**

The threshold should be higher for high-stakes domains:
- Medical / legal / financial: confidence > 0.85 to answer confidently; < 0.85 → "Based on available documents, but please verify with a professional"
- Internal policy Q&A: confidence > 0.60 to answer confidently — users expect partial answers
- Customer-facing chatbot: confidence > 0.70 to answer; lower → "I'll connect you with a human agent"

**What happens when the LLM says "I don't know" too often:**

Analyse the queries that triggered refusals. If they are legitimate questions the corpus should answer, the problem is retrieval (not enough k, poor chunking, vocabulary mismatch) — not calibration. Fix retrieval first, then recalibrate.

---

## Q30. LLM parametric knowledge vs retrieved context — what happens when they conflict?

**Answer:**

LLMs are trained on large text corpora and have strong parametric beliefs about factual matters. When the retrieved document contradicts the LLM's parametric knowledge, the LLM may:
- Prefer the retrieved context (correct behaviour)
- Prefer its parametric knowledge (hallucination)
- Average or hedge ("some sources say X while others say Y") — often wrong

**When parametric knowledge wins (failure):**

A medical RAG system retrieves a passage: "The recommended dosage for Drug X is 5mg daily." The LLM's training data contains conflicting studies showing 10mg is also used. The LLM may "soften" the retrieved information to align with its broader knowledge, generating: "The typical dosage is 5–10mg daily" — which is not what the document says.

**Forcing context preference:**

```
System: You are a document Q&A assistant. Your ONLY source of truth is the provided 
        context. If your training knowledge conflicts with the context, ALWAYS defer 
        to the context. If the context is unambiguous, do not introduce uncertainty 
        that is not present in the context itself.
```

**Why this still fails:**

Strong parametric beliefs are deeply embedded in model weights. System prompt instructions reduce but do not eliminate the influence of parametric knowledge. A highly confident wrong LLM belief can override even explicit instructions.

**More reliable approaches:**

*Temperature 0:* Removes random sampling. The LLM generates the most probable token at each step, which for a RAG task is most likely to be grounded in the context (which is more immediate than parametric memory).

*Instruction fine-tuning for RAG:* Fine-tune the model specifically to prefer provided context over parametric knowledge. Models trained with this objective ("when context is provided, prefer context") are significantly better at this than instruction-prompted general-purpose models.

*Citation forcing:* If the LLM must cite a source for every claim, it cannot use parametric knowledge without fabricating a citation — and citation fabrication is detectable.

**Token cost implication of parametric override prevention:**
Temperature 0 is free. Citation forcing adds response length (~20% more tokens) — a small cost for a significant reliability improvement. Fine-tuning adds upfront cost but reduces per-query hallucination and thus reduces the need for expensive faithfulness checking.
