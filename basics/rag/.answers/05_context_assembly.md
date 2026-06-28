# Context Assembly — Answers

## Q21. How do you construct the context window from retrieved chunks — ordering, deduplication, formatting?

**Answer:**

Context assembly is the least glamorous part of RAG and among the most impactful. The same chunks ordered differently produce different generation quality.

**Deduplication:**
Before ordering, remove near-duplicate chunks. This is common when:
- Chunk overlap causes the same passage to appear in two adjacent chunks
- Multi-query retrieval returned the same chunk for two different query variants
- A document was indexed twice (ingestion idempotency failure)

```python
def deduplicate_chunks(chunks: list[Chunk], similarity_threshold=0.95) -> list[Chunk]:
    unique = [chunks[0]]
    for candidate in chunks[1:]:
        if all(cosine_similarity(candidate.embedding, u.embedding) < similarity_threshold
               for u in unique):
            unique.append(candidate)
    return unique
```

Deduplication must happen before ordering — ordering a set that includes duplicates wastes context budget and confuses the LLM.

**Ordering:**

The LLM's attention is not uniform across the context window. Empirically, it attends most strongly to the beginning and end of the context, and least to the middle (the "lost-in-the-middle" phenomenon, Nelson et al. 2023). This has a direct implication: the most relevant chunk should be placed first or last, not buried at position 8 of 10.

*Strategy 1 — Relevance order (naive):* Place the highest-scoring chunk first. Simple, but if the chunks form a narrative (e.g., a multi-part policy), this breaks the logical flow.

*Strategy 2 — Book-end order (optimal for diverse chunks):* Place the most relevant chunk first, the second-most relevant chunk last, fill the middle with lower-ranked chunks. This maximises LLM attention on the two most important passages.

*Strategy 3 — Document order (for coherent passages):* If multiple chunks come from the same document (adjacent sections), sort them by their position within the document (page number, section order). The LLM then reads them as a continuous text rather than out-of-order fragments.

**Formatting:**

Chunk presentation matters. Compare:

*Bad (no structure):*
```
Employees are entitled to 20 days annual leave. The manager must approve leave requests 
within 5 business days. Contract employees are not eligible for paid leave. Employees 
must notify HR of any leave exceeding 5 consecutive days.
```

*Good (with source attribution):*
```
[Source: Employee Handbook 2024 | Section: Leave Policy | Page: 34]
Employees are entitled to 20 days annual leave.
The manager must approve leave requests within 5 business days.
---
[Source: Contractor Policy v2.1 | Section: Benefits | Page: 12]
Contract employees are not eligible for paid leave.
```

The separator and source metadata serve two purposes: the LLM uses them for citation generation, and they signal to the LLM that the content comes from different sources (reducing the risk of cross-contaminating facts).

---

## Q22. 20 chunks × 600 tokens, 128k context window. What goes in?

**Answer:**

20 chunks × 600 tokens = 12,000 tokens — well within 128k. The question is not about context window capacity; it is about cost, attention quality, and signal-to-noise ratio.

**The token cost calculation:**
12,000 input tokens × $0.015/1K tokens = $0.18 per query.
At 10,000 queries/day: $1,800/day = $54,000/month just for context tokens.

With 5 carefully selected chunks (3,000 tokens): $0.045/query = $450/day = $13,500/month.
Selecting fewer, better chunks saves $40,500/month at this scale.

**Selection criteria when you have more chunks than you want to include:**

*Criterion 1 — Reranker score:* After cross-encoder reranking, the top-5 by reranker score are almost always the most relevant. If your reranker is well-calibrated, trust it.

*Criterion 2 — Marginal relevance (MMR):* Maximal Marginal Relevance selects chunks that are both relevant to the query AND different from already-selected chunks. This maximises information density in the context.

```python
def mmr_select(query_emb, chunk_embs, k=5, lambda_param=0.7):
    selected = []
    remaining = list(range(len(chunk_embs)))
    for _ in range(k):
        scores = [
            lambda_param * cosine_similarity(query_emb, chunk_embs[i])
            - (1 - lambda_param) * max(cosine_similarity(chunk_embs[i], chunk_embs[s])
                                        for s in selected) if selected else 0
            for i in remaining
        ]
        best = remaining[scores.index(max(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected
```

*Criterion 3 — Confidence threshold:* Exclude any chunk whose reranker score is below 0.3 (poor relevance), even if k hasn't been reached. Including low-confidence chunks dilutes the signal and increases hallucination risk.

*Criterion 4 — Query-type-aware k:*
- Factual lookup ("What is the CEO's name?") → k=3
- Comparison ("Compare policy A to policy B") → k=8–10
- Synthesis ("Summarise the key risks in the report") → k=12–15

**What to do with the 128k window:**
Reserve it for genuinely multi-document synthesis tasks where you need broad coverage. For the majority of queries, 3–8 chunks is optimal. Using the full 128k context window for every query is expensive, slow, and usually counterproductive.

---

## Q23. Two retrieved chunks contain contradictory information. How do you handle it?

**Answer:**

Contradictions in the retrieved context are common in real corpora: updated policies that coexist with their predecessors, market data that has been revised, conflicting expert opinions. The LLM exposed to contradictory context may: ignore one source, average them (subtly wrong), or hallucinate a resolution.

**Detection:**

After context assembly but before generation, run a contradiction check:

```python
def detect_contradiction(chunks: list[Chunk], llm) -> list[tuple[Chunk, Chunk, str]]:
    contradictions = []
    for i, chunk_a in enumerate(chunks):
        for chunk_b in chunks[i+1:]:
            if topic_overlap(chunk_a, chunk_b) > 0.7:  # only check related chunks
                response = llm.complete(
                    f"Do these two passages contradict each other? Answer yes/no and explain:\n\n"
                    f"Passage A: {chunk_a.content}\n\nPassage B: {chunk_b.content}"
                )
                if response.startswith("yes"):
                    contradictions.append((chunk_a, chunk_b, response))
    return contradictions
```

*Cost:* O(k²) LLM calls for pairwise comparison. With k=10, that's 45 comparisons — not practical at scale. Mitigation: only check pairs with high topic overlap (embedding cosine similarity > 0.7). Typically reduces to 3–5 pairs for most query sets.

**Handling detected contradictions:**

*Option 1 — Surface the contradiction to the LLM explicitly:*
```
[IMPORTANT: The following two sources contain conflicting information. 
 Use the more recent source (Source B, dated 2024-03-01) as authoritative 
 unless the question specifically asks about historical policy.]

[Source A: Employee Handbook 2022]
Annual leave is 15 days per year.

[Source B: Employee Handbook 2024 (SUPERSEDES Source A)]
Annual leave is 20 days per year.
```

Adding the metadata about which source is more recent helps the LLM make the right choice.

*Option 2 — Use document recency as a tiebreaker:*
Every chunk has a `last_modified` date. If two chunks contradict, prefer the more recent one. Inject recency metadata into the context and instruct the LLM to prefer newer information.

*Option 3 — Surface the contradiction to the user:*
"I found conflicting information in your knowledge base: [Source A] states X while [Source B] states Y. [Source B] is more recent. Would you like to see both?"

This is the most transparent approach — appropriate for compliance-critical or high-stakes domains.

**Prevention at ingestion time:**
Track document versions explicitly. When a new version of a document is ingested, mark old version chunks as `superseded=true` and filter them from retrieval by default. This prevents the contradiction from reaching the LLM at all.

---

## Q24. How do you preserve document structure — tables, code, lists, math — through chunking and context assembly?

**Answer:**

Structure carries meaning. A table with headers "Q1", "Q2", "Q3" and rows "Revenue", "Expenses" is meaningless if the cells are extracted out of context. A code snippet split at line 47 of a 60-line function is syntactically broken.

**Tables:**

*Strategy A — Prose conversion:*
Convert the table to natural language prose at ingestion: "In Q1, Revenue was $4.2M and Expenses were $2.1M. In Q2, Revenue was $4.8M and Expenses were $2.3M." The prose version is embeddable and fully preserves the values. Loss: complex multi-row tables become verbose.

*Strategy B — Structured representation in context:*
Store the table as markdown in the chunk:
```
| Quarter | Revenue | Expenses | Margin |
|---------|---------|----------|--------|
| Q1 2024 | $4.2M   | $2.1M    | 50%    |
| Q2 2024 | $4.8M   | $2.3M    | 52%    |
```
Markdown tables are well-understood by modern LLMs and preserve column relationships. The embedding quality for markdown tables is reasonable with current models.

*Strategy C — Metadata indexing for structured retrieval:*
For tables that need to be queried structurally ("What was Q3 margin?"), extract table data as structured JSON alongside the text embedding. Route structured queries to a SQL layer, not the RAG pipeline.

**Code:**

Never split code mid-function. Use AST-based splitting: functions and classes are the natural chunk units. Preserve the full function signature, docstring, and body as a single chunk. For context assembly, include the code in a fenced code block:

```
[Source: payment_service.py | Function: process_refund]
```python
def process_refund(transaction_id: str, amount: float) -> RefundResult:
    """Process a refund for a given transaction."""
    ...
```
```

**Mathematical notation:**

LaTeX in documents is often mangled by PDF extractors. Strategies:
- Use a PDF extractor that handles LaTeX (mathpix, nougat)
- For equations central to the query, convert to verbose natural language ("the probability P of event A given B equals the probability of A and B divided by the probability of B")

**Lists:**

Numbered and bulleted lists must be kept together as a single chunk. A policy with 10 requirements split across two chunks loses the enumeration context — chunk 2 starts with "7. Employees must..." with no reference to requirements 1–6.

---

## Q25. What is the lost-in-the-middle problem and how do you mitigate it?

**Answer:**

**What it is:**
Nelson et al. (2023) empirically demonstrated that LLMs perform significantly worse when the relevant information is in the middle of a long context versus at the beginning or end. In a 10,000-token context with the answer at position 8,000, LLM accuracy drops by 10–20pp compared to when the answer is at position 0 or 10,000.

**Why it happens:**
Attention in transformer architectures is not uniform. The "primacy effect" (the model attends more to early tokens) and "recency effect" (attends more to recent tokens) both degrade middle-context attention. For long contexts, the middle effectively has lower resolution.

**Impact on RAG:**
If you place your most relevant chunk at position 5 of 10 in the context, the LLM may generate an answer ignoring that chunk — instead using whichever chunk appears first or last. This means a correct retrieval can produce an incorrect answer due to poor context ordering.

**Mitigation 1 — Book-end ordering (most effective):**
Place the highest-reranker-scored chunk first, the second-highest-scored chunk last. Fill the middle with lower-ranked chunks. This maximises the signal that reaches the LLM:

```python
def book_end_order(chunks_by_score: list[Chunk]) -> list[Chunk]:
    if len(chunks_by_score) <= 2:
        return chunks_by_score
    best = chunks_by_score[0]
    second_best = chunks_by_score[1]
    middle = chunks_by_score[2:]
    return [best] + middle + [second_best]
```

**Mitigation 2 — Reduce k:**
Fewer chunks means the answer is more likely to be near the beginning or end. If 5 chunks are sufficient, do not include 15. Token savings and attention quality both improve.

**Mitigation 3 — Context compression:**
Before assembly, compress each retrieved chunk to only its most relevant sentences. This reduces overall context length, keeping more of the context in the high-attention zones.

**Mitigation 4 — Chunk repeating (expensive):**
For critical passages, include them at both the start and end of the context. The LLM sees the important information twice — at the beginning (primacy) and at the end (recency). Doubles the token cost for that chunk.

**Mitigation 5 — Model selection:**
Newer models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) exhibit significantly reduced lost-in-the-middle degradation compared to earlier models. If your application requires long contexts, evaluate models specifically for long-context performance, not just short-context benchmarks.

**Token cost implication:**
Book-end ordering and k reduction are free — they improve quality and reduce cost simultaneously. Context compression adds one LLM call per chunk (compression model) but reduces downstream LLM input tokens by 40–60%. For expensive generation models, compression often reduces total cost despite the additional compression call.
