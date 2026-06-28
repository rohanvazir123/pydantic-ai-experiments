# Retrieval — Answers

## Q16. Walk me through your full reranking architecture. What does a cross-encoder add?

**Answer:**

Retrieval is a two-stage process in production RAG: coarse retrieval (fast, approximate) followed by reranking (slower, more precise).

**Stage 1 — Bi-encoder retrieval (embedding similarity):**
The query and documents are encoded independently into vectors. Similarity is computed by dot product. This is fast (sub-millisecond per query on an HNSW index) but imprecise: the bi-encoder cannot model the interaction between query and document — it must capture each independently.

Result: top-100 candidates ranked by cosine similarity.

**Stage 2 — Cross-encoder reranking:**
The cross-encoder takes a (query, document) pair as input simultaneously. It can model fine-grained interactions between them. Output: a relevance score for each pair.

```
Input:  [CLS] "What is the parental leave policy?" [SEP] "Employees with 12 months tenure 
         are eligible for 16 weeks paid parental leave..." [SEP]
Output: relevance score = 0.94
```

The cross-encoder reads both texts in a single forward pass, attending across both simultaneously. This captures nuances the bi-encoder misses: negation ("the policy does NOT apply to contractors"), conditional statements, and precise factual alignment.

**What the cross-encoder adds:**
- Recall improvement: the bi-encoder may rank a vaguely related but verbose chunk above a short but precisely relevant chunk. The cross-encoder fixes this.
- Precision improvement: bi-encoder similarity scores are not calibrated for relevance — two similar-looking scores may correspond to very different actual relevance. Cross-encoder scores are better calibrated.
- Typical improvement: 5–15pp recall@5 improvement over bi-encoder alone on domain-specific queries.

**What it costs:**

| Component | Latency | Cost per 1000 queries |
|-----------|---------|----------------------|
| Bi-encoder retrieval (top-100) | 10–50ms | ~$0.01 (embedding) |
| Cross-encoder reranking (top-100 → top-10) | 100–400ms | ~$0.05–0.20 (inference) |
| Full pipeline (retrieval + rerank) | 150–450ms | ~$0.06–0.21 |

The cross-encoder runs N forward passes (N = size of candidate set). Limit N to 50–100 — running the cross-encoder on 1,000 candidates takes 1–4 seconds and is usually not worth it. The bi-encoder's top-100 should be high enough recall that the cross-encoder can find the right answer within them.

**Latency mitigation:**
Reranking can run in parallel with early stages of prompt construction. If the bi-encoder retrieval returns at T=50ms and reranking takes 300ms, the total pipeline latency only increases by 250ms (not 300ms) if you overlap reranking with other processing.

---

## Q17. How do you tune the top-k retrieval parameter?

**Answer:**

k is the number of chunks retrieved before reranking. It directly affects retrieval recall, context token cost, and LLM quality.

**The failure modes:**

*k too low (k < 5):*
High miss rate. If the relevant chunk is ranked 6th by the embedding model, it is excluded. Typical bi-encoder retrieval has 70–80% recall@5 on diverse query sets — meaning 20–30% of queries don't have the relevant chunk in the retrieved set. Low k is the most common reason RAG systems produce hallucinated answers: the LLM generates an answer because it can't say "I don't know" with no context, but the context doesn't actually contain the answer.

*k too large (k > 20 without reranking):*
Context dilution. Including 20 chunks means 20 × 512 tokens = 10,240 tokens of context. The LLM processes all of this but the relevant passage is diluted by 19 other chunks. Answer quality degrades because the LLM cannot reliably identify which chunk contains the authoritative answer (lost-in-the-middle). Token cost increases proportionally: 10,240 input tokens per query × 10,000 queries/day × $0.015/1K tokens = $1,536/day for context alone.

**Optimal k strategy:**

*Phase 1 — Retrieval k (before reranking):* Set high (50–100) to maximise recall. This is the candidate set. At this stage, you care about recall, not precision.

*Phase 2 — Reranked k (after reranking):* Reduce to 5–10 for context assembly. After reranking, precision is high — the top-5 reranked chunks are almost certainly the most relevant.

**Dynamic k:**
Not all queries need the same k. A factual lookup ("What is the CEO's name?") needs k=3 — the answer is in one place. A synthesis query ("Summarise the key risks mentioned in the due diligence report") needs k=15+. Use a query complexity classifier to set k dynamically.

**Token cost impact of k:**
Context assembly cost = k × avg_chunk_tokens × input_token_price. With k=10 and 512-token chunks at $0.015/1K tokens:
- 10 × 512 = 5,120 tokens per query
- 10,000 queries/day = 51.2M tokens/day
- Cost: $768/day

With k=5: $384/day. k is a significant cost lever at scale — tune it carefully.

---

## Q18. Query transformation — HyDE, query expansion, multi-query retrieval.

**Answer:**

Query transformation addresses a fundamental mismatch: the user's query is expressed in the user's vocabulary, but the indexed documents use different vocabulary. Transformation bridges this gap.

**HyDE (Hypothetical Document Embeddings):**

Instead of embedding the query directly, ask the LLM to generate a hypothetical document that would answer the query, then embed the hypothetical document.

```
Query: "What are the benefits of transformer architecture in NLP?"
Hypothetical document: "Transformer architecture offers several advantages in NLP tasks. 
                         The self-attention mechanism allows the model to capture long-range 
                         dependencies efficiently, unlike RNNs which suffer from vanishing 
                         gradients. Transformers are also highly parallelisable..."
Embed: the hypothetical document (not the query)
Retrieve: documents similar to the hypothetical document
```

*Why it works:* The hypothetical document uses the vocabulary, style, and structure of actual documents in the corpus — it is a better match for the embedding space than the bare query.
*Fails when:* The LLM generates a hypothetical document that is confidently wrong (hallucinated facts that match irrelevant documents). Works best for informational queries; less useful for navigational or factual lookup queries.
*Cost:* One additional LLM call per query (~$0.01–0.05 depending on model). Adds 200–800ms latency.

**Query expansion:**

Augment the query with synonyms, related terms, and alternative phrasings before retrieval.

```
Query: "employee leave policy"
Expanded: "employee leave policy, vacation time, annual leave, PTO, paid time off, 
           absence management, holiday entitlement"
Retrieve: using the expanded query embedding
```

*Why it works:* Coverage of vocabulary variations. If the document uses "holiday entitlement" and the query uses "vacation", expansion bridges the gap.
*Fails when:* Expansion adds irrelevant terms that shift the embedding away from the true intent. "AI model" expanded to "AI model, fashion model, model train" would degrade retrieval badly.
*Cost:* Either rule-based (synonymy lookup, free) or LLM-based (one additional LLM call, $0.01–0.05). LLM-based expansion is higher quality.

**Multi-query retrieval:**

Generate N reformulations of the query, retrieve for each, deduplicate and merge results.

```
Original: "What are the performance review criteria?"
Variant 1: "How are employees evaluated in performance reviews?"
Variant 2: "What factors determine an employee's performance rating?"
Variant 3: "Performance appraisal criteria and scoring rubric"
```

Retrieve top-k for each variant. Deduplicate by chunk ID. Rerank the combined set.

*Why it works:* Different phrasings retrieve different chunks. A query about "performance review criteria" phrased as "evaluation factors" may retrieve a chunk that the original phrasing missed. Union of retrievals has higher recall than any single retrieval.
*Cost:* N embedding calls + N ANN searches. With N=3 and 50ms per retrieval: adds 100–150ms. One LLM call to generate variants: adds 200–500ms. Total added cost: 300–650ms per query. Significant — only justified when single-query recall is demonstrably insufficient.

**When to use each:**
- HyDE: informational queries in specialised domains with vocabulary mismatch
- Query expansion: short queries (< 5 words), diverse vocabulary corpus
- Multi-query: complex questions with multiple facets, when recall@10 < 75%

---

## Q19. How do you detect when retrieval has failed?

**Answer:**

Retrieval failure is the most dangerous failure mode in RAG because it is often invisible: the LLM generates a confident-sounding answer using whatever context it received, even if none of it contains the actual answer.

**Detection method 1 — Query-context similarity threshold:**
After retrieval, compute the maximum cosine similarity between the query embedding and each retrieved chunk embedding. If the maximum similarity is below a threshold (empirically, < 0.3–0.4), no retrieved chunk is semantically close to the query.

```python
def is_retrieval_failure(query_embedding, chunk_embeddings, threshold=0.35):
    max_similarity = max(cosine_similarity(query_embedding, c) for c in chunk_embeddings)
    return max_similarity < threshold
```

*Limitation:* Cosine similarity is a weak proxy for relevance. A high similarity does not guarantee the chunk answers the question; a low similarity does not guarantee the chunk is irrelevant. This is a coarse filter.

**Detection method 2 — LLM faithfulness score:**
After generation, ask a small LLM: "Does this answer follow from the provided context? (yes/no/partially)"

```
System: You are a faithfulness evaluator.
Context: {retrieved_chunks}
Answer: {generated_answer}
Question: Is every factual claim in the answer directly supported by the context? 
          Reply: yes, no, or partially.
```

If "no" — retrieval likely failed. The LLM generated an answer from parametric knowledge because the context was insufficient.
*Cost:* One additional LLM call per query. For a fast small model (Haiku, GPT-4o-mini): ~10ms latency, ~$0.001/query. Run this on 100% of queries and log the results. Use the "no" rate as a real-time retrieval health metric.

**Detection method 3 — Empty or very short retrieved set:**
If retrieval returns 0 chunks (no results above the similarity threshold), retrieval has definitively failed.

**Detection method 4 — Citation verification failure:**
If the generated answer contains citations, verify that each cited chunk actually supports the cited claim. If citation verification fails for > 50% of citations, the LLM generated content not grounded in the retrieved context.

**What to do when retrieval fails:**
Do not generate an answer. Instead:
1. Try query transformation (HyDE, expansion) and re-retrieve
2. If re-retrieval also fails, return: "I couldn't find relevant information in the knowledge base to answer this question."
3. Log the query for corpus gap analysis — a retrieval failure on a legitimate question signals a gap in the indexed corpus.

---

## Q20. A query requires multi-hop reasoning across three documents. How does your retrieval handle it?

**Answer:**

Multi-hop queries require connecting information from multiple sources where the connection is not explicit in any single document. Example: "What is the maximum expense amount an employee can approve without a VP signature, given our expense policy and the VP delegation schedule?"

This requires:
- Document 1 (expense policy): "Expenses above $5,000 require senior approval"
- Document 2 (delegation schedule): "VP approval threshold is $10,000; Director threshold is $5,000"
- Synthesising: Directors can approve up to $5,000; VPs can approve up to $10,000

**Why naive RAG fails for multi-hop:**
A single retrieval pass retrieves the chunks most similar to the original query. The query may retrieve the expense policy chunk but not the delegation schedule chunk (it is semantically less similar to the query). The LLM then answers with incomplete information or hallucinates the delegation details.

**Solution 1 — Iterative retrieval:**
Retrieve → generate intermediate answer → use intermediate answer as the next query → retrieve again → final generation.

```
Step 1: Query: "expense approval limits"
        Retrieved: expense policy chunk ("expenses > $5,000 require senior approval")
        Intermediate: "Senior approval is required above $5,000. What constitutes 'senior'?"

Step 2: Query: "VP Director approval thresholds delegation"
        Retrieved: delegation schedule chunk
        Final generation: combines both chunks
```

*Cost:* 2× LLM calls, 2× retrieval calls. Adds 1–3s latency. Only justified for queries that are explicitly multi-step.

**Solution 2 — Knowledge graph retrieval:**
Pre-build a knowledge graph that explicitly connects entities across documents: Employee → ExpensePolicy → ApprovalThreshold → DelegationSchedule. A graph traversal can find the multi-hop path without multiple LLM calls.

*Cost:* Significant upfront investment in graph construction and maintenance. Only justified when multi-hop queries are frequent and the entity relationships are stable.

**Solution 3 — Pre-built dense retrieval with rich metadata:**
Index chunks with entity metadata: "This chunk is about: [expense policy, approval thresholds, Director-level employees]". At query time, retrieve based on the union of all entity mentions in the query. This brings in related chunks without multi-hop reasoning.

**Solution 4 — Long context + full document retrieval:**
For known multi-document queries (detected by query classification), retrieve entire documents rather than chunks and include them in a long-context LLM. Only feasible when documents are short and the LLM has a 128k+ context window.

**The honest answer:**
Multi-hop reasoning is an unsolved problem for RAG at scale. Iterative retrieval is the most reliable approach for production but doubles or triples latency. For most business RAG use cases, careful corpus design (pre-computing cross-document summaries, building a Q&A document that synthesises related policies) is more practical than a sophisticated multi-hop architecture.

---

## Query normalization: what gets normalized, what stays raw

A common design question is whether to normalize the user query (lemmatize, lowercase, strip punctuation) before passing it to the LLM. The answer is: **normalize for retrieval internals, keep raw for the LLM**.

### What the normalized query is used for

Normalization (e.g. spaCy lemmatization: "What are the PTO policies?" → "what be the pto policy") is applied inside the retriever, before:

- **L2 cache key** — the sha256 hash is computed from the normalized query, so "PTO Policy" and "pto policy" hit the same Redis entry
- **Embedding call** — the embedding is computed from the normalized form, so semantically equivalent queries produce the same (or very close) vector, improving L3 semantic cache hit rate
- **Full-text search (tsvector/BM25)** — normalized tokens reduce surface-form mismatch against indexed content

### What receives the raw query

- **Intent classifier** — the nano model classifying intent should see the original phrasing; lemmatized text ("what be the pto policy?") is harder for a small model to interpret correctly
- **Main LLM** — the LLM receives the original user query and the retrieved context. Passing "what be the pto policy" to the LLM would produce stilted, unnatural responses

### Why normalization is local to the retriever

The normalization `query = normalize_query(query)` is a local rebind inside `retriever.retrieve()`. The `query` variable in the calling pipeline is unaffected. This is intentional: the retriever can use whatever internal representation improves cache hit rate and retrieval quality, while the rest of the pipeline handles the user's actual words.

### Summary

| Component | Query form |
|-----------|-----------|
| L2 Redis cache key | normalized |
| Embedding / vector search | normalized |
| L3 semantic cache lookup | normalized (via embedding) |
| Full-text search (BM25/tsvector) | normalized |
| Intent classifier | raw |
| Main LLM generation | raw |
| User-facing response | raw (user sees their own words reflected back) |
