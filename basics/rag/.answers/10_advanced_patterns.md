# Advanced Patterns — Answers

## Q47. What is HyDE and when does it improve retrieval?

**Answer:**

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) inverts the standard retrieval approach: instead of embedding the query and finding similar documents, it asks the LLM to generate a hypothetical document that would answer the query, then embeds that document and retrieves similar real documents.

**The intuition:**

A user query ("benefits of attention mechanism in transformers") is short and uses layman phrasing. The documents ("The self-attention mechanism allows the model to attend to all positions in the sequence simultaneously, enabling the capture of long-range dependencies...") are long and use technical phrasing. The embedding representations of these two texts may be semantically close but not as close as two document-style texts on the same topic would be.

HyDE bridges this by creating a document-style text on the query side:

```
Query: "benefits of attention mechanism in transformers"

LLM generates: "Attention mechanisms in transformers offer several key advantages. 
                The self-attention operation allows each position in the sequence to 
                attend to all other positions, enabling the capture of long-range 
                dependencies that recurrent networks struggle with. Unlike RNNs, 
                transformers are fully parallelisable, reducing training time..."

Embed the generated text → retrieve documents similar to the generated text
```

**When it improves retrieval:**

1. Short, abstract queries (< 5 words) where the query embedding is under-specified
2. Specialised domains where user vocabulary differs significantly from document vocabulary
3. Queries asking "how" or "why" — the answer style matches document style more than question style
4. When recall@10 on bare query retrieval is < 70% on your evaluation set

**When it hurts:**

1. Factual lookup queries ("What is the CEO's name?") — the LLM may generate a confident but wrong fact, and retrieving documents similar to the wrong hypothetical document gives worse results than the original query
2. Queries about company-specific or proprietary information the LLM has never seen in training — the hypothetical document will be generic and may retrieve generic rather than specific chunks
3. Time-critical queries — HyDE adds one LLM call (200–800ms) before retrieval even starts

**Token cost:**

One additional LLM call per query. With GPT-4o-mini: ~200–400 token output × $0.0006/1K output tokens = $0.0002/query. Negligible cost. Latency is the real constraint.

---

## Q48. RAG fusion and multi-query retrieval — when is it worth the cost?

**Answer:**

**Multi-query retrieval:** Generate N reformulations of the query, retrieve for each, deduplicate and merge.

**RAG fusion:** A specific implementation of multi-query retrieval that uses Reciprocal Rank Fusion (RRF) to merge the ranked lists from multiple retrievals into a single unified ranking.

```python
def rag_fusion(query: str, llm, retriever, n_variants=4, k=10):
    # Step 1: Generate query variants
    variants = llm.generate_variants(query, n=n_variants)
    # e.g., ["What are the expense approval thresholds?",
    #         "Maximum amount an employee can expense without approval?",
    #         "Expense policy authorization limits by seniority"]
    
    # Step 2: Retrieve for each variant
    all_results = {}
    for variant in [query] + variants:
        for rank, chunk in enumerate(retriever.search(variant, k=k)):
            if chunk.id not in all_results:
                all_results[chunk.id] = {"chunk": chunk, "rrf_score": 0}
            all_results[chunk.id]["rrf_score"] += 1 / (60 + rank)
    
    # Step 3: Re-rank by RRF score
    return sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)[:k]
```

**When it's worth the cost:**

1. Multi-faceted queries that can be decomposed into sub-questions: "What are the tax implications of employee stock options for both the company and the employee?" — generates separate retrievals for company-side and employee-side tax treatment.

2. Queries where single-pass retrieval recall@10 is < 75%. Multi-query increases recall by covering vocabulary variations that the original query misses.

3. Synthesis queries where breadth of coverage matters more than precision.

**When it is not worth the cost:**

1. Simple factual lookups. "What is the CEO's name?" does not benefit from 4 reformulations — the answer is in one place.

2. Latency-critical applications. Multi-query adds N embedding calls + N retrieval calls in parallel (adds 50–100ms) + N-1 LLM calls to generate variants (adds 200–800ms). Total added latency: 300–900ms.

3. Corpora with uniform vocabulary. If the document vocabulary is consistent and closely matches user phrasing, vocabulary bridging is unnecessary.

**Token cost:**

N LLM calls for variant generation: N × 100 tokens × $0.0006/1K = $0.00006 per variant. For N=4: $0.00024/query. Negligible. The real cost is latency.

---

## Q49. Agentic RAG — the system decides when and what to retrieve.

**Answer:**

In basic RAG, retrieval happens exactly once per query. Agentic RAG gives the system tools (retrieve, search, summarise, calculate) and lets the LLM decide when to call them and how many times.

**Architecture:**

```python
def agentic_rag(query: str, max_iterations=5):
    context = []
    history = []
    
    for iteration in range(max_iterations):
        # LLM decides what to do
        action = llm.think(query, context, history, tools=[
            "retrieve(search_query)",
            "summarise(document_id)",
            "answer(final_answer)"
        ])
        
        if action.type == "retrieve":
            chunks = retriever.search(action.query, k=5)
            context.extend(chunks)
            history.append(f"Retrieved {len(chunks)} chunks for: {action.query}")
        
        elif action.type == "summarise":
            summary = summariser.summarise(action.document_id)
            context.append(summary)
        
        elif action.type == "answer":
            return action.final_answer
    
    return "Max iterations reached — unable to fully answer"
```

**When to use:**

1. Multi-hop queries where the next retrieval depends on what the previous one found: "Find the policy that applies to the customer segment that has the highest churn rate."
   - Retrieve churn data → find highest-churn segment → retrieve the policy for that segment
   
2. Queries requiring iterative refinement: initial retrieval is insufficient, so the agent retrieves again with a more specific query.

3. Tool-using scenarios where retrieval is one of several tools (alongside SQL queries, calculations, API calls).

**Failure modes of agentic loops:**

*Infinite loops:* The agent keeps retrieving without converging on an answer. Mitigate with `max_iterations` and a hard stop.

*Query drift:* Each retrieval generates a new query that drifts further from the original intent. By iteration 5, the agent is retrieving content completely unrelated to the original question. Mitigate by always including the original query in the LLM's context.

*Confirmation bias:* The agent only retrieves information that confirms its initial hypothesis, missing contradictory evidence. Mitigate by including a "search for conflicting information" step in the system prompt.

*Token cost explosion:* Each iteration adds LLM calls + retrieval. At 5 iterations: 5× the base RAG cost. At $0.10/query base cost: $0.50/query for agentic RAG. At 10,000 queries/day: $5,000/day. Only use agentic RAG when single-pass RAG demonstrably fails.

---

## Q50. Documents longer than the context window — books, contracts, manuals.

**Answer:**

A 500-page legal contract (≈ 200,000 tokens) cannot fit in any current context window as a single block. You must decompose it.

**Strategy 1 — Standard chunking + RAG:**
Split into chunks of 512–1,024 tokens, embed, index. Retrieval finds the relevant clauses. This is the right strategy for most long documents.

*Works for:* Contracts where specific clauses answer specific questions.
*Fails for:* Questions requiring understanding of the document as a whole ("Does this contract create any unusual obligations?"), or questions where the answer spans many non-adjacent sections.

**Strategy 2 — Hierarchical summarisation:**
Build a summary hierarchy:
```
Document (200,000 tokens)
→ Chapter summaries (10 × 2,000 tokens = 20,000 tokens)
→ Document-level summary (2,000 tokens)
```

Index summaries at each level. For high-level questions, retrieve the document-level summary. For specific questions, retrieve chapter summaries, then retrieve specific chunks from the relevant chapter.

*Cost:* One LLM call per chapter to generate summaries. For 10 chapters: 10 × $0.10 = $1.00 per document. For 500K documents: $500K for initial summarisation. Only justified for long documents; skip for short ones.

**Strategy 3 — Late chunking (JinAI approach):**
Embed the full document using a long-context embedding model, then pool the token embeddings at chunk boundaries rather than re-embedding each chunk. This preserves global document context in the chunk embeddings.

*Requires:* A model with long-context embedding support (jina-embeddings-v3 supports up to 8,192 tokens with late chunking).

**Strategy 4 — Sliding window with large overlap:**
For contracts where adjacent sections are highly cross-referential, use very large chunks (2,048 tokens) with 50% overlap. Expensive in storage and embedding cost, but preserves more context.

**Token cost consideration:**
Every strategy except standard chunking adds significant upfront cost. Profile your actual long-document query volume before investing in hierarchical summarisation infrastructure. If only 5% of documents are > 10 pages, standard chunking handles 95% of cases efficiently.

---

## Q51. Self-RAG — when does the model decide to retrieve?

**Answer:**

Self-RAG (Asai et al. 2023) trains a model to generate special reflection tokens at inference time: `[Retrieve]` (should retrieval happen?), `[IsREL]` (is the retrieved passage relevant?), `[IsSUP]` (is the response supported by the passage?), `[IsUSE]` (is the response useful?).

**How it works:**

```
User: "What is the capital of France?"
Model generates: "Paris" (no [Retrieve] token — parametric knowledge sufficient)

User: "What does section 4.2 of our employee handbook say about parental leave?"
Model generates: "[Retrieve]"
System: run retrieval, inject top-k chunks
Model generates: "According to the handbook... [IsSUP: supported] [IsREL: fully relevant]"
```

**What it requires:**

Fine-tuning the base LLM on a dataset that teaches when retrieval is necessary. The training data must include examples of: queries answerable from parametric knowledge, queries requiring retrieval, and queries where retrieved context is irrelevant or insufficient. This is a significant training investment — appropriate for research or large-scale deployments, not standard business RAG.

**When it is useful:**

High-volume deployments where a significant fraction of queries (> 30%) could be answered from parametric knowledge without retrieval. Retrieval for these queries wastes latency and cost. Self-RAG's `[Retrieve]` token acts as a query routing mechanism: retrieve only when needed.

**Practical alternative for most teams:**
A lightweight intent classifier that runs before retrieval: "Does this query require information from the knowledge base, or can it be answered from general knowledge?" This achieves similar routing without fine-tuning — just prompt engineering + a small classifier.

---

## Q52. Combining RAG over documents with structured data (SQL, knowledge graphs).

**Answer:**

Most enterprise knowledge lives in two forms: unstructured documents (policies, reports, emails) and structured databases (transaction records, CRM data, product catalogs). A complete knowledge system needs both.

**The retrieval routing problem:**

"Show me the refund policy for orders over $500 placed by Gold-tier customers in Q3 2024."

This requires:
- Policy document: "What is the refund policy?" → RAG
- Structured data: "What orders over $500 were placed by Gold-tier customers in Q3?" → SQL

**Architecture — Hybrid retrieval router:**

```python
def hybrid_retrieve(query: str, user_context: dict) -> list[Context]:
    # Route to appropriate retrieval mechanism
    route = query_router.classify(query)
    
    if route == "DOCUMENT_ONLY":
        return rag_retrieve(query)
    
    elif route == "SQL_ONLY":
        sql = nl2sql.generate(query, schema)
        return [structured_query(sql)]
    
    elif route == "HYBRID":
        doc_results = rag_retrieve(query)
        sql_results = nl2sql_retrieve(query)
        return merge_contexts(doc_results, sql_results)
```

**Merging structured and unstructured results:**

Context assembly must present structured data clearly to the LLM:
```
[Retrieved Documents]
Refund Policy (Section 3.2): Orders can be refunded within 30 days if the customer
is a Gold-tier member. No restocking fee applies.

[Structured Data]
Query: Orders > $500, Gold-tier customers, Q3 2024
Results: 142 orders, total value $89,340, of which 23 have pending refund requests
```

The LLM then synthesises: "Based on the policy, Gold-tier customers can request refunds within 30 days. The 23 pending refund requests from Q3 2024 are eligible under this policy."

**When hybrid makes sense:**

- Questions combining policy + data: "Are there any orders that should have qualified for a refund but don't show one?"
- Questions requiring corroboration: "The contract says delivery should be within 5 business days — how are we actually performing?"
- Knowledge graph: for entity-relationship queries ("What other products are made by the vendor who supplies product X?"), a knowledge graph provides better precision than document retrieval.

**Token cost of hybrid:**
Two retrieval paths add overhead but both are cheap relative to LLM generation. The main cost is additional LLM calls (one for SQL generation, one for synthesis). At $0.01 per LLM call: ~$0.02 additional per hybrid query. Acceptable given the quality improvement.
