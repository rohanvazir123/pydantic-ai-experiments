# Latency and Performance — Answers

## Q42. Latency components of a RAG pipeline — what is on the critical path?

**Answer:**

**Typical production RAG latency breakdown:**

| Component | Latency | On critical path? | Parallelisable? |
|-----------|---------|------------------|----------------|
| Query embedding | 20–50ms | Yes | No (must precede retrieval) |
| Query transformation (HyDE/expansion) | 200–800ms | Yes (if enabled) | Partial |
| ANN retrieval | 10–50ms | Yes | No |
| Reranking | 100–400ms | Yes | Partially (batch) |
| Context assembly | 5–10ms | Yes | No |
| LLM generation (TTFT) | 200–800ms | Yes | No |
| LLM generation (full) | 500ms–5s | Yes (streaming) | N/A |
| Post-gen faithfulness check | 100–500ms | No (async) | Yes |
| Citation verification | 50–200ms | No (async) | Yes |

**Critical path:** query embedding → [transformation] → retrieval → reranking → context assembly → LLM generation.

**Where to parallelise:**

*Query embedding + query transformation:* If you use HyDE (generate a hypothetical document then embed it), the hypothetical generation and the direct embedding of the original query can run in parallel:
```
Parallel:
  Thread A: embed(original_query) → retrieve(original_embedding)
  Thread B: generate(hypothetical_doc) → embed(hypothetical_doc) → retrieve(hypothetical_embedding)
Merge results, deduplicate, rerank combined set
```

*Sparse + dense retrieval:* BM25 and ANN search run in parallel, merge results with RRF.

*Faithfulness check + citation verification:* Run asynchronously after streaming the answer to the user. These don't need to complete before the user sees the response — they feed the quality monitoring dashboard.

**Where you cannot parallelise:**
Retrieval must come after embedding. Reranking must come after retrieval. Context assembly must come after reranking. LLM generation must come after context assembly. The core pipeline is sequential.

**Token cost and latency trade-off:**

Every token in the context adds latency. A 10,000-token context takes longer to process than a 3,000-token context — both TTFT and generation time scale with input length. The latency cost of a large context window is approximately:
- TTFT: +1–3ms per 1,000 additional input tokens
- Generation time: unaffected (depends only on output length)

Reducing context from 10,000 to 3,000 tokens saves ~20–30ms and reduces input token cost by 70%.

---

## Q43. What can be cached in a RAG system — and what invalidation is needed?

**Answer:**

**Layer 1 — Query embedding cache:**

Identical query strings (after normalization) produce identical embeddings. Cache: `query_text → embedding_vector`. TTL: indefinitely (embeddings don't change unless the model changes). Invalidation: when the embedding model is updated.

*Hit rate:* 20–30% for BI-style applications (users ask the same question repeatedly). Near 0% for conversational assistants (every query is unique).
*Storage:* 1,536 floats × 4 bytes = 6KB per cached embedding. 100K cached queries = 600MB.

**Layer 2 — Retrieval result cache:**

Cache: `(query_embedding_cluster, document_filter) → list[chunk_ids]`. TTL: tied to document update frequency. Invalidation: when any document in the relevant namespace is updated.

*Risk:* A cached retrieval result may include a chunk from a document that has since been deleted or updated. The cache must be invalidated when any document referenced in the cached chunk list changes.
*Implementation:* Store, for each cached retrieval result, the set of document IDs it includes. When a document is updated, invalidate all cached retrieval results that include that document ID.

**Layer 3 — Generated answer cache:**

Cache: `(canonical_query, document_version_hash) → generated_answer`. TTL: tied to both query frequency and document staleness.

*Correctness risk:* A cached answer may become wrong if the underlying documents change. The `document_version_hash` must change whenever any indexed document is updated. But computing this hash for a large corpus is expensive — use a per-namespace version counter instead: increment the counter on every document update, include it in the cache key.

*Token cost savings:* A cached answer skips all LLM generation tokens. For a 500-token generated response at $0.06/1K output tokens: $0.03/query saved. At 30% cache hit rate and 10,000 queries/day: $90/day savings.

**Layer 4 — Chunk content cache:**

When a chunk is retrieved and its content is fetched from the vector store or database, cache the content in a fast store (Redis) by chunk ID. Avoids repeated database fetches for the same popular chunks.

**What you must not cache:**

Generated answers for user-specific queries with personal data. Retrieval results for queries with access control filtering — the cache key must include the user's permission scope to avoid serving Tenant A's cached results to Tenant B.

---

## Q44. How does streaming change the RAG pipeline?

**Answer:**

Streaming means the LLM starts sending response tokens to the user before the full response is generated. For RAG, this is almost always the right default for interactive applications.

**Architecture change for streaming:**

```
Non-streaming (sequential, user waits):
[embed] → [retrieve] → [rerank] → [assemble] → [LLM call] → [return full response]
Total wait: 2–5 seconds

Streaming (user sees tokens as they arrive):
[embed] → [retrieve] → [rerank] → [assemble] → [start LLM stream] → [user sees first token]
Time to first token: 500ms–1.5s
User sees response arriving in ~200ms chunks
```

Time to first meaningful token (TTFT) is the SLA for streaming systems — users perceive this as "responsiveness."

**What streaming enables:**

Early abandonment: if the LLM starts generating a clearly wrong answer, the user can interrupt before the full response is delivered. Requires a stop/cancel mechanism in the UI.

Progressive rendering: the UI can show citations and sources as they appear in the stream, rather than waiting for the full response.

**Cases where streaming is harmful:**

*Faithfulness post-processing:* If you run a faithfulness check after generation and only serve the response if it passes, you cannot stream — the full response must be generated before verification. Solution: stream to the user and flag potential issues asynchronously (show a "Verifying sources..." indicator that resolves after the check).

*Citation injection:* If citations are added by a post-processing step (not generated inline by the LLM), streaming the response before post-processing is complete means the user sees an uncited response. Solution: ask the LLM to generate citations inline within the streamed response.

*Tool-using agents:* If the RAG system is agentic (retrieves → evaluates → re-retrieves), streaming intermediate results to the user is confusing. Stream only the final synthesis step.

---

## Q45. High concurrency — thousands of users simultaneously. What breaks first?

**Answer:**

**Bottleneck 1 — Embedding API rate limits:**

Most embedding APIs have rate limits (tokens/minute). At 1,000 concurrent users × 500 token queries × 1 embedding call = 500K tokens/minute. OpenAI text-embedding-3-small allows 1M tokens/minute at the default tier. At 5,000 concurrent users, you hit the limit. Solution: deploy a self-hosted embedding model (nomic-embed-text, BGE) on GPU. Latency drops from 50ms (API) to 5–10ms (local), and rate limits disappear.

**Bottleneck 2 — Vector store query throughput:**

Vector databases have throughput limits. Pinecone: ~200 QPS per pod. Qdrant/pgvector on a single instance: 50–200 QPS depending on index size and hardware. At 1,000 concurrent users with 1 retrieval/query: you need 1,000 QPS capacity. Solution: horizontal scaling of the vector store (Pinecone pods, Qdrant sharding) or read replicas.

**Bottleneck 3 — LLM API concurrency limits:**

LLM APIs have concurrency limits (simultaneous requests) and tokens/minute limits. GPT-4o: typically 500 RPM (requests per minute) at tier 2. At 1,000 concurrent users all querying simultaneously, requests queue. Solution: use multiple API keys (if permitted), deploy a load balancer across multiple LLM providers, or self-host the LLM.

**Bottleneck 4 — Cross-encoder reranking GPU:**

If reranking runs on a single GPU, it can handle ~50–100 concurrent requests before queueing builds up. Solution: horizontal scaling — deploy multiple reranking instances behind a load balancer.

**Bottleneck 5 — Database connection pool exhaustion:**

The chunk content database (PostgreSQL or equivalent) has a connection pool limit. At high concurrency, pool exhaustion causes queries to queue. Solution: PgBouncer connection pooling, or move chunk content to Redis (in-memory, higher throughput).

**Load test design:**

Before any launch, run a load test at 2× the expected peak concurrent users. Identify the bottleneck (use distributed tracing to see where latency accumulates under load). Fix it. Repeat until the system handles 2× peak without degradation.

---

## Q46. When does pre-generating answers for expected queries make sense?

**Answer:**

Pre-generation (generating and caching answers for anticipated queries before users ask them) is a specific optimisation for predictable, high-value query patterns.

**When it makes sense:**

*High query concentration:* If 30% of your queries are "What is the refund policy?", pre-generating this answer and serving it from cache delivers sub-50ms response time for 30% of traffic. The pre-generated answer is freshened every time the policy document is updated.

*Scheduled reports:* A dashboard that displays the same aggregated data to hundreds of users every morning. Pre-generate the RAG response at 7am before users arrive.

*FAQ systems:* A customer support chatbot where 80% of questions come from a known list of 200 FAQs. Pre-generating answers for all 200 is trivial (200 × $0.01 = $2) and delivers instant responses.

**When it does not make sense:**

*Conversational / personalised queries:* Every query is unique ("What was my contract renewal date?"). Pre-generation is impossible.

*Rapidly changing knowledge base:* If the corpus updates hourly, pre-generated answers become stale quickly. The cost of keeping pre-generated answers fresh exceeds the latency benefit.

*Long-tail queries:* If only 10% of queries repeat, pre-generation has low cache hit rates and high generation cost for limited benefit.

**Staleness risk of pre-generated answers:**

This is the critical risk. A pre-generated answer is correct at generation time and wrong after a document update.

*Mitigation:* Tie pre-generated answer validity to the document version hash. When any document in the relevant set is updated, invalidate all pre-generated answers that cited that document. Trigger re-generation automatically.

*For FAQ systems:* Manually review pre-generated answers after any knowledge base update. A pre-generated answer that is confidently wrong is worse than no answer — it provides stale information at high confidence.

**Token cost of pre-generation:**

Pre-generating 200 FAQ answers: 200 × ~1,000 tokens context + 300 tokens output = 200 × $0.015 + 200 × $0.06 = $3 + $12 = $15 total. Refreshed daily: $450/month. This eliminates LLM costs for 80% of queries if those FAQs are actually what users ask. ROI is high.
