# Latency SLAs — Answers

## Q33. You have a p99 latency SLA of 3 seconds. You're already at 2.8s with zero margin. What do you do?

**Answer:**

2.8s with zero margin means one slow LLM response or one cache miss blows the SLA. This requires structural changes, not tuning.

**Architectural decisions to hold the SLA under load:**

*Decision 1 — Decompose the SLA into sub-component budgets:*
The 3-second budget must be explicitly partitioned: schema retrieval < 150ms, prompt construction < 20ms, LLM generation < 2,200ms, SQL validation < 50ms, result formatting < 80ms. Each component owner is accountable to their budget. Without explicit budgets, every team optimizes locally and nobody owns the overall SLA.

*Decision 2 — Maximize parallelism:*
As noted earlier, schema retrieval and LLM call can be partially parallelized. More specifically: start the LLM call with a preliminary prompt (system instructions + query, no schema yet). While the LLM processes the system prefix, retrieval runs. When retrieval completes, inject the schema context and the LLM continues. This "pre-warming" approach saves 100–150ms of retrieval time from the critical path.

*Decision 3 — Tiered model strategy:*
Route queries by complexity: simple queries (single table, no join, pattern matches a known template) go to a smaller, faster model (7B fine-tuned, inference < 800ms). Complex queries go to the large model. This requires a lightweight complexity classifier upfront, but it moves 40–60% of traffic to a path with a 1s budget instead of 2.2s.

*Decision 4 — Aggressive caching:*
Target 30–40% cache hit rate. At 2.8s uncached, a 35% cache hit rate at 100ms brings the average response time to ~1.8s. The p99 only improves if you can also cache the slow tail — queries that historically take > 2.5s are candidates for mandatory caching with longer TTLs.

**What to drop when you still can't hold the SLA:**

In priority order:
1. Drop the reranking step — use embedding similarity ranking directly (saves 100–200ms, accuracy cost is moderate)
2. Reduce k (tables in context) from 15 to 8 (saves ~300ms in LLM processing, accuracy cost on complex joins)
3. Drop self-consistency checking / retry logic — first-attempt SQL only (saves 1.5–2s on retries, accuracy cost is measurable)
4. Return generated SQL without execution — let the user decide when to run it (moves execution off the critical path entirely)

Never drop security validation — it is not a latency optimization target.

---

## Q34. How do you set a latency SLA for NL2SQL in the first place?

**Answer:**

Setting an SLA requires separating the controllable latency (SQL generation) from the uncontrollable latency (SQL execution).

**Two separate SLAs:**

*SLA 1 — Time-to-SQL:* The time from query submission to a valid, validated SQL query being ready for execution. This is entirely within the system's control. A reasonable target: p50 < 1s, p99 < 3s. This is the SLA the NL2SQL team owns.

*SLA 2 — Time-to-results:* The time from SQL being ready to results being returned. This depends on the database, query complexity, and warehouse load. This SLA is owned jointly with the data infrastructure team. Warehouse queries can take 10ms (cached) or 20 minutes (full scan). Setting a sub-2s SLA here requires either restricting the query complexity or having a highly performant warehouse.

**Why conflating these into one SLA is a mistake:**
If your SLA is "under 5 seconds total", the NL2SQL team has no way to hit it independently — a slow warehouse makes them miss the SLA even if their pipeline is instant. Separate the SLA to create clear ownership.

**What to communicate to users when execution exceeds the SLA:**

Don't show a loading spinner for 45 minutes. Surface the SQL immediately after generation (SLA 1 is met). Show a progress indicator for execution with an estimated completion time (from the EXPLAIN plan). Allow the user to cancel the query and run a modified, cheaper version. This converts a "system is slow" experience into a "system is working on something complex" experience.

---

## Q35. Schema retrieval has p50 of 80ms but p99 of 900ms. What causes the tail?

**Answer:**

A 10x gap between p50 and p99 in vector search is larger than typical and suggests something beyond normal query variance.

**Likely causes:**

*Cause 1 — Cold cache on the embedding model:*
If the embedding model (for query embedding) is loaded on demand, the first request after a cold start incurs model loading latency. This shows up as p99 spikes, not as elevated p50. Solution: keep the embedding model warm with synthetic pings, or pre-load it at service startup.

*Cause 2 — Index fragmentation or suboptimal HNSW parameters:*
Vector indexes (HNSW, IVF) have parameters (ef_search, nprobe) that trade accuracy for speed. High ef_search values give better recall but increase per-query latency. If the vector index was configured for accuracy without latency constraints, the p99 reflects worst-case graph traversal. Solution: tune ef_search to balance recall and latency; validate that recall at the optimized setting is still acceptable.

*Cause 3 — Long-tail query embedding time:*
Complex, long queries take longer to embed. A 20-token query embeds in 5ms; a 200-token query takes 40ms. If the p99 queries are the longest queries (verbose, multi-part questions), the embedding step itself is the bottleneck. Solution: truncate or summarize long queries before embedding.

*Cause 4 — Database connection pool exhaustion:*
If the vector database (Pinecone, pgvector, Weaviate) is accessed through a connection pool, pool exhaustion under load causes queuing. The p99 includes the queue wait time. Solution: increase pool size, or implement a timeout with a fallback to a degraded retrieval (fewer results, lower accuracy).

*Cause 5 — Reranking on large candidate sets:*
If the reranker receives a large candidate set (top-100 instead of top-40), reranking latency scales with candidate count. Solution: reduce the coarse retrieval candidate set, or run reranking asynchronously while returning the top-5 embedding results immediately.

---

## Q36. You want to offer a "fast mode" that sacrifices accuracy for latency. What do you cut?

**Answer:**

Cuts are ordered by latency savings vs. accuracy cost. Never cut security validation.

**Cut 1 — Skip reranking (save 100–200ms, accuracy cost: low-moderate):**
Reranking improves schema retrieval precision by 5–15% on complex queries. Skipping it returns the raw embedding-ranked results. The accuracy cost is concentrated on queries that require tables ranked > 5 by embedding similarity but < 3 by the reranker. For simple queries (top-1 table is obvious), skipping reranking has no accuracy impact.

**Cut 2 — Reduce schema context (save 200–400ms in LLM processing, accuracy cost: moderate):**
Reduce from 15 tables to 8 tables in context. Reduce full DDL to column-name-only for all but the top-2 tables. Reduces prompt token count by 40–50%, directly reducing LLM TTFT and generation time. Accuracy cost is measurable on queries requiring detailed column-level knowledge of lower-ranked tables.

**Cut 3 — Route to a smaller model (save 500–1000ms, accuracy cost: moderate-significant):**
A 7B or 13B fine-tuned model generates SQL 3–5x faster than a 70B model. Accuracy on simple queries is comparable; accuracy on complex multi-join queries with ambiguous column names is meaningfully lower. Route only simple queries (single table, pattern matches common templates) to the small model in fast mode.

**Cut 4 — Skip self-consistency / retry (save 1.5–2s, accuracy cost: significant for ambiguous queries):**
In standard mode, if the first SQL fails validation, the system retries with modified prompts. Removing retries means the first attempt is the final answer. On queries where the first attempt historically succeeds > 90% of the time, this has minimal impact. On complex queries where retries are common, this significantly reduces accuracy.

**Measuring before shipping:**
For each cut, compute the accuracy delta on your evaluation set by query type. Report: "Fast mode reduces average latency by Xms at the cost of Y percentage points of execution accuracy on multi-join queries." Let the product team make the trade-off decision with real numbers.
