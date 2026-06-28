# Latency and Performance — Answers

## Q20. A user expects under 2 seconds. Your LLM call alone takes 1.5 seconds. Walk me through every latency lever.

**Answer:**

With 1.5s for the LLM call, you have 500ms for everything else. Every millisecond must be accounted for. Here are the levers in order of impact:

**Lever 1 — Parallelize schema retrieval and prompt construction with the LLM call:**
The typical sequential pipeline is: classify → retrieve schema → construct prompt → call LLM. Schema retrieval takes 200–400ms. If you can start retrieval at the same time as any pre-processing step, you reclaim that time. In practice: kick off the embedding of the user query and the retrieval immediately on request receipt. While retrieval runs, do any lightweight preprocessing (session lookup, cache check). By the time retrieval completes, prompt construction starts instantly.

**Lever 2 — Semantic cache for repeated questions:**
Cache the (query, schema_version) → SQL mapping. For repeated queries (which can be 30–40% of traffic in a BI context), the LLM call is skipped entirely. Response time drops to < 100ms. Key: the cache key must include the schema version — a cached SQL is invalid if the schema has changed. Correctness risk: near-duplicate queries that differ slightly in phrasing may be assigned to the wrong cache entry. Mitigate with semantic similarity matching (embed the new query and compare against cached query embeddings) rather than exact string matching.

**Lever 3 — Model size and inference infrastructure:**
1.5s for an LLM call is typical for a large model (70B+) via an API. Options:
- Use a smaller fine-tuned model for NL2SQL specifically (7B–13B fine-tuned models can match 70B performance on NL2SQL benchmarks at 3–5x lower latency)
- Use streaming generation so the user sees the SQL appearing token by token — perceived latency is much lower even if total latency is the same
- Route simple queries (single table, no joins) to a faster/cheaper model; complex queries to the large model

**Lever 4 — Prompt length reduction:**
Token count directly affects LLM latency (time to first token + generation time). Reducing prompt length from 4,000 tokens to 2,000 tokens roughly halves generation latency for the same model. Achieve this by: (1) aggressive schema truncation (only include columns relevant to the query, not all 50 columns in every table), (2) compressed few-shot examples (1–2 examples instead of 5), (3) tighter system prompt.

**Lever 5 — Speculative decoding / prefix caching:**
Many LLM inference frameworks (vLLM, TGI) support prefix caching — if the system prompt is identical across requests (which it is in NL2SQL), the key-value cache for the system prompt is reused across requests, reducing TTFT (time-to-first-token) significantly.

**Lever 6 — SQL execution latency (the 500ms budget):**
Schema retrieval: 80–150ms (vector search, well-optimized)
Prompt construction: < 10ms
SQL validation (parse + schema check): < 10ms
SQL execution (OLAP query): 50ms–minutes (this is the non-deterministic part)

For the SQL execution, you cannot guarantee 500ms. Solution: decouple SQL generation from SQL execution. Return the generated SQL to the user immediately (the "thinking is done" moment), then stream the result rows as they come in. The user sees progress immediately. The 2-second SLA applies to time-to-SQL, not time-to-results.

---

## Q21. You want to cache generated SQL for repeated questions. What is your cache key and invalidation strategy?

**Answer:**

**Cache key design:**

Naive approach: hash the natural language query string. This fails because "show me revenue" and "show revenue" are different strings but should hit the same cache entry.

Better approach: a composite key with three components:
1. **Semantic query embedding** (bucketed into discrete cluster IDs) — two queries with cosine similarity > 0.95 share a cache bucket
2. **Schema version hash** — a hash of the schema fingerprint (table names, column names, types) for the tables relevant to this query class. Changes when the schema changes.
3. **User authorization scope hash** — if row-level security means different users see different data, the cache must be scoped to the user's permissions. A query that returns different rows for different users cannot be cached globally.

For exact cache hits: `hash(normalized_query) + schema_version_hash + auth_scope_hash`
For semantic cache hits: `nearest_cluster_id(embed(query)) + schema_version_hash + auth_scope_hash`

**What goes wrong without proper invalidation:**

A schema migration renames `revenue_amount` to `net_revenue`. The cached SQL contains `SELECT revenue_amount FROM ...`. This executes and throws a column-not-found error — or worse, if a new `revenue_amount` column was added for a different purpose, it silently returns the wrong data.

**Invalidation strategy:**

*Event-driven invalidation (preferred):*
Subscribe to schema change events from the data platform (DDL audit logs, migration framework hooks, information_schema change detection). On any schema change, invalidate all cache entries whose schema version hash includes the changed table. This is surgical — it only invalidates cache entries that could be affected.

*TTL-based fallback:*
Set a maximum TTL of 24 hours as a backstop, even if no schema change event is received. This bounds the staleness window in case the event-driven system misses a change.

*Zero-TTL for high-risk tables:*
For tables that change frequently or whose correctness is business-critical (financial data, compliance tables), disable caching entirely. The latency cost is worth the correctness guarantee.

---

## Q22. How does streaming change the user experience and system architecture for NL2SQL?

**Answer:**

**User experience impact:**

Streaming SQL token-by-token gives the user immediate feedback that something is happening. Perceived latency drops dramatically — a 2-second response feels instant if tokens start appearing at 100ms. For NL2SQL specifically, streaming has an additional benefit: the user can read the SELECT clause and FROM clause as they appear and immediately spot if the LLM is querying the wrong table — catching errors before the query even finishes generating.

**System architecture changes:**

Streaming requires SSE (Server-Sent Events) or WebSocket instead of a simple HTTP request-response. The pipeline becomes:
1. Validate input → start LLM stream → validate SQL in flight (run parse check once you have enough tokens for a complete statement) → execute → stream result rows

**Cases where streaming is harmful:**

*Case 1 — SQL validation depends on the complete query:*
You cannot safely validate SQL until the full statement is generated. If you stream tokens to the user before validation completes, you may display a partial SQL that will be rejected. Solution: buffer the complete SQL, validate it, then stream it to the user in a second step. The user sees a "generating..." indicator, then the full validated SQL appears — this is still faster-feeling than waiting for results.

*Case 2 — Multi-step pipelines with retry logic:*
If SQL generation fails validation and the system retries with a modified prompt, streaming the first attempt to the user is confusing — they see SQL being generated and then it disappears. For systems with retry loops, do not stream until the final validated SQL is produced.

*Case 3 — Result streaming on non-streaming-capable databases:*
Some databases don't support streaming result sets — they buffer the full result before returning it. Streaming the SQL immediately is still valuable, but result streaming requires cursor-based pagination at the application layer.
