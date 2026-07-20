# Token Cost Analysis — NL2SQL Systems

Every component of an NL2SQL pipeline consumes tokens. At scale, token cost is often the largest operational expense — larger than infrastructure, larger than human review. This file breaks down the cost at each stage, shows how it compounds at scale, and gives concrete optimisation strategies with accuracy trade-off analysis.

## Table of Contents

- [Cost Components Overview](#cost-components-overview)
- [Stage-by-Stage Cost Breakdown](#stage-by-stage-cost-breakdown)
- [Cost at Scale — Example Calculations](#cost-at-scale--example-calculations)
- [The Three Biggest Cost Levers](#the-three-biggest-cost-levers)
- [Fine-Tuning vs Few-Shot vs RAG — Cost Comparison](#fine-tuning-vs-few-shot-vs-rag--cost-comparison)
- [Caching — The Highest-ROI Cost Reduction](#caching--the-highest-roi-cost-reduction)
- [Fast Mode — Explicit Cost vs Accuracy Trade-offs](#fast-mode--explicit-cost-vs-accuracy-trade-offs)
- [Cost Monitoring and Alerting](#cost-monitoring-and-alerting)
- [Rate Limiting — TPM/RPM as a Throughput Constraint](#rate-limiting--tpmrpm-as-a-throughput-constraint)
- [Cost Optimisation Checklist](#cost-optimisation-checklist)

---

## Cost Components Overview

An NL2SQL pipeline has eight distinct token-consuming stages:

```
User query
    │
    ▼
[1] Query embedding          → embedding tokens (input only)
    │
    ▼
[2] Schema retrieval         → no LLM tokens (ANN search)
    │
    ▼
[3] Reranking                → cross-encoder inference (not token-priced, GPU compute)
    │
    ▼
[4] SQL generation           → input tokens (query + schema + few-shot) + output tokens (SQL)
    │
    ▼
[5] SQL validation (optional LLM)  → input tokens (SQL + schema) + output tokens (verdict)
    │
    ▼
[6] Retry on failure (optional)    → repeat of stage 4 cost
    │
    ▼
[7] Confidence scoring (optional)  → input tokens (query + SQL + context)
    │
    ▼
[8] Result explanation (optional)  → input tokens (query + SQL + result) + output tokens (NL)
```

Stages 4 and 8 dominate. Stage 4 (SQL generation) is the core cost; stage 8 (result explanation) doubles or triples it.

---

## Stage-by-Stage Cost Breakdown

Reference pricing (June 2026 approximate — verify current rates):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $0.80 | $4.00 |
| text-embedding-3-small | $0.02 | — |
| text-embedding-3-large | $0.13 | — |

### Stage 1 — Query Embedding

```
Cost = query_tokens × embedding_price
     = 20 tokens × $0.02/1M = $0.0000004 per query
```

**Effectively free.** Embedding is never the cost bottleneck. A billion queries costs $400 in embedding alone.

### Stage 4 — SQL Generation (the dominant cost)

```
Input tokens = system_prompt + schema_context + few_shot_examples + user_query
             = 300 + 3,000 + 1,500 + 50
             = ~4,850 tokens (typical)

Output tokens = generated SQL
              = 50–300 tokens (simple query to complex multi-join)
              = ~150 tokens average

Cost per query (GPT-4o):
  Input:  4,850 × $2.50/1M  = $0.012
  Output:   150 × $10.00/1M = $0.0015
  Total:                      $0.014 per query

Cost per query (GPT-4o-mini):
  Input:  4,850 × $0.15/1M  = $0.00073
  Output:   150 × $0.60/1M  = $0.000090
  Total:                      $0.00082 per query
```

**The model choice is a 17× cost difference.** GPT-4o vs GPT-4o-mini: $0.014 vs $0.00082 per query.

### Stage 4 — Token breakdown of input prompt

| Component | Typical tokens | % of prompt |
|-----------|---------------|------------|
| System prompt (instructions) | 200–400 | 5–8% |
| Schema context (tables, columns) | 1,500–5,000 | 35–60% |
| Few-shot examples (3–5 examples) | 800–2,000 | 20–30% |
| User query | 20–100 | 1–3% |
| Chat history (multi-turn) | 0–2,000 | 0–25% |

Schema context is the largest single component — it directly scales with k (tables in context) and the verbosity of the schema representation.

### Stage 5 — LLM SQL Validation (optional)

Only used when you ask the LLM to validate its own output (not recommended — parse-based validation is better and free). If used:

```
Cost = ~500 input tokens + ~50 output tokens
     ≈ $0.001 per query (GPT-4o-mini)
     ≈ $0.015 per query (GPT-4o)
```

Avoid LLM validation — a parser (sqlglot) validates syntax for free in 1ms. LLM validation is expensive, slower, and less reliable.

### Stage 6 — Retry on Failure

Every retry is a full repeat of stage 4. If 10% of queries require one retry:

```
Effective cost = (1.0 + retry_rate) × stage_4_cost
               = 1.10 × $0.014 = $0.0154 per query (GPT-4o)
```

High retry rates (> 15%) signal a prompt or retrieval problem, not a cost problem. Fix the root cause.

### Stage 8 — Natural Language Result Explanation (optional)

When the system generates a plain-English explanation alongside the SQL result:

```
Input:  query (50) + SQL (150) + result_sample (300) + instructions (200) = 700 tokens
Output: explanation = 150–300 tokens average

Cost (GPT-4o): 700 × $2.50/1M + 250 × $10/1M = $0.0018 + $0.0025 = $0.0043
```

Adds ~30% to the per-query cost for a significant UX improvement. Usually worth it for executive-facing deployments.

---

## Cost at Scale — Example Calculations

### Scenario A: Small internal BI tool (500 queries/day)

```
Model: GPT-4o (quality priority)
Queries/day: 500
SQL generation cost: 500 × $0.014 = $7/day = $210/month
With result explanation: 500 × $0.0043 = $2.15/day additional
Total: ~$280/month
```

At this scale, model quality matters more than cost optimisation. Use GPT-4o.

### Scenario B: Mid-size analytics platform (10,000 queries/day)

```
Model: GPT-4o
SQL generation: 10,000 × $0.014 = $140/day = $4,200/month

Optimisations applied:
  30% cache hit rate: 10,000 × 0.70 × $0.014 = $98/day
  Reduce schema context 50% (3,000 → 1,500 tokens): saves $0.004/query
  Route simple queries to GPT-4o-mini (40% of traffic):
    Simple:  4,000 × $0.00082 = $3.28/day
    Complex: 6,000 × $0.014   = $84/day
    Total:   $87.28/day

Before optimisation: $4,200/month
After optimisation:  $2,620/month  (37% reduction)
```

### Scenario C: High-volume data platform (100,000 queries/day)

```
Model: GPT-4o (unoptimised) = $1,400/day = $42,000/month

Optimisations required:
  50% cache hit rate: 50,000 queries at full cost
  Model routing (60% mini, 40% GPT-4o):
    Mini:    60,000 × $0.00082 = $49.20/day
    GPT-4o:  40,000 × $0.014  = $560/day
    Total:   $609.20/day before cache

  With 50% cache hit:
    50,000 × $0.00082 (avg, post-routing) + 50,000 × $0 (cached) = $41/day
    Wait — routing saves money but the remaining 50K must be split by model:
    
  Full calculation:
    100K queries/day × 50% cache hit = 50K LLM calls
    Of 50K: 60% mini (30K × $0.00082 = $24.60) + 40% GPT-4o (20K × $0.014 = $280)
    Total LLM: $304.60/day
    
After optimisation: $304.60/day = $9,140/month
vs unoptimised: $42,000/month
Savings: $32,860/month (78% reduction)
```

### The schema context multiplier

Schema context is the single largest token consumer per query. Every 1,000 tokens of schema = $0.0025 per query at GPT-4o rates.

```
Schema context: 5,000 tokens → $0.0125/query input cost for schema alone
Schema context: 1,500 tokens → $0.00375/query

Difference: $0.009/query
At 100K queries/day: $900/day = $27,000/month from schema context alone

Reducing k from 15 tables to 8 tables (approx 40% schema token reduction):
  Saves: 0.40 × $27,000 = $10,800/month
```

This is why smart schema retrieval (retrieve only what's needed) pays for itself immediately.

---

## The Three Biggest Cost Levers

### Lever 1 — Model routing (biggest single lever)

Route queries by complexity to the appropriate model:

```python
def route_query(query: str, schema_complexity: int) -> str:
    complexity_score = compute_complexity(query, schema_complexity)
    
    if complexity_score < 0.3:
        return "gpt-4o-mini"    # single table, simple filter, common pattern
    elif complexity_score < 0.7:
        return "gpt-4o-mini"    # medium complexity — fine-tuned mini handles this
    else:
        return "gpt-4o"         # complex multi-join, window functions, subqueries

# Complexity features:
# - Number of entity mentions
# - Presence of join-indicating language ("by", "across", "linked")
# - Multi-step logic keywords ("who have X but not Y")
# - Historical accuracy of mini model on similar queries
```

| Distribution | Cost per 10K queries |
|-------------|---------------------|
| 100% GPT-4o | $140 |
| 60% mini, 40% GPT-4o | $55 |
| 80% mini, 20% GPT-4o | $33 |

The trade-off: GPT-4o-mini accuracy on complex queries is 10–15pp lower than GPT-4o. Only route simple queries to mini. Simple = single table, common template, historical success > 90%.

### Lever 2 — Schema context compression (second biggest lever)

The schema context is the largest input token component. Two compression strategies:

**A. Column-level relevance filtering:**
Instead of including all columns for every retrieved table, include only columns relevant to the query:

```python
def compress_schema(tables: list[Table], query: str, embedder) -> str:
    query_embedding = embedder.embed(query)
    result = []
    for table in tables:
        # Score each column against the query
        relevant_cols = [
            col for col in table.columns
            if cosine_similarity(query_embedding, col.embedding) > 0.3
        ]
        # Always include primary key and foreign keys regardless
        pk_fk_cols = [c for c in table.columns if c.is_pk or c.is_fk]
        final_cols = list(set(relevant_cols + pk_fk_cols))
        result.append(format_table_ddl(table, final_cols))
    return "\n\n".join(result)
```

Average token reduction: 40–60% of schema tokens, with minimal accuracy loss on simple-to-medium queries. Accuracy loss on complex queries with obscure columns: 3–8pp.

**B. Tiered schema detail:**
```
Top 2 tables: full DDL (all columns, types, comments, sample values)
Tables 3–8:   column names and types only (no comments, no sample values)
Tables 9–15:  table name and description only (1 line each)
```

Token savings vs full DDL for all 15 tables: 50–65%. Accuracy impact: minimal for queries involving the top tables, measurable for queries requiring specific knowledge of lower-ranked tables.

**Combined savings at 100K queries/day, GPT-4o:**
```
Full schema (5,000 tokens): $27,000/month in schema tokens alone
Compressed schema (2,000 tokens): $10,800/month
Savings: $16,200/month from schema compression
```

### Lever 3 — SQL caching

Every cached response saves 100% of generation tokens. The cache key must be precise enough to avoid serving wrong cached SQL.

```python
cache_key = hash(
    normalize_query(user_query),      # lowercase, strip whitespace, normalise synonyms
    schema_version_hash,               # invalidates on schema change
    user_auth_scope_hash,             # different users may have different data access
    model_version,                    # invalidates on model update
)
```

**Cache hit rate by application type:**

| Application type | Expected cache hit rate | Daily savings at 10K queries, GPT-4o |
|-----------------|------------------------|--------------------------------------|
| BI dashboards (few standard questions) | 50–70% | $70–98 |
| Internal analytics (power users) | 20–35% | $28–49 |
| Customer-facing chatbot (open-ended) | 5–15% | $7–21 |
| Ad-hoc exploration | < 5% | < $7 |

---

## Fine-Tuning vs Few-Shot vs RAG — Cost Comparison

### Few-shot prompting (static examples in prompt)

```
Per-query cost = SQL generation with examples in prompt
Input overhead from 5 examples: ~1,500 tokens × $2.50/1M = $0.00375/query overhead

At 10K queries/day with GPT-4o: $37.50/day in few-shot overhead alone
At 100K queries/day: $375/day = $11,250/month in few-shot overhead
```

Few-shot overhead is significant at scale. Compress to 3 examples or switch to dynamic retrieval.

### RAG over query library (dynamic few-shot selection)

```
Additional cost vs static few-shot:
  Query embedding (already done): $0
  Query library retrieval (ANN): $0 (compute, not token cost)
  Examples retrieved: same token cost as static, but fewer needed (3 vs 5)

Savings: replacing 5 static examples with 3 dynamically selected ones
  Saves: 2 examples × ~300 tokens × $2.50/1M = $0.0015/query
  At 100K queries/day: $150/day = $4,500/month saved vs static 5-shot
  Plus: higher accuracy from relevance-matched examples
```

RAG over query library pays for its infrastructure cost in reduced prompt length within months at scale.

### Fine-tuning

```
Upfront cost:
  Training dataset preparation: ~40 hours engineering @ internal rate
  Fine-tuning API cost (GPT-4o-mini): ~$25 per run, ~$200 for 8 iterations
  Total upfront: ~$200–500

Per-query cost after fine-tuning:
  No few-shot examples needed in prompt: saves 1,500 tokens
  Savings per query: 1,500 × $0.15/1M = $0.000225 (GPT-4o-mini)
  At 100K queries/day: $22.50/day = $675/month

Break-even: $200 upfront / $22.50/day savings = 9 days
```

Fine-tuning on GPT-4o-mini breaks even in under two weeks at 100K queries/day, and significantly reduces per-query cost while improving accuracy. At scale, fine-tuning is almost always cost-justified.

---

## Caching — The Highest-ROI Cost Reduction

### Three-tier cache with cost accounting

```
Tier 1: Exact query cache (< 1ms)
  Hit: saves 100% of generation cost
  Miss rate: depends on application (see table above)
  Storage cost: negligible (SQL strings are small)

Tier 2: Semantic query cache (< 50ms)
  Hit: saves 100% of generation cost
  Miss rate: 10–15% higher hit rate vs exact cache
  Storage: embed + store all cached queries ($0.0000004/query to embed)
  Risk: semantically similar queries may need different SQL — use similarity threshold > 0.95

Tier 3: Schema-aware partial cache
  Store (table_set, query_type) → SQL template
  Fill in parameter values at retrieval time
  Useful for parameterised queries ("top N customers by X")
  Risk: template instantiation can produce wrong SQL for edge cases
```

### Cache invalidation cost at scale

Every schema change invalidates N cached queries. With 10K cached queries and 5 schema changes/week:

```
Invalidation events: 5/week × [fraction of queries affected per change]
If each schema change affects 20% of cached queries:
  5 × 2,000 = 10,000 cache invalidations/week
  = 10,000 queries that must be regenerated: 10,000 × $0.014 = $140/week

This is negligible compared to the weekly cache savings.
```

### Cache ROI calculation

```
Application: 50K queries/day, 30% cache hit rate, GPT-4o

Without cache: 50,000 × $0.014 = $700/day
With 30% cache: 35,000 × $0.014 + 0 = $490/day
Savings: $210/day = $6,300/month

Cache infrastructure cost (Redis Enterprise): ~$500/month
Net savings: $5,800/month
ROI: first month is breakeven; every subsequent month is net $5,800 saved
```

---

## Fast Mode — Explicit Cost vs Accuracy Trade-offs

Every "fast mode" cut reduces token cost AND latency. The table shows real cost impact:

| Cut | Tokens saved | Cost reduction/query (GPT-4o) | Accuracy impact |
|-----|-------------|-------------------------------|----------------|
| Skip reranking | 0 (reranking is GPU compute, not tokens) | $0 token cost | Moderate — 3–8pp retrieval precision loss |
| Reduce k: 15 → 8 tables | ~2,000 input tokens | $0.005 | Moderate — misses required tables more often |
| Column filtering (40% reduction) | ~1,200 input tokens | $0.003 | Low for simple queries, moderate for complex |
| 3 few-shot examples → 1 | ~600 input tokens | $0.0015 | Low for common patterns, high for rare ones |
| Skip retries | 0 tokens saved (saves retry cost) | $0.014 × retry_rate | Significant — first-attempt-only accuracy |
| Route to GPT-4o-mini | Full model switch | $0.014 → $0.00082 (-94%) | 10–20pp on complex queries |
| GPT-4o-mini + all above cuts | All combined | ~$0.0005/query | Acceptable for simple queries only |

**The cheapest configuration that still works for simple queries:**

```
Model: GPT-4o-mini
Schema: 8 tables, column-level filtering (top relevant cols only)
Few-shot: 1 dynamically selected example
Retries: disabled (first attempt only)
Cache: enabled

Cost: ~$0.0005/query (vs $0.014 for full GPT-4o)
28× cheaper
Accuracy: 85–90% on simple single-table queries
          60–70% on complex multi-join queries (not acceptable for complex)

Use case: route 50–60% of production traffic here (the simple queries)
          Route the remaining 40–50% to full GPT-4o configuration
```

---

## Cost Monitoring and Alerting

### Metrics to track per day

```python
daily_cost_metrics = {
    "total_llm_cost_usd": sum(query.llm_cost for query in daily_queries),
    "cost_per_query_p50": percentile([q.llm_cost for q in daily_queries], 50),
    "cost_per_query_p99": percentile([q.llm_cost for q in daily_queries], 99),
    "cache_hit_rate": cached_queries / total_queries,
    "retry_rate": retried_queries / total_queries,
    "model_routing_distribution": {
        "gpt-4o-mini": mini_queries / total_queries,
        "gpt-4o":      gpt4o_queries / total_queries,
    },
    "schema_tokens_avg": mean(q.schema_input_tokens for q in daily_queries),
    "output_tokens_avg": mean(q.output_tokens for q in daily_queries),
}
```

### Alert thresholds

```
Alert: daily LLM cost > 150% of 7-day rolling average
  Cause: likely a cache invalidation storm, retry loop, or traffic spike
  
Alert: p99 cost per query > 5× p50 cost
  Cause: long-tail queries consuming disproportionate tokens
  (a query with 15K input tokens at GPT-4o costs $0.038 vs $0.014 average)

Alert: retry rate > 15%
  Cause: systematic SQL generation failure — schema or prompt issue
  
Alert: cache hit rate drops > 20pp week-over-week
  Cause: schema change invalidating cache, or query distribution shift
```

---

## Rate Limiting — TPM/RPM as a Throughput Constraint

Token **cost** ($ per token) and token **throughput** (tokens processed per minute) are separate constraints. A system can be well within its monthly cost budget and still get `429`'d by the LLM provider because it burst past the account's TPM (tokens-per-minute) or RPM (requests-per-minute) limit. Cost dashboards do not catch this — you need TPM/RPM tracked as their own metric.

### Provider limit tiers (illustrative — GPT-4o style tiering)

| Tier | RPM | TPM |
|------|-----|-----|
| Tier 1 | 500 | 30,000 |
| Tier 2 | 5,000 | 450,000 |
| Tier 3 | 5,000 | 800,000 |
| Tier 5 | 10,000 | 2,000,000 |

At ~5,000 tokens/query (4,850 input + 150 output, from Stage 4 above), a Tier 2 account (450K TPM) saturates at roughly 90 queries/minute. A workload sized at "10,000 queries/day" sounds nowhere near that — but a burst of 100+ queries landing in the same 60-second window (a dashboard refresh storm, a batch job, a traffic spike) will 429 well before the daily average would suggest any problem.

### Metrics to track (in addition to cost)

```python
daily_cost_metrics["tokens_per_minute_peak"] = max_over_windows(
    tokens_used, window="1m"
)
daily_cost_metrics["requests_per_minute_peak"] = max_over_windows(
    request_count, window="1m"
)
daily_cost_metrics["rate_limit_headroom_pct"] = (
    tokens_per_minute_peak / provider_tpm_limit
)
daily_cost_metrics["rate_limit_429_count"] = count(
    responses.status == 429
)
```

`rate_limit_headroom_pct` is the one that matters operationally — it tells you how close to the ceiling you are before a single traffic spike pushes you over.

### Alert thresholds

```
Alert: rate_limit_headroom_pct > 80% for 5 consecutive minutes
  Cause: traffic burst approaching the provider's TPM ceiling
  Action: engage backpressure (queue/delay) before 429s start cascading

Alert: rate_limit_429_count > 0 in the last hour
  Cause: provider-side throttling occurred
  Action: confirm retry-with-backoff absorbed it without a user-visible failure;
          if 429s are recurring, this is a capacity problem, not a bug
```

### Client-side token-bucket limiter (proactive, not reactive)

Waiting for a 429 and then backing off is reactive — by the time it arrives, the request has already failed. A token-bucket limiter sized to the provider's TPM enforces the ceiling client-side, so requests queue smoothly instead of bursting into throttling:

```python
class TokenBucketLimiter:
    def __init__(self, tpm_limit: int, refill_interval_s: float = 1.0) -> None:
        self.capacity = tpm_limit
        self.tokens = tpm_limit
        self.refill_per_tick = tpm_limit / (60 / refill_interval_s)

    async def acquire(self, estimated_tokens: int) -> None:
        while self.tokens < estimated_tokens:
            await asyncio.sleep(0.1)
            self.tokens = min(self.capacity, self.tokens + self.refill_per_tick)
        self.tokens -= estimated_tokens
```

Estimate `estimated_tokens` from the assembled prompt (schema context + few-shot + query) before the call, not after — the bucket has to gate admission, not just record usage.

### Retries compound TPM pressure

A retry triggered by a 429 re-spends tokens against the *same* one-minute window that just got throttled. Retrying immediately without backoff makes the throttle worse, not better. Exponential backoff with jitter is necessary but not sufficient on its own — the token-bucket limiter above prevents most 429s from happening in the first place, which is strictly better than handling them well after the fact.

### Scaling past a single key's TPM ceiling

At high volume, the account's TPM ceiling — not cost — becomes the hard limit on throughput. Options, in order of operational complexity:
1. **Multiple API keys behind a router**, load-balanced by remaining headroom per key (not round-robin — round-robin bursts a key that's already near its ceiling)
2. **Multi-provider fallback** — route overflow to a secondary model/vendor when the primary is saturated (accuracy/consistency trade-off, but keeps the system available)
3. **Negotiated enterprise throughput** — dedicated capacity agreements once traffic consistently exceeds standard tiers

This is a distinct scaling axis from model routing (Lever 1 above) — model routing reduces *cost per query*, TPM sharding increases *available throughput*. A system can need both simultaneously: cheap enough per query, but still capacity-constrained in aggregate.

### Self-hosted LLMs — no quota-based 429, but the same physical ceiling

Everything above assumes a hosted provider (OpenAI, Anthropic) enforcing TPM/RPM as an account-level billing quota. Self-hosted inference (vLLM, TGI, Triton, Ollama) has no such quota — there's no contract to enforce — but it does have a real physical ceiling: GPU memory for KV-cache, max batch size, max concurrent sequences.

The failure mode is different, not absent:

- **Hosted:** exceed the quota → immediate `429`, request rejected, latency unaffected.
- **Self-hosted:** exceed capacity → most serving frameworks queue the request via continuous batching rather than reject it. There's no error — instead, p99 latency quietly climbs as the queue grows. Some deployments do surface an explicit `429`/`503`, but only if something in front of the model (an API gateway, a configured max-concurrency limit) is enforcing one; the inference server itself usually won't on its own.

Practical implications:
- Size the token-bucket limiter against **measured tokens/sec throughput of the GPU fleet** (batch_size × tokens/sec/sequence, benchmarked empirically), not a vendor-published TPM number — there isn't one.
- Alert on **queue depth** and **time-in-queue**, not `rate_limit_429_count` — for a self-hosted deployment that metric may stay at zero even while the system is saturated and users are timing out.
- If you need a hard backpressure signal (fail fast instead of degrade silently), you have to add it yourself — e.g. reject at the gateway once queue depth exceeds a threshold, rather than relying on the inference server to do it.

### Cost attribution

At multi-tenant scale, attribute cost to tenant and query type:

```python
def log_query_cost(query: Query, result: SQLResult):
    db.insert("query_costs", {
        "tenant_id":      query.tenant_id,
        "query_type":     classify(query.text),   # aggregation, join, lookup, etc.
        "model":          result.model_used,
        "input_tokens":   result.input_tokens,
        "output_tokens":  result.output_tokens,
        "cost_usd":       compute_cost(result),
        "cached":         result.from_cache,
        "retry_count":    result.retry_count,
        "timestamp":      now(),
    })
```

This enables: per-tenant cost billing, identifying which query types are most expensive, and spotting individual tenants with abnormally high token consumption (potential misuse or query pattern that needs a prompt fix).

---

## Cost Optimisation Checklist

Run through this checklist before declaring the system ready for production at scale:

**Schema compression:**
- [ ] Column-level relevance filtering implemented (target: 40%+ schema token reduction)
- [ ] Tiered schema detail by retrieval rank (full DDL for top-3, condensed for rest)
- [ ] k tuned to minimum needed per query complexity tier (not a fixed k=15 for all)

**Model routing:**
- [ ] Query complexity classifier trained and deployed
- [ ] Simple queries (estimated: 50–60% of traffic) routed to mini model
- [ ] Accuracy baseline established per query type for both models
- [ ] Hard guardrail: never route queries requiring window functions or 4+ table joins to mini

**Caching:**
- [ ] Exact query cache with schema version hash in cache key
- [ ] Semantic cache for near-duplicate queries (cosine > 0.95 threshold)
- [ ] Cache hit rate measured per application type (target: 25%+ for analytics, 50%+ for BI)
- [ ] Cache invalidation wired to schema change events (not just TTL)
- [ ] Cache key includes auth scope hash (no cross-tenant cache pollution)

**Prompt engineering for cost:**
- [ ] Few-shot examples reduced to 3 (dynamically selected) from 5 (static)
- [ ] System prompt compressed (target: under 300 tokens)
- [ ] Chat history truncation: only last N turns, not full conversation

**Retry control:**
- [ ] Retry rate monitored (alert if > 15%)
- [ ] Maximum 2 retries per query (not unlimited)
- [ ] Retry with modified prompt, not identical prompt (avoid paying twice for same failure)

**Operational:**
- [ ] Per-query cost logged and attributed to tenant + query type
- [ ] Daily cost dashboard with 7-day trend and alerting on 150% spike
- [ ] Monthly cost review: which tenants / query types are driving cost?
