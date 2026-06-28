# NL2SQL System Design — Top 20 Curated Topics

The 20 highest-signal topics selected from the full set of 55. Each one either reveals a fundamental design instinct, exposes a specific blind spot, or forces a real architectural trade-off decision. Covers the full breadth of the system — retrieval, generation, evaluation, latency, security, and drift.

Full detailed answers for each are in the individual section files linked below.

## Table of Contents

- [Selection Criteria](#selection-criteria)
- [The Top 20](#the-top-20)
- [Quick Study Map](#quick-study-map)

---

## Selection Criteria

A topic made this list if it satisfies at least two of:
- Tests a design decision with no obviously correct answer
- Exposes a failure mode that is easy to miss until it hits production
- Requires knowledge that spans multiple layers of the system
- Has a real trade-off where the right answer depends on context

---

## The Top 20

---

### 1. End-to-end pipeline architecture and failure detection

> Walk me through the end-to-end architecture of a production NL2SQL system. Where are the failure modes, and how do you detect them at each stage without a human in the loop?

**Why it's top 20:** Sets the baseline. A strong answer treats each stage as an independent failure domain with its own detection mechanism — not a single end-to-end black box. Weak answers describe the happy path only.

**What a strong answer covers:**
- Five distinct stages: intent classification → schema retrieval → SQL generation → validation → execution
- Specific, observable failure signal at each stage (parse failure, hallucination check, row count anomaly)
- A cross-cutting confidence score that aggregates signals into a review queue
- Explicit distinction between "user sees an error" failures and "user sees wrong data" failures — the second is more dangerous

**Full answer:** [01_pipeline_architecture.md](01_pipeline_architecture.md#q1)

---

### 2. The LLM vs deterministic code boundary

> How do you decide where to draw the boundary between what the LLM handles and what deterministic code handles? What are the consequences of getting that boundary wrong in each direction?

**Why it's top 20:** Reveals system design instinct. Candidates who put security validation or SQL parsing into the LLM don't understand the reliability requirements of production systems.

**What a strong answer covers:**
- The test: if the output can be verified by running code, make it deterministic
- Security rules, SQL parsing, cost guardrails — always deterministic
- Language understanding, schema linking, join path selection — LLM
- Specific failure for each wrong direction: non-determinism in safety-critical stages vs. hand-written rules that don't scale

**Full answer:** [01_pipeline_architecture.md](01_pipeline_architecture.md#q2)

---

### 3. Schema linking — the hardest sub-problem

> How do you map "top performing reps last quarter" to the correct tables, columns, and time logic in a schema you've never seen?

**Why it's top 20:** Schema linking is where most NL2SQL systems actually fail in production. A shallow answer says "the LLM handles it." A strong answer has a specific multi-layer architecture.

**What a strong answer covers:**
- Three layers: lexical matching → semantic embedding matching → LLM-based explicit linking
- Join path discovery for tables not mentioned by name
- Deterministic post-processing for time expressions (never trust the LLM to compute date bounds)
- Business glossary for KPI terms that have no schema representation

**Full answer:** [01_pipeline_architecture.md](01_pipeline_architecture.md#q3)

---

### 4. Large schema retrieval — 400 tables, 6,000 columns, 15 fit in context

> You can't fit it all in context. How do you decide what to include, and what do you do when the relevant tables aren't retrieved?

**Why it's top 20:** This is the central engineering problem in NL2SQL at scale. Every production deployment hits it. Candidates who say "just use a larger context window" don't understand cost, latency, or the lost-in-the-middle attention problem.

**What a strong answer covers:**
- Multi-stage funnel: coarse ANN retrieval (top-40) → cross-encoder reranking (top-15) → join graph expansion → budget-aware truncation
- Column-level relevance filtering to fit more tables in fewer tokens
- Post-retrieval check: any table in generated SQL not in retrieved set = hallucination flag
- Curated overrides for systematically hard-to-retrieve tables

**Full answer:** [02_schema_representation.md](02_schema_representation.md#q5)

---

### 5. A trusted evaluation metric — beyond execution accuracy and exact match

> Execution accuracy and exact match accuracy both have serious flaws. Describe a metric or evaluation framework you'd actually trust for a production system.

**Why it's top 20:** Reveals whether the candidate has actually run an NL2SQL system in production or is only familiar with benchmark papers. Execution accuracy on a small test database is a notoriously misleading metric.

**What a strong answer covers:**
- Execution accuracy flaw: plausible-wrong SQL returns data, empty results are undetectable
- Exact match flaw: enormous surface variation in equivalent SQL
- Layered metric: execution success + result set equivalence on adversarial synthetic data + human spot-check on stratified sample + implicit production signals (re-query rate, export rate)
- Honest acknowledgement of what no automated metric catches (semantic errors on realistic data)

**Full answer:** [03_accuracy_evaluation.md](03_accuracy_evaluation.md#q10)

---

### 6. The hardest failure class — semantically wrong SQL that executes cleanly

> A query runs and returns results, but answers a different question than the user asked. How do you detect this?

**Why it's top 20:** Most candidates focus on preventing syntax errors and execution failures. Semantic errors are invisible to every standard check and can have serious downstream consequences in analytical or financial contexts.

**What a strong answer covers:**
- Why this is undetectable by syntax, execution, and schema checks
- Schema-level structural sanity checks (aggregation type matches question intent, GROUP BY matches "by" clause)
- Result distribution anomaly detection for recurring query patterns
- Self-consistency via two independent SQL generations
- The honest answer: reliable detection requires human review or ground truth

**Full answer:** [03_accuracy_evaluation.md](03_accuracy_evaluation.md#q11)

---

### 7. 85% benchmark accuracy — good enough to ship?

> Your PM says 85% is good enough. What do you tell them?

**Why it's top 20:** Tests product and business judgment alongside technical depth. A candidate who says "yes, 85% is fine" or "no, we need 99%" without nuance is missing the point.

**What a strong answer covers:**
- 15% failure rate at 1,000 queries/day = 150 wrong results, potentially acted on
- Why benchmark accuracy is typically 5–15pp above production accuracy
- Why 85% might actually be optimistic (benchmark queries are cleaner than real queries)
- Why 85% might be fine if failures are concentrated in detectable/obvious error types
- The actual path to ship: shadow pilot, confidence threshold, restricted rollout, analyst-assist framing not analyst-replace

**Full answer:** [03_accuracy_evaluation.md](03_accuracy_evaluation.md#q12)

---

### 8. Full defence against prompt injection

> A malicious user crafts a question designed to exfiltrate data or drop tables. Walk me through every layer.

**Why it's top 20:** Security in NL2SQL is often underestimated. The naive solution ("the LLM won't do bad things") fails immediately under adversarial input. This tests whether the candidate treats LLM output as untrusted.

**What a strong answer covers:**
- Six independent layers that each must hold: input sanitization, structured prompting with delimiters, AST-based output validation (allowlist of SELECT only), read-only DB service account, database-level RLS as last resort, audit logging with anomaly detection
- Why AST-based validation is essential and regex is insufficient (regex can be bypassed)
- Why the database permission layer is a fallback, not a primary control
- The inference attack gap: RLS doesn't prevent deriving sensitive information from allowed data

**Full answer:** [05_execution_safety_security.md](05_execution_safety_security.md#q16)

---

### 9. Missing WHERE clause on a 2-billion-row table — $400 query

> How do you prevent this, and how do you balance guardrails against legitimate analytical questions?

**Why it's top 20:** A practical, consequential failure mode that every data warehouse NL2SQL system must address. Tests whether the candidate thinks about cost and infrastructure impact, not just correctness.

**What a strong answer covers:**
- Table size metadata registry: flag large tables without WHERE clause before execution
- EXPLAIN-based cost check before execution
- Automatic LIMIT injection for interactive queries
- Distinction between interactive queries (guardrailed) and scheduled/export queries (not guardrailed)
- Per-query cost cap with user-facing explanation

**Full answer:** [05_execution_safety_security.md](05_execution_safety_security.md#q19)

---

### 10. Sub-2-second end-to-end latency when LLM call is 1.5 seconds

> Walk me through every latency lever — what can you parallelize, what can you cache, and what are the correctness risks?

**Why it's top 20:** Latency is a first-class product requirement that shapes the entire architecture. A candidate who says "use a faster model" without a systematic framework is not thinking at the right level.

**What a strong answer covers:**
- Parallelize schema retrieval with LLM call pre-warming
- Semantic cache for 30–40% of traffic at < 100ms
- Tiered model routing: simple queries → small model (800ms), complex → large model
- Prompt length reduction: column-level relevance filtering, compressed few-shot, tighter system prompt
- Prefix caching / KV cache reuse for fixed system prompt
- Decouple SQL generation SLA from execution SLA — return SQL immediately, stream results

**Full answer:** [06_latency_performance.md](06_latency_performance.md#q20)

---

### 11. SQL cache key design and invalidation

> What is your cache key, and what goes wrong if schema migration happens and the cache isn't invalidated?

**Why it's top 20:** Caching is the highest-ROI latency optimization, but a poorly designed cache key or invalidation strategy causes silent wrong results — the worst kind of production failure.

**What a strong answer covers:**
- Composite cache key: semantic query embedding cluster ID + schema version hash + auth scope hash
- Why exact string matching fails (paraphrase variation) and why embedding similarity matching is necessary
- Event-driven invalidation from DDL audit logs as the primary mechanism
- Short TTL (15–30 min) as a backstop
- Zero-TTL for high-stakes tables (financial, compliance)
- Per-table cache invalidation, not full cache flush

**Full answer:** [06_latency_performance.md](06_latency_performance.md#q21)

---

### 12. Fine-tuning vs few-shot prompting vs RAG — when does each win?

> Lay out the trade-offs across accuracy, maintainability, schema adaptability, and operational cost.

**Why it's top 20:** The most common architectural decision in NL2SQL. Candidates who say "just fine-tune" or "just use RAG" without understanding the trade-offs are not ready for a production deployment decision.

**What a strong answer covers:**
- Few-shot: highest schema adaptability, per-query token cost, good for early-stage and new schemas
- RAG over query library: better than static few-shot for diverse patterns, requires library maintenance
- Fine-tuning: highest accuracy ceiling, poorest schema adaptability, justified only with 5,000+ examples and stable schema
- Production recommendation: RAG as primary, fine-tune for base SQL patterns when volume justifies it, use RAG for schema-specific adaptation on top of the fine-tuned model

**Full answer:** [07_model_prompt_strategy.md](07_model_prompt_strategy.md#q23)

---

### 13. Building a feedback loop without explicit user feedback

> Most users won't tell you the result was wrong — they'll rephrase or give up. What implicit signals can you collect?

**Why it's top 20:** Production NL2SQL improvement is fundamentally a signal collection problem. Candidates who rely on explicit ratings have not shipped a real system.

**What a strong answer covers:**
- Immediate re-query with semantic similarity as the strongest negative signal
- SQL edit as the highest-quality correction signal
- Result export/download as a positive signal
- Session abandonment as a noisy signal (use in aggregate, not per-query)
- Why you never train directly on implicit signals — use them to triage to a human review queue
- The feedback loop: implicit signal → triage → human label → training data

**Full answer:** [08_feedback_loops.md](08_feedback_loops.md#q27)

---

### 14. p99 latency SLA of 3 seconds — you're at 2.8s with zero margin

> Walk me through every architectural decision to hold the SLA under load, and what you drop when you can't.

**Why it's top 20:** Tests whether the candidate can reason about a system under real constraints — not just design for the happy path.

**What a strong answer covers:**
- Explicit sub-component budget: retrieval < 150ms, LLM < 2,200ms, validation < 50ms, formatting < 80ms
- Parallelism: LLM pre-warm while retrieval runs
- Tiered routing: 40–60% of traffic to a faster small model
- Aggressive caching targeting 30–40% hit rate
- Ordered cuts: reranking first, then schema context reduction, then smaller model, then skip retries
- Hard rule: never drop security validation

**Full answer:** [10_latency_slas.md](10_latency_slas.md#q33)

---

### 15. Separating time-to-SQL SLA from time-to-results SLA

> How do you set a latency SLA for NL2SQL when query execution time is non-deterministic?

**Why it's top 20:** This is a fundamental framing mistake most teams make. Conflating SQL generation time with execution time creates an unownable SLA.

**What a strong answer covers:**
- Two separate SLAs: time-to-SQL (owned by NL2SQL team, controllable) and time-to-results (owned jointly with data infrastructure)
- Why conflating them is wrong: warehouse latency can vary from 10ms to 20 minutes depending on query complexity and cluster load
- User experience when execution exceeds SLA: return SQL immediately, show progress with ETA from EXPLAIN plan, allow query cancellation
- The framing shift: "system is slow" → "system is working on something complex"

**Full answer:** [10_latency_slas.md](10_latency_slas.md#q34)

---

### 16. "Show me the sales numbers" — vague in 5 dimensions

> How does your system decide whether to ask a clarifying question, make a default assumption, or return multiple interpretations?

**Why it's top 20:** Vague queries are the majority of real user input. A system that either blindly picks one interpretation or asks the user 5 questions before answering is not usable.

**What a strong answer covers:**
- Ambiguity scoring per dimension (metric, time period, region, product, granularity)
- Routing logic: below threshold → default with annotation, above threshold → single highest-stakes clarifying question
- Defaults are context-aware (user history, persona, time of day)
- Annotate every assumption in plain language with a quick-change affordance
- Different thresholds by persona: analysts get defaults, executives get clarification for metric definitions

**Full answer:** [11_vague_queries.md](11_vague_queries.md#q37)

---

### 17. "Who are our best customers?" — the KPI definition problem

> How does your system know which definition to use, and what happens when that definition changes?

**Why it's top 20:** Business KPIs are the most common source of semantically wrong results in NL2SQL. The LLM's general-knowledge definition ("highest revenue") is almost never the company's actual definition.

**What a strong answer covers:**
- Business glossary with KPI-to-table mapping as the authoritative source
- Pre-computed KPI columns in the warehouse as the preferred resolution path
- Query history as a revealed-preference signal for implicit defaults
- Clarification as the fallback with schema-derived options
- Versioning the KPI definition and surfacing it in every result — this is how changes become visible before they cause wrong decisions

**Full answer:** [11_vague_queries.md](11_vague_queries.md#q40)

---

### 18. A table renamed, column deprecated, new table added — all in one migration

> Walk me through every place this breaks and your strategy for detecting and recovering without manual re-deployment.

**Why it's top 20:** Schema drift is the leading cause of silent production failures in deployed NL2SQL systems. This tests whether the candidate has designed for change, not just for a static schema.

**What a strong answer covers:**
- Five break points: vector index, SQL cache, few-shot library, business glossary, in-flight conversations
- Event-driven invalidation as the primary mechanism (DDL audit log hooks)
- Cascading invalidation: re-embed the table, invalidate affected cache entries, flag affected examples
- The 5-minute recovery window: acceptable staleness between schema change and full invalidation
- Multi-turn conversation handling: alert user when schema version in conversation context differs from current

**Full answer:** [12_schema_drift.md](12_schema_drift.md#q42)

---

### 19. Wrong 15 tables retrieved — plausible SQL, wrong data, no error

> You have 500 tables, retrieval picks the wrong 15, the model generates plausible-looking SQL, it executes, and returns wrong data. How do you detect this and fix the retrieval step?

**Why it's top 20:** The combination of retrieval miss + plausible execution + silent wrong result is the most dangerous failure mode in production. It cannot be detected by any execution-level check.

**What a strong answer covers:**
- Post-retrieval table membership check: any table in generated SQL not in retrieved set = flag immediately
- Result distribution anomaly detection for recurring query patterns
- Self-consistency: generate SQL twice, divergent table references signal ambiguous retrieval
- Root cause fix: enrich the embedding document with historical query patterns, business synonyms, join path descriptions
- Curated overrides for systematically hard-to-retrieve tables that bypass semantic ranking

**Full answer:** [13_schema_subset_llm.md](13_schema_subset_llm.md#q48)

---

### 20. Multi-tenancy — Tenant A and Tenant B both have a `payments` table

> Your retrieval and prompting must be tenant-scoped with zero cross-contamination. Walk me through the architecture.

**Why it's top 20:** Multi-tenancy is not an afterthought — it must be designed into every layer from the start. A candidate who says "just filter by tenant_id" without addressing retrieval, caching, few-shot examples, and the audit trail is missing the depth of the problem.

**What a strong answer covers:**
- Tenant-namespaced vector store partitions: Tenant A queries never search Tenant B's namespace
- Tenant isolation enforced at the retrieval service layer, not just the caller — omitting tenant_id returns zero results, not cross-tenant results
- Tenant-scoped few-shot example library with the same isolation guarantee
- Cache key includes tenant_id as a mandatory component
- Service account permissions scoped to the tenant's tables at the database level
- Audit logging of retrieval events with tenant_id — any cross-namespace access is a security incident

**Full answer:** [13_schema_subset_llm.md](13_schema_subset_llm.md#q55)

---

## Quick Study Map

| # | Topic | Section file | Core concept |
|---|-------|-------------|--------------|
| 1 | End-to-end architecture + failure detection | [01](01_pipeline_architecture.md) | Per-stage failure domain |
| 2 | LLM vs deterministic boundary | [01](01_pipeline_architecture.md) | "Can code verify this?" |
| 3 | Schema linking | [01](01_pipeline_architecture.md) | 3-layer lexical→semantic→LLM |
| 4 | 400 tables, 15 fit in context | [02](02_schema_representation.md) | Multi-stage retrieval funnel |
| 5 | Trusted evaluation metric | [03](03_accuracy_evaluation.md) | Layered metric, no single number |
| 6 | Semantically wrong SQL that executes | [03](03_accuracy_evaluation.md) | Invisible failure class |
| 7 | 85% accuracy, good enough to ship? | [03](03_accuracy_evaluation.md) | Shadow pilot, confidence threshold |
| 8 | Prompt injection full defence | [05](05_execution_safety_security.md) | 6 independent layers |
| 9 | Missing WHERE on 2B-row table | [05](05_execution_safety_security.md) | EXPLAIN + cost guardrails |
| 10 | Sub-2s latency, LLM takes 1.5s | [06](06_latency_performance.md) | Parallelize + cache + tier |
| 11 | SQL cache key + invalidation | [06](06_latency_performance.md) | Event-driven + schema version hash |
| 12 | Fine-tuning vs few-shot vs RAG | [07](07_model_prompt_strategy.md) | Trade-off matrix |
| 13 | Feedback loop without explicit signals | [08](08_feedback_loops.md) | Implicit signals → review queue |
| 14 | p99 3s SLA, at 2.8s | [10](10_latency_slas.md) | Sub-component budgets |
| 15 | Time-to-SQL vs time-to-results SLA | [10](10_latency_slas.md) | Two separate SLAs |
| 16 | "Show me the sales numbers" | [11](11_vague_queries.md) | Harm-weighted ambiguity score |
| 17 | "Who are our best customers?" | [11](11_vague_queries.md) | Business glossary + KPI versioning |
| 18 | Table renamed, column deprecated | [12](12_schema_drift.md) | Event-driven cascading invalidation |
| 19 | Wrong 15 tables, plausible SQL, wrong data | [13](13_schema_subset_llm.md) | Table membership check |
| 20 | Multi-tenancy, same table name | [13](13_schema_subset_llm.md) | Isolation at every layer |
