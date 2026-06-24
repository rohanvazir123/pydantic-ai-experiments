# Pipeline Architecture — Answers

## Q1. Walk me through the end-to-end architecture of a production NL2SQL system. Where are the failure modes, and how do you detect them at each stage without a human in the loop?

**Answer:**

A production NL2SQL pipeline has five distinct stages, each with its own failure class:

**Stage 1 — Query Understanding and Intent Classification**
The raw user query is classified before anything else. Is it analytical (requires SQL), conversational (requires a response without SQL), or ambiguous? A lightweight classifier — either a fine-tuned small model or a few-shot prompt — handles this. Failure mode: misclassifying an analytical question as conversational and returning a plain-text response. Detection: monitor the rate at which users immediately rephrase after a response — high rephrasing rate on a query pattern signals classification error.

**Stage 2 — Schema Retrieval**
The classified query is used to retrieve the relevant subset of the schema from a vector store. Failure mode: the correct table is not in the top-k retrieved results. Detection: embed a post-retrieval check — after SQL is generated, verify that the tables referenced in the generated SQL are a subset of the retrieved schema. Any mismatch is a retrieval failure. Log these for offline analysis.

**Stage 3 — SQL Generation**
The LLM receives the query, retrieved schema, and few-shot examples and generates SQL. Failure modes: syntactically invalid SQL, semantically wrong SQL (correct syntax, wrong logic), hallucinated column or table names. Detection for syntax: run a parse step (e.g. `sqlglot.parse`) before execution — reject and retry or return an error. Detection for hallucination: check every table and column name in the generated SQL against the retrieved schema — any name not present is a hallucination.

**Stage 4 — SQL Validation and Guardrails**
Before execution: check for missing WHERE clauses on large tables, detect DML statements (INSERT, UPDATE, DELETE, DROP), check query cost via EXPLAIN. Failure mode: a valid but destructive or expensive query slips through. Detection: static analysis rules + cost-based thresholds derived from the query plan.

**Stage 5 — Execution and Result Presentation**
The SQL runs, results come back. Failure modes: empty result set (may signal wrong filtering), unexpectedly large result set (may signal missing filter), runtime error (table not found, type mismatch). Detection: instrument every execution — log row counts, execution time, and error codes. Empty results on a query class that historically returns data is an anomaly worth alerting on.

**Cross-cutting detection without humans:** Build a confidence scoring layer that aggregates signals from all five stages (retrieval rank of top table, parse success, hallucination check result, execution success, row count anomaly) into a single score. Queries below a threshold are flagged for async human review rather than rejected — this gives you a feedback queue without blocking the user.

---

## Q2. How do you decide where to draw the boundary between what the LLM handles and what deterministic code handles?

**Answer:**

The boundary should be drawn at the point where the problem stops being a pattern-matching / language understanding problem and starts being a correctness-verifiable problem.

**What the LLM should handle:**
- Mapping natural language intent to a structured query (the core NL→SQL translation)
- Disambiguating entity references ("top customers" → relevant columns)
- Selecting the appropriate join path from multiple valid options
- Handling paraphrase and synonym variation

**What deterministic code should handle:**
- Schema retrieval and ranking (vector similarity is deterministic given fixed embeddings)
- SQL parsing and validation (a parser is 100% reliable; an LLM is not)
- Security rules — never let the LLM decide whether a query is safe to execute
- Cost guardrails — EXPLAIN plan analysis, row count checks
- Cache lookup — exact-match and semantic cache are deterministic
- Result formatting and pagination

**The consequence of getting this wrong in each direction:**

Too much to the LLM: you introduce non-determinism into stages that need reliability. Asking the LLM to validate its own SQL is a known failure — it will confidently say invalid SQL is fine. Asking the LLM to enforce security rules is a prompt injection attack waiting to happen.

Too little to the LLM: you lose the core value. Trying to handle synonym resolution or join path selection with hand-written rules doesn't scale — that's exactly the problem LLMs solve well.

The practical test: if the output of a stage can be verified by running code (parse it, look it up, execute an EXPLAIN), make it deterministic. If the output requires language understanding to verify, the LLM is already doing the work.

---

## Q3. Schema linking is often cited as the hardest sub-problem in NL2SQL. Describe your approach.

**Answer:**

Schema linking is the problem of mapping spans in the natural language query to specific tables, columns, and values in the database schema. It is hard because the mapping is rarely lexical — "top performing reps last quarter" requires mapping "top performing" to a performance metric column, "reps" to a sales representatives table, and "last quarter" to a time-bounded filter on a date column.

**A production approach has three layers:**

**Layer 1 — Lexical matching:** Exact and fuzzy string match between query tokens and table/column names. Fast, handles cases where the user uses the exact column name. Covers maybe 30% of real queries.

**Layer 2 — Semantic matching:** Embed the query and embed table/column descriptions (including aliases and business glossary terms). Use cosine similarity to score relevance. "Reps" matches a table described as "sales representatives" even with zero lexical overlap. This is the core retrieval step.

**Layer 3 — LLM-based linking:** Pass the top-k retrieved schema elements plus the query to the LLM and ask it to explicitly identify which tables and columns are needed, with reasoning. This handles compositional cases — "top performing reps last quarter" requires the LLM to understand that "last quarter" is a time filter that needs a date column AND that "top performing" requires an aggregation on a metric column that may be in a different table requiring a join.

**The specific hard cases:**

*Join path discovery:* "top performing reps" might require joining `sales_rep → opportunity → revenue_fact`. None of these tables are mentioned by name. The system must know the join graph and infer the required path from the semantic intent. Solution: pre-compute all join paths in the schema graph, embed the semantic meaning of each path, and retrieve paths alongside tables.

*Time expressions:* "last quarter", "YTD", "same period last year" require knowing the current date and translating to exact date bounds. Handle this in deterministic post-processing, not in the LLM — inject the current date into the prompt and use a date expression library to expand these terms.

*Business KPIs:* "top performing" means nothing without a definition. This must come from a business glossary layer, not the LLM's general knowledge.

---

## Q4. How does your pipeline behave when the user's question is genuinely ambiguous?

**Answer:**

First, distinguish between two types of ambiguity:

**Type 1 — Resolvable with context:** "Show me their orders" — "their" can be resolved from conversation history. This is not true ambiguity; it is a co-reference problem. Resolve it before reaching the ambiguity handler.

**Type 2 — Genuinely ambiguous:** "Show me the top customers" — top by revenue, by order count, by margin. No amount of context resolves this; it requires a business decision.

**The three options and their trade-offs:**

**Option A — Pick one and annotate it:** Generate SQL using the most common interpretation (by revenue, as the default) and surface the assumption explicitly in the UI: "Showing top customers by total revenue. [Change metric ▾]". This is the highest-throughput option and works well for casual analytics. Risk: the user doesn't notice the annotation and makes a business decision on the wrong metric.

**Option B — Ask for clarification:** Return a structured clarifying question before generating SQL. Works well in a chat-style interface where latency tolerance is higher. Risk: users find it frustrating if it happens frequently. Use this only when the ambiguity is in a high-stakes dimension (date range, metric definition) rather than a stylistic one.

**Option C — Return multiple interpretations:** Generate two or three SQL variants with labels ("By revenue", "By order count") and let the user pick. Works in a dashboard-style interface. Risk: confusing for non-technical users; adds UI complexity.

**How to decide programmatically:** Train a classifier on query + schema to predict ambiguity type. If the ambiguity is in a low-stakes stylistic dimension, use Option A. If it touches a metric definition or date range, use Option B. Reserve Option C for power-user interfaces.

**The correct answer for a financial reporting system** is almost always Option B — wrong assumptions in finance have real consequences. For a BI tool used by data analysts who expect to refine queries, Option A is usually right.
