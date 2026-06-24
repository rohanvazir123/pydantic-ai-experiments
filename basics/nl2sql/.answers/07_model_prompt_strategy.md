# Model and Prompt Strategy — Answers

## Q23. Fine-tuning vs few-shot prompting vs RAG over a query library — lay out the trade-offs.

**Answer:**

These are not mutually exclusive — a production system typically combines all three. The question is which is primary and which is supplementary.

**Few-shot prompting:**
You include 3–10 example (question, SQL) pairs in the prompt at inference time.

*Accuracy:* High for query patterns that match the examples; degrades sharply for patterns not represented.
*Maintainability:* Easy — add or modify examples without redeployment. Examples can be curated by domain experts without ML expertise.
*Schema adaptability:* Excellent — the examples can be schema-specific. Swap examples when you deploy to a new schema.
*Operational cost:* High per-query cost — examples consume tokens on every request.
*When it wins:* Early-stage deployment where you have < 500 example queries and need to iterate quickly. New schema domains where you haven't built a training set yet.

**RAG over a query library:**
A library of (question, SQL) pairs is embedded and stored. At inference time, retrieve the most semantically similar examples to the current query and include them in the prompt.

*Accuracy:* Better than static few-shot for diverse query patterns — the retrieved examples are more relevant to the specific query. Degrades when the query is unlike anything in the library.
*Maintainability:* Medium — requires maintaining the query library and keeping its embeddings current. Adding new examples requires re-embedding.
*Schema adaptability:* Good — the library is schema-specific, so adapting to a new schema means building a new library.
*Operational cost:* Same token cost as few-shot, plus retrieval overhead (50–100ms). But higher accuracy means fewer retries, so effective cost is lower.
*When it wins:* Systems with a well-curated query library of 1,000+ examples. The retrieval step provides a meaningful quality boost over static examples.

**Fine-tuning:**
The model weights are updated on a dataset of (question, schema, SQL) triples.

*Accuracy:* Highest ceiling for in-distribution queries — the model has internalized patterns that few-shot examples can only approximate.
*Maintainability:* Highest cost — requires a curated dataset, training runs, evaluation, and redeployment when the schema or query patterns change significantly. A schema change can invalidate a fine-tuned model.
*Schema adaptability:* Poor without re-training. A model fine-tuned on schema A will hallucinate column names from schema A when deployed on schema B.
*Operational cost:* Lowest per-query inference cost — no examples in the prompt, shorter prompts, faster inference.
*When it wins:* Mature deployment with a stable schema, high query volume (the inference savings justify the training cost), and a large curated dataset (5,000+ high-quality examples).

**Production recommendation:** Start with RAG over a growing query library. Fine-tune when you have > 5,000 examples and the schema has stabilized. Use fine-tuning for the base model and RAG for schema-specific adaptation — the fine-tuned model handles general SQL patterns, the retrieved examples handle schema-specific conventions.

---

## Q24. How do you construct few-shot examples for a schema the model has never seen?

**Answer:**

**Selection criteria:**
The goal is maximum pattern coverage with minimum redundancy.

*Step 1 — Identify query pattern taxonomy:*
Enumerate the query types the system must handle: simple SELECT with filter, aggregation with GROUP BY, multi-table JOIN, subquery, window function, date arithmetic, HAVING clause, UNION, nested aggregation. Each pattern type should have at least one example.

*Step 2 — Write examples manually for this specific schema:*
The examples must use the actual table and column names from the schema. Generic examples about a hypothetical schema do not transfer — they actively harm performance because the LLM may import column names from the example into queries about the real schema. Write 1–2 examples per pattern type using real tables and columns.

*Step 3 — Select examples dynamically based on the query:*
Include examples that are semantically similar to the user's query (using embedding similarity) rather than a fixed set. A query about date arithmetic should retrieve the date arithmetic example; a query about aggregation should retrieve the aggregation example.

**How many examples:**
3–5 is typically optimal. More than 7–8 adds token cost without meaningful accuracy improvement — the LLM's attention on early context weakens for very long prompts, and the examples you most need (the last few) may be the most diluted.

**The risk of closely matching examples:**
If an example is a near-duplicate of the user's query but uses a slightly different table or filter condition, the LLM may copy the example structure wholesale — including the wrong table name. This is a form of overfitting to the example. Mitigation: when retrieved examples are very similar (cosine similarity > 0.95) to the user's query, include only the most similar one and fill remaining slots with diversity-maximizing examples.

---

## Q25. Your prompt averages 8,000 tokens. What do you cut, and how do you measure the impact?

**Answer:**

Token cost and latency scale linearly with input length (approximately). Halving prompt length roughly halves TTFT and reduces cost by ~50%.

**What to cut, in order of safety:**

*Cut 1 — Schema detail for low-relevance tables (save 1,000–2,000 tokens):*
The top-1 or top-2 retrieved tables get full DDL. Tables ranked 3–10 get column names and types only, no comments or sample values. Tables ranked 11–15 get table name and description only. Accuracy impact: minimal for queries that primarily use the top tables; measurable for queries that require precise knowledge of lower-ranked tables.

*Cut 2 — Few-shot example compression (save 500–1,000 tokens):*
Shorten example SQL by removing comments and formatting whitespace. Replace 5 examples with 3 dynamically selected ones. Accuracy impact: varies by query type — complex query types suffer more from fewer examples.

*Cut 3 — System prompt tightening (save 200–500 tokens):*
Eliminate verbose instructions that can be compressed. "Generate a syntactically valid SQL SELECT statement using only the tables and columns provided in the schema below, without using any columns or tables not explicitly listed" can be replaced with tighter phrasing.

*Cut 4 — Remove sample values from schema (save 500–1,500 tokens):*
Sample values (e.g., `status IN ('active', 'churned', 'trial')`) are helpful for value-filtering queries but expensive in tokens. Move sample values to the few-shot examples for the cases that need them, rather than including them for all columns.

**Measuring the impact of each cut:**
Run an A/B test on your evaluation set: baseline prompt vs. reduced prompt. Measure execution accuracy per query type. The cuts that hurt most are the ones to restore first. Build a prompt size vs. accuracy Pareto curve — find the point where cutting further causes accuracy to drop non-linearly. That is your operating point.

---

## Q26. How do you handle multiple SQL dialects?

**Answer:**

Postgres, BigQuery, Spark SQL, and T-SQL differ in: date functions (`DATE_TRUNC` vs `DATEADD` vs `TRUNC`), string functions, LIMIT vs TOP syntax, array types, window function syntax, and identifier quoting conventions.

**Option A — Single model, dialect in the prompt:**
Include a line in the system prompt: "Generate SQL compatible with BigQuery Standard SQL." The LLM's general knowledge of SQL dialects is usually sufficient for common functions. Failure modes: for dialect-specific features (BigQuery ARRAY_AGG, T-SQL CROSS APPLY), the model may generate the wrong syntax if its pre-training underrepresents that dialect.

**Option B — Dialect-specific fine-tuned models:**
Train one model per dialect on dialect-specific example queries. Highest accuracy for each dialect; highest operational cost (maintaining N model versions, N deployment pipelines).

**Option C — Generate dialect-neutral SQL, then transpile:**
Generate SQL in a canonical dialect (Postgres or ANSI SQL) and use a transpiler (sqlglot supports 20+ dialects) to convert to the target dialect. This is attractive because: (1) the LLM only needs to know one dialect well, (2) transpilation is deterministic and testable, (3) transpilation errors are easily caught.

Failure modes of transpilation: some constructs have no equivalent in the target dialect (BigQuery doesn't support recursive CTEs in all configurations; T-SQL has quirky GROUP BY behavior). The transpiler may fail or produce incorrect output for these cases.

**Production recommendation:** Option C (generate + transpile) for the majority of queries. Add dialect-specific few-shot examples for the constructs where transpilation is known to fail. Monitor transpilation failure rate per dialect — a high rate signals that you need dialect-specific examples or a specialized model for that dialect.
