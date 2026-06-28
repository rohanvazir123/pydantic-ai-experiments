# Passing the Right Subset of Schemas to the LLM — Answers

## Q48. You have 500 tables, context fits 15. Retrieval picks wrong 15. SQL looks plausible. Wrong data. How?

**Answer:**

This is the most dangerous failure in NL2SQL: the query executes, returns data, and everything looks correct — but the SQL ran against a wrong table.

**Detection:**

*Detection 1 — Post-retrieval table membership check:*
After SQL is generated, compare every table name in the generated SQL against the set of tables that were in the retrieval context. Any table in the SQL that was not in the top-15 retrieved tables is either a hallucination (the LLM invented it) or a retrieval miss (it exists but wasn't retrieved). Either way, flag and reject.

This catches the case where the LLM generates SQL against `orders_international` but only `orders` was in the context. The flag: "Generated SQL references table not in context: orders_international."

*Detection 2 — Result distribution anomaly:*
For recurring query patterns, compare result distributions against historical results. If "total revenue" typically returns values between $1M and $10M but today returns $47, the query is probably running against the wrong table. This requires historical result tracking and anomaly thresholds per query pattern.

*Detection 3 — Self-consistency check:*
Generate the SQL twice with slightly different prompts (different few-shot examples). If the two runs reference different tables, retrieval was ambiguous — flag for human review or clarification.

**Fixing the retrieval step:**

*Root cause analysis:* Understand why the correct table ranked 18th. Was it a naming issue (the table name has no semantic relationship to the query)? A description gap (the table has no description and is only indexed by its name)? A crowding issue (too many similar tables diluting the signal)?

*Targeted enrichment:* Add the specific terms that would have matched the query to the table's embedding metadata. "The `partner_revenue_fact` table contains sales revenue for channel partners" — add synonyms: "channel sales", "partner sales", "reseller revenue".

*Domain pinning:* For known query patterns that consistently retrieve the wrong table, add a deterministic override: queries that match the pattern "partner revenue" always include `partner_revenue_fact` regardless of its retrieval rank. This is a curated shortlist that bypasses retrieval for known problematic cases.

---

## Q49. Walk me through your complete schema retrieval pipeline — where does each component fail?

**Answer:**

**Stage 1 — Query embedding (5–20ms):**
The user query is embedded using a text embedding model (e.g., text-embedding-3-small, nomic-embed-text).

*Failure mode:* The query uses domain-specific terminology the embedding model has rarely seen in pre-training ("ARR", "ACV", "NRR" in SaaS). The embedding places the query in a generic "business" vector space rather than the specific "SaaS metrics" space, degrading retrieval relevance.

*Mitigation:* Fine-tune the embedding model on domain-specific corpora. Alternatively, expand the query with synonyms before embedding ("ARR" → "ARR annual recurring revenue subscription").

**Stage 2 — Approximate nearest neighbor search (30–150ms):**
The query embedding is compared against all stored table embeddings using ANN search (HNSW or IVF index).

*Failure mode:* High-volume concurrent queries degrade ANN latency. The index was built with parameters optimized for accuracy on the original 200-table schema; at 500 tables, recall at k=15 has degraded but nobody re-tuned the parameters.

*Mitigation:* Monitor recall@k monthly against a golden set. Re-tune HNSW `ef_construction` and `M` parameters when schema grows significantly.

**Stage 3 — Cross-encoder reranking (80–300ms):**
The top-40 ANN results are reranked using a cross-encoder model that considers the (query, table_description) pair jointly.

*Failure mode:* The reranker was trained on generic (query, document) pairs and doesn't understand SQL-specific relationships. A query about "customer lifetime value" correctly ranks `customer` table first, but the reranker doesn't know that CLV calculations typically require `orders` and `payments` tables too.

*Mitigation:* Fine-tune the reranker on (query, table, relevance_label) triples where relevance is labeled by domain experts or inferred from historical query execution.

**Stage 4 — Join graph expansion (10–50ms):**
For the top-15 reranked tables, traverse the FK graph one hop. Add tables that are direct join targets of selected tables and that have high co-occurrence with those tables in historical queries.

*Failure mode:* The FK graph is incomplete (many production schemas don't enforce FK constraints). Bridge tables (many-to-many junction tables) are two hops away and are not surfaced.

*Mitigation:* Build a "logical FK graph" from query log co-occurrence rather than relying solely on declared FK constraints. Tables that frequently appear together in historical JOINs have an implied relationship.

**Stage 5 — Budget-aware selection and truncation (< 5ms):**
From the expanded candidate set, select the final subset that fits within the token budget, applying full DDL to top tables and abbreviated schemas to lower-ranked ones.

*Failure mode:* A table with many columns (the `customer` table has 80 columns) consumes the entire token budget, leaving no room for the join table.

*Mitigation:* Apply column-level relevance filtering: for each table in the context, include only the columns that are semantically relevant to the query (embed columns, rank by similarity to the query, include top-N). This can reduce a 80-column table to 8 relevant columns for a specific query.

---

## Q50. How do you embed a table schema for retrieval?

**Answer:**

Raw column names and types are low-signal for embedding because they have minimal natural language content. A table with columns `id, c_id, amt, ts, flg` is essentially unembeddable without enrichment.

**What to include in the embedding document, in order of signal value:**

*1. Table name (normalized):* Convert `cust_ord_trans` to "customer order transactions" using abbreviation expansion. This alone can dramatically improve retrieval quality.

*2. Table description:* A 1–3 sentence natural language description of what the table contains, generated by LLM from the column names + sample data. "This table records individual customer transactions, including the transaction amount, timestamp, and fulfillment status."

*3. Column names (normalized and expanded):* Include expanded forms of abbreviated column names.

*4. Sample column values for categorical columns:* `status IN ('active', 'churned', 'trial')` tells the embedding model this is a status/lifecycle column. This helps queries like "show active customers" match the right table.

*5. Business synonyms from the glossary:* If the business calls this table's data "bookings" but the table is named `revenue_transactions`, add "bookings" to the embedding document.

*6. Historical query patterns:* The top 10 NL queries that historically retrieved this table, distilled to a set of key phrases. This is the highest-signal feature — it directly captures what human users think this table answers.

*7. Foreign key relationship summary:* "This table joins to: customers (via customer_id), products (via product_id)." This helps retrieve the table for join-requiring queries.

**What NOT to include:**
- Every single column value sample (too noisy)
- SQL DDL verbatim (the embedding model doesn't read DDL well; transform it to prose first)
- Internal technical metadata (table owner, creation date, storage size)

**Embedding strategy:**
Use a single embedding per table (the full enriched document as above). For large schemas, also embed at the column level and store column embeddings separately — this enables column-level retrieval for queries that name specific concepts that map to specific columns rather than tables.

---

## Q51. A 4-way join with no direct FK relationships — bridge tables are semantically invisible. How do you surface them?

**Answer:**

This is the hardest case in schema retrieval. The user asks about "customer lifetime value by acquisition channel" — this requires: `customers` → `orders` → `order_items` → `marketing_attribution`. The bridge table (`order_items`) has no semantic relationship to "acquisition channel" or "lifetime value."

**Approach 1 — Join path pre-computation:**
Offline, compute all valid join paths in the schema graph using BFS/DFS from every table. For each path, generate a natural language description of what the path enables: "customers → orders → order_items → marketing_attribution: enables analysis of customer revenue by marketing channel." Embed these path descriptions. At query time, retrieve paths by embedding similarity alongside tables.

*Failure case:* The path description for `order_items` mentions "order details" but not "marketing attribution" — the path embedding doesn't match the query.

**Approach 2 — Query log co-occurrence graph:**
From historical SQL queries, build a co-occurrence matrix: how often does table A appear in the same query as table B? Tables that always appear together have a strong co-occurrence relationship. When `customers` and `marketing_attribution` are both retrieved, automatically include `orders` and `order_items` because they have high co-occurrence with both.

*Strength:* This is empirically grounded — it captures the actual join patterns humans use, not just the schema structure.

**Approach 3 — LLM-based join graph reasoning:**
After initial retrieval, pass the retrieved tables to the LLM and ask: "To answer this question, which additional tables might be needed to join these together?" The LLM can infer missing bridge tables from its understanding of relational data patterns. Include only tables that the LLM suggests AND that exist in the actual schema (validated by a lookup).

**Production recommendation:** Approach 2 is the most reliable in production because it's data-driven. Implement Approach 1 as a structural safety net for tables that don't appear in query history. Use Approach 3 as a last-resort fallback for complex queries where 1 and 2 don't surface the necessary bridge tables.

---

## Q52. How do you decide the right k (tables in context)?

**Answer:**

k is not a single fixed value — it has an optimal range that varies by query complexity and schema characteristics.

**The cost of k being too low:**
Missing the correct table means the LLM either hallucinates it (references a table not in context) or generates incomplete SQL (uses only the retrieved tables, producing wrong results). The miss rate at k=5 is significantly higher than at k=15 for a 500-table schema.

**The cost of k being too high:**
At k=20 with full DDL, a prompt might be 10,000+ tokens. Problems: (1) LLM attention degrades on very long prompts — tables near the end of the prompt receive less "attention" than tables near the beginning (the lost-in-the-middle problem); (2) higher token cost and latency; (3) more irrelevant schema context can confuse the model.

**Optimal k for different scenarios:**
- Single-table queries: k=5 is sufficient. Including 15 tables wastes context.
- Two-table joins: k=8 is sufficient.
- Complex multi-join queries: k=15–20 is needed.

**Dynamic k:**
Use a query complexity classifier to predict the number of tables the query will require, then set k = predicted_table_count × 2 (giving a 2x safety margin). Cap at k=20. This requires the classifier to run before retrieval, which means it must be fast (< 20ms) — use a small fine-tuned model or a rule-based classifier.

**Tuning k empirically:**
Run your evaluation set at k = {5, 8, 10, 12, 15, 20}. Plot execution accuracy vs. k for each query complexity tier. Find the point where increasing k no longer improves accuracy — that's your optimal k. Typically this is k=12–15 for complex schemas, k=8 for simple schemas.

---

## Q53. Fixed k wastes context on simple queries and truncates complex ones. How do you build dynamic k selection?

**Answer:**

**Query complexity features:**
- Number of distinct entity mentions (tables, metrics, filters, grouping dimensions) in the NL query
- Presence of join-indicating language ("by", "across", "linked to", "with their")
- Presence of aggregation ("total", "average", "count", "sum")
- Presence of multi-step logic ("customers who have X and have done Y but not Z")
- Query length (proxy for complexity)

**Model for dynamic k:**
Train a lightweight regression or ordinal classifier: input = complexity features, output = predicted number of tables needed (1–8). Set k = predicted_count + safety_margin (2–3 tables).

This is a small model (logistic regression or a 3-layer MLP) that adds < 5ms to the pipeline. Alternatively, use a rules-based system: if join-indicating language is present, set k=12; otherwise k=6.

**Validating correctness of dynamic k:**
For a held-out evaluation set, compare: recall@dynamic_k vs. recall@fixed_k_15. Dynamic k should achieve similar recall for complex queries (by setting k high when needed) and lower token cost for simple queries (by setting k low). The metric: mean recall * (1 - token_overhead_ratio).

---

## Q54. "Show me the churn rate by cohort" — semantically distant from any column name. How do you bridge the gap?

**Answer:**

This is a domain knowledge problem. "Churn rate by cohort" requires knowing: (1) what "churn" means in the company's data (what event constitutes churn?), (2) what "cohort" means (signup cohort? acquisition channel cohort? first purchase month?), (3) which tables track these concepts.

**Layer 1 — Business glossary with KPI-to-table mapping:**
The glossary must contain: `churn_rate: requires customers table (status column) + subscription_events table (event_type = 'cancelled'). Cohort: defined by customers.signup_month.`

When the query embedding matches the glossary term "churn rate", the KPI-to-table mapping directly includes the relevant tables, bypassing the semantic retrieval step. This is the highest-reliability solution for known KPIs.

**Layer 2 — Query pattern library:**
The query library (used for few-shot RAG) should contain examples of churn rate queries with the correct SQL. When "churn rate by cohort" is embedded and matched against the query library, the historical example surfaces the correct tables as context for the LLM.

**Layer 3 — Augmented query expansion:**
Before embedding the query for retrieval, expand it with domain-specific terminology: "churn rate by cohort" → "churn rate cancellation rate by cohort signup month customer retention". The expanded query has higher lexical overlap with table descriptions that mention cancellation, retention, and signup.

**Layer 4 — Column-level retrieval:**
Index columns individually, not just tables. "Churn" as a concept might match a column called `is_churned`, `churn_date`, or `subscription_status`. Column-level retrieval surfaces these and the tables they belong to.

**The honest answer:**
For queries that require deep domain knowledge not present in the schema text, technical retrieval cannot fully close the gap. The business glossary is the essential component — it translates business concepts to schema elements with high precision. Without it, every new domain concept requires ad-hoc handling.

---

## Q55. Multi-tenancy in schema retrieval — Tenant A and Tenant B have different `payments` tables. How?

**Answer:**

Multi-tenant schema retrieval requires tenant isolation at every layer: indexing, retrieval, generation, and execution.

**Indexing:**
Each tenant's schema is indexed separately in the vector store, tagged with `tenant_id`. Never mix embeddings from different tenants in the same index partition. Use namespacing in the vector store: queries from Tenant A only search the Tenant A namespace. This is a hard requirement — without it, a query for Tenant A might retrieve Tenant B's schema description and generate SQL against Tenant B's column names.

Implementation in Pinecone: use namespaces. In pgvector: add a `tenant_id` column to the embeddings table and filter `WHERE tenant_id = ?` on every retrieval query.

**Retrieval:**
Every retrieval call includes `tenant_id` as a mandatory filter. This is not optional or best-effort — it must be enforced in the retrieval service layer, not left to the caller. A bug in the caller that omits `tenant_id` should return zero results, not cross-tenant results.

**SQL Generation:**
The few-shot examples are tenant-specific. Tenant A's examples use Tenant A's `payments` columns; Tenant B's examples use Tenant B's `payments` columns. The example library is indexed by `tenant_id` and retrieved with the same isolation guarantee.

**Execution:**
The SQL executes against Tenant A's database or Tenant A's schema within a shared database. Service account permissions are scoped to the tenant's tables — the database enforces tenant isolation even if the SQL is wrong.

**Schema cache isolation:**
The cache key includes `tenant_id` as a mandatory component. A cache hit from Tenant A cannot be served to Tenant B, even if they asked the same natural language question. The underlying tables are different; the SQL must be different.

**The audit requirement:**
Log every retrieval event with `tenant_id`. Any retrieval event that touches a namespace not matching the authenticated `tenant_id` is a security incident — alert on it immediately.
