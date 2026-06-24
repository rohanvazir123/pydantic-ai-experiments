# Schema Representation and Context — Answers

## Q5. A schema has 400 tables, 6,000 columns — you can't fit it all in context. How do you decide what to include?

**Answer:**

This is fundamentally a retrieval problem, not a prompting problem. The answer is a multi-stage funnel that progressively narrows from 400 tables to the 10–15 that actually go into the prompt.

**Stage 1 — Offline schema indexing:**
For each table, generate a rich semantic representation: table name, description, column names, column types, sample values (non-sensitive), foreign keys, synonyms from a business glossary, and embeddings of historical queries that used this table. Store these in a vector database.

**Stage 2 — Query-time coarse retrieval:**
Embed the user query and run approximate nearest-neighbor search against the table embeddings. Retrieve top-40. This is fast (< 50ms) and has high recall — the correct table is almost always in the top-40.

**Stage 3 — Reranking:**
Run a cross-encoder reranker (a small fine-tuned model that takes (query, table_description) as input and outputs a relevance score) over the top-40 to get a more accurate ranking. This is slower but more precise. Narrow to top-15.

**Stage 4 — Join graph expansion:**
For the top-15 tables, traverse the foreign key graph one hop out. Any table that is a direct join target of a selected table and has not been selected yet is a candidate for inclusion — score it by how many selected tables reference it. This surfaces bridge tables that are semantically invisible but structurally necessary.

**Stage 5 — Budget allocation:**
Not all tables need the same level of detail in the prompt. The top-3 most relevant tables get full DDL (all columns, types, comments, sample values). Tables ranked 4–10 get column names and types only. Tables 11–15 get just the table name and description. This fits more tables into a fixed token budget without losing the most important information.

**When the correct table is still not retrieved:**
This is the hard failure case. Mitigations:
1. Track execution failures where the generated SQL references a table not in the retrieved set — this surfaces retrieval misses without user reports.
2. Offer a "tables not found?" UI escape hatch where the user can name a specific table or concept, bypassing retrieval.
3. Build a query routing layer: queries that pattern-match known hard-to-retrieve domains (e.g. billing, compliance) always include domain-specific tables regardless of retrieval rank.

---

## Q6. Column names like `amt`, `flg`, `cd` are meaningless to an LLM. How do you enrich schema metadata at scale?

**Answer:**

Manual annotation doesn't scale. The approach is a combination of automated enrichment with human validation on the tail.

**Automated enrichment pipeline:**

*Step 1 — LLM-based description generation:*
For each column, send the LLM: table name, column name, column type, a sample of 10 non-sensitive distinct values, and the names of other columns in the same table. Ask it to generate a one-sentence description and a list of synonyms. This is a batch offline job, not a real-time operation. Cost: roughly $0.01–$0.05 per table for a large model; for 400 tables it is trivially cheap.

*Step 2 — Query log mining:*
Parse historical SQL queries from your warehouse's query log. Extract how each column is referenced — in WHERE clauses, aggregations, JOINs. A column that always appears in `WHERE flg = 1` alongside a `status` column is probably a boolean status flag. A column called `amt` that appears in SUM() aggregations alongside `revenue_type` columns is probably a monetary amount. This behavioral signal is often more accurate than LLM inference.

*Step 3 — Synonym expansion from business glossary:*
If the company has a data dictionary or BI tool with metric definitions (Looker, dbt docs, Confluence), parse these and extract synonyms automatically. "amt" might map to "amount", "value", "total", "revenue" depending on context.

*Step 4 — Human validation on low-confidence descriptions:*
Any description with a confidence score below a threshold goes into a review queue for data stewards. This is typically 10–15% of columns, making manual review tractable.

**Keeping enrichment fresh:**
Trigger a re-enrichment job on any schema change event (detected via information_schema diff or DDL event hooks). New columns get auto-enriched immediately. Changed columns (type changes, renamed columns) invalidate their existing description and enter the review queue. The system should never silently use a stale description after a schema change.

---

## Q7. How do you represent foreign key relationships and join paths to the model?

**Answer:**

The representation choice has a significant impact on the model's ability to construct correct multi-table queries.

**Option A — Prose description:**
"The orders table can be joined to the customers table using orders.customer_id = customers.id."

*Pros:* Natural for the LLM to parse; handles non-standard join conditions.
*Cons:* Verbose; for a schema with 50 join paths, this consumes enormous token budget; LLMs can get confused when multiple valid join paths exist.

**Option B — DDL with FOREIGN KEY constraints:**
```sql
CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(id),
  ...
);
```
*Pros:* Compact; LLMs pre-trained on SQL have seen this format extensively.
*Cons:* Many production schemas don't have actual FK constraints (especially data warehouses); the LLM must infer the join type (inner, left) from context.

**Option C — Explicit join path catalog in the prompt:**
```
JOIN PATHS:
- orders → customers via orders.customer_id = customers.id (many-to-one)
- orders → products via order_items.product_id = products.id (many-to-many, through order_items)
```
*Pros:* Most explicit; handles many-to-many through bridge tables; documents join type.
*Cons:* Consumes token budget; must be maintained as the schema evolves.

**Option D — Graph-based retrieval, not in-prompt representation:**
Pre-compute all join paths offline as a graph. At query time, retrieve only the join paths needed for the selected tables and inject them as a minimal join path catalog. The LLM never sees the full graph.

**Production recommendation:** Use Option B (DDL) as the base format — it's the most token-efficient and leverages the LLM's pre-training. Supplement with an explicit join path catalog (Option C) for tables that involve many-to-many relationships or non-obvious join keys. Use graph-based retrieval (Option D) to limit what gets included to only the paths needed for the retrieved tables.

---

## Q8. How does your schema retrieval strategy change between a data warehouse and an OLTP system?

**Answer:**

They have fundamentally different characteristics that require different retrieval strategies.

**Data Warehouse (star/snowflake schema, denormalized):**
- Fewer tables (50–200 vs 500+ in OLTP)
- Tables have clear semantic ownership: one fact table, several dimension tables
- Join paths are predictable and follow a star pattern
- Column names are often more descriptive (it was designed for querying)
- Most queries involve 1–3 tables

*Retrieval strategy:* Fact tables and their dimension tables should be grouped and retrieved together. If the query retrieves `sales_fact`, the retrieval system should automatically surface `date_dim`, `product_dim`, and `customer_dim` — even if the query doesn't mention "date" or "product" — because virtually all sales fact queries join these dimensions. This is a schema-topology-aware retrieval override.

**OLTP System (3NF normalized):**
- Many more tables (often 300–1000)
- Tables represent fine-grained entities and their relationships
- Join paths are deep (often 4–5 hops to get meaningful data)
- Many queries require traversing bridge/association tables that have no semantic meaning to a user
- Column names are often cryptic (legacy naming conventions)

*Retrieval strategy:* Pure semantic retrieval fails more often because the tables needed are structurally necessary but semantically invisible (bridge tables). You need: (1) join graph traversal to surface bridge tables after semantic retrieval, (2) richer column-level retrieval rather than table-level (a query about "customer email preferences" might land on a column in a `party_contact_method` table with no obvious semantic connection to "preferences"), (3) heavier reliance on query log mining to understand which tables historically co-occur.

**Does the same stack work for both?** The core LLM generation step is the same. The retrieval pipeline needs to be configurable: data warehouses benefit from topology-aware retrieval; OLTP systems benefit from deeper join graph traversal and column-level semantic indexing. Treat these as tunable parameters in the retrieval configuration, not two separate systems.
