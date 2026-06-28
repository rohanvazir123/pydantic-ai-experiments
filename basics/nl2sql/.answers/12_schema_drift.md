# Schema Changes and Drift — Answers

## Q42. A table renamed, a column deprecated, a new table added — all in one migration. Walk me through every place this breaks.

**Answer:**

A compound schema migration breaks the NL2SQL system in multiple layers simultaneously, which is why event-driven invalidation is essential.

**Where it breaks, layer by layer:**

*Break 1 — Schema embedding index:*
The vector store contains embeddings for the old table name and column names. Queries that semantically match the old name (`revenue_fact`) will still retrieve it — and the retrieval will succeed with no error, because the embedding still exists. The LLM then generates SQL referencing the old table name, which fails at execution time. Worse: the new table (`revenue_fact_v2`) is not yet indexed — queries that should use it will fail retrieval silently.

*Break 2 — SQL cache:*
Any cached (question → SQL) pairs that reference the renamed table or deprecated column are now stale. They will hit the cache, bypass SQL generation, and execute against the new schema — either failing with a "table not found" error or, if the name collision happens to resolve to a different table, silently returning wrong data.

*Break 3 — Few-shot example library:*
If any cached few-shot examples reference the old table or column name, every query that retrieves those examples will be prompted with stale schema context. The LLM will generate SQL based on the stale examples, referencing tables that no longer exist.

*Break 4 — Business glossary / KPI definitions:*
If a KPI definition references `revenue_fact.gross_amount` and that column has been split into two new columns, the KPI definition is wrong. Queries using that KPI will generate SQL against non-existent columns.

*Break 5 — Multi-turn conversation context:*
Any in-flight conversation that has already established context using the old schema is now operating on stale context. The user continues the conversation and the system generates SQL that references deprecated columns.

**Recovery strategy without manual re-deployment:**

*Automated schema change detection:*
Run a diff job against `information_schema` (or the DDL audit log) on a schedule (every 5 minutes for production) or triggered by a migration framework hook. The diff identifies: renamed tables, added columns, dropped columns, type changes.

*Cascading invalidation:*
On detection of any change to table T:
1. Remove all embeddings for table T from the vector store; re-embed and re-index within 60 seconds
2. Invalidate all cache entries whose schema version hash includes table T
3. Flag all few-shot examples that reference table T for re-validation
4. Notify the business glossary system of the change for manual review
5. Invalidate all active conversation contexts that reference table T

This cascade must complete in < 5 minutes — that is the acceptable window during which the system may produce wrong results after a schema change.

---

## Q43. How do you version your schema representation?

**Answer:**

Schema versioning is critical for two scenarios: auditing past query results, and multi-step conversations that span a schema change.

**Schema version as a content hash:**
The schema version is a hash of the canonical schema representation (table names + column names + types, sorted deterministically). This hash changes automatically when any structural change occurs, without requiring manual version bumping. Store this hash in the retrieval metadata for every cached query and every conversation session.

**Versioned schema storage:**
Keep N versions of the schema representation in the vector store, tagged with their version hash and a timestamp. When a query comes in:
- For new queries: use the current schema version
- For multi-turn continuation: use the schema version from the first turn of the conversation

**Handling cross-version conversations:**
If a schema change occurs mid-conversation, the system faces a choice:
1. Continue with the old schema version — the conversation remains coherent but references stale tables
2. Switch to the new schema version — the conversation may lose coherence if prior SQL context references old tables
3. Alert the user: "The schema has changed since your conversation started. Some results may be affected. [Start fresh]"

Option 3 is the most honest and the least risky. Implement it by detecting, at each turn, whether the schema version in the conversation context matches the current schema version. If not, surface the alert before generating SQL.

**For audit trails:**
Log every query execution with: the SQL generated, the schema version used, the user ID, and the timestamp. If a financial report is questioned months later, you can reproduce exactly what schema the query ran against and verify the SQL was correct at that point in time.

---

## Q44. Your schema cache has a 24-hour TTL. A critical table is renamed at 9am. How do you prevent all-day wrong SQL?

**Answer:**

A 24-hour TTL on schema cache is incompatible with production schema changes in a business that runs migrations during business hours. The architecture is wrong.

**The correct invalidation architecture:**

*Primary mechanism — Event-driven invalidation:*
Subscribe to DDL events from the data platform:
- PostgreSQL: use `pg_event_trigger` on `ddl_command_end` to fire a notification
- Snowflake: subscribe to `QUERY_HISTORY` for DDL statements
- BigQuery: subscribe to Cloud Audit Logs for `tables.update` events
- dbt: hook into `dbt run` completion events

When a DDL event for table T is received, immediately invalidate all cache entries whose schema version hash includes T. This is millisecond-latency invalidation — the system responds to the migration within seconds of it completing.

*Secondary mechanism — Short TTL as a backstop:*
Set the schema cache TTL to 15–30 minutes, not 24 hours. This bounds the maximum staleness window in case the event-driven system fails (missed event, network partition, message queue lag). The cost: slightly higher cache miss rate; the benefit: a schema change is always reflected within 30 minutes regardless of event delivery.

*Tertiary mechanism — Validation on cache hit:*
Before serving a cached SQL result, validate that all table and column names in the cached SQL still exist in the current schema. This adds 5–10ms to every cache hit but catches stale entries that slipped through. If validation fails, treat it as a cache miss and regenerate.

**Operational trade-offs of more aggressive invalidation:**
More aggressive invalidation increases cache miss rate, which increases LLM calls, which increases cost and latency. Quantify this: a schema with 400 tables that changes 5 tables per week at 15-minute TTL means 5/400 * 100% * (15min/24hr) = essentially zero increase in miss rate during non-change periods. The cost is near-zero.

---

## Q45. Some schema changes are backwards compatible, some are breaking. How does your system distinguish them?

**Answer:**

This matters because the handling should differ: backwards-compatible changes can be handled gracefully; breaking changes require immediate invalidation and potentially user notification.

**Classification:**

*Backwards compatible (no immediate impact on generated SQL):*
- Adding a new nullable column to an existing table
- Adding a new table
- Adding an index
- Loosening a constraint (NOT NULL → nullable)
- Adding a column default

Handle: re-index the schema in the background. Existing cached SQL continues to work. No user notification needed.

*Breaking changes (invalidate immediately):*
- Renaming a table or column
- Dropping a table or column
- Changing a column's data type in a way that affects the SQL (varchar → int)
- Tightening a constraint that existing queries may violate
- Moving a column to a different table

Handle: immediate cache invalidation for affected tables. If in-flight conversations reference the changed entity, surface an alert. Re-embed and re-index the affected tables within 60 seconds.

**Detection:**

Run a diff between the previous schema snapshot and the current schema snapshot on every schema check cycle. Compare:
1. Table set: tables removed = breaking (DROP TABLE), tables added = compatible
2. Column set per table: columns removed = breaking, columns added = compatible check (is it nullable?)
3. Column types per column: type changed = breaking
4. Table/column names: name changed = breaking (implies rename)

Classify the migration as a whole as "breaking" if any individual change is breaking. Apply breaking-change handling to all tables involved in the migration, not just the specific table with the breaking change (migrations are often transactional and related tables may also be affected).

---

## Q46. `revenue` split into `gross_revenue` and `net_revenue`. Users are still asking about "revenue."

**Answer:**

This is a semantic deprecation — the column name no longer exists but the concept does, in two forms. The system must handle it without silently generating SQL against a non-existent column.

**Immediate detection:**
After the schema change, any query for "revenue" will attempt to retrieve a `revenue` column in schema linking. The schema validation step will flag "revenue" as a hallucinated column — the column doesn't exist in the current schema. This is the detection mechanism.

**Handling in the business glossary:**
Add a deprecation entry to the business glossary:
```
Term: "revenue"
Status: DEPRECATED
Replaced by: gross_revenue (definition: ...), net_revenue (definition: ...)
Migration note: Use net_revenue for P&L reporting; gross_revenue for volume analysis
```

When the schema linking step encounters "revenue" and finds no matching column, it falls back to the business glossary and finds the deprecation entry. The system then asks a clarifying question: "The 'revenue' column has been split. Did you mean gross revenue or net revenue?"

**Pre-empting the clarification:**
If a specific business context always uses one definition (e.g., "revenue" in a finance dashboard always means `net_revenue`), configure a context-specific default in the glossary. The clarification is skipped and the assumption is annotated in the result.

**Preventing silent wrong SQL:**
The key safeguard is the column hallucination check: any column name in the generated SQL that does not exist in the retrieved schema is flagged as a hallucination before execution. This prevents the scenario where the LLM generates `SELECT revenue FROM ...` (which would fail at execution) — the system catches it at validation time and either retries with the correct column or asks for clarification.

---

## Q47. Your schema has 40 tables added per quarter. How does retrieval quality degrade?

**Answer:**

Adding 40 tables per quarter means 160 new tables per year. At 400 starting tables, you reach 560 tables in a year and 800 in two years. Retrieval quality degrades in two ways: crowding and specificity loss.

**Crowding effect:**
As the schema grows, more tables compete for the top-k retrieval slots. If k=15 is fixed and there are now 800 candidate tables instead of 400, the probability that the correct table is in the top-15 decreases — especially if many new tables have similar names or descriptions (common in large enterprises: `orders`, `orders_v2`, `orders_archive`, `orders_international`).

**Specificity loss:**
New tables often have overlapping semantic descriptions. A new `customer_interactions` table may be semantically similar to the existing `customer_activity` and `customer_events` tables. The embedding-based retrieval cannot reliably distinguish between them without fine-grained discriminating features.

**Detecting degradation before users notice:**

*Automated retrieval quality monitoring:*
For a golden set of (query, expected_table) pairs, run retrieval hourly and track the rank of the expected table. If the average rank increases monotonically over weeks, retrieval quality is degrading. Alert when the expected table drops from top-3 to top-10 on average.

*Schema growth rate monitoring:*
Track the rate of new table additions. When the schema exceeds a size threshold (e.g., doubles in size since the last embedding model evaluation), trigger a retrieval quality audit — run the golden query set and measure recall@k for various k values.

**Mitigation strategies:**
1. Hierarchical retrieval: first retrieve the schema domain (finance, sales, operations), then retrieve tables within the domain. This keeps the candidate set small within each domain.
2. Increase k dynamically as the schema grows — but this increases prompt token cost.
3. Schema clustering: group semantically similar tables (orders, order_items, order_status) into logical clusters. Retrieve clusters, then retrieve tables within the matched cluster.
4. Periodic re-evaluation of embedding quality: as the schema evolves, the pre-trained embedding model's ability to distinguish new table names may degrade. Fine-tune the embedding model on the current schema's terminology annually.
