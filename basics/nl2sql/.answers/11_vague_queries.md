# Correctness Around Vague Queries — Answers

## Q37. "Show me the sales numbers" — vague in 5 dimensions. How does your system decide what to do?

**Answer:**

The five dimensions of vagueness: metric (revenue? units? deals closed?), time period (all time? this quarter? trailing 12 months?), region (global? by region?), product line (all? specific?), granularity (daily? monthly? total?).

**The decision tree:**

*Step 1 — Ambiguity detection and scoring:*
Run the query through an ambiguity classifier that scores each dimension: 0 (fully specified), 0.5 (partially specified or inferrable from context), 1.0 (completely unspecified). "Sales numbers" has metric=1.0, time=0.8 (no period mentioned, but "current" is a reasonable default), region=1.0, product=1.0, granularity=0.8.

*Step 2 — Ambiguity threshold and routing:*
If the sum of ambiguity scores exceeds a threshold (e.g., 3.0 out of 5.0 possible), the query is too ambiguous to answer with a single SQL — route to clarification. If below threshold, resolve using defaults and proceed.

*Step 3 — Default resolution with annotation:*
For the dimensions below threshold, apply defaults: time → last 30 days, granularity → aggregate total. Generate the SQL with these defaults. Annotate the response: "Showing total revenue (all regions, all products) for the last 30 days. [Refine ▾]"

*Step 4 — Clarification for high-ambiguity dimensions:*
For dimensions above threshold, ask. But don't ask about all of them at once — users find multi-question dialogs frustrating. Ask only about the dimension with the highest ambiguity score and the highest business impact. "What metric would you like — revenue, units sold, or number of deals?" Once answered, proceed with defaults for the remaining dimensions.

**What signals drive the routing decision:**
- User persona: analysts get defaults with annotation; executives get clarification for high-stakes metrics
- Query history: if this user has previously queried with a specific metric definition, apply it as their personal default
- Time of day / context: a query right before a quarterly business review is more likely to be asking about the current quarter than a query sent at random

---

## Q38. When your system makes an assumption to resolve a vague query, how and where do you surface it?

**Answer:**

Surfacing assumptions is a UI and trust problem. The wrong approach surfaces too much (every query has a disclaimer that users learn to ignore) or too little (users don't know the assumption was made).

**The right level of surfacing:**

*For consequential assumptions (metric definition, date range):*
Surface explicitly before or alongside the result, in plain language. Not "Generated SQL uses `SUM(revenue_amount) WHERE created_date >= DATEADD(month, -1, GETDATE())`" — instead: "Showing total revenue for the last 30 days." Make it actionable: provide a dropdown or quick-tap alternatives ("Switch to: this quarter | last 12 months | all time").

*For low-consequence assumptions (ordering, default LIMIT):*
Surface passively in the result metadata, not as a primary UI element. "Sorted by revenue, highest first. Showing top 100 results." This appears below the result, not above it.

*For aggregation-level assumptions (daily vs. monthly granularity):*
Show the granularity in the column header or chart axis. If the result is monthly, the date column says "Month" — this is self-documenting.

**The risk of surfacing too prominently:**
If every query result starts with "I assumed X, Y, Z", users learn to skim the assumptions — even when they matter. Reserve prominent surfacing for metric definition assumptions (high stakes) and date range assumptions (commonly wrong). For other assumptions, use passive surfacing.

**The risk of surfacing too subtly:**
A user who doesn't notice the "last 30 days" annotation and presents "total revenue" in a board meeting has been harmed. For executive personas or any query that pattern-matches to "report", "board", "present", "share", force the assumption into the result headline, not the footnote.

---

## Q39. How do you define a correctness SLA for vague queries?

**Answer:**

For vague queries, there is no single ground truth SQL — multiple queries are defensibly correct. Traditional execution accuracy (compared to a ground truth query) doesn't apply.

**A correctness SLA for vague queries has three components:**

*Component 1 — Intent preservation:*
The generated SQL must answer *a* reasonable interpretation of the question, not a random or clearly wrong one. Measure this by having domain experts rate whether the generated SQL is a reasonable interpretation of the NL query, on a 1–5 scale. Target: mean rating > 3.5, < 5% of queries rated 1 (clearly wrong interpretation).

*Component 2 — Assumption transparency:*
Every assumption made to resolve vagueness must be surfaced to the user and must be factually accurate. If the system says "Showing revenue for last 30 days" but the SQL actually uses `last_90_days`, this is a transparency failure. Measure: audit rate of assumption descriptions vs. actual SQL — they must match 100% of the time.

*Component 3 — User acceptance rate:*
Track the percentage of vague queries where the user accepts the result without rephrasing (as a proxy for satisfaction). For vague queries, target > 60% acceptance (vs. > 80% for precise queries). The gap reflects the inherent ambiguity — some users will always have meant something different.

**What this SLA still misses:**
A user who accepts a result that happens to be correct by coincidence looks the same as a user who verified the result. You cannot distinguish these with implicit signals. This is why the SLA has three components rather than one — no single metric is sufficient.

---

## Q40. "Who are our best customers?" — how does your system know which definition to use?

**Answer:**

"Best customers" is a business KPI definition, not a schema property. The LLM's general training data contains a common-sense definition (highest revenue), which may be completely wrong for a specific company.

**Where business logic lives — in order of preference:**

*Layer 1 — Company data dictionary / business glossary:*
If the company has defined "best customer" in a data catalog (Collibra, Atlan, dbt docs), the system must retrieve that definition and inject it into the prompt. This is the authoritative source. The NL2SQL system must have a business glossary retrieval step parallel to schema retrieval — when a KPI term is detected in the query, retrieve its definition.

*Layer 2 — Pre-computed KPI columns:*
If "customer score" or "customer tier" is a computed column in the data warehouse (a materialized column updated nightly), using it is preferable to recomputing the KPI from first principles in the query. The schema enrichment step should flag such columns with a note: "This column represents [definition]. Use it for KPI X queries."

*Layer 3 — Query history:*
If queries about "best customers" historically use `ORDER BY lifetime_value DESC`, that is a revealed preference that can inform the default. Mine query logs for KPI-related patterns and surface them as defaults.

*Layer 4 — Clarification:*
If none of the above exist, ask the user which definition to use — and offer the options you know are plausible from the schema (revenue, order count, margin). Store their answer as a session or user-level preference for future queries.

**When the definition changes:**
This is the dangerous case. If the company changes its definition of "best customer" from "highest revenue" to "highest margin" and the old definition was cached in the system, the system silently produces wrong results using the old definition. Mitigations: (1) tie the KPI definition to a version in the data dictionary; any version change invalidates cached interpretations, (2) surface the definition used in every result ("Best customers by lifetime revenue — [See definition ▾]"), making changes noticeable to users.

---

## Q41. At what point do you reject a query as too vague to answer?

**Answer:**

The threshold for rejection (or mandatory clarification) should be set based on the expected harm of a wrong default assumption, not just the degree of vagueness.

**The harm-weighted ambiguity score:**

Each unresolved dimension gets an ambiguity score (0–1) weighted by the consequence of getting it wrong:

| Dimension | Raw ambiguity | Harm weight | Weighted score |
|-----------|--------------|-------------|----------------|
| Metric definition | 1.0 | 2.0 (financial decisions) | 2.0 |
| Time period | 0.8 | 1.5 | 1.2 |
| Grouping dimension | 0.7 | 1.0 | 0.7 |
| Sort order | 0.5 | 0.2 | 0.1 |

Sum of weighted scores > threshold → clarification required.

**What the system returns when rejecting:**

Not an error message — a structured clarification. Return the partial interpretation you can make plus the specific question you need answered:
"I can show you customer data — I just need one more detail: what does 'best' mean in your context? [By revenue] [By order frequency] [By margin] [Something else]"

Give the user clickable options derived from the schema (you know what metrics are available), not a free-text field. This makes the clarification feel like guided refinement rather than failure.

**Calibrating the threshold:**
Too low a threshold (too much clarification) → users find the system frustrating and stop using it. Too high a threshold (too little clarification) → the system confidently returns wrong answers. A/B test different thresholds on a per-persona basis: analysts tolerate more defaults, executives need more clarification. Start with a conservative threshold and relax it as you build confidence in the quality of your defaults.
