# Query Intent Detection — Ambiguous and Out-of-Scope Queries

Two distinct detection problems that must be solved before SQL generation even starts:

1. **Ambiguity detection** — the query is analytically intentioned but under-specified. Multiple valid SQL interpretations exist.
2. **Scope detection** — the query is unrelated to the tenant's data, partially related, or referencing entities/concepts the schema does not contain.

Both must be detected early, before schema retrieval or SQL generation, because letting an ambiguous or out-of-scope query reach the LLM produces confidently wrong output with no signal that something went wrong.

## Table of Contents

- [Why This Is Hard](#why-this-is-hard)
- [Stage 1 — Intent Classification](#stage-1--intent-classification)
- [Ambiguity Detection — Concrete Mechanism](#ambiguity-detection--concrete-mechanism)
  - [Dimension-Level Ambiguity Scoring](#dimension-level-ambiguity-scoring)
  - [Schema-Grounded Ambiguity](#schema-grounded-ambiguity)
  - [What to Do With an Ambiguous Query](#what-to-do-with-an-ambiguous-query)
- [Out-of-Scope Detection — Concrete Mechanism](#out-of-scope-detection--concrete-mechanism)
  - [Fully Unrelated Queries](#fully-unrelated-queries)
  - [Entity Mismatch — Query References Data the Tenant Doesn't Have](#entity-mismatch--query-references-data-the-tenant-doesnt-have)
  - [Partial Scope — Mixed Analytical and Conversational](#partial-scope--mixed-analytical-and-conversational)
  - [Schema Coverage Miss — Topic Exists but Data Doesn't](#schema-coverage-miss--topic-exists-but-data-doesnt)
- [Putting It Together — The Pre-Generation Gate](#putting-it-together--the-pre-generation-gate)
- [Concrete Examples With Classifications](#concrete-examples-with-classifications)
- [What to Return to the User for Each Case](#what-to-return-to-the-user-for-each-case)
- [Common Mistakes](#common-mistakes)

---

## Why This Is Hard

Both problems look easy until you see real user queries:

**Ambiguity is not just vague language.** "Show me revenue last quarter" sounds specific but is ambiguous if the company has five different revenue definitions in five different tables. The query is precise in natural language but maps to multiple SQL queries, all defensible.

**Out-of-scope is not just off-topic.** "Who are our top customers in Germany?" is fully in-scope for a company with German customers and completely out-of-scope for a US-only business with no geography dimension in its schema. The same query is in-scope or out-of-scope depending on the tenant's data — which means detection must be schema-aware, not query-only.

The failure if you get this wrong:
- Miss an ambiguous query → system picks one interpretation silently, user acts on wrong data
- Miss an out-of-scope query → system generates SQL that either fails at execution or returns plausible-but-meaningless results

---

## Stage 1 — Intent Classification

Before ambiguity or scope checks, classify the query into one of four intents:

```
ANALYTICAL   — answerable by querying the tenant's database
CONVERSATIONAL — a question or statement requiring a natural language response, not SQL
OUT_OF_SCOPE — unrelated to the tenant's data domain
AMBIGUOUS    — analytical intent but insufficient specification to generate a unique SQL
```

**Classifier architecture:**

A two-stage classifier:

*Stage A — Fast rule-based pre-filter (< 1ms):*
Handle the obvious cases without an LLM call:
- Query is a greeting, thank-you, or one-word input → CONVERSATIONAL
- Query contains SQL keywords and looks like a direct SQL question ("what does SELECT do?") → CONVERSATIONAL
- Query is clearly general knowledge ("what is the GDP of France", "who won the 2024 election") → OUT_OF_SCOPE

*Stage B — Embedding similarity classifier (< 20ms):*
Embed the query. Compare against:
1. A labeled set of known ANALYTICAL queries for this tenant (from query history or golden set)
2. A labeled set of CONVERSATIONAL queries (domain-agnostic)
3. A labeled set of OUT_OF_SCOPE queries (domain-agnostic)

Compute cosine similarity to each cluster centroid. The highest similarity determines the intent class. If all similarities are below 0.5, the query is novel — escalate to the LLM classifier.

*Stage C — LLM classifier for novel queries (< 500ms, only when stages A and B are inconclusive):*

```
System: You are classifying user queries for a business analytics system.
        The system can query a database containing: {brief_schema_summary}.
        Tenant domain: {tenant_description}.

Classify this query as one of:
  ANALYTICAL   - can be answered by querying the database
  CONVERSATIONAL - requires a natural language response, not a database query  
  OUT_OF_SCOPE - unrelated to the tenant's data or business domain
  AMBIGUOUS    - analytical intent but missing critical specification

Query: "{user_query}"

Output JSON: {"intent": "...", "confidence": 0.0-1.0, "reason": "..."}
```

**Why a three-stage approach rather than always using the LLM:**
The LLM classifier is accurate but adds 300–500ms. For the 70% of queries that are clearly ANALYTICAL (follow patterns the system has seen before), rule-based + embedding classification handles them in < 20ms. Reserve the LLM call for the ambiguous edge cases.

---

## Ambiguity Detection — Concrete Mechanism

A query passes intent classification as ANALYTICAL but may still be too under-specified to generate a correct SQL. Ambiguity detection runs after intent classification, before schema retrieval.

### Dimension-Level Ambiguity Scoring

Ambiguity exists along six independent dimensions. Before defining the scoring, it is critical to anchor what **score = 0.0 (non-vague / fully specified)** means for each dimension. Without these anchors the scoring is arbitrary.

---

#### What non-vagueness means per dimension

**Metric (what is being measured):**

Score = 0.0 when:
- The query names a single, unambiguous column or a term that resolves to exactly one column in the business glossary: `"net_revenue"`, `"arr"`, `"number of closed deals"`
- The metric term appears verbatim as a column name or glossary alias
- Context from prior turns has established which metric is in use

Score > 0.0 when:
- The term maps to multiple columns: `"revenue"` → `gross_revenue`, `net_revenue`, `arr`, `mrr` → score 0.9
- The term is a high-level category without a clear default: `"performance"`, `"numbers"`, `"figures"` → score 1.0
- The term exists in the glossary but has conditional definitions (e.g., "revenue" means ARR for SaaS contracts and one-time payment for professional services) → score 0.7

**Time period:**

Score = 0.0 when:
- An explicit, unambiguous date range is given: `"Q3 2024"`, `"January 1 to March 31 2024"`, `"the week of June 10"`
- A relative expression resolves to a unique range regardless of fiscal/calendar calendar: `"yesterday"`, `"last 7 days"`, `"MTD"` (month-to-date is always calendar)
- A prior turn established the time period and this query continues it without changing it

Score > 0.0 when:
- No time expression at all: `"show me revenue"` → score 0.7 (will default, but user may not expect current period)
- Relative expression that is ambiguous given the tenant's fiscal calendar: `"last year"` when fiscal year ≠ calendar year → score 0.8
- Vague recency: `"recent"`, `"latest"`, `"current"` → score 0.6 (defaults to last 30 days, but user intent varies)
- Relative to an unclear reference point: `"same period last year"` when the current period has not been established → score 0.9

**Region / Geography:**

Score = 0.0 when:
- An exact region value is named that exists in the schema: `"North America"`, `"EMEA"`, `"Germany"` — and that value appears as a distinct value in the region column
- The schema has only one geographic dimension and the user specifies a value from it

Score > 0.0 when:
- No region mentioned and the schema has a region dimension → score 0.5 (implies all regions, but user may have meant a specific one)
- Region named but maps to multiple hierarchy levels: `"West"` could be a region, a sub-region, or a country depending on the schema → score 0.7
- Region term is informal and does not exactly match schema values: `"the western markets"` → score 0.6

**Product / Product Line:**

Score = 0.0 when:
- A specific product name or SKU is given that maps to exactly one row or group in the product dimension: `"iPhone 15"`, `"Product SKU-4821"`, `"Enterprise plan"`
- The query says "all products" — this is unambiguous (no filter needed)

Score > 0.0 when:
- No product specified and the schema has a product dimension → score 0.4 (implies all products; lower harm than metric or time)
- Product name is informal or ambiguous across multiple product lines: `"our flagship product"` → score 0.8
- Product name exists in multiple product tables (active vs discontinued) → score 0.6

**Grouping / Breakdown dimension:**

Score = 0.0 when:
- The query explicitly names the breakdown: `"by region"`, `"by month"`, `"per sales rep"`, `"broken down by product category"`
- The query is explicitly asking for a total with no breakdown: `"total revenue"`, `"aggregate count"`

Score > 0.0 when:
- No grouping specified for an aggregation query → score 0.5 (will default to aggregate, but user may want a breakdown)
- Ambiguous grouping level: `"by area"` when the schema has `region`, `sub_region`, `territory`, `country` → score 0.7

**Granularity / Time grain:**

Score = 0.0 when:
- Explicit time grain stated: `"monthly"`, `"weekly"`, `"daily"`, `"quarterly"`, `"annual"`
- The query is for a point-in-time value where granularity is irrelevant: `"what is today's headcount"`

Score > 0.0 when:
- No granularity and query spans a period → score 0.5 (default to aggregate total, but user may want a trend)
- Vague granularity: `"over time"`, `"trend"`, `"historically"` → score 0.6

---

#### Scoring implementation with anchored definitions

```python
class AmbiguityScorer:

    def score(self, query: str, ctx: TenantContext) -> AmbiguityReport:
        return AmbiguityReport(
            metric       = self._score_metric(query, ctx),
            time_period  = self._score_time(query, ctx),
            region       = self._score_region(query, ctx),
            product      = self._score_product(query, ctx),
            grouping     = self._score_grouping(query, ctx),
            granularity  = self._score_granularity(query),
        )

    def _score_metric(self, query: str, ctx: TenantContext) -> float:
        terms = extract_metric_terms(query)
        if not terms:
            return 1.0  # no metric at all — completely unspecified

        candidates = ctx.glossary.resolve_all(terms)  # returns list of matching columns

        if len(candidates) == 0:
            return 1.0  # metric term not in glossary — unknown
        if len(candidates) == 1:
            return 0.0  # resolves to exactly one column — non-vague
        if len(candidates) <= 3:
            return 0.5  # a few candidates — moderately ambiguous
        return 0.9       # many candidates — highly ambiguous

    def _score_time(self, query: str, ctx: TenantContext) -> float:
        expr = extract_time_expression(query)  # returns parsed time range or None

        if expr is None:
            return 0.7  # no time expression — will default to current period

        if expr.is_absolute:
            # "Q3 2024", "January 1–March 31 2024" — unambiguous regardless of calendar
            return 0.0

        if expr.is_relative_unambiguous:
            # "yesterday", "last 7 days", "MTD" — calendar-agnostic
            return 0.0

        if expr.requires_fiscal_calendar and ctx.fiscal_year_differs_from_calendar:
            # "last year", "last quarter", "YTD" when fiscal ≠ calendar
            return 0.8

        if expr.is_vague:
            # "recent", "latest", "current"
            return 0.6

        return 0.2  # relative but probably resolvable with standard calendar

    def _score_region(self, query: str, ctx: TenantContext) -> float:
        regions = extract_region_references(query)

        if not regions:
            # No region mentioned
            if ctx.schema.has_region_dimension:
                return 0.5  # user might have meant all regions — annotate assumption
            return 0.0      # schema has no region dimension — dimension irrelevant

        for region in regions:
            matches = ctx.schema.find_region_values(region)
            if len(matches) == 0:
                return 0.8  # region not found in schema values
            if len(matches) > 1:
                return 0.7  # maps to multiple hierarchy levels
        return 0.0  # all regions resolve to exactly one schema value

    def _score_product(self, query: str, ctx: TenantContext) -> float:
        products = extract_product_references(query)

        if not products:
            if ctx.schema.has_product_dimension:
                return 0.4  # no product filter — implies all, lower harm than time/metric
            return 0.0

        for product in products:
            candidates = ctx.schema.find_products(product)
            if len(candidates) == 0:
                return 0.8  # product name not in schema
            if len(candidates) > 1:
                return 0.6  # matches multiple product rows or tables
        return 0.0

    def _score_grouping(self, query: str, ctx: TenantContext) -> float:
        grouping = extract_grouping_expression(query)

        if grouping is None:
            # Aggregation query with no breakdown specified
            if is_aggregation_query(query):
                return 0.5
            return 0.0  # non-aggregation queries don't need grouping

        candidates = ctx.schema.find_grouping_columns(grouping)
        if len(candidates) > 1:
            return 0.7  # "by area" → region, sub_region, territory ambiguous
        return 0.0

    def _score_granularity(self, query: str) -> float:
        grain = extract_time_grain(query)

        if grain is not None:
            return 0.0  # "monthly", "weekly", "daily" — unambiguous

        time_expr = extract_time_expression(query)
        if time_expr and time_expr.spans_period:
            return 0.5  # query covers a period but grain unspecified
        return 0.0      # point-in-time or granularity irrelevant
```

**Harm-weighted aggregate score:**

Not all dimensions are equally consequential. Weight by the impact of getting it wrong:

```python
HARM_WEIGHTS = {
    "metric":       2.0,  # wrong metric → wrong business decision
    "time_period":  1.5,  # wrong time → potentially large numerical error
    "entity_scope": 1.2,  # wrong entity → plausible but wrong subset
    "grouping":     0.8,  # wrong grouping → reorganised but data is there
    "comparison":   0.7,
    "granularity":  0.5,  # wrong granularity → correct data, wrong resolution
}

def weighted_ambiguity_score(report: AmbiguityReport) -> float:
    return sum(
        getattr(report, dim) * weight
        for dim, weight in HARM_WEIGHTS.items()
    )

# Threshold: > 2.0 requires clarification, 1.0–2.0 uses default + annotation, < 1.0 proceeds
```

### Schema-Grounded Ambiguity

Pure language analysis misses ambiguity that only exists relative to the schema. Two examples:

**Example 1 — Column collision:**
"Show me the status of open deals" — unambiguous in natural language. But the schema has:
- `opportunities.status` (sales stage: Prospect, Demo, Proposal, Closed)
- `opportunities.deal_status` (financial status: Open, Invoiced, Paid)

Both columns exist. Both are "status". Neither name matches "status" exactly. The query is schema-ambiguous even though it is language-clear.

**Detection:** After the NL analysis, run schema retrieval and check whether the retrieved tables have multiple columns that could satisfy the same semantic role in the query. If `resolve_column("status", table="opportunities")` returns > 1 candidate, flag it as schema-grounded ambiguity.

**Example 2 — Table collision:**
"Show me the top customers" — but the schema has:
- `customers` (active accounts)
- `customer_archive` (churned accounts)
- `prospect_customers` (pipeline, not yet converted)

Which population does "customers" mean? Language is clear; schema creates the ambiguity.

**Detection:** When schema retrieval returns multiple tables matching the same entity term, and those tables represent different populations (not just different aspects), flag as ambiguous. Heuristic: if two tables share a name prefix and both have a `customer_id` or `id` column, they likely represent different populations of the same entity.

---

### What to Do With an Ambiguous Query

Three responses, chosen by weighted ambiguity score and user persona:

**Score < 1.0 — Proceed with defaults, annotate:**
```
User: "Show me revenue last year"
System: Showing total net revenue for calendar year 2024.
        [Switch to: fiscal year 2024 | gross revenue | by month]
```
Annotation appears inline, above the result. Quick-switch links let the user correct assumptions without re-typing.

**Score 1.0–2.0 — Ask one clarifying question (highest-weight dimension only):**
```
User: "Show me revenue numbers"
System: I can pull the revenue data — which metric would you like?
        [Total revenue] [ARR] [Net revenue] [Gross revenue]
```
Only the highest-harm ambiguous dimension is asked about. Everything else defaults and is annotated in the result. Do NOT ask about all dimensions at once.

**Score > 2.0 — Full clarification before proceeding:**
```
User: "Show me the numbers"
System: I need a bit more detail to pull the right data:
        1. What metric? [Revenue] [Orders] [Customers] [Something else]
```
Even here: ask one question, not five. The user answers, which resolves the top dimension; re-score with the new context. If score is now < 2.0, proceed with defaults for the rest.

---

## Out-of-Scope Detection — Concrete Mechanism

Four distinct out-of-scope patterns, each with a different detection mechanism and response.

### Fully Unrelated Queries

Queries with no analytical intent and no connection to business data.

**Examples:**
- "What is the capital of France?"
- "Write me a Python function to sort a list"
- "Who is the CEO of Apple?"
- "Tell me a joke"

**Detection — embedding distance:**
Embed the query. Compute cosine similarity to the centroid of the tenant's known analytical query embeddings. If similarity < 0.25, the query is in a completely different semantic space — flag as OUT_OF_SCOPE.

Supplement with a keyword blocklist for common patterns: general knowledge questions (who, what is, define, explain), code generation requests, creative requests.

**Response:**
```
System: I can only answer questions about your business data.
        This question is outside what I can help with here.
```
Short, direct, no apology. Do not attempt to generate SQL.

---

### Entity Mismatch — Query References Data the Tenant Doesn't Have

The query is analytically intended but references an entity, dimension, or metric that doesn't exist in the tenant's schema.

**Examples (for a US-only SaaS company):**
- "Show me revenue by country" — no geography dimension in schema
- "What is our NPS score this quarter?" — no NPS data in the database
- "Show me inventory levels" — not an inventory business

**Detection — schema coverage check:**

After intent classification confirms ANALYTICAL intent, run a fast schema coverage check before full retrieval:

```python
def check_schema_coverage(query: str, schema_index: SchemaIndex) -> CoverageResult:
    # Extract key concepts from the query
    concepts = extract_concepts(query)  # ["country", "revenue", "geography"]
    
    # Check each concept against schema index
    unresolvable = []
    for concept in concepts:
        matches = schema_index.find(concept, similarity_threshold=0.4)
        if not matches:
            unresolvable.append(concept)
    
    if len(unresolvable) / len(concepts) > 0.5:
        return CoverageResult(
            status="OUT_OF_SCOPE",
            missing_concepts=unresolvable,
            message=f"Your database doesn't appear to contain data about: {', '.join(unresolvable)}"
        )
    
    if unresolvable:
        return CoverageResult(
            status="PARTIAL_SCOPE",
            missing_concepts=unresolvable,
        )
    
    return CoverageResult(status="IN_SCOPE")
```

**The important threshold:** `similarity_threshold=0.4` is intentionally low — you want high recall here (don't miss out-of-scope queries). False positives (flagging in-scope queries as out-of-scope) are easily handled by the user saying "actually, I meant X." False negatives (missing out-of-scope queries) generate SQL that executes against the wrong or nonexistent data.

**Response for entity mismatch:**
```
User: "Show me revenue by country"
System: Your database doesn't appear to have a geographic breakdown for revenue.
        I can show you revenue by: [region] [product line] [sales rep] [customer segment]
        
        Or did you mean something different by "country"?
```

Offer what IS available as alternatives. Users often ask for a dimension by the wrong name.

---

### Partial Scope — Mixed Analytical and Conversational

The query contains both an answerable analytical component and a component that cannot be answered by SQL.

**Examples:**
- "Show me our Q3 revenue and also explain why we might have underperformed"
- "Who are our top customers and what do you think we should do about retention?"
- "Give me the sales numbers for last month and compare to industry benchmarks"

**Detection:** After intent classification returns ANALYTICAL, check for conjunctive structure that combines a data request with an open-ended or external question. Patterns:
- "... and also [explain/suggest/recommend/compare to industry/...]"
- "... and what should we do about ..."
- "... vs industry average / competitor / market"

**Response — answer the answerable part:**
```
User: "Show me Q3 revenue and why we underperformed"
System: Here is Q3 revenue by product line:
        [result table]
        
        Note: I can show you historical data but can't analyze the reasons for performance — 
        that would need context beyond what's in your database.
        
        Related data that might help: [YoY comparison] [by sales rep] [deal stage breakdown]
```

Never refuse the entire query because part of it is out of scope. Answer the answerable part and clearly delineate what you couldn't answer.

---

### Schema Coverage Miss — Topic Exists but Data Doesn't

The most subtle case. The tenant's business domain includes the concept, but the data has not been loaded, the table exists but is empty, or the relevant time range has no data.

**Examples:**
- "Show me support ticket volume" — `support_tickets` table exists but has 0 rows (data not ingested yet)
- "Show me revenue for 2019" — data only goes back to 2021
- "Show me mobile app usage" — table exists, data collection started last month

**Detection — data availability check:**

After schema retrieval confirms the relevant tables exist, run a quick data availability check before SQL generation:

```python
def check_data_availability(
    tables: list[str],
    query_context: QueryContext,
    db: Database
) -> AvailabilityResult:
    
    for table in tables:
        row_count = db.scalar(f"SELECT COUNT(*) FROM {table}")
        if row_count == 0:
            return AvailabilityResult(
                status="NO_DATA",
                table=table,
                message=f"The {table} table exists but contains no data yet."
            )
        
        # Check date range availability if query has time filter
        if query_context.has_time_filter:
            available_range = db.scalar(
                f"SELECT MIN(created_at), MAX(created_at) FROM {table}"
            )
            if not query_context.time_range_overlaps(available_range):
                return AvailabilityResult(
                    status="DATE_OUT_OF_RANGE",
                    table=table,
                    available_range=available_range,
                    message=f"Data for {table} is only available from {available_range.min} to {available_range.max}."
                )
    
    return AvailabilityResult(status="AVAILABLE")
```

**Response for data availability miss:**
```
User: "Show me support ticket volume for last quarter"
System: The support ticket data is in the system but hasn't been populated yet.
        The table currently has 0 records.
        
        Is there another way I can help? I can show you: [customer feedback] [NPS scores] [escalation data]
```

This is far better than generating SQL that returns an empty result with no explanation — the user would assume the answer is "zero tickets" rather than "the data isn't here."

---

## Putting It Together — The Pre-Generation Gate

Every query passes through this gate in sequence before schema retrieval or SQL generation begins:

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1 — Intent Classification (< 20ms)                    │
│ Output: ANALYTICAL | CONVERSATIONAL | OUT_OF_SCOPE | AMBIGUOUS│
└──────────────────────────────┬──────────────────────────────┘
                               │
           ┌───────────────────┼────────────────────┐
           ▼                   ▼                    ▼
    CONVERSATIONAL         OUT_OF_SCOPE           ANALYTICAL
    Return NL response     Return scope           Continue
                           message                    │
                                                      ▼
                                   ┌──────────────────────────────────┐
                                   │ STAGE 2 — Schema Coverage Check  │
                                   │ (< 30ms, embedding-based)        │
                                   │ Output: IN_SCOPE | PARTIAL |     │
                                   │         OUT_OF_SCOPE             │
                                   └──────────────────┬───────────────┘
                                                      │
                                                 IN_SCOPE
                                                      │
                                                      ▼
                                   ┌──────────────────────────────────┐
                                   │ STAGE 3 — Ambiguity Scoring      │
                                   │ (< 50ms)                         │
                                   │ Weighted score across 6 dims     │
                                   └──────────────────┬───────────────┘
                                                      │
                              ┌───────────────────────┼──────────────────────┐
                              ▼                       ▼                      ▼
                         Score > 2.0           Score 1.0–2.0           Score < 1.0
                         Clarify first         Default + annotate      Proceed to
                                                                       retrieval
                                                      │
                                                      ▼
                                   ┌──────────────────────────────────┐
                                   │ STAGE 4 — Data Availability Check│
                                   │ (< 50ms, after retrieval)        │
                                   │ Table empty? Date range missing?  │
                                   └──────────────────┬───────────────┘
                                                      │
                                                      ▼
                                               SQL Generation
```

**Total gate latency: < 100ms** for stages 1–3. Stage 4 runs after retrieval (adds 30–50ms). The LLM classifier in Stage 1 is only invoked for novel queries — the common case (embedded similarity) completes in < 20ms.

---

## Concrete Examples With Classifications

| Query | Intent | Ambiguity score | Scope result | Action |
|-------|--------|-----------------|--------------|--------|
| "Show me Q3 revenue by region" | ANALYTICAL | 0.3 | IN_SCOPE | Proceed to generation |
| "Show me the numbers" | ANALYTICAL | 3.8 | IN_SCOPE | Clarify: what metric? |
| "Revenue last year" with fiscal≠calendar | ANALYTICAL | 1.6 | IN_SCOPE | Default to calendar, annotate |
| "Show me revenue by country" (no geo dim) | ANALYTICAL | 0.2 | OUT_OF_SCOPE (entity mismatch) | "No geography data, here's what we have" |
| "Who won the Super Bowl?" | CONVERSATIONAL | — | OUT_OF_SCOPE | "I can only query your business data" |
| "Show me revenue and explain the dip" | ANALYTICAL | 0.5 | PARTIAL_SCOPE | Answer data part, note limit |
| "Support ticket volume Q2" (empty table) | ANALYTICAL | 0.1 | IN_SCOPE → no data | "Table exists but has 0 rows" |
| "Show me churn for top customers" | ANALYTICAL | 1.2 | IN_SCOPE | Clarify: define "top" customers |
| "partners" (one word) | AMBIGUOUS | 4.5 | IN_SCOPE | Ask: what do you want to know about partners? |
| "revenue vs industry benchmark" | ANALYTICAL | 0.4 | PARTIAL_SCOPE | Show internal revenue, note benchmark unavailable |

---

## What to Return to the User for Each Case

**CONVERSATIONAL:**
Natural language response. No SQL. No apology. Just answer the question or redirect.

**OUT_OF_SCOPE — fully unrelated:**
```
"I can only answer questions about your business data in [system name]."
```
One sentence. No hedging.

**OUT_OF_SCOPE — entity mismatch:**
```
"Your database doesn't have [X]. I can show you [closest available alternatives]."
```
Always offer alternatives. Never a dead end.

**OUT_OF_SCOPE — data not available:**
```
"The [table] data is available from [start date] to [end date]. Your query asks for [requested range]."
```
State what IS available. Let the user decide whether to query the available range.

**AMBIGUOUS — high score:**
```
"I need one more detail: [single highest-harm dimension question]"
[Option A] [Option B] [Option C] [Something else]
```
One question. Clickable options derived from the schema. Not a free-text prompt.

**AMBIGUOUS — medium score:**
```
[Result table]
Showing [metric] for [time period]. [Change ▾]
```
Result first, assumption annotation second. Never block the user — let them correct if needed.

---

## Common Mistakes

**Mistake 1 — Running ambiguity detection after SQL generation:**
If you detect ambiguity only after the LLM has already generated SQL (and the SQL happens to pick one interpretation), you either discard valid work or let a wrong assumption through. Run ambiguity scoring before the LLM call.

**Mistake 2 — Schema coverage check is query-only, not schema-grounded:**
"Show me NPS scores" is only out-of-scope if the tenant has no NPS data. A query-only classifier would flag it as analytical (because NPS is a business term). A schema-grounded check catches it because `nps_scores` table doesn't exist.

**Mistake 3 — Asking all ambiguous dimensions at once:**
"I need to know: what metric, what time period, which region, and what granularity?" is a UX failure. Users abandon. Ask the single most consequential question and default the rest.

**Mistake 4 — Treating empty results as out-of-scope:**
A query that returns zero rows is not out-of-scope — it is a valid result. The data availability check is specifically for tables with zero rows or date ranges with no data, not for filters that happen to match nothing.

**Mistake 5 — Using the same out-of-scope message for all cases:**
"I can't answer that" is unhelpful for entity mismatch (the user might mean something different) and confusing for data availability issues (the user thinks the data should exist). Tailor the message to the specific reason.

**Mistake 6 — No logging of rejected queries:**
Every query that is classified as OUT_OF_SCOPE or AMBIGUOUS is a signal. Log them. If 20% of queries ask about "country" and you have no geography data, that is a schema gap worth filling — or a UI hint that should tell users what data is available before they ask.
