# NL2SQL Eval Pipeline Design

How to build and operate an evaluation pipeline for a production NL2SQL system — not just what to measure, but how to architect the system that measures it continuously.

## Table of Contents

- [What the Existing Coverage Misses](#what-the-existing-coverage-misses)
- [Eval Pipeline Architecture](#eval-pipeline-architecture)
- [Component-Level Evaluation](#component-level-evaluation)
  - [Schema Retrieval Eval](#schema-retrieval-eval)
  - [SQL Generation Eval](#sql-generation-eval)
  - [End-to-End Eval](#end-to-end-eval)
- [Test Dataset Design and Management](#test-dataset-design-and-management)
- [Adversarial Test Cases — Concrete Examples](#adversarial-test-cases--concrete-examples)
- [Continuous Evaluation and Regression Detection](#continuous-evaluation-and-regression-detection)
- [Human-in-the-Loop Evaluation](#human-in-the-loop-evaluation)
- [Sandbox and Pilot Customer Testing](#sandbox-and-pilot-customer-testing)
  - [Sandbox Environment Design](#sandbox-environment-design)
  - [Shadow Mode Testing](#shadow-mode-testing)
  - [Pilot Customer Program](#pilot-customer-program)
  - [What to Measure During the Pilot](#what-to-measure-during-the-pilot)
  - [Graduation Criteria — Pilot to Full Rollout](#graduation-criteria--pilot-to-full-rollout)
- [Metrics Tracking and Alerting](#metrics-tracking-and-alerting)
- [Eval in CI/CD](#eval-in-cicd)
- [Common Mistakes](#common-mistakes)

---

## What the Existing Coverage Misses

The accuracy section (`03_accuracy_evaluation.md`) answers: what metrics to use, how to build a ground-truth set, how to detect semantic errors, and what accuracy threshold means for shipping.

This file answers the engineering question: **how do you build a system that runs evaluation continuously, detects regressions automatically, separates component failures from end-to-end failures, and integrates with your deployment pipeline?**

---

## Eval Pipeline Architecture

The eval pipeline runs in three modes:

```
┌────────────────────────────────────────────────────────────────┐
│  Mode 1: Pre-deployment gate (runs in CI on every PR)          │
│  Mode 2: Scheduled regression run (daily, against production)  │
│  Mode 3: Shadow eval (every production query, async)           │
└────────────────────────────────────────────────────────────────┘
```

**Physical architecture:**

```
Test Dataset Store (S3 / git-lfs)
        │
        ▼
Eval Runner (Python service)
   ├── Schema Retrieval Evaluator      → recall@k per query
   ├── SQL Generation Evaluator        → parse success, hallucination rate, result equivalence
   ├── End-to-End Evaluator            → execution accuracy, semantic check, latency
   └── Human Review Queue (async)      → low-confidence results flagged for labelling
        │
        ▼
Metrics Store (Prometheus / BigQuery)
        │
        ▼
Dashboard (Grafana) + Alerting (PagerDuty)
```

The eval runner is a standalone service — it does not share code with the production NL2SQL pipeline. This prevents the eval from being accidentally fixed by the same change that introduces a bug.

---

## Component-Level Evaluation

Evaluate each stage of the pipeline independently. End-to-end accuracy conflates retrieval failures with generation failures — you cannot improve what you cannot isolate.

### Schema Retrieval Eval

**What it measures:** Given a natural language query, does the schema retrieval step surface the correct tables within the top-k results?

**Golden dataset structure:**
```json
{
  "query": "Show me the monthly churn rate for enterprise customers",
  "required_tables": ["customers", "subscription_events", "customer_segments"],
  "required_columns": {
    "subscription_events": ["event_type", "event_date", "customer_id"],
    "customer_segments": ["segment_name"]
  },
  "difficulty": "hard",
  "tags": ["multi-join", "churn", "date-aggregation"]
}
```

**Metrics:**
- `recall@k` for k ∈ {5, 10, 15}: percentage of test cases where all required tables appear in the top-k results. Track per difficulty tier.
- `mean_rank` of the required table set: average rank position of the lowest-ranked required table. Lower is better.
- `false_positive_rate`: percentage of retrieved tables that are not required. High false positive rate wastes context window and confuses SQL generation.

**Example eval run output:**
```
Schema Retrieval Eval — 2024-03-15
──────────────────────────────────
recall@5:  0.61  (↓ from 0.65 last week — REGRESSION)
recall@10: 0.79
recall@15: 0.88

By difficulty:
  easy:   recall@5 = 0.91
  medium: recall@5 = 0.68
  hard:   recall@5 = 0.32  ← hard queries are the bottleneck

Top missed tables:
  subscription_events (missed in 41% of churn-related queries)
  order_items (missed in 28% of revenue attribution queries)

Action: Enrich 'subscription_events' embedding with synonyms: 
        'cancellation', 'churn event', 'customer lifecycle'
```

**Regression threshold:** Alert if recall@10 drops > 3pp week-over-week or > 5pp vs. the 30-day baseline.

---

### SQL Generation Eval

**What it measures:** Given a query and the ground-truth schema context (bypassing retrieval), how well does the SQL generation step perform? This isolates generation quality from retrieval quality.

**Why inject ground-truth schema:** If retrieval is wrong and generation is wrong, you don't know which one to fix. By feeding the correct schema directly, you get a clean signal on generation quality.

**Metrics:**

*Parse success rate:* Percentage of generated SQL that parses without error. Should be > 98%. Anything below signals a fundamental prompt or model problem.

*Hallucination rate:* Percentage of generated SQL that references a table or column not present in the injected schema. Should be < 2%.

*Structural correctness:* Rules-based AST check for alignment between NL intent and SQL structure:
```python
def check_structural_correctness(nl_query: str, sql_ast: AST) -> list[str]:
    violations = []
    
    # "total X" → must have SUM or COUNT aggregation
    if "total" in nl_query.lower() and not has_aggregation(sql_ast, ["SUM", "COUNT"]):
        violations.append("missing_aggregation_for_total")
    
    # "by X" → must have GROUP BY on a column matching X
    if "by region" in nl_query.lower() and not has_group_by(sql_ast, "region"):
        violations.append("missing_group_by_for_by_clause")
    
    # "last quarter" → must have a date filter
    if "last quarter" in nl_query.lower() and not has_date_filter(sql_ast):
        violations.append("missing_date_filter_for_time_expression")
    
    return violations
```

*Result set equivalence on adversarial data:* Run both ground-truth SQL and generated SQL on a synthetic database designed to expose common errors (see adversarial test cases below). Compare result sets.

**Example failing case:**
```
Query: "Customers who placed more than 3 orders"
Ground truth SQL: SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 3
Generated SQL:    SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) >= 3

Exact match: FAIL
Execution on standard test DB (customers with 1,2,3,4,5 orders): PASS (both return same rows)
Execution on adversarial DB (one customer with exactly 3 orders): FAIL
  Ground truth returns: 0 rows
  Generated SQL returns: 1 row (the customer with exactly 3 orders)
```

This is why adversarial test databases matter. Standard test databases mask off-by-one errors.

---

### End-to-End Eval

**What it measures:** The full pipeline with real retrieval, real generation, real execution against a test database.

**Test database requirements:**
- Representative schema (same structure as production)
- Synthetic data that is specifically designed to make wrong queries fail (not just convenient test data)
- Data that covers edge cases: NULL values, empty FK relationships, zero-row result sets that should be non-empty, boundary values for date filters

**End-to-end metrics:**
- `execution_accuracy`: percentage of queries that execute without error
- `result_equivalence`: percentage where result set matches ground truth (on the adversarial test DB)
- `latency_p50/p95/p99`: percentile latency of the full pipeline per query complexity tier
- `confidence_calibration`: does a confidence score of 0.8 correspond to ~80% accuracy? Plot confidence vs. accuracy bucketed by score range.

---

## Test Dataset Design and Management

### Structure

Store the dataset as versioned JSON files in git (or git-lfs for large datasets):

```
eval/
├── datasets/
│   ├── golden_set_v1.json          # Original curated set
│   ├── golden_set_v2.json          # v2 adds production-derived queries
│   └── golden_set_current -> v2    # Symlink to active version
├── adversarial/
│   ├── boundary_conditions.json    # Off-by-one, NULL handling
│   ├── ambiguity_cases.json        # Vague queries
│   ├── schema_linking_hard.json    # Semantically distant tables
│   └── multi_join_complex.json     # 4+ table joins
├── databases/
│   ├── standard_test.sql           # Standard test database
│   └── adversarial_test.sql        # Adversarial test database
└── schemas/
    └── schema_v*.json              # Versioned schema snapshots
```

### Test case schema

Every test case must have:

```json
{
  "id": "tc_0042",
  "query": "Show me revenue by product category for Q3 2024",
  "ground_truth_sql": "SELECT p.category, SUM(o.amount) AS revenue FROM orders o JOIN products p ON o.product_id = p.id WHERE o.created_at BETWEEN '2024-07-01' AND '2024-09-30' GROUP BY p.category ORDER BY revenue DESC",
  "required_tables": ["orders", "products"],
  "required_columns": ["orders.amount", "orders.created_at", "products.category"],
  "difficulty": "medium",
  "query_type": ["aggregation", "multi-join", "date-filter"],
  "expected_row_count_range": [1, 50],
  "adversarial_db_expected_result": [
    {"category": "Electronics", "revenue": 45000},
    {"category": "Apparel", "revenue": 23000}
  ],
  "known_failure_modes": ["wrong date boundary", "joining to wrong products table"],
  "added_by": "domain_expert",
  "added_date": "2024-01-15",
  "production_frequency": "high"
}
```

### Dataset versioning and growth strategy

- **Start:** 100–200 expert-curated queries before launch
- **After 30 days:** Add production-derived queries (from the human review queue) — target 50–100 new cases per month
- **Ongoing:** Weight by production query frequency. If 30% of production queries ask about revenue, 30% of the eval set should too.
- **Never delete test cases** — deprecate them by setting `"active": false`. This preserves the history of what the system could handle at each point in time.
- **Version the dataset** when the schema changes — a test case written for schema v1 must be re-validated after schema v2.

---

## Adversarial Test Cases — Concrete Examples

These are the test cases that distinguish a mature eval pipeline from a superficial one. Each targets a specific, common failure mode.

### 1. Off-by-one in boundary conditions

```json
{
  "query": "Show customers who placed exactly 3 orders",
  "ground_truth_sql": "SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) = 3",
  "adversarial_db": "Contains customers with 2, 3, and 4 orders",
  "why_adversarial": "Tests COUNT(*) = 3 vs >= 3 vs > 2 — all wrong alternatives return the same result on most test data",
  "common_wrong_sql": "HAVING COUNT(*) >= 3"
}
```

### 2. Date range boundary — inclusive vs exclusive

```json
{
  "query": "Revenue for Q1 2024 (Jan 1 to Mar 31)",
  "ground_truth_sql": "WHERE created_at >= '2024-01-01' AND created_at < '2024-04-01'",
  "adversarial_db": "Contains one order timestamped exactly '2024-03-31 23:59:59' and one at '2024-04-01 00:00:01'",
  "why_adversarial": "LLMs commonly generate BETWEEN '2024-01-01' AND '2024-03-31' which excludes datetime values on Mar 31 after midnight",
  "common_wrong_sql": "WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31'"
}
```

### 3. NULL handling in aggregations

```json
{
  "query": "Average order value per customer",
  "ground_truth_sql": "SELECT customer_id, AVG(amount) FROM orders WHERE amount IS NOT NULL GROUP BY customer_id",
  "adversarial_db": "Contains orders with NULL amount values for cancelled orders",
  "why_adversarial": "AVG() ignores NULLs by default, but if the question implies 'including cancelled', the semantics are different. Tests whether the LLM handles NULL correctly for the specific intent.",
  "common_wrong_sql": "SELECT customer_id, SUM(amount) / COUNT(*) FROM orders GROUP BY customer_id"
}
```

### 4. Semantically equivalent tables with different meanings

```json
{
  "query": "Show the top 10 customers by lifetime revenue",
  "schema_trap": "Database has both 'customers' (current active) and 'customer_archive' (churned). Both have revenue columns.",
  "ground_truth_sql": "Uses UNION ALL across both tables or a view that includes all customers",
  "adversarial_condition": "customer_archive has 40% of total revenue",
  "why_adversarial": "LLM often picks only 'customers' table, silently excluding 40% of revenue — results look plausible but are systematically wrong",
  "expected_detection": "Result row count on adversarial DB differs between correct and wrong SQL"
}
```

### 5. Aggregation scope error — global vs partitioned

```json
{
  "query": "For each region, show the percentage of orders that were returned",
  "ground_truth_sql": "SELECT region, SUM(CASE WHEN returned THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS return_rate FROM orders GROUP BY region",
  "common_wrong_sql": "SELECT region, COUNT(*) FILTER (WHERE returned) * 100.0 / (SELECT COUNT(*) FROM orders) FROM orders GROUP BY region",
  "why_wrong": "The wrong SQL divides by total orders globally, not orders per region. On a dataset where regions have equal order counts it passes; on unequal distribution it fails.",
  "adversarial_db": "West has 1000 orders, East has 100 orders. Same 10% return rate. Wrong SQL returns East at 1%, West at 10%."
}
```

### 6. JOIN directionality — LEFT vs INNER

```json
{
  "query": "Show all customers and their total order value (include customers with no orders)",
  "ground_truth_sql": "SELECT c.id, c.name, COALESCE(SUM(o.amount), 0) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
  "common_wrong_sql": "SELECT c.id, c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
  "why_wrong": "INNER JOIN silently drops customers with no orders. On a standard test DB where every customer has at least one order, both queries return identical results.",
  "adversarial_db": "10% of customers have never placed an order.",
  "detection": "Correct SQL returns N rows (all customers), wrong SQL returns N*0.9 rows — easy to detect by row count comparison."
}
```

### 7. Schema linking — column in unexpected table

```json
{
  "query": "Show the email address of customers who churned last month",
  "schema_trap": "Email is stored in 'contact_info' table, not 'customers' table. Churn status is in 'subscription_events'.",
  "ground_truth_sql": "SELECT ci.email FROM customers c JOIN contact_info ci ON c.id = ci.customer_id JOIN subscription_events se ON c.id = se.customer_id WHERE se.event_type = 'churned' AND se.event_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
  "common_wrong_sql": "SELECT email FROM customers WHERE churned_date >= ...",
  "why_wrong": "LLM assumes 'email' is in 'customers' table. Generates SQL that fails with column-not-found — detectable. The harder failure: if 'customers' has a stale 'email' column from a legacy schema, the query executes but returns outdated emails."
}
```

### 8. Fiscal vs calendar year

```json
{
  "query": "Show revenue for last year",
  "ambiguity": "Company fiscal year runs April 1 to March 31. 'Last year' is ambiguous.",
  "ground_truth_sql_calendar": "WHERE created_at BETWEEN '2023-01-01' AND '2023-12-31'",
  "ground_truth_sql_fiscal": "WHERE created_at BETWEEN '2023-04-01' AND '2024-03-31'",
  "why_adversarial": "Both queries execute cleanly. The difference is $4M in revenue. Tests whether the system surfaces the ambiguity or silently picks one.",
  "expected_behavior": "System should ask clarifying question or annotate assumption clearly"
}
```

### 9. Window function vs GROUP BY confusion

```json
{
  "query": "For each month, show revenue and the running total revenue year-to-date",
  "ground_truth_sql": "SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS monthly_revenue, SUM(SUM(amount)) OVER (PARTITION BY DATE_TRUNC('year', created_at) ORDER BY DATE_TRUNC('month', created_at)) AS ytd_revenue FROM orders GROUP BY 1",
  "common_wrong_sql": "SELECT DATE_TRUNC('month', created_at), SUM(amount), (SELECT SUM(amount) FROM orders o2 WHERE DATE_TRUNC('year', o2.created_at) = DATE_TRUNC('year', orders.created_at) AND o2.created_at <= orders.created_at) FROM orders GROUP BY 1",
  "why_wrong": "The subquery approach is semantically correct but O(n²) — it will time out on large tables. Tests whether the LLM knows to use window functions for running totals.",
  "detection": "Both produce correct results on small datasets. EXPLAIN plan shows the performance difference."
}
```

### 10. Multi-step filter — users who did X but not Y

```json
{
  "query": "Show customers who placed an order in 2023 but have not placed an order in 2024",
  "ground_truth_sql": "SELECT customer_id FROM orders WHERE YEAR(created_at) = 2023 EXCEPT SELECT customer_id FROM orders WHERE YEAR(created_at) = 2024",
  "common_wrong_sql_1": "SELECT customer_id FROM orders WHERE YEAR(created_at) = 2023 AND customer_id NOT IN (SELECT customer_id FROM orders WHERE YEAR(created_at) = 2024)",
  "common_wrong_sql_2": "SELECT customer_id FROM orders WHERE YEAR(created_at) IN (2023) AND YEAR(created_at) NOT IN (2024)",
  "why_wrong_2": "SQL_2 is logically impossible — a row cannot have created_at in both 2023 and 2024. Returns correct result by accident on a single-row-per-customer table but fails on multi-order tables.",
  "adversarial_db": "100 customers ordered in both 2023 and 2024. 50 ordered only in 2023. 50 ordered only in 2024."
}
```

---

## Continuous Evaluation and Regression Detection

### What triggers an eval run

| Trigger | What runs | Latency requirement |
|---------|-----------|-------------------|
| PR opened against main | Schema retrieval eval + SQL gen eval on golden set (fast subset) | < 10 min |
| Merge to main | Full eval pipeline including end-to-end | < 30 min |
| Daily scheduled job | Full eval + comparison to 7-day rolling baseline | No hard limit |
| Schema change detected | Retrieval eval only, scoped to changed tables | < 5 min |
| Model version update | Full eval + A/B comparison to current model | < 60 min |

### Regression detection logic

```python
class RegressionDetector:
    THRESHOLDS = {
        "retrieval_recall@10":     {"absolute_drop": 0.03, "relative_drop": 0.05},
        "execution_accuracy":      {"absolute_drop": 0.02, "relative_drop": 0.03},
        "hallucination_rate":      {"absolute_increase": 0.01},
        "parse_success_rate":      {"absolute_drop": 0.01},
        "latency_p99_ms":          {"absolute_increase": 500},
    }
    
    def detect(self, baseline: Metrics, candidate: Metrics) -> list[Regression]:
        regressions = []
        for metric, thresholds in self.THRESHOLDS.items():
            delta = candidate[metric] - baseline[metric]
            if "absolute_drop" in thresholds and delta < -thresholds["absolute_drop"]:
                regressions.append(Regression(metric, delta, severity="blocking"))
            if "absolute_increase" in thresholds and delta > thresholds["absolute_increase"]:
                regressions.append(Regression(metric, delta, severity="blocking"))
        return regressions
```

A blocking regression fails the CI gate. A warning regression goes into the deployment notes but does not block.

### Segmented regression tracking

Track metrics per segment, not just in aggregate. An overall accuracy improvement can mask a regression on a critical query class:

```
Eval run: v1.4.2 vs v1.4.1
──────────────────────────────────────────────────
Overall execution accuracy:  +1.2pp  ✓
By query type:
  single-table:              +0.5pp  ✓
  aggregation:               +3.1pp  ✓
  multi-join:                -2.8pp  ⚠ REGRESSION
  date arithmetic:           +0.9pp  ✓
By schema domain:
  finance:                   -4.1pp  ⚠ BLOCKING REGRESSION
  sales:                     +2.3pp  ✓
  hr:                        +1.0pp  ✓

Decision: DO NOT DEPLOY — finance domain regression is blocking
```

---

## Human-in-the-Loop Evaluation

Automated metrics cannot detect all failure classes. Build a structured human review process.

### What goes into human review

1. **All queries below confidence threshold** (confidence < 0.6) — high prior probability of error
2. **Queries where generated SQL and ground-truth SQL diverge** but result sets match — may indicate SQL is correct by coincidence
3. **Queries from new query patterns** not yet represented in the golden set — review to decide if they should be added
4. **Randomly sampled 2% of all production queries** — statistical baseline for actual production accuracy
5. **All queries flagged by implicit feedback signals** (immediate re-query, SQL edit)

### Review task design

Show the reviewer:
- The original natural language query
- The generated SQL (formatted, with table names highlighted)
- The result set (first 10 rows)
- The ground-truth SQL if available (for golden set reviews)

Ask three binary questions:
1. Does the SQL correctly answer the question? (yes/no)
2. If no: is the error in table selection, column selection, filtering logic, aggregation, or join type?
3. If yes: is there a simpler or more efficient SQL that would be equivalent?

Target: 2 independent reviewers per query, with disagreement resolution by a third reviewer. Track inter-reviewer agreement — a Kappa < 0.7 on a query type means the question is too ambiguous to reliably evaluate.

---

## Sandbox and Pilot Customer Testing

Automated eval on a golden set tells you how well the system performs on curated queries against a synthetic database. Sandbox and pilot testing tells you how it performs on **real queries from real people against real data** — a completely different signal. No eval pipeline is complete without this stage.

### Sandbox Environment Design

The sandbox is a fully operational instance of the NL2SQL system connected to a **copy of production data**, isolated from the production system. It is not a test database with synthetic data — it is real data in a separate environment.

**Why a copy of production data and not synthetic data:**
- Real schema complexity: production schemas have 15 years of organic growth, inconsistent naming, and nullable columns that synthetic data doesn't replicate
- Real data distributions: aggregation queries that return sensible numbers on production may return $0 or 10^9 on synthetic data, making quality hard to judge
- Real edge cases: NULL concentrations, duplicate rows, orphaned FK references — these only exist in real data and they expose bugs synthetic data cannot

**Sandbox infrastructure requirements:**

```
Sandbox Environment
├── NL2SQL Service         ← identical binary to production, different config
├── Schema Registry        ← copy of production schema metadata, refreshed daily
├── Vector Store           ← copy of production embeddings, refreshed on schema change
├── Read-only DB replica   ← daily snapshot of production, PII masked
│     └── PII masking:     names → synthetic names, emails → fake@example.com,
│                          phone → 000-000-0000, SSN → XXX-XX-XXXX
├── Eval Recorder          ← logs every query, generated SQL, result, latency
└── Human Review UI        ← internal tool for reviewing sandbox sessions
```

**PII masking is non-negotiable.** Never expose real customer names, emails, or sensitive identifiers to internal testers or pilot customers. Use a masking library (Faker, Presidio) that applies consistent masking — the same real name always maps to the same fake name, so JOIN relationships are preserved and the data remains analytically coherent.

**Sandbox refresh schedule:**
- Schema: refresh whenever production schema changes (event-driven)
- Data: daily snapshot at 2am, masking applied before copy
- Embeddings: refresh after every schema change and after every model update

---

### Shadow Mode Testing

Before any pilot customer sees a single result, run the new system version in **shadow mode** against production traffic for 48–72 hours.

**How shadow mode works:**

```
Production request
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
Current production system                   Shadow (new) system
  - generates SQL                             - generates SQL
  - executes query                            - executes query (read-only)
  - returns result to user                   - result goes to eval recorder only
                                             - user never sees it
```

The user interacts with the current system. The shadow system runs in parallel, asynchronously, without affecting the user experience or adding latency to the production path.

**What shadow mode catches that CI eval cannot:**

- **Real query distribution:** production users ask things that never appear in the golden set. Shadow mode surfaces these immediately.
- **Schema-specific failures:** a column that exists in your test schema but not production, or vice versa. CI runs against a test database; shadow runs against the real schema.
- **Latency under real load:** CI runs sequentially against a quiet database. Shadow runs against production load patterns — you see the real p99 under concurrent requests.
- **Cold start issues:** model loading, vector index warm-up, cache miss patterns on real query distribution.

**Shadow mode eval metrics:**

```
Shadow Mode Report — New Model v1.4.2 vs Production v1.4.1
Duration: 72 hours | Shadow queries: 14,847

SQL Parse Success:           99.2% vs 99.0%  ✓
Table hallucination rate:     0.9% vs 1.1%   ✓ (improvement)
Result equivalence (sampled): 84.1% vs 82.6% ✓
Latency p50:                  920ms vs 890ms  ⚠ +30ms
Latency p99:                2,810ms vs 2,640ms ⚠ +170ms

Query patterns in shadow not in golden set:
  - "Compare X to last year same period" (87 occurrences, 71% correct)
  - "Top N by X excluding Y" (43 occurrences, 64% correct) ← ADD TO GOLDEN SET
  - Queries with company-specific acronyms (29 occurrences, 41% correct) ← ENRICH GLOSSARY

Decision: Latency increase acceptable. Proceed to limited pilot.
          Add 43 "excluding" pattern queries to golden set before next release.
```

**Shadow mode exit criteria before pilot:**
- Parse success rate ≥ production baseline
- Hallucination rate ≤ production baseline
- Latency p99 within +200ms of production baseline
- No new systematic failure patterns identified

---

### Pilot Customer Program

The pilot is a **structured, consent-based rollout to a small, selected set of real users** who interact with the system as they normally would, knowing they are in a pilot.

**Pilot size and selection:**

Start with 5–10 users. Selection criteria:

| Criterion | Why |
|-----------|-----|
| Power users (high query volume) | Maximises signal per day of pilot |
| Domain diversity (finance + ops + sales) | Ensures cross-domain coverage |
| SQL-literate users | They can evaluate SQL correctness independently, giving you richer feedback |
| Users with known, recurring query patterns | You can pre-verify their common queries work correctly before they do |
| Users who have explicitly agreed to give feedback | Consent is required — users who don't know they're in a pilot give no feedback |

**What pilot users should NOT be:**
- Your most important executive stakeholders (if the system fails during their pilot session it creates trust issues disproportionate to the technical failure)
- Users with access to the most sensitive data (limit blast radius of any accidental data exposure)
- Users who will immediately share results externally (financial reports sent to investors, regulatory submissions)

**Pilot onboarding checklist:**

```
Before pilot user's first session:
  ☐ Verify their most common 10 queries work correctly in sandbox
  ☐ Confirm PII masking is complete for their data domain
  ☐ Set up their session in the eval recorder
  ☐ Brief them on what they're testing and how to flag issues
  ☐ Give them a direct feedback channel (Slack DM or email to the team)
  ☐ Confirm rollback is ready and takes < 5 minutes
```

**Pilot user experience:**

The pilot user should experience the system as a real product, not a test harness. Do not ask them to fill out a survey after every query — that destroys the natural usage pattern. Instead:

- One optional "report a problem" button per result (low friction)
- A weekly 15-minute conversation with the team to discuss what surprised them
- A Slack channel where they can drop screenshots of anything unexpected

**Running the pilot in phases:**

```
Phase 1 — Observe only (week 1):
  Pilot users use the system. Team watches the eval recorder.
  No changes to the system during this phase.
  Goal: understand the real query distribution and where the system struggles.

Phase 2 — Targeted fixes (week 2):
  Fix the top 3 failure patterns identified in Phase 1.
  Re-run those query patterns in the pilot.
  Goal: verify the fixes work on real queries, not just synthetic ones.

Phase 3 — Expanded pilot (week 3–4):
  Expand to 20–30 users if Phase 2 exit criteria are met.
  Include at least one user who is not SQL-literate.
  Goal: test the full user experience including error messages, confidence indicators, UI.
```

---

### What to Measure During the Pilot

The pilot produces three types of signal unavailable from automated eval:

**Signal 1 — Real query distribution:**
Log every query (anonymized if needed). After 500 queries, you have a real distribution to compare against your golden set. Queries that appear frequently in production but are absent from the golden set should be added — these are coverage gaps that inflate your benchmark accuracy.

```python
# After pilot, compare distributions
pilot_query_types = classify_queries(pilot_log)
golden_set_distribution = classify_queries(golden_set)

gaps = {
    query_type: pilot_freq - golden_freq
    for query_type, pilot_freq in pilot_query_types.items()
    if pilot_freq - golden_set_distribution.get(query_type, 0) > 0.05
}
# gaps shows query types that are 5%+ more common in production than in golden set
```

**Signal 2 — Qualitative failure taxonomy:**
Have a domain expert review every query the system got wrong during the pilot. Classify each failure:

| Failure class | Example | Fix |
|--------------|---------|-----|
| Schema linking | Wrong column for "revenue" | Enrich schema metadata |
| Business term unknown | "ARR" not in glossary | Add to business glossary |
| Time expression | "Last fiscal year" → calendar year | Add fiscal calendar logic |
| Missing context | "Same as last time" in first query | Improve multi-turn detection |
| Correct SQL, wrong UX | Right result but confusing presentation | UX fix, not model fix |

**Signal 3 — Trust and confidence calibration in real users:**
Ask pilot users: "When the system shows a confidence indicator, do you check the SQL?" If 90% say no, the indicator is not working as a trust calibration tool. If 90% say always, it's creating friction. Target: users check SQL for low-confidence results (< 0.6) and trust high-confidence results (> 0.85) without checking.

Track this by comparing the rate of SQL edits at different confidence score bands:

```
Confidence 0.0–0.4:  SQL edit rate = 42%  ← users correctly skeptical
Confidence 0.4–0.6:  SQL edit rate = 28%  ← reasonable
Confidence 0.6–0.8:  SQL edit rate = 11%  ← reasonable
Confidence 0.8–1.0:  SQL edit rate =  4%  ← users trust high confidence

If high-confidence queries have >10% edit rate, confidence is miscalibrated — recalibrate.
```

---

### Graduation Criteria — Pilot to Full Rollout

Do not graduate to full rollout based on a fixed time period ("we ran the pilot for 2 weeks"). Graduate based on meeting specific, measurable criteria:

```
Required for graduation (all must be met):

Quantitative:
  ☐ Execution accuracy ≥ 85% on pilot queries (measured by eval recorder)
  ☐ User-reported error rate < 10% (from "report a problem" button)
  ☐ SQL edit rate < 15% overall
  ☐ No P0 incidents during pilot (wrong data presented as fact in any external report)
  ☐ Latency p99 < 3s on production-equivalent load
  ☐ Confidence calibration error < 10pp (confidence 0.8 → actual accuracy 70–90%)

Qualitative:
  ☐ At least 3 pilot users have said unprompted that the system saves them time
  ☐ No pilot user has lost trust in the system due to a failure they did not catch
  ☐ Team has classified every failure during the pilot and has a fix or mitigation for each

Coverage:
  ☐ Pilot has covered all major schema domains (finance, sales, ops, HR)
  ☐ At least 500 unique queries processed in the pilot
  ☐ At least one non-SQL-literate user has successfully used the system
```

**Rollout strategy after graduation:**

```
Week 1:  5% of users (random sample, not hand-selected)
Week 2:  20% of users (if week 1 metrics hold)
Week 3:  50% of users (if week 2 metrics hold)
Week 4:  100% of users

At each step: monitor re-query rate, SQL edit rate, latency p99.
Automatic rollback if any metric regresses > 20% relative to the prior step's baseline.
```

**The circuit breaker:** Define in advance what triggers an immediate full rollback from any rollout stage — not a team discussion, an automatic trigger. Example: if re-query rate on any 1-hour window exceeds 25% (2x the baseline), automatically roll back to the previous version and page on-call. Speed of rollback matters more than the trigger threshold — a rollback that takes 20 minutes causes real user harm; one that takes 60 seconds does not.

---

### Metrics to track over time

```
System health metrics (alert on regression):
  - retrieval_recall@10_p50    # median recall across query types
  - execution_accuracy_p50     # median execution accuracy
  - hallucination_rate         # should be < 2%
  - parse_success_rate         # should be > 98%
  - latency_p99_ms             # end-to-end pipeline latency

Quality trend metrics (track but don't alert):
  - accuracy_by_query_type[]   # per-type breakdown
  - accuracy_by_difficulty[]   # easy/medium/hard
  - accuracy_by_schema_domain[]
  - human_review_agreement     # inter-reviewer Kappa

Production behavior metrics:
  - re_query_rate              # implicit negative signal
  - sql_edit_rate              # explicit correction rate
  - export_rate                # implicit positive signal
  - cache_hit_rate             # efficiency metric
  - confidence_calibration_error # |predicted_accuracy - actual_accuracy|
```

### Dashboard layout

```
┌─────────────────────────────────────────────────────────────┐
│ SYSTEM HEALTH               LAST 7 DAYS    TREND            │
│ Execution Accuracy          87.3%          ↑ +0.8pp         │
│ Retrieval Recall@10         91.2%          → flat           │
│ Hallucination Rate           0.8%          ↓ -0.2pp (good)  │
│ Parse Success               99.1%          → flat           │
│ Latency p99                 2,340ms        ↑ +180ms ⚠       │
├─────────────────────────────────────────────────────────────┤
│ ACCURACY BY QUERY TYPE                                       │
│ Single table                96.1%                           │
│ Aggregation                 91.4%                           │
│ Multi-join                  79.2%          ← focus area     │
│ Date arithmetic             84.3%                           │
│ Subquery                    71.8%          ← focus area     │
├─────────────────────────────────────────────────────────────┤
│ PRODUCTION SIGNALS (trailing 24h)                            │
│ Re-query rate               12.4%          ↑ +2.1pp ⚠       │
│ SQL edit rate                3.2%          → flat           │
│ Export rate                 34.1%          → flat           │
└─────────────────────────────────────────────────────────────┘
```

---

## Eval in CI/CD

### Gate structure

```yaml
# .github/workflows/eval.yml
jobs:
  fast-eval:
    # Runs on every PR — must pass for merge
    steps:
      - run: python eval/run.py --suite fast --threshold blocking
    # fast suite: 50 golden queries, < 5 min
    # blocking threshold: no regression on core metrics

  full-eval:
    # Runs on merge to main — must pass for deployment
    steps:
      - run: python eval/run.py --suite full --compare-to production
    # full suite: all 500+ golden queries, end-to-end
    # compare-to: generates diff report against current production model

  shadow-eval:
    # Runs daily against production traffic (async, doesn't block)
    steps:
      - run: python eval/run.py --suite shadow --output metrics-store
    # shadow: samples 2% of production queries, human review queue
```

### Eval result as a PR artifact

Every PR that touches the NL2SQL pipeline includes an eval summary:

```
Eval Summary for PR #847 (Update retrieval reranker)
──────────────────────────────────────────────────────
Fast eval: PASSED (47/50 queries, 94.0%)

Changes vs. baseline:
  retrieval_recall@10:  +3.2pp  ✓ (0.879 → 0.911)
  execution_accuracy:   +0.8pp  ✓ (0.865 → 0.873)
  hallucination_rate:   -0.1pp  ✓ (0.009 → 0.008)
  latency_p99:          +120ms  ⚠ (2,220ms → 2,340ms) — within threshold

Query types improved: multi-join (+4.1pp), schema-linking-hard (+6.2pp)
Query types regressed: none above threshold

Reviewer: @eval-bot | Full report: eval-results/pr-847/
```

---

## Common Mistakes

**Mistake 1 — Eval database with no adversarial data:**
A test database where every customer has at least one order, every date filter returns results, and all NULL values are cleaned up will make your eval numbers optimistic. Deliberately add edge cases.

**Mistake 2 — Golden set constructed by the same team that built the system:**
The team will unconsciously write queries that the system handles well. Require at least 20% of the golden set to be written by domain experts who have never seen the system.

**Mistake 3 — Treating exact match as a metric:**
SQL has too much surface variation. A query with different alias names, different column ordering, or `WHERE x != 'inactive'` instead of `WHERE x = 'active'` will fail exact match but is semantically correct. Use result set equivalence on adversarial data instead.

**Mistake 4 — Not segmenting by query type:**
An overall accuracy number hides whether multi-join queries have 50% accuracy. Always report per-type and per-difficulty breakdowns. Overall accuracy is a headline; segments are where you find the problems.

**Mistake 5 — Eval set that never grows:**
A static eval set from day 1 will overfit to the eval set as the team iterates. Add production-derived queries monthly. The eval set should reflect the current distribution of production traffic, not the distribution of queries at launch.

**Mistake 6 — Evaluating on production data directly:**
Running eval against the live production database is dangerous — a buggy eval query can be destructive or expensive. Always use a dedicated test database with synthetic, representative data.

**Mistake 7 — Not tracking confidence calibration:**
A system that assigns 0.9 confidence to queries where it is only 60% accurate is worse than useless — it actively misleads users. Track calibration error (expected accuracy at each confidence bucket vs. actual accuracy) and optimize for calibration alongside accuracy.
