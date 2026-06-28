# Accuracy, Evaluation, and Trust — Answers

## Q9. How do you build a ground-truth evaluation set for a domain you're deploying into for the first time?

**Answer:**

Building a ground-truth eval set before you have user data requires being deliberately adversarial about the biases you're introducing.

**Construction approach:**

*Step 1 — Domain expert query elicitation:*
Work with the domain experts (analysts, product managers, finance team) who know the data best. Ask them: "What are the ten questions you answer every week?" and "What is the hardest question you have ever tried to answer with this data?" This gives you coverage of common patterns and hard edge cases. Write the SQL for these queries yourself — do not use the NL2SQL system to generate the ground truth.

*Step 2 — Schema-driven systematic coverage:*
For every table in the schema, generate at least: one simple single-table query, one aggregation, one filter with a date range, and one join to the most common related table. This prevents coverage gaps where your eval set happens to skip an entire area of the schema.

*Step 3 — Adversarial query generation:*
Generate queries specifically designed to be hard: ambiguous time references ("last year" — calendar year or fiscal year?), entity names that appear in multiple tables, queries requiring 3+ table joins, queries with business KPI definitions that differ from column names.

**Biases this eval set carries:**

*Selection bias:* Domain experts ask the questions they know are answerable. Hard-to-answer questions (those requiring missing data or complex business logic) are underrepresented. Your accuracy numbers will be optimistic.

*Persona bias:* Expert-generated queries are more precisely phrased than real user queries. Real users say "show me how we're doing" — experts say "show me Q3 revenue by region compared to Q3 last year." Your eval set is harder to answer correctly than it appears.

*Schema coverage bias:* Systematically generated queries cover each table equally — but in production, 20% of tables generate 80% of queries. Over-representing rare tables inflates the difficulty of your benchmark.

**What to do about it:** Weight your eval set by expected query frequency once you have production data. Treat initial accuracy numbers as upper bounds on real-world performance, not representative estimates.

---

## Q10. Execution accuracy and exact match accuracy both have serious flaws. Describe a metric you'd actually trust.

**Answer:**

**Why execution accuracy is insufficient:**
Two SQL queries can return the same result set for different reasons — one is correct by accident (the data distribution happens to make both queries equivalent on the test database but not in general). Also, empty result sets are undetectable — a wrong filter that returns zero rows matches a correct query that should also return zero rows.

**Why exact match is worse:**
SQL has enormous surface variation. `WHERE status = 'active'` and `WHERE status != 'inactive'` are semantically equivalent on most data. Ordering of SELECT columns, use of aliases, different but equivalent JOIN syntax — all of these cause exact match to fail on correct queries. Exact match is systematically biased against SQL style variation.

**What I'd actually trust — a layered metric:**

*Layer 1 — Execution success rate:*
Did the query execute without error? Necessary but not sufficient.

*Layer 2 — Result set equivalence:*
Run both the generated SQL and the ground-truth SQL on a canonical test database. Compare result sets (as sets, not ordered sequences). This catches semantic errors that execution success misses. Limitation: requires a populated test database that is representative — if the test database has no rows where a buggy filter would show its effect, the bug is invisible.

*Layer 3 — Result set equivalence on adversarial data:*
Generate synthetic data specifically designed to differentiate incorrect queries from correct ones. For example, if the question is "customers who placed more than 3 orders", generate exactly one customer with 3 orders and one with 4 — if the query uses >= instead of >, this catches it.

*Layer 4 — Human spot-check on a stratified sample:*
For a random sample of 5% of queries (or all queries below a confidence threshold), have a domain expert review the generated SQL and result. This is the only layer that catches semantic errors on realistic data distributions.

*Layer 5 — Production implicit feedback:*
Track whether users immediately re-query after seeing a result (suggests dissatisfaction), whether they edit the SQL (indicates error), and whether they export/act on the result (suggests satisfaction). These are noisy but scale to 100% of production traffic.

**What this still misses:** Queries that are technically correct but return too many rows to be useful (missing a filter), or queries that return plausible-looking but subtly wrong results (off-by-one in date range, wrong granularity). No automated metric catches these reliably without human review.

---

## Q11. A query executes successfully but returns semantically wrong results. How do you detect this?

**Answer:**

This is the hardest failure class in NL2SQL because it is invisible to all syntactic and execution-level checks. A query that runs cleanly and returns data looks identical to a correct query from the system's perspective.

**Detection approaches, ranked by reliability:**

**1. Schema-level sanity checks:**
After generating SQL, check: Does the aggregation match the question intent? ("total revenue" should have SUM, not COUNT or AVG). Does the grouping dimension match the "by" clause in the question? ("revenue by region" should have a GROUP BY on a region column). Does the date filter match the time expression? These are structural rules derivable from the question without knowing the ground truth.

**2. Result distribution anomaly detection:**
For queries that historically return data in a known range, flag results that fall outside expected bounds. "Monthly revenue" should never be negative; if it is, something is wrong. Build these bounds from historical query results. Works well for recurring analytical queries; fails for ad-hoc queries with no history.

**3. Self-consistency check:**
Generate two independent SQL queries from the same question (different few-shot examples, slightly different prompt) and compare results. Divergence signals that at least one is wrong. This doubles LLM cost so it should be reserved for high-stakes queries.

**4. Chain-of-thought verification:**
Ask the LLM to explain in natural language what the generated SQL does, then ask a second LLM call: "Does this explanation match the original question?" This is imperfect but catches gross semantic errors where the SQL does something completely different from what was asked.

**5. User feedback signals:**
The most reliable signal is implicit — a user who downloads the result and uses it has validated it; a user who re-queries within 30 seconds has rejected it. Build this feedback into your accuracy tracking.

**Honest answer:** You cannot reliably detect semantic errors at scale without human review or ground-truth data. The best you can do is layer probabilistic signals and build a review queue for queries where multiple signals suggest something is wrong.

---

## Q12. Your system has 85% execution accuracy on your benchmark. Your PM says that's good enough to ship. What do you tell them?

**Answer:**

85% execution accuracy sounds high until you think about what the 15% failure rate means in practice.

**The numbers argument:**
If your system processes 1,000 queries per day, 150 of them return wrong results. If users act on those results without realizing they're wrong — making business decisions based on incorrect data — the downstream cost can be enormous, especially in finance, operations, or compliance contexts.

**Why 85% on the benchmark may be optimistic:**
- Benchmark queries are typically cleaner and more precisely phrased than real user queries
- Benchmark databases are often smaller and simpler than production databases
- Benchmark eval sets have schema coverage bias — they don't include the hardest real-world queries because those are hard to write ground-truth SQL for
- Real accuracy on production is typically 5–15 percentage points below benchmark accuracy

**Why 85% may also be pessimistic on certain dimensions:**
- If the 15% failures are concentrated in a specific query class (e.g. date arithmetic), that's fixable before ship
- If the 15% failures produce obvious errors (empty results, SQL errors) rather than silently wrong results, users self-correct and the business impact is lower
- If you add a confidence threshold and only show results for the 70% of queries above the threshold, your effective accuracy on shown results might be 95%+

**What to actually tell the PM:**
"85% benchmark accuracy with unknown semantic error rate is not sufficient to ship without safeguards. The right path is: (1) run a production shadow pilot with human review of all outputs before users see them, (2) instrument implicit feedback signals, (3) add a confidence indicator in the UI so users know when to double-check, (4) restrict the initial rollout to query types where you have the highest accuracy, and expand from there." Ship it as a tool that assists analysts, not one that replaces their judgment.
