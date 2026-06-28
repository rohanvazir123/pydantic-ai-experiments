# Feedback Loops and Continuous Improvement — Answers

## Q27. How do you build a feedback loop when most users won't tell you the result was wrong?

**Answer:**

Explicit feedback (thumbs up/down) has < 5% participation rate in most BI tools. You need implicit signals that cover 100% of traffic.

**Implicit signals and their reliability:**

*Signal 1 — Immediate re-query (highest signal):*
If a user submits a new query within 30–60 seconds of receiving results, and the new query is semantically similar to the first (same topic, different phrasing or filter), the first result was likely wrong or incomplete. Similarity threshold: cosine similarity > 0.7 between consecutive queries.

*Signal 2 — Query editing:*
If the interface allows the user to see and edit the generated SQL, an edit is the strongest possible implicit signal — the user is explicitly correcting the output. The edit delta (original SQL vs. edited SQL) is a training signal: the edit shows exactly what was wrong and what the correct version looks like.

*Signal 3 — Result download / export:*
A user who downloads the results to CSV or exports to a dashboard has effectively validated the query. Weight this as positive signal.

*Signal 4 — Session abandonment:*
A user who gets a result and closes the session within 10 seconds (without export or further queries) may have found what they needed — or may have given up. Ambiguous signal; use in aggregate (high abandonment rate on a query class) rather than per-query.

*Signal 5 — Empty result acknowledgment:*
If the query returns zero rows and the user does not query again or refine, they either accepted it (unlikely for analytical queries) or gave up. If the query *should* return data based on schema statistics, zero rows is likely a wrong filter.

**Avoiding training on noisy labels:**

Never train directly on implicit signals. Instead, build a human review queue: queries with high negative signal scores (multiple rephrases, no export, quick re-query) are surfaced for human review. A domain expert labels them as correct/incorrect and provides the correct SQL. Only these human-labeled examples enter the training set.

This keeps the training data high-quality while the implicit signals serve as a cost-efficient triage mechanism.

---

## Q28. A user edits the generated SQL before running it. How do you use that signal?

**Answer:**

A SQL edit is the highest-quality training signal you can get — it is a direct correction from a user who understood both the generated SQL and the correct SQL.

**Extracting the signal:**

*Step 1 — Diff analysis:*
Compute the semantic diff between original and edited SQL. Classify the edit type:
- **Column substitution:** `revenue` → `net_revenue` (schema linking error — wrong column was chosen)
- **Table substitution:** `orders` → `order_items` (schema retrieval miss — wrong table selected)
- **Filter addition:** no WHERE → `WHERE region = 'West'` (missing filter — likely an ambiguity resolution issue)
- **Aggregation change:** COUNT → SUM (wrong aggregation — LLM misunderstood the metric)
- **Join addition:** single table → multi-table (join path miss)

Each edit type points to a specific failure in the pipeline, allowing targeted improvement.

**Distinguishing "SQL was wrong" from "user changed their mind":**

This is hard. Heuristics:
- If the user edited within 30 seconds of receiving the SQL, and the edit is significant (more than whitespace or alias changes), it is likely a correction.
- If the edit happens after > 5 minutes, the user may have refined their question — not necessarily a model error.
- If the natural language query and the edited SQL are semantically consistent, it is a correction. If the edited SQL does something the original query didn't ask for, the user changed their mind.

Use a small classifier trained on labeled edit pairs to distinguish corrections from intent changes. The classifier input is: time-to-edit, edit magnitude, semantic similarity between NL query and edited SQL.

**Using corrections for improvement:**

Add (original_query, schema, edited_SQL) as a training pair. Flag it as higher quality than automatically generated pairs. If the same edit pattern appears across 10+ different users on the same query class, it is a systematic model error — prioritize it for a prompt or fine-tuning fix.

Do not blindly add user-edited SQL to training — verify that the edited SQL is actually correct (passes schema validation, executes successfully, returns non-trivially different results from the original). Noisy training data degrades the model.

---

## Q29. How do you safely deploy a model update that improves some patterns but regresses others?

**Answer:**

Any model update in NL2SQL is a double-edged sword: improvements in one query class often come at the cost of regressions in others. Safe deployment requires treating this as a scientific experiment, not a push.

**Pre-deployment evaluation:**

Run the new model against your full evaluation set (not just the queries it was fine-tuned on). Segment the results by query type (single table, multi-join, aggregation, date arithmetic, etc.) and by schema domain. For each segment, compute execution accuracy delta (new model minus current model). Any segment with a regression > 2 percentage points is a red flag.

**Canary deployment:**

Route 1–5% of production traffic to the new model. Use a consistent hash of the user ID so the same users always hit the canary — this ensures a user isn't context-switching between two different model behaviors in the same session. Monitor the canary using implicit feedback signals (re-query rate, edit rate, export rate) compared to the baseline. Run for 24–48 hours before expanding.

**Shadow mode:**

Run the new model in shadow mode — it generates SQL but doesn't execute it. Compare the new model's SQL to the current model's SQL for every query. Queries where they diverge are surfaced for human review. This detects regressions before users ever see them.

**Rollout strategy with circuit breaker:**

Expand the canary incrementally: 1% → 5% → 20% → 50% → 100%, with hold periods between each step. At each step, if the implicit feedback metrics (re-query rate, error rate) for the canary exceed the baseline by more than a threshold, automatically roll back to the current model and alert the team. This circuit breaker must be automated — a manual rollback in a production incident is too slow.

**Dealing with regressions that can't be fixed pre-launch:**

If the new model improves accuracy on 80% of query types but regresses on 20%, consider deploying it with a query router: new model handles query types where it is better; current model handles query types where it regresses. The router uses query classification to determine which model to invoke. This is more complex to maintain but avoids forcing users to choose between two imperfect systems.
