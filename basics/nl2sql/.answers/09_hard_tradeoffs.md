# Hard Trade-offs — Answers

## Q30. Accuracy vs. explainability — you can't have both equally. How do you decide?

**Answer:**

The trade-off is real: larger models are more accurate but harder to introspect; chain-of-thought (CoT) reasoning improves interpretability but adds 200–500ms latency and can slightly decrease accuracy (the model sometimes reasons itself into a wrong answer it wouldn't have made with direct generation).

**The decision framework:**

*Factor 1 — Who is the user?*
For data analysts who can read SQL and evaluate correctness directly: explainability of the *reasoning process* matters less. The SQL itself is the explanation — a correct SQL is more valuable than a narrated incorrect one. Optimize for accuracy.

For executives or business users who cannot evaluate SQL: they need to trust the system without being able to verify it. Explainability matters more — a system that shows its reasoning ("I'm interpreting 'top customers' as highest total revenue in the last 12 months") builds trust even if it adds latency.

*Factor 2 — Stakes of the domain:*
In financial reporting, audit, or compliance: explainability is non-negotiable. You need an audit trail of why a specific SQL was generated. Use CoT reasoning logged as metadata even if it's not shown to the user — it is available for compliance review.

In a casual analytics context (exploratory data analysis, ad-hoc queries): accuracy matters more. Users are exploring and will manually verify interesting findings. Optimize for throughput and accuracy.

*Factor 3 — Error consequences:*
If a wrong answer leads to a bad business decision that can't be undone (a financial report that's already been sent to investors), explainability is a risk mitigation tool. If a wrong answer is easily noticed and corrected (an analyst queries again), accuracy is the better investment.

**Practical resolution:**
You don't have to choose globally. Build a tiered system: casual queries use a fast, high-accuracy model with minimal CoT; high-stakes queries (detected by keyword pattern or user persona) use a CoT-generating model with explicit reasoning shown to the user. The incremental cost applies only where it matters.

---

## Q31. Should your system tell the user when it is uncertain, or silently return its best guess?

**Answer:**

The answer is context-dependent but the default should always be to surface uncertainty — silently returning wrong answers is worse than surfacing uncertainty in almost every case.

**The case for surfacing uncertainty:**

In a financial reporting context, a user who acts on wrong data without knowing the system was uncertain has been harmed by the system's overconfidence. A confidence indicator gives the user the information they need to decide whether to verify. "I'm 60% confident this SQL answers your question — please verify the date range logic" is actionable.

**The case for silent best-guess:**

In a casual analytics context, confidence indicators add cognitive load. A user exploring data doesn't want to evaluate a confidence score on every query — they want results. If the system is wrong, they will notice and rephrase. Surfacing low confidence on every mildly complex query trains users to ignore the indicator, making it useless when it actually matters.

**Building a confidence model:**

Confidence should not be self-reported by the LLM ("I am 80% confident"). LLM self-reported confidence is poorly calibrated. Instead, derive confidence from observable pipeline signals:
- Schema retrieval rank of the top-retrieved table (higher rank = higher confidence)
- SQL parse success on the first attempt (retry = lower confidence)
- Hallucination check result (hallucinated column = low confidence)
- Similarity of the query to high-accuracy training examples
- Semantic consistency between the NL query and the generated SQL (re-embed both, compute similarity)

Calibrate this signal against ground truth: a confidence score of 0.8 should correspond to ~80% execution accuracy on your evaluation set.

**The minimum viable approach:**
Show a low-confidence indicator only when confidence drops below a threshold (e.g., below 0.6). Don't show it at all for high-confidence queries. This minimizes noise while surfacing the cases that actually need user attention.

---

## Q32. SQL-fluent analysts and non-technical executives in the same system — how do you handle both personas without two systems?

**Answer:**

This is a UI and output-formatting problem more than a model problem. The SQL generation pipeline can be the same; the presentation layer adapts.

**What differs by persona:**

| Dimension | Analyst | Executive |
|-----------|---------|-----------|
| SQL visibility | Want to see, edit, and run the SQL | Don't want to see SQL at all |
| Error messages | Technical (column not found, join condition) | Plain English ("I couldn't find data for that time period") |
| Confidence indicators | Want to know specifics (which table was uncertain) | Want binary: "ready to share" vs "needs review" |
| Result format | Raw table with full columns | Summarized, formatted, key metrics highlighted |
| Query refinement | Will write SQL directly to refine | Needs natural language refinement loop |
| Latency tolerance | Higher (they're iterating) | Lower (they need quick answers) |

**Architecture to support both:**

*Single generation pipeline:* The NL→SQL generation is identical. The system prompt does not change by persona.

*Persona-aware output formatter:* After SQL is generated and executed, pass the result through a persona-specific formatter:
- Analyst view: raw SQL visible and editable, full result table, column metadata visible
- Executive view: SQL hidden (available via "see query" disclosure), result rendered as a formatted table or chart, natural language summary of what the query found

*Prompt adaptation for error messages:* When SQL generation fails, the error message is rendered by the formatter, not the LLM. The formatter maps technical error codes to persona-appropriate language.

*Query refinement loop:* For analysts, refinement happens via SQL edit. For executives, refinement happens via a follow-up NL query. Both feed the same multi-turn context system.

**Persona detection:**
Infer from user role (pulled from the auth token) rather than asking users to self-identify. Don't force users into a persona — allow analysts to switch to the simplified view for demo purposes, and allow executives to expose the SQL if they want to.

The key insight: the hard engineering is in the SQL generation and retrieval layers. The persona adaptation is presentation logic — keep it out of the core pipeline so the pipeline remains testable and maintainable independently.
