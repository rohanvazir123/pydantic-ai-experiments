# Multi-Turn and Context — Answers

## Q13. A user says "now filter that by region" as a follow-up. How do you resolve "that" to the previous query's context?

**Answer:**

Multi-turn resolution requires maintaining a structured conversation state, not just appending raw message history to the prompt.

**Conversation state structure:**
For each turn, store: the original natural language query, the generated SQL, the tables and columns used, the result schema (column names and types of the result set), and a natural language summary of what was returned ("This query showed monthly revenue by product category for Q3").

**Resolving "filter that by region":**

*Step 1 — Identify the referent:* "that" refers to the most recent query result. The previous SQL becomes the base context.

*Step 2 — Apply the modification:* The new request is "add a filter by region." Determine whether "region" maps to a column already in the previous query's tables or requires a new join. If the previous query's FROM clause already includes a table with a region column, add `WHERE region = ?` — but we don't know which region, so this becomes an additional ambiguity. If the region column requires a new table join, expand the query.

*Step 3 — Reconstruct the full query:* Do not append the modification to the previous SQL as a patch. Regenerate the full SQL from scratch using the accumulated intent: "Show monthly revenue by product category for Q3, filtered to region X." This is more reliable than SQL patching, which requires the LLM to correctly modify arbitrary SQL.

**What breaks with 10 deep turns and a topic change:**

After 10 turns, the accumulated context is large. If the user changed topics mid-conversation ("actually, forget that, show me employee headcount"), the previous SQL context is no longer relevant. The system must detect topic shifts.

Detection: Compute semantic similarity between the new query and the most recent N queries. If similarity drops below a threshold, treat this as a new conversation and reset the SQL context. Retain the natural language history for co-reference resolution (the user might still say "compared to last year" which refers to a time expression from earlier) but do not carry forward the SQL.

The practical limit for reliable multi-turn SQL context is 3–5 turns. Beyond that, the compounding ambiguity makes reliable generation much harder, and you should surface a "start fresh" option to the user.

---

## Q14. How do you handle pronoun and entity co-reference across turns?

**Answer:**

Co-reference resolution in NL2SQL is harder than in general NLP because the entities being referenced are database entities (table rows, filtered result sets, metric definitions) rather than named entities in prose.

**Architecture for co-reference state:**

Maintain a structured entity registry per conversation session:
```
{
  "mentions": [
    { "turn": 1, "phrase": "top 10 customers", "resolved_to": "CustomerID IN (SELECT TOP 10 CustomerID FROM ...)", "type": "result_set" },
    { "turn": 2, "phrase": "their orders", "resolved_to": "orders WHERE customer_id IN <above>", "type": "derived" },
    { "turn": 3, "phrase": "Acme Corp", "resolved_to": "customer_name = 'Acme Corp'", "type": "entity_value" }
  ]
}
```

When a new query arrives, run a co-reference resolution step that maps pronouns and definite noun phrases ("their", "those customers", "that period") to entries in the entity registry before SQL generation.

**"Show me their top customers" where "their" refers to a company three turns ago:**

*Step 1:* Identify "their" as a pronoun requiring resolution. Scan the entity registry for the most recent entity that could be referred to by a possessive. Three turns ago: "Acme Corp" was mentioned and resolved to `customer_name = 'Acme Corp'`. "Their top customers" means the top customers associated with Acme Corp.

*Step 2:* This requires knowing what "customers of Acme Corp" means in the schema — is Acme Corp a vendor, a reseller, or a partner? This is where the schema matters. If Acme Corp appears in a `partners` table with a FK to `customers`, the resolution requires a JOIN. The co-reference resolution must be schema-aware, not just text-aware.

*Step 3:* Inject the resolved entity context into the SQL generation prompt: "Find the top customers linked to partner 'Acme Corp' (partners.partner_id = customers.partner_id), ranked by [metric]."

**State management at scale:**
The entity registry is stored per-session in Redis with a TTL equal to the session timeout. It is passed to the LLM as structured context, not as raw conversation history — raw history grows unboundedly and the LLM's attention degrades on distant context.

---

## Q15. When should your system refuse to carry context forward and ask the user to re-state?

**Answer:**

The system should refuse to carry context forward in these specific situations:

**1. Detected topic shift:**
Semantic similarity between the new query and the accumulated SQL context drops below a threshold (empirically, cosine similarity < 0.4 on sentence embeddings is a reliable signal). Rather than silently resetting, surface it: "It looks like you're starting a new question. I'll start fresh — does that work?"

**2. Conflicting constraints:**
The new query introduces a constraint that contradicts a prior constraint. "Show me revenue where region = 'West'" followed by "exclude the West region" — these constraints are contradictory. Carrying both forward silently produces wrong SQL. Flag the conflict explicitly.

**3. Reference to a dropped result set:**
The user refers to "those results" but the referenced query returned zero rows, or the result set was too large and was truncated. There is no meaningful result set to reference. Ask the user to re-state what they want to filter or modify.

**4. Context chain length exceeds threshold:**
After 6–8 turns of accumulated modifications, the compound intent becomes unreliable to represent in a single SQL query. The LLM's ability to correctly incorporate 6 successive modifications degrades. At this point, summarize the current SQL in natural language and ask the user to confirm: "Your current query is: monthly revenue by product, filtered to West region, excluding refunds, for Q3 2024 only. Is that correct?" This serves as both a context reset and a correctness check.

**5. Schema change detected:**
If the underlying schema has changed since the conversation started (a table was renamed, a column removed), all context that references the old schema is stale. Reset and re-generate from the current intent.

**How to make this decision programmatically:**
Expose a context-reset decision as a lightweight classifier that runs before every multi-turn resolution step. It takes as input: the new query, the previous SQL, the entity registry, and the conversation length. Binary output: carry context or reset. Train this classifier on conversations where human reviewers marked context as stale or conflicting.
