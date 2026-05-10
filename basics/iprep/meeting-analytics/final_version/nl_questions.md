# Questions Leadership Would Actually Ask
Audience: product and engineering leadership panel (per req.md)
Company: AegisCloud — four products: Detect, Protect, Comply, Identity

Questions are framed around **decisions**, not data descriptions.
Each question states what a leader needs to decide and which tables answer it.

---

## Engineering / CTO

> Core concern: are we shipping reliable software, and what is it costing us commercially?

**E1 — Which product is generating the most customer-visible technical failures?**
_Decision: where to focus reliability investment next quarter._
Tables: `key_moments` (moment_type = `technical_issue`) + `meeting_summaries.products`

**E2 — Which technical issues appear in meetings that also have churn signals — the bugs actively costing us customers?**
_Decision: the real P0 list. Not the sprint backlog — the issues that are killing renewals._
Tables: `key_moments` filtered on both `technical_issue` and `churn_signal` per meeting, joined through `semantic_meeting_themes`

**E3 — How many renewal and commercial meetings were contaminated by the Detect outage?**
_Decision: business case for investing in redundancy and SLA improvement. Quantifies the revenue cost of the incident._
Tables: `semantic_meeting_themes` (`call_type = 'external'`, `'Detect' = ANY(products)`) + `key_moments` (churn_signal)

**E4 — Which technical issues did we resolve on the call vs leave unresolved?**
_Decision: where support teams are equipped and where they aren't. Surfaces escalation gaps._
Tables: `key_moments` — meetings with `technical_issue` AND `positive_pivot` vs meetings with only `technical_issue`

**E5 — Which product has the cleanest support conversations — fewest technical issues, highest sentiment?**
_Decision: where engineering quality is strongest and which practices to replicate across other products._
Tables: `semantic_meeting_themes` (call_type = `support`, sentiment_score) + `meeting_summaries.products` + `key_moments` (technical_issue count per meeting)

---

## Product / CPO

> Core concern: what should be on the roadmap, and are we betting on the right things?

**P1 — Which product area has the most feature gap requests, and are those customers blocked (negative sentiment) or growing (positive)?**
_Decision: P0 blockers vs roadmap wishlist. Same feature request means different priority depending on customer sentiment._
Tables: `key_moments` (moment_type = `feature_gap`) + `meeting_summaries` (products, sentiment_score)

**P2 — What are customers asking for specifically in commercial and renewal meetings — the features they'd pay for?**
_Decision: revenue-blocking gaps. Filter to external call type to separate commercial asks from support complaints._
Tables: `key_moments` (feature_gap, text) + `semantic_meeting_themes` (call_type = `external`) — use `action_items_by_theme` view for owner/theme context

**P3 — Which product generates the most praise — where is our product-market fit strongest?**
_Decision: where to double down and what to use in competitive positioning. Comply consistently scores highest._
Tables: `key_moments` (moment_type = `praise`) + `meeting_summaries.products`

**P4 — Is the Comply v2 bet validated? How often does compliance come up as a buying criterion in external meetings?**
_Decision: whether Comply v2 is the right counter-narrative to the Detect outage for at-risk account communications._
Tables: `semantic_meeting_themes` (`'Comply' = ANY(products)`, call_type = `external`) + `meeting_summaries.overall_sentiment`

**P5 — Which product features are customers actively praising in their own words?**
_Decision: what's working that we must not break. Also the input for case studies, marketing copy, and competitive positioning._
Tables: `key_moments` (moment_type = `praise`, text) + `meeting_summaries.products` — read verbatims, not just counts

---

## Sales & CS

> Core concern: which accounts are at risk and what do we do about it this week?

**R1 — Which specific meetings have churn signals and negative sentiment — who needs a call today?**
_Decision: immediate CSM and AE outreach list. Named accounts, ranked by churn signal count._
Tables: `semantic_meeting_themes` (sentiment_score, is_primary) + `key_moments` (churn_signal) + `meetings` (title, organizer_email)

**R2 — Which organizers and contacts keep appearing in high-risk meetings?**
_Decision: who to escalate to exec level. Repeat appearance = systemic account problem, not a one-off._
Tables: `meetings.organizer_email` + `meeting_participants` filtered to high-risk meeting_ids

**R3 — In pricing conversations, is the customer in a strong position or a weak one?**
_Decision: deal strategy. Pricing moments paired with churn signals = customer has leverage. Paired with praise = expansion opportunity._
Tables: `key_moments` — meetings with `pricing_offer`, joined to `key_moments` for co-occurring `churn_signal` or `praise`

**R4 — How many renewal meetings were contaminated by Detect reliability discussions?**
_Decision: quantify ARR at risk from the March outage. Input to executive account recovery plan._
Tables: `semantic_meeting_themes` (`call_type = 'external'`) + `semantic_clusters` (Reliability-related themes) + `key_moments` (churn_signal)

**R5 — Which accounts show strong positive signals — potential references, expansion candidates, or early Comply v2 champions?**
_Decision: where to invest relationship time for upside, not just risk mitigation. Positive external meetings with praise moments and no churn signals._
Tables: `semantic_meeting_themes` (call_type = `external`, sentiment_score > 3.5) + `key_moments` (praise) + `meetings` (organizer_email, title)

---

## Support / Operations

> Core concern: where is support capacity going and where is follow-through breaking down?

**S1 — Which product drives the most support escalations, and are the same issues repeating?**
_Decision: knowledge base investment and L1 deflection opportunities. Repeat issues = documentation gap._
Tables: `semantic_meeting_themes` (call_type = `support`) + `meeting_summaries.products` + `key_moments` (technical_issue, text)

**S2 — Are reliability meetings running longer than other call types?**
_Decision: operational cost of outages in support hours. Feeds resource planning for incident response._
Tables: `meetings.duration_minutes` + `semantic_meeting_themes` (call_type, is_primary)

**S3 — Which action item owners are most overloaded across themes?**
_Decision: accountability and capacity. Identifies whether follow-through bottlenecks fall in engineering, CS, or sales._
Tables: `action_items_by_theme` view (owner, theme_title, count)

---

## Cross-cutting

**X1 — Which themes consistently appear together — are any customers dealing with compound failures?**
_Decision: account triage prioritisation. A customer with both Reliability and Customer Retention themes is more at risk than one with either alone._
Tables: `semantic_meeting_themes` self-join on meeting_id where cluster_id differs — `action_items_by_theme` for follow-up burden

**X2 — What is the sentiment split across call types — how much of our customer conversation is in distress vs healthy?**
_Decision: posture check. If >50% of external meetings are negative or mixed-negative, the pipeline needs attention before the next QBR._
Tables: `semantic_meeting_themes` (call_type, overall_sentiment, is_primary)

---

## Future Improvement

LLM-generated question discovery: inject schema context, product names, sample findings, and a stakeholder persona prompt into an LLM to generate additional questions. Useful for discovering blind spots that a human analyst wouldn't think to ask. Not implemented — low priority given time constraints, but straightforward to add as a notebook section.
