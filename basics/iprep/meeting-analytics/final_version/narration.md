# Video Narration Guide
> Use this as a cheat sheet, not a script. Know the key number for each chart and the "so what", then talk through it naturally in your own voice.

## Three numbers to memorise
- **1.04** — churn signals per meeting for reliability calls (vs 0.71 for retention)
- **38** — high-risk meetings on the watchlist
- **26** — themes discovered, no K specified upfront

---

## Opening (30 sec)
"I'm going to walk through an analysis of 100 AegisCloud customer meetings — support calls, account renewals, and internal syncs. The goal was to classify themes across those meetings and surface actionable insights. I'll show you the methodology, then the findings."

---

## Setup cell
"First cell — connect to the database and define two helper functions. `q()` runs any SQL query and returns a dataframe. `save()` exports each chart. That's all the infrastructure we need."

## Helpers cell
"These four functions load all the raw data directly into Postgres — meeting JSONs, sentiment scores, clustering results. Everything is self-contained, no external scripts."

## Load cell *(watch it print live)*
"Running that now — 100 meetings, 311 participants, 4,313 transcript lines, 26 discovered themes. All loaded in a few seconds."

## Verify cell
"Quick sanity check — all 9 tables have the exact row counts we expect. Clean data, ready to analyse."

---

## Chart 1 — Dataset Overview
"The dataset splits roughly half support calls, a quarter external account meetings, and a quarter internal. Detect and Comply show up in the most meetings — they're the two products driving most of the conversation volume."

## Chart 2 — UMAP Scatter
"This is a 2D map of the 343 topic phrases extracted from meeting summaries — each dot is a phrase, colored by cluster. HDBSCAN found 26 clusters naturally without us specifying a number upfront. Detect-related phrases cluster tightly on the left — that density reflects how dominant the reliability narrative is. Comply phrases are a separate island on the right."

## Chart 3 — 26 Themes Table
"Here are all 26 themes with the number of meetings each one dominates. Incidents and reliability sit at the top. The audience column shows who each theme is most relevant to — engineering, sales, customer success."

## Chart 4 — Sentiment by Call Type
"Support calls are the most negative — expected. But the warning is external calls, the renewal and account meetings. Those should skew positive. Mixed sentiment there tells us the Detect outage is contaminating commercial conversations that should be about growth."

## Chart 5 — Theme × Sentiment Heatmap *(pause here longest)*
"This is the headline chart. Every row is a theme, every column a sentiment bucket, sorted most negative at the top. Detect-related themes are almost entirely red. Comply is almost entirely green. One product is on fire, the other is the strongest asset in the portfolio."

## Chart 6 — Churn Density
"This is the commercial risk chart. Reliability calls generate 1.04 churn signals per meeting. Customer retention calls — the ones explicitly about keeping customers — generate only 0.71. Outage conversations are more commercially dangerous than renewal conversations."

## Chart 7 — High-Risk Watchlist
"These are the specific meetings with both negative sentiment and explicit churn signals. Named meetings, named organizers, ranked by risk. This isn't a trend chart, it's an action list. These accounts need a call this week."

## Chart 8 — Product Signals
"Breaking down technical issues and churn signals by product. Detect leads on both — the March outage created a long tail of support and commercial follow-up that's still running through the pipeline."

## Chart 9 — Positive Signals
"The counter-narrative. Comply has the highest praise density of any product and its external meetings skew strongly positive. Comply v2 — on separate infrastructure, unaffected by the Detect outage — is the forward-looking story to pair with the recovery message for at-risk accounts."

## Chart 10 — E3/R4 Detect Contamination
"Quantifying the commercial damage. Of the external Detect-tagged meetings — the renewal and account calls — over half carry an explicit churn signal. Conversations that should be about expanding the contract are now about damage control."

## Chart 11 — Feature Gaps by Product
"Feature gaps split by customer sentiment. Gaps raised in negative meetings are P0 — customers who are blocked and at risk. Gaps in positive meetings are roadmap investments. Same backlog item, completely different urgency depending on the account posture."

## Chart 12 — Action Items by Owner
"Which themes generate the most follow-up work and who owns it. Engineering carries the heaviest load on Detect. If Detect reliability slips further, engineering capacity gets consumed and Comply v2 delivery slips — which is exactly what the transcripts show happening."

---

## Closing (20 sec)
"Three things to act on: fix Detect's pipeline before the next QBR, use Comply v2 as the forward-looking story for at-risk accounts, and work through that watchlist this week. Everything here came directly from the transcripts — no manual tagging, no survey data."
