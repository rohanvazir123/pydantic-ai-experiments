# Slide Walkthrough — Quick Reference
Keep it brief. One sentence per bullet, then move on.

---

**Slide 1 — Title**
"I analysed 100 AegisCloud customer meetings to classify themes and surface actionable insights for different teams."

---

**Slide 2 — Approach 1: Rule-Based**
"First approach — hand-coded rules. Every theme has to be anticipated manually.
The moment a customer says the same thing in a different way, the rules miss it.
Not scalable. Breaks silently."

---

**Slide 3 — Approach 2: TF-IDF + KMeans**
"Second approach — machine learning, but you still have to guess the number of clusters upfront.
Short topic phrases give very weak signals. Theme names come out as mechanical word lists.
Better, but not good enough."

---

**Slide 4 — Approach 3: Embeddings + HDBSCAN + LLM**
"Final approach — embed each topic phrase as a vector, let HDBSCAN find the clusters from the data.
26 themes emerged naturally. No K required.
The LLM only names what HDBSCAN already grouped — it does not do the clustering."

---

**Slide 5 — Database Schema**
"Two-layer design. Raw meeting data in 6 base tables. Clustering outputs in 3 semantic tables.
Every insight query joins both layers in one schema."

---

**Slide 6 — Meeting Breakdown**
"100 meetings — roughly half support calls, a quarter renewals, a quarter internal.
Detect and Comply appear most often across the dataset."

---

**Slide 7 — UMAP Scatter**
"Each dot is a topic phrase from meeting summaries, coloured by cluster.
26 clusters found — no K was specified. Detect phrases are tightly packed on the left. Comply is a separate island."

---

**Slide 8 — Theme Sentiment Heatmap**
"Most negative themes at the top, most positive at the bottom.
Detect is almost entirely red. Comply is almost entirely green.
One product is in crisis. The other is the strongest asset in the portfolio."

---

**Slide 9 — Churn Signals by Product**
"Churn signals are moments where customers explicitly expressed risk of leaving.
Detect generates the most — by a large margin.
This drives account prioritisation."

---

**Slide 10 — Accounts Needing Immediate Follow-Up**
"38 meetings with negative sentiment and at least one churn signal.
Ranked by urgency. Top rows need a call this week.
This is an action list, not a trend."

---

**Slide 11 — Which Products Are Generating the Most Risk?**
"Three bars per product — total meetings, technical issues, churn signals.
Detect leads on both. The March outage created a long tail of follow-up work."

---

**Slide 12 — Where Customers Are Happy**
"Comply has the highest praise density of any product.
Comply renewal conversations are the most positive in the dataset.
This is the good news story to lead with for at-risk Detect accounts."

---

**Slide 13 — How the Detect Outage Affected Renewal Calls**
"Renewal meetings that mention Detect are significantly more negative than those that don't.
Over half carry an explicit churn signal.
Outage conversations are bleeding into commercial calls."

---

**Slide 14 — What Customers Are Asking For and How Urgently**
"Same feature request, completely different urgency.
Blocked in red — at-risk customer, act now, P0.
Growing in green — healthy account, add to roadmap."

---

**Slide 15 — Who Owns the Follow-Up Work?**
"Left: which themes generate the most follow-up actions.
Right: which team owns that work by product.
Engineering carries the heaviest Detect load — if reliability slips, Comply v2 delivery slips with it."

---

**Slide 16 — Three Things to Act On**
"One: fix Detect reliability before the next QBR.
Two: call the 38 high-risk accounts this week.
Three: lead with Comply v2 for at-risk Detect accounts — it is the strongest counter-narrative in the data."
