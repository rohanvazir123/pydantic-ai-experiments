# Video Narration Guide

Target length: **7–10 minutes**. ~30–40 seconds per slide. Don't rush — pause after key numbers.

---

## Recording Setup

**Screen layout:**
- Full screen: Jupyter notebook in browser — this is what OBS captures
- Second monitor or phone: narration.md open for reference

No camera. OBS captures screen + Yeti audio only.
In OBS set the source to **Window Capture → Jupyter browser window**, not full screen.
Narration doc on a separate screen or device — glance at it freely while talking.

---

## OBS + Yeti Setup

**Before you start:**
- Yeti: set dial on back to **Cardioid** (circle icon)
- Speak into the **front face** of the mic, not the top — 6–8 inches away
- OBS Audio Mixer: levels should hit **-12 to -6 dB** (green/yellow). Never red.
- OBS Source: **Window Capture** → select the browser with the slides (cleaner than full screen)
- Mute Desktop Audio in the mixer (no system sounds)
- Do a 10-second test recording and play it back before starting for real

**To record:**
- Click **Start Recording** in OBS
- Recording saves to the path set in **Settings → Output**

**After recording:**
- Trim silence at the start/end in any video editor (Photos app on Mac can do basic trim)
- No need to edit further — one clean take is fine

---

## Slide Walkthrough

---

**Slide 1 — Title** `~20 sec`

"Hi, this is an analysis of 100 customer meetings transcripts to classify themes and surface actionable insights
for different teams — engineering, product, sales, and customer success."
My name is Vikrant Potnis, I'm presenting this for the end client Rubrik brought to me by Aziro.

---

**Slide 2 — Approach 1: Rule-Based** `~30 sec`

"First approach — hand-crafted rules. Every theme is defined manually using keywords.

Simple and deterministic, but not scalable. Every new theme or shift in customer language needs a manual update.
And it breaks silently — a misclassified meeting produces no error, just wrong output.
The same concept expressed differently — 'pipeline failure' versus 'ingestion outage' — gets treated as completely different."

---

**Slide 3 — Approach 2: TF-IDF + KMeans** `~40 sec`

"Second approach — represent each meeting as a bag of words and cluster with KMeans. More systematic than rules.

Three problems.
First: you have to decide upfront how many themes there are — say 8. But what if there are actually 26?
Second: it treats words literally. 'Outage remediation' and 'post-mortem' mean the same thing to a human — the algorithm sees no connection because they share no words.
Third: theme names are just the top words from each cluster — you get labels like 'renewal / competitive / pricing / outage' which tell you nothing useful."

---

**Slide 4 — Approach 3: Embeddings + HDBSCAN + LLM** `~40 sec`

"Final approach — embed each topic phrase as a vector, let HDBSCAN find the clusters from the data.
26 themes emerged naturally. No K required, density-adaptive.
The LLM names the clusters — it does not form them. The heavy lifting is done by HDBSCAN.
This is why a small local model produces clean, readable labels."

---

**Slide 5 — Database Schema** `~40 sec`

"Two-layer design, nine tables total.

Six base tables loaded from raw meeting JSON — meetings, participants, summaries, key moments, action items, and transcripts.
These capture everything in the transcripts: sentiment scores, signal moments like churn or praise, and the topic phrases from each summary.

Three semantic tables are generated after the clustering and LLM labeling runs.
The clusters table holds the 26 discovered themes. The phrases table maps each topic phrase to its cluster.
The meeting themes table maps each meeting to its themes, with call type and sentiment.

Two separate layers means base data and clustering are independent — you can reload raw data without touching the clustering, and rebuild clustering without re-running the pipeline."

---

**Slide 6 — Meeting Breakdown by Call Type and Product** `~30 sec`

"100 meetings — roughly half support, a quarter renewals, a quarter internal.
Detect and Comply appear in the most meetings across the dataset.

Call type split comes from the meeting themes table. Product mentions come from the meeting summaries table —
the products column is populated at load time by scanning each summary for known product names."

---

**Slide 7 — UMAP Scatter** `~30 sec`

"Each dot is a topic phrase from meeting summaries, coloured by cluster.
26 themes emerged naturally — no K was specified.
Detect phrases cluster tightly together. Comply sits as a separate island.

The 2D coordinates are pre-computed for visualisation only.
The actual clustering ran on a 10-dimensional reduction to get better separation."

---

**Slide 8 — Theme Sentiment Heatmap** `~40 sec`

"Sorted most negative at the top, most positive at the bottom.
Detect-related themes dominate the red zone. Comply dominates the green zone.
One product is in crisis — the other is the strongest asset in the portfolio.

This comes from joining meeting themes with the clusters table to get each meeting's primary theme and sentiment.
It is the single most important chart in the deck — it shows the full picture in one view."

---

**Slide 9 — Churn Signals by Product** `~30 sec`

"Churn signals are moments in the transcripts where customers explicitly expressed risk of leaving.
Detect generates far more than any other product.
Direct input for account prioritisation and recovery planning.

This joins the products column in meeting summaries with the key moments table,
filtering to churn signal moments only, grouped by product."

---

**Slide 10 — Accounts Needing Immediate Follow-Up** `~40 sec`

"38 meetings with negative sentiment and at least one explicit churn signal.
Ranked by severity — top rows are the most urgent accounts.
This is an action list, not a trend. These accounts need a call this week.

Built by joining meetings, meeting themes, and key moments — filtering to primary themes
with negative sentiment that also have churn signal key moments. The signal count is what ranks the list."

---

**Slide 11 — Which Products Are Generating the Most Risk?** `~30 sec`

"Three signals per product: total meetings, technical issues, and churn signals.
Detect leads on both — the March outage created a long tail of follow-up.
This shows which product needs reliability investment first.

Product attribution comes from meeting summaries. The two signal types — technical issues and churn signals —
are each counted separately from the key moments table."

---

**Slide 12 — Where Customers Are Happy** `~30 sec`

"Comply has the highest praise density of any product.
Comply renewal conversations are the most positive in the dataset.
This is the good news story to lead with for at-risk Detect accounts.

Left chart: praise moments from the key moments table, grouped by product.
Right chart: meeting themes filtered to Comply, external calls only — these are the renewal and account conversations."

---

**Slide 13 — How the Detect Outage Affected Renewal Calls** `~40 sec`

"Renewal meetings that mention Detect are significantly more negative than those that don't.
Over half carry an explicit churn signal.
Outage conversations are bleeding into commercial calls that should be about growth.

This filters meeting themes to external calls only, then splits by whether Detect appears in the products column.
The pie chart counts churn signal key moments within the Detect-tagged group."

---

**Slide 14 — What Customers Are Asking For and How Urgently** `~40 sec`

"The same feature request carries completely different urgency depending on the account situation.
Blocked in red — at-risk customer, treat as P0, act immediately.
Growing in green — healthy account asking for more, add it to the roadmap.

This joins the products column with feature gap key moments, then groups by the overall sentiment of that meeting.
The sentiment of the meeting — not the request itself — is what determines the urgency.
Same feature, completely different priority."

---

**Slide 15 — Who Owns the Follow-Up Work?** `~40 sec`

"Left panel shows which themes generate the most follow-up actions.
Engineering carries the heaviest Detect load — if reliability slips, Comply v2 delivery slips with it.
This identifies where bottlenecks sit before they become a problem.

This uses a view that joins action items to meeting themes on the primary theme flag, then to the clusters table for the label.
The right panel adds the products column from meeting summaries to show ownership broken down by product."

---

**Slide 16 — Three Things to Act On** `~40 sec`

"One: fix Detect reliability before the next QBR — it is the number one churn driver across the dataset.

Two: call the 38 high-risk accounts this week. The list is already ranked by urgency — start from the top.

Three: lead with Comply v2 for at-risk Detect accounts. When a customer is upset about the Detect outage,
don't walk into that renewal call with only an apology. Open with the Comply v2 story —
give them a forward-looking reason to stay before you address what went wrong.
Comply has the highest satisfaction of any product in the dataset. Pair the bad news with good news.

Everything here came directly from the transcripts — no manual tagging, no survey data."
