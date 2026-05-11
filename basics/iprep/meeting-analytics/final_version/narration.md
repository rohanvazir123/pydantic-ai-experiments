# Video Narration Guide

Target length: **7–10 minutes**. ~30–40 seconds per slide. Don't rush — pause after key numbers.

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
- Trim silence at the start/end in any video editor (Photos app on Windows can do basic trim)
- No need to edit further — one clean take is fine

---

## Slide Walkthrough

Keep it brief. One sentence per bullet, then move on.

---

**Slide 1 — Title** `~20 sec`

"
Hi, my name is XYZ <spell it out> this work is to analyze meeting transcripts 
classify themes and surface actionable insights for different teams."

---

**Slide 2 — Approach 1: Rule-Based Keyword Matching** `~30 sec`
I evaluated 3 approaches.

"First approach — hand-crafted rules. Every theme has to be created manually.
No LLM involved.


TODO: List  the pros and the cons clearly in simple words
---

**Slide 3 — Approach 2: TF-IDF + KMeans** `~30 sec`

"Second approach — use K-means clustering but you still have to guess the number of clusters upfront.
Bag-of-words misses meaning — 'outage remediation' and 'post-mortem' look unrelated.
Theme names come out as mechanical word lists."

---

**Slide 4 — Approach 3: Embeddings + HDBSCAN + LLM** `~40 sec`

"Final approach — embed each topic phrase as a vector, let HDBSCAN find the clusters from the data.
26 themes emerged naturally. No K required, density-adaptive.
The LLM names the clusters — it does not form them. The heavy lifting is done by HDBSCAN."

---

**Slide 5 — Database Schema** `~30 sec`

"Two-layer design. Six base tables loaded from raw JSON.
Three semantic tables loaded from clustering.
All insight queries join both layers in one schema — no cross-database joins."

TODO: explain the schema, what are the basic tables, what tables are generated after HDBSCAN + LLM
Why did we choose the schema (see faq and design md files under basics\iprep\meeting-analytics\final_version )

---

**Slide 6 — Meeting Breakdown by Call Type and Product** `~30 sec`

"100 meetings — roughly half support, a quarter renewals, a quarter internal.
Detect and Comply appear in the most meetings across the dataset.
This sets the scale before any analysis begins."


TODO: explain which tables these insights are generated from and why it is important

---

**Slide 7 — UMAP Scatter: 343 Topic Phrases, 26 Clusters** `~40 sec`

"Each dot is a topic phrase from meeting summaries, coloured by cluster.
26 themes emerged naturally — no K was specified.
Detect phrases cluster tightly together. Comply sits as a separate island."

---

**Slide 8 — Theme Sentiment Heatmap** `~40 sec`

"Sorted most negative at the top, most positive at the bottom.
Detect-related themes dominate the red zone. Comply dominates the green zone.
One product is in crisis — the other is the strongest asset in the portfolio."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 9 — Churn Signals by Product** `~30 sec`

"Churn signals are moments where customers explicitly expressed risk of leaving.
Detect generates far more than any other product.
Direct input for account prioritisation and recovery planning."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 10 — Accounts Needing Immediate Follow-Up** `~40 sec`

"38 meetings with negative sentiment and explicit churn signals.
Ranked by severity — top rows are the most urgent accounts.
This is an action list, not a trend. These accounts need a call this week."


TODO: explain which tables these insights are generated from and why it is important

---

**Slide 11 — Which Products Are Generating the Most Risk?** `~30 sec`

"Three signals per product: total meetings, technical issues, and churn signals.
Detect leads on both technical issues and churn — the outage created a long tail.
This shows which product needs reliability investment first."


TODO: explain which tables these insights are generated from and why it is important

---

**Slide 12 — Where Customers Are Happy** `~30 sec`

"Comply has the highest praise density of any product.
Comply renewal conversations are the most positive in the dataset.
Comply v2 is the good news story to pair with the Detect recovery message."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 13 — How the Detect Outage Affected Renewal Calls** `~40 sec`

"Renewal meetings mentioning Detect are significantly more negative than those that do not.
Over half of Detect renewal meetings carry an explicit churn signal.
Outage conversations are bleeding into commercial calls that should be about growth."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 14 — What Customers Are Asking For and How Urgently** `~40 sec`

"The same feature request carries completely different urgency depending on the account situation.
Blocked in red — at-risk customer, treat as P0, act immediately.
Growing in green — healthy account asking for more, add it to the roadmap."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 15 — Who Owns the Follow-Up Work?** `~40 sec`

"Left panel shows which themes generate the most follow-up actions across all meetings.
Engineering carries the heaviest Detect load — reliability slips delay Comply v2.
This identifies where capacity bottlenecks sit before they become a problem."

TODO: explain which tables these insights are generated from and why it is important

---

**Slide 16 — Three Things to Act On** `~40 sec`

"One: fix Detect reliability before the next QBR — it is the number one churn driver across the dataset.
Two: call the 38 high-risk accounts this week — negative sentiment, explicit churn signals.
Three: lead with Comply v2 for at-risk Detect accounts — it is the strongest counter-narrative in the data."

TODO: explain which tables these insights are generated from and why it is important

