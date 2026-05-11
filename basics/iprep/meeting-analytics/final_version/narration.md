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

"Hi, I analysed 100 AegisCloud customer meetings to classify themes and surface actionable insights
for different teams — engineering, product, sales, and customer success."

---

**Slide 2 — Approach 1: Rule-Based Keyword Matching** `~30 sec`

"First approach — hand-crafted rules. Every theme is defined manually using keywords and regex matching.

Simple to understand and fully deterministic — same input always gives the same output, no models required.

But it is not scalable. Every new theme, product, or shift in customer language needs a manual update.
And it breaks silently — a misclassified meeting produces no error, just wrong output.
The same concept expressed differently — 'pipeline failure' versus 'ingestion outage' — gets treated as completely different."

---

**Slide 3 — Approach 2: TF-IDF + KMeans** `~30 sec`

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

The six base tables are loaded directly from the raw meeting JSON — meetings, participants, summaries,
key moments, action items, and transcript lines. These capture everything in the transcripts:
sentiment scores, signal moments like churn or praise, and the topic phrases extracted during summarisation.

The three semantic tables are generated after HDBSCAN and the LLM run:
semantic_clusters holds the 26 discovered themes with their LLM-generated title and audience.
semantic_phrases maps each of the 343 topic phrases to its cluster.
semantic_meeting_themes maps each meeting to its themes, with call type and sentiment.

The reason for two separate layers: base tables can be reloaded from raw JSON at any time
without touching the clustering. Semantic tables can be rebuilt from the clustering outputs without
re-running the full pipeline. And all insight queries join both layers in a single schema —
no cross-database joins needed."

---

**Slide 6 — Meeting Breakdown by Call Type and Product** `~30 sec`

"100 meetings — roughly half support, a quarter renewals, a quarter internal.
Detect and Comply appear in the most meetings across the dataset.
This sets the scale before any analysis begins.

This comes from two tables: semantic_meeting_themes for the call type split,
and meeting_summaries for the product mentions — the products array is extracted
at load time by scanning each summary for known product names."

---

**Slide 7 — UMAP Scatter: 343 Topic Phrases, 26 Clusters** `~40 sec`

"Each dot is a topic phrase from meeting summaries, coloured by cluster.
26 themes emerged naturally — no K was specified.
Detect phrases cluster tightly together. Comply sits as a separate island.

The coordinates come from a pre-computed UMAP run — a separate 2D reduction done for
visualisation only. The actual clustering ran on a 10-dimensional UMAP to avoid the
curse of dimensionality."

---

**Slide 8 — Theme Sentiment Heatmap** `~40 sec`

"Sorted most negative at the top, most positive at the bottom.
Detect-related themes dominate the red zone. Comply dominates the green zone.
One product is in crisis — the other is the strongest asset in the portfolio.

This joins semantic_meeting_themes with semantic_clusters to get the overall_sentiment
for each meeting's primary theme. It is the single most important chart in the deck —
it shows the full picture in one view."

---

**Slide 9 — Churn Signals by Product** `~30 sec`

"Churn signals are moments where customers explicitly expressed risk of leaving.
Detect generates far more than any other product.
Direct input for account prioritisation and recovery planning.

This joins the meeting_summaries products array with key_moments filtered to
moment_type equals churn_signal. The products array was extracted at load time —
so we know which product each meeting is about."

---

**Slide 10 — Accounts Needing Immediate Follow-Up** `~40 sec`

"38 meetings with negative sentiment and explicit churn signals.
Ranked by severity — top rows are the most urgent accounts.
This is an action list, not a trend. These accounts need a call this week.

This joins meetings, semantic_meeting_themes, semantic_clusters, and key_moments.
The filter is: primary theme with negative sentiment AND at least one churn_signal key moment.
The churn signal count is what ranks the list."

---

**Slide 11 — Which Products Are Generating the Most Risk?** `~30 sec`

"Three signals per product: total meetings, technical issues, and churn signals.
Detect leads on both technical issues and churn — the March outage created a long tail.
This shows which product needs reliability investment first.

The source is meeting_summaries for product attribution and key_moments for the two signal types —
technical_issue and churn_signal — each counted separately per product."

---

**Slide 12 — Where Customers Are Happy** `~30 sec`

"Comply has the highest praise density of any product.
Comply renewal conversations are the most positive in the dataset.
Comply v2 is the good news story to pair with the Detect recovery message.

Left chart: key_moments filtered to moment_type equals praise, grouped by product.
Right chart: semantic_meeting_themes filtered to Comply in the products array,
external call type only — these are the renewal and account conversations."

---

**Slide 13 — How the Detect Outage Affected Renewal Calls** `~40 sec`

"Renewal meetings mentioning Detect are significantly more negative than those that do not.
Over half of Detect renewal meetings carry an explicit churn signal.
Outage conversations are bleeding into commercial calls that should be about growth.

This filters semantic_meeting_themes to external call type only, then splits by whether Detect
appears in the products array. The pie chart queries key_moments for churn_signal
in that Detect-tagged external cohort."

---

**Slide 14 — What Customers Are Asking For and How Urgently** `~40 sec`

"The same feature request carries completely different urgency depending on the account situation.
Blocked in red — at-risk customer, treat as P0, act immediately.
Growing in green — healthy account asking for more, add it to the roadmap.

This joins the meeting_summaries products array with key_moments filtered to feature_gap,
then groups by the overall_sentiment of that meeting. The sentiment of the meeting — not the request —
is what determines urgency. Same feature, completely different priority."

---

**Slide 15 — Who Owns the Follow-Up Work?** `~40 sec`

"Left panel shows which themes generate the most follow-up actions across all meetings.
Engineering carries the heaviest Detect load — reliability slips delay Comply v2.
This identifies where capacity bottlenecks sit before they become a problem.

This uses the action_items_by_theme view, which joins action_items to semantic_meeting_themes
on is_primary equals true, then to semantic_clusters for the theme label.
The right panel adds the products array from meeting_summaries to show the product breakdown per department."

---

**Slide 16 — Three Things to Act On** `~40 sec`

"One: fix Detect reliability before the next QBR — it is the number one churn driver across the dataset.

Two: call the 38 high-risk accounts this week — these meetings have negative sentiment and explicit churn signals. The list is already ranked by urgency.

Three: lead with Comply v2 for at-risk Detect accounts. Here is what this means in practice — when a customer is upset about the Detect outage, do not walk into that renewal call with only an apology and a recovery plan. Open the conversation with the Comply v2 story. Give them a forward-looking reason to stay before you address what went wrong. The data shows Comply has the highest customer satisfaction of any product. Customers asking about Comply are in growth mode. Pair the bad news with good news.

Everything on these slides came directly from the transcripts — no manual tagging, no survey data."
