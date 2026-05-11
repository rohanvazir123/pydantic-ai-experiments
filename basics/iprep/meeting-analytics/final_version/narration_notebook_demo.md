# Notebook Demo — Narration Guide

Recording: screen (Jupyter notebook) + Yeti audio only. No camera.
Target length: **7–10 minutes**. Follow the notebook top to bottom, Shift+Enter through each cell.

---

## Recording Setup

- Full screen: Jupyter notebook in browser — this is what OBS captures
- Second monitor or phone: this file open for reference
- OBS source: **Window Capture → Jupyter browser window only**
- No camera, audio only

---

## OBS + Yeti

- Yeti dial: **Cardioid** (circle icon), speak into the front face, 6–8 inches away
- OBS Audio Mixer: levels **-12 to -6 dB**. Never red. Mute Desktop Audio.
- Do a 10-second test recording before starting for real

---

## Walkthrough

---

### Title cell `~15 sec`

*Scroll past the watermark line to the title.*

"Hi, this is an analysis of 100 AegisCloud customer meeting transcripts to classify themes
and surface actionable insights for different teams.
My name is Vikrant Potnis, presenting for the end client Rubrik, brought to me by Aziro."

---

### Design Decisions cell `~2 min`

*Scroll slowly through the cell. Pause on each approach heading.*

**On Approach 1:**
"First approach — hand-crafted rules. Simple and deterministic, but not scalable.
Every new theme or shift in customer language needs a manual update.
It also breaks silently — a misclassified meeting produces no error, just wrong output.
'Pipeline failure' and 'ingestion outage' mean the same thing — the rules treat them as different."

**On Approach 2:**
"Second approach — KMeans clustering on bag-of-words vectors. More systematic, but three problems.
You have to guess the number of themes upfront. It treats words literally, so 'outage remediation'
and 'post-mortem' look completely unrelated. And theme names come out as mechanical word lists
like 'renewal / competitive / pricing / outage' — which tells you nothing."

**On Approach 3:**
"Final approach — embed each topic phrase as a vector, let HDBSCAN find the clusters from the data.
26 themes emerged naturally. No K required.
The LLM names the clusters — it does not form them. The heavy lifting is done by HDBSCAN."

**On Database Schema:**
"Two-layer design. Six base tables loaded from the raw JSON — meetings, summaries, key moments,
action items, participants, transcripts.
Three semantic tables generated after the clustering runs — the 26 clusters, the 343 topic phrases,
and the meeting-to-theme assignments.
Two separate layers means you can reload raw data without touching clustering, and vice versa."

---

### Setup cell `~10 sec`

*Run the cell. It prints 'Ready.'*

"This sets up the database connection and the helper functions we use throughout."

---

### Helpers + Load cells `~20 sec`

*Run helpers cell (prints 'Helpers defined.'), then the load cell.*

"These functions read the raw JSON folders and load all nine tables into Postgres.
Running it now — you can see the row counts as each table is populated."

*Wait for it to finish.*

"100 meetings, 311 participants, 4,313 transcript lines — all loaded."

---

### Verify cell `~10 sec`

*Run it. All PASS.*

"Quick sanity check — all nine tables have the expected row counts. All pass. Ready to analyse."

---

### Chart Summary `~15 sec`

*Scroll to the Chart Summary table.*

"Here is an overview of all eleven charts we will go through — what each one shows and which team it is for.
The links are clickable if you want to jump directly to a section."

---

### Section 3 — Meeting Breakdown `~30 sec`

*Run the cell. Chart appears.*

"100 meetings — roughly half support calls, a quarter renewal and account meetings, a quarter internal.
Detect and Comply appear in the most meetings across the dataset.
This sets the scale before any analysis begins.

Call type split comes from the meeting themes table. Product mentions come from meeting summaries —
the products column is extracted at load time by scanning each summary for known product names."

---

### Section 4 — UMAP Scatter `~30 sec`

*Run the cell. Scatter plot appears.*

"Each dot is a topic phrase from meeting summaries, coloured by cluster.
26 themes emerged naturally — no K was specified.
Detect phrases cluster tightly in one region. Comply sits as a completely separate island.

The 2D coordinates are pre-computed for visualisation only.
The actual clustering ran on a 10-dimensional reduction to get better separation."

---

### 26 Themes Table `~20 sec`

*Run the cell. Styled table appears.*

"Here are all 26 discovered themes, ranked by how many meetings each one dominates.
The bar indicator on the right shows relative meeting count.
Audience column shows which team each theme is most relevant to."

---

### Section 5 — Theme Sentiment Heatmap `~40 sec`

*Run the cell. Heatmap appears.*

"Sorted most negative at the top, most positive at the bottom.
Detect-related themes dominate the red zone. Comply dominates the green zone.
One product is in crisis — the other is the strongest asset in the portfolio.

This comes from joining the meeting themes table with the clusters table to get each meeting's
primary theme and sentiment. It is the single most important chart —
it shows the full sentiment picture across all 26 themes in one view."

---

### Section 6.1 — Churn Signals by Product `~30 sec`

*Run the cell.*

"Churn signals are moments in the transcripts where customers explicitly expressed risk of leaving.
Detect generates far more than any other product.
Direct input for account prioritisation and recovery planning.

This joins the products column in meeting summaries with the key moments table,
filtering to churn signal moments only, grouped by product."

---

### Section 6.2 — Accounts Needing Immediate Follow-Up `~40 sec`

*Run the cell. Table appears.*

"38 meetings with negative sentiment and at least one explicit churn signal.
Ranked by severity — top rows are the most urgent accounts.
This is an action list, not a trend. These accounts need a call this week.

Built by joining meetings, meeting themes, and key moments — filtering to primary themes
with negative sentiment that also have churn signal moments. The signal count is what ranks the list."

---

### Section 6.3 — Which Products Are Generating the Most Risk? `~30 sec`

*Run the cell.*

"Three signals per product: total meetings, technical issues, and churn signals.
Detect leads on both — the March outage created a long tail of follow-up work.
This shows which product needs reliability investment first.

Product attribution comes from meeting summaries. Technical issues and churn signals
are counted separately from the key moments table."

---

### Section 6.4 — Where Customers Are Happy `~30 sec`

*Run the cell.*

"Comply has the highest praise density of any product.
Comply renewal conversations are the most positive in the dataset.
This is the good news story to lead with for at-risk Detect accounts.

Left chart: praise moments from key moments, grouped by product.
Right chart: meeting themes filtered to Comply, external calls only — the renewal conversations."

---

### Section 7.1 — How the Detect Outage Affected Renewal Calls `~40 sec`

*Run the cell.*

"Renewal meetings that mention Detect are significantly more negative than those that don't.
Over half carry an explicit churn signal.
Outage conversations are bleeding into commercial calls that should be about growth.

This filters meeting themes to external calls only, then splits by whether Detect appears in the products column.
The pie chart counts churn signal key moments within the Detect-tagged group."

---

### Section 7.2 — What Customers Are Asking For and How Urgently `~40 sec`

*Run the cell.*

"The same feature request carries completely different urgency depending on the account situation.
Blocked in red — at-risk customer, treat as P0, act immediately.
Growing in green — healthy account asking for more, add it to the roadmap.

This joins the products column with feature gap key moments, then groups by the overall sentiment of that meeting.
The sentiment of the meeting — not the request itself — determines the urgency.
Same feature, completely different priority."

---

### Section 7.3 — Who Owns the Follow-Up Work? `~40 sec`

*Run the cell.*

"Left panel shows which themes generate the most follow-up actions.
Engineering carries the heaviest Detect load — if reliability slips, Comply v2 delivery slips with it.
This identifies where bottlenecks sit before they become a problem.

This uses a view that joins action items to meeting themes on the primary theme,
then to the clusters table for the label. The right panel adds the products column
to show ownership broken down by product."

---

### Closing `~30 sec`

*Stay on the last chart.*

"Three things to act on.
Fix Detect reliability before the next QBR — it is the number one churn driver across the dataset.
Call the 38 high-risk accounts this week — the list is already ranked by urgency.
And lead with Comply v2 for at-risk Detect accounts — when a customer is upset about the outage,
open with the Comply v2 story. Give them a forward-looking reason to stay, not just an apology.

Everything here came directly from the transcripts — no manual tagging, no survey data."
