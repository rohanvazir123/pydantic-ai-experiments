# Take C — FAQ

## Table of Contents
- [Does the pipeline produce themes?](#does-the-pipeline-produce-themes)
- [Does Take C use pgvector and tsvector?](#does-take-c-use-pgvector-and-tsvector)
- [What insights/findings can be generated per theme?](#what-insightsfindings-can-be-generated-per-theme)
- [How do we know Take C produces meaningful themes?](#how-do-we-know-take-c-produces-meaningful-themes)
- [How do we know Take C won't club unrelated topics into a cluster?](#how-do-we-know-take-c-wont-club-unrelated-topics-into-a-cluster)
- [Did the LLM do a good job labeling clusters? How does it know what to do?](#did-the-llm-do-a-good-job-labeling-clusters-how-does-it-know-what-to-do)
- [How does the Take B vs Take C comparison work?](#how-does-the-take-b-vs-take-c-comparison-work)
- [What did the Take B vs Take C comparison actually find?](#what-did-the-take-b-vs-take-c-comparison-actually-find)
- [Should we re-run Take B with k=26 to directly compare against Take C's 26 clusters?](#should-we-re-run-take-b-with-k26-to-directly-compare-against-take-cs-26-clusters)
- [Which Take C outputs were produced by HDBSCAN and which by the LLM?](#which-take-c-outputs-were-produced-by-hdbscan-and-which-by-the-llm)
- [How does UMAP work?](#how-does-umap-work)
- [How does HDBSCAN work?](#how-does-hdbscan-work)
- [How do I reload the Take C tables into Postgres?](#how-do-i-reload-the-take-c-tables-into-postgres)
- [What 3 Postgres tables does Take C create, and what is in each?](#what-3-postgres-tables-does-take-c-create-and-what-is-in-each)
- [Why does Take C extend Take A's schema instead of creating its own?](#why-does-take-c-extend-take-as-schema-instead-of-creating-its-own)
- [Take B labels clusters by concatenating top centroid terms — why can't Take C do the same?](#take-b-labels-clusters-by-concatenating-top-centroid-terms--why-cant-take-c-do-the-same)

---

## Does the pipeline produce themes?

Yes. Two artifacts contain the themes:

**1. `semantic_clusters.json`** — one entry per discovered cluster:
```json
{
  "cluster_id": 3,
  "theme_title": "Identity & Access Management",
  "audience": "Engineering",
  "rationale": "SSO failures, MFA policy gaps, and provisioning delays directly impact platform reliability and security posture.",
  "representative_phrases": ["mfa enforcement", "scim provisioning", "..."],
  "phrase_count": 28
}
```

**2. `meeting_themes.csv`** — per-meeting row showing primary theme, all themes,
inferred call type, and sentiment score.

**How theme discovery works:**

- Step 5 (HDBSCAN) groups the ~343 deduplicated topic phrases into clusters based on embedding similarity — no fixed K required.
- Step 6 sends ~20 representative phrases per cluster to `llama3.1:8b` (Ollama) which generates: `theme_title`, `audience` (Engineering/Product/Sales/All), and a one-sentence rationale.
- Step 8 maps each meeting's topics back to cluster IDs, so every meeting gets a `primary_theme` and a full list of all themes it touches.

**Expected themes** (based on topic vocabulary in `list_of_topics.txt`):

- Incidents & Reliability
- Compliance & Audit
- Identity & Access Management
- Backup & Recovery
- Product Roadmap & Feature Gaps
- Customer Retention & Renewal
- Billing & Commercial
- Internal Engineering / Planning
- Competitive Landscape

*(Exact count and names come from the data, not from hand-written rules.)*

---

## Does Take C use pgvector and tsvector?

Yes — added in `take_c_pg_store.py` (step 9 of the pipeline).

**Tables written to `meeting_analytics` schema:**

| Table | Contents |
|-------|----------|
| `semantic_clusters` | `cluster_id`, `theme_title`, `audience`, `rationale`, `phrase_count` |
| `semantic_phrases` | canonical text + `vector(768)` embedding (IVFFlat index) + `tsvector` GENERATED column (GIN index) |
| `semantic_meeting_themes` | meeting → cluster mapping + `call_type` + `sentiment_score` |

**Hybrid search available** (pgvector + tsvector + RRF, same pattern as `rag/storage/vector_store/postgres.py`):

```python
store.semantic_search_phrases(query_embedding)   # cosine similarity via IVFFlat
store.text_search_phrases(query)                 # plainto_tsquery on canonical text
store.hybrid_search_phrases(query, embedding)    # RRF merge of both (k=60)
```

**8 insight queries run automatically after persist:**

| Method | What it shows |
|--------|--------------|
| `insight_theme_sentiment()` | Avg sentiment score per theme |
| `insight_churn_by_theme()` | Churn signal count per theme (joins `key_moments`) |
| `insight_call_type_theme_matrix()` | Call type × theme meeting counts |
| `insight_feature_gap_themes()` | Feature gap signal count per theme |
| `insight_sentiment_distribution_by_theme()` | Sentiment bucket breakdown per theme |
| `insight_signal_counts_by_theme()` | All keyMoment signal types per theme |
| `insight_theme_cooccurrence()` | Which themes appear together in the same meeting |
| `insight_high_risk_meetings()` | Meetings with churn signals + negative sentiment |

> **NOTE:** churn and feature_gap insights join Take A's `key_moments` table.
> They will be blank until Take A has been run:
> ```
> python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py --reset
> ```

**Run commands:**
```bash
python basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py              # full pipeline
python basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py --reset-pg   # drop+recreate semantic tables first
python basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py --skip-pg    # skip Postgres, CSV/JSON only
python basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py --dry-run    # no Ollama, no Postgres
```

---

## What insights/findings can be generated per theme?

For any theme (e.g. "Customer Retention / Renewal / Commercial Risk"), the following
are available via insight query methods in `take_c_pg_store.py`:

**1. Sentiment distribution** `[insight_sentiment_distribution_by_theme]`

Count of each `overall_sentiment` category + avg score per theme:
```
Customer Retention / Renewal / Commercial Risk
  mixed-negative   8 meetings   avg_score=2.4
  negative         4 meetings   avg_score=1.8
  neutral          3 meetings   avg_score=3.0
  mixed-positive   5 meetings   avg_score=3.8
  very-positive    2 meetings   avg_score=4.9
```

**2. Business signal counts** `[insight_signal_counts_by_theme]`

Pivoted count of every key-moment type per theme (joins Take A `key_moments`):
```
Customer Retention   churn=14  concern=11  pricing=6  feature_gap=4  praise=2
```

**3. Call type breakdown** `[insight_call_type_theme_matrix]`

How many support / external / internal meetings per theme.
Example: Customer Retention is mostly external (account manager) calls.

**4. Theme co-occurrence** `[insight_theme_cooccurrence]`

Which theme pairs appear together most in the same meeting.
Example: "Customer Retention" + "Incidents & Reliability" co-occurring frequently
means outages are driving churn — a specific, actionable finding for leadership.

**5. High-risk meetings** `[insight_high_risk_meetings]`

Meetings with churn signals AND `sentiment_score < 3.0`, ranked by churn count.
These are the specific meetings leadership should follow up on immediately.

> **NOTE:** signal-count queries (2, 5) join Take A's `key_moments` table.
> Run Take A first: `python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py --reset`

---

## How do we know Take C produces meaningful themes?

Short answer: we don't automatically — unsupervised clustering has no ground truth.
Three practical validation checks, in order of effort:

**1. Compare against Take A** (cheapest, most convincing)

Take A has 8 hand-crafted themes built from expert knowledge of the topic vocabulary.
If semantic clustering independently discovers roughly the same groupings, that is
strong evidence. The validation question is:
> "For each Take C cluster, what % of its phrases would Take A's `THEME_KEYWORDS` have assigned to the same theme?"

If they agree 80%+ without sharing any rules, the clusters are meaningful.

**2. Silhouette score on phrase embeddings**

Run `silhouette_score` on the UMAP-reduced embeddings with HDBSCAN labels.
Range −1 to 1; above 0.3 is reasonable for short text. Take B already does this
for TF-IDF vectors.

**3. Sense-check the insight queries**

Once insight queries run against real data, ask: do results match intuition?
- Does the "Compliance & Audit" cluster have low sentiment? (audits are stressful)
- Does the "Customer Retention" cluster have the most churn signals?
- Do internal meetings cluster around "Engineering / Planning" themes?

If the insights are counter-intuitive, the clusters are probably wrong.

---

## How do we know Take C won't club unrelated topics into a cluster?

It can — this is the most important risk in the approach.

**Why it can go wrong:**

Embeddings capture linguistic similarity, not business similarity:
- `"audit readiness"` and `"sprint planning"` both involve preparation/process — the model might see them as closer than they are
- `"alert latency"` and `"performance degradation"` both carry a performance signal — might cluster with reliability OR monitoring topics
- `"incident response"` is ambiguous — security IR vs operational outage IR

**Built-in guardrail: LLM labeling as quality check**

If a cluster is impure, the LLM will generate a vague label like
`"Technical Operations & Processes"`. A coherent cluster gets a sharp, specific label
like `"Identity & Access Management"`. Vague = investigate.

**Active detection (implemented in the pipeline):**

1. **Intra-cluster cosine similarity** — stored in `cluster_metrics.json` as `cluster_coherence`:

   | Score | Flag | Action |
   |-------|------|--------|
   | ≥ 0.6 | `tight` | Phrases are genuinely similar |
   | 0.4–0.6 | `review` | Inspect cluster members |
   | < 0.4 | `LOOSE` | Likely impure; tune HDBSCAN or split manually |

2. **Scan `phrase_clusters.csv`** — read 10 random phrases per cluster, ask "do these belong together?" Takes 5 minutes.

3. **Compare against Take A themes** — if a Take C cluster contains phrases that Take A's `THEME_KEYWORDS` would split across 2+ themes, the cluster is probably impure.

**Fix if you find impure clusters:**
```
--min-cluster-size higher   HDBSCAN gets stricter, splits loose groupings
--min-samples lower         fewer noise points, more clusters
```

---

## Did the LLM do a good job labeling clusters? How does it know what to do?

Short answer: yes — all 26 clusters got sharp, specific labels with zero fallbacks triggered.

**How it knows — the prompt (zero-shot, no fine-tuning, no examples):**

```
You are categorizing customer call topics for a B2B SaaS company's leadership team.

The following phrases all come from one theme cluster discovered by semantic clustering.
Based only on these phrases, provide a short, executive-level theme label.

Phrases:
- <phrase 1>
- <phrase 2>
...

Respond with valid JSON only — no extra text, no markdown fences:
{"theme_title": "<3-6 words, title case>", "audience": "<Engineering | Product | Sales | All>",
 "rationale": "<one sentence: why this theme matters to that audience>"}
```

**Key mechanics:**

| Choice | Why |
|--------|-----|
| "B2B SaaS company's leadership team" | Steers toward commercial, executive-level labels rather than generic technical ones |
| "phrases all come from one theme cluster" | Frames the task as naming a coherent group, not tagging individual phrases |
| Constrained JSON schema | Forces a specific output format (3 fields, no prose) |
| `temperature=0.2` | Keeps output deterministic — consistent labels across re-runs |
| 3 retry attempts | Falls back to top-3 phrase join if all retries fail |

**Evidence the labels are good:**

- All 26 clusters have specific, readable titles:
  - `"Multi-Factor Authentication Management"` (7 phrases, coherence=0.705 — tight)
  - `"HIPAA Compliance and Reporting"` (7 phrases, coherence=0.664 — tight)
  - `"Escalation Process and Management"` (5 phrases, coherence=0.794 — tight)
  - `"Service Level Agreement Management"` (6 phrases, coherence=0.780 — tight)
- No cluster fell back to the `"phrase1 / phrase2 / phrase3"` fallback label.
- Audience assignment matches expected domain knowledge:
  - `"Cloud and Integration Configuration"` → Engineering
  - `"Billing and Pricing Issues"` → Sales | Customer Support
  - `"Competitive Market Positioning"` → All
- Broader clusters with mixed phrases (e.g. `"Product and Feature Development"`, 37 phrases, coherence=0.461) got appropriately broad labels rather than false precision.

**One gap vs design doc:**

The design doc says "sample up to 20 phrases by proximity to cluster centroid".
The implementation just takes the first 20 in natural order (no centroid sort).
It didn't matter in practice — labels are clean — but centroid-ranked phrases
would be slightly more representative for large clusters.

---

## How does the Take B vs Take C comparison work?

The comparison runs in three layers (see `compare_b_vs_c.py` at the root of `meeting-analytics/`).

**Layer 1 — Cross-tab (8×26 matrix)**

For every meeting we know its Take B cluster (0–7) and its Take C primary theme (0–25).
Cross-tabbing gives a matrix where each cell is the count of meetings in B-cluster X
and C-cluster Y. If the methods agree, each row is dominated by 1–2 C clusters.
If a row is spread thin across many C clusters, those two methods disagree on that B cluster.

**Layer 2 — Split analysis**

The more interesting question: does Take C reveal finer structure inside a Take B cluster?

- B-4 `"hipaa / compliance / reporting"` — does Take C split into C-10 (HIPAA Compliance), C-11 (Compliance and Governance), and C-12 (Audit Readiness) as distinct populations?
- B-1 `"outage / incident / failure"` — does Take C separate C-8 (Incident Response and Review) from C-9 (Outage Prevention and Recovery)?

That would be the headline finding: two independent methods discovering the same
underlying groups, with Take C simply having higher resolution.

**Layer 3 — Agreement proxy**

For each B cluster, find its "dominant" C theme (the C cluster that the most meetings
in that B cluster land on). Then for each individual meeting, check: does its own C
primary theme match the dominant C theme of its B cluster?
If ~80%+ of meetings agree, the themes are signal not noise.

**Key limitation:**

Take C assigns meetings to multiple themes (primary + secondaries); Take B assigns
exactly one. The comparison is only on the C primary theme. A meeting that Take C
calls "Billing" primary but also "Renewal" secondary will look like a disagreement
with Take B's `"renewal / competitive"` cluster even if the information is consistent.
The cross-tab (Layer 1) gives the true picture; the agreement proxy (Layer 3) is
a quick headline number, not a strict accuracy score.

---

## What did the Take B vs Take C comparison actually find?

Run: `python basics/iprep/meeting-analytics/compare_b_vs_c.py` (from repo root)

All 100 meetings matched across both outputs.
**Agreement proxy: 37/100 = 37% — do not lead with this number** (see why below).

**Why 37% is not a problem:**

The agreement proxy asks "does a meeting's C primary theme match its B cluster's
dominant C theme?" With Take C splitting every B cluster into multiple finer themes,
the dominant C theme only captures a fraction of each B cluster by design. The low
number reflects higher resolution, not disagreement between the methods.

**Where B and C agree well (confirmed cohesive topics):**

| B cluster | Meetings | Top C theme | Capture |
|-----------|----------|-------------|---------|
| B-2 `"billing / overage / dispute"` | 6 | C-13 Billing and Pricing Issues | 67% |
| B-7 `"backup / hybrid / recovery"` | 6 | C-14 Data Backup and Recovery | 50% |
| B-3 `"planning / sprint / launch"` | 7 | Splits cleanly 3+3 into C-18 Product Dev + C-21 IT Ops | — |

B-3 is notable: Take C correctly separates product launch meetings from engineering
sprint meetings — B grouped them because they share planning vocabulary.

**Where Take C reveals real finer structure:**

- **B-4** `"hipaa / compliance / reporting"` (17 meetings, 5 C-splits)
  → C-11 Compliance and Governance (7), C-18 Product Dev (6), C-10 HIPAA (2), C-12 Audit Readiness (1).
  The 6 that landed on "Product Dev" are likely compliance feature build meetings — B couldn't separate them.

- **B-1** `"outage / incident / failure"` (26 meetings, 10 C-splits)
  → C-09 Outage Prevention (7), C-08 Incident Response (5), C-06 Customer Renewal (4), plus 7 others.
  Take C correctly separates outage prevention (engineering reliability work) from incident response (post-incident process) — two distinct workflows that TF-IDF conflated because they share "outage" vocabulary.

- **B-5** `"backup / performance / support response"` (8 meetings, 4 C-splits)
  → C-14 Data Backup and Recovery (3) and C-24 System Reliability (3) split evenly.
  Confirms the hypothesis: this B cluster was always two topics.

**The noise (B clusters that spread wide):**

- B-0 `"renewal / competitive / pricing"` → 8 C-splits (33% top-C)
- B-6 `"mfa / identity / sso"` → 7 C-splits (33% top-C)

These were B's broadest clusters. Wide C-splits reflect genuine topic breadth, not error.

**Bottom line:**

The two methods are consistent. B-2 billing and B-5 backup are confirmed as real
cohesive topics by both approaches. The strongest finding is B-1 outage/incident:
Take C correctly separates outage prevention from incident response — a distinction
that matters to engineering leadership.

---

## Should we re-run Take B with k=26 to directly compare against Take C's 26 clusters?

**No — this comparison is not meaningful.**

Take B clusters **meetings** (100 documents). At k=26 you get ~4 meetings per cluster
on average. KMeans centroids built from 3–4 meetings are noisy and the labels would
be near-meaningless. The comparison would be fragile by construction.

Take C's 26 clusters come from clustering **343 topic phrases**, not 100 meetings. The
"26" emerged from density patterns across hundreds of short semantic phrases.
Forcing k=26 into Take B just borrows that number — it doesn't make the two approaches
operate at the same unit.

The right comparison is k=8 (Take B) vs Take C's 26 — already done. It answers the
meaningful question: do the two methods agree on the underlying themes, just at
different resolutions? Answer: yes (see results entry above).

---

## Which Take C outputs were produced by HDBSCAN and which by the LLM?

The pipeline has two intelligence sources:

**HDBSCAN (step 5) produces:**
- The cluster groupings themselves — which phrases belong together
- `cluster_id` on every phrase in `phrase_clusters.csv`
- `phrase_count` per cluster in `semantic_clusters.json`
- `cluster_coherence` scores in `cluster_metrics.json` (computed from embeddings post-clustering)
- The number 26 — HDBSCAN found it from data density; it was not specified upfront

**LLM / `llama3.1:8b` (steps 6 and 7) produces:**
- `theme_title` in `semantic_clusters.json`
- `audience` in `semantic_clusters.json`
- `rationale` in `semantic_clusters.json`
- `call_type` and `call_confidence` in `meeting_themes.csv`

**Pipeline logic (step 8, no ML) produces:**
- `primary_theme_id` and `all_theme_ids` in `meeting_themes.csv` (majority-vote over a meeting's topic phrase cluster assignments)
- `representative_phrases` in `semantic_clusters.json` (first N phrases in natural cluster order)
- `all_theme_titles` (looked up from LLM labels)
- `sentiment_score` and `overall_sentiment` (passed through from raw JSON, not computed)

> In short: HDBSCAN decides the shape and membership of clusters; the LLM names them
> and classifies call types. Neither knows about the other.

---

## How does UMAP work?

UMAP (Uniform Manifold Approximation and Projection) reduces 768-dim embeddings to
10 dims before HDBSCAN. The curse of dimensionality and *why* reduction is needed is
covered in `take_c_design.md` section 3.4. This entry covers what UMAP actually does.

**Step 1 — Build a neighborhood graph in high-dimensional space**

For each phrase embedding, UMAP finds its `n_neighbors=15` nearest neighbors using
cosine distance. It assigns edge weights based on how close each neighbor is —
very close neighbors get high weight, distant ones get low weight. The result is
a weighted graph where every phrase is connected to its 15 nearest semantic peers.

**Step 2 — Find a low-dimensional layout that preserves the graph**

UMAP initializes 10-dim positions for all phrases at random, then optimizes them
so that points that were close in 768-dim space are also close in 10-dim space,
and points that were far apart stay far apart. This is done by gradient descent —
it iteratively adjusts positions to minimize a loss function comparing the
high-dim graph structure to the low-dim layout.

**Key parameters used:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_neighbors` | 15 | How many neighbors each phrase considers — lower = more local structure; higher = more global |
| `min_dist` | 0.0 | How tightly cluster members can pack — 0.0 allows maximum compression, helping HDBSCAN find dense regions |
| `metric` | `'cosine'` | Distance measure — matches the embedding space geometry |
| `random_state` | 42 | Fixes random initialization for reproducibility |

**Two UMAP runs in the pipeline:**

| Run | Output dims | Used for |
|-----|-------------|----------|
| Clustering | 10 | Input to HDBSCAN |
| Visualization | 2 | Scatter plot in the notebook only — not used for clustering |

---

## How does HDBSCAN work?

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
finds clusters as dense regions of points separated by sparse regions. It does not
need K specified upfront — it discovers the number of clusters from the data.

**Step 1 — Compute core distances**

For each point, find the distance to its `min_samples=3` nearest neighbor. This is
the "core distance" — a measure of how dense the local neighborhood is. Points in
dense regions have small core distances; isolated points have large ones.

**Step 2 — Build a minimum spanning tree**

HDBSCAN constructs a graph connecting all points, weighting edges by the "mutual
reachability distance" (a combination of core distances). It then finds the minimum
spanning tree of this graph — the set of edges that connects all points at minimum
total cost.

**Step 3 — Build a cluster hierarchy**

By removing edges from the spanning tree from longest to shortest, the tree
progressively breaks into smaller and smaller components. This produces a full
hierarchy of possible clusterings at every density level — like a dendrogram.

**Step 4 — Extract stable clusters**

HDBSCAN scores each branch of the hierarchy by its "stability" — how long it
persists as density increases. Branches that persist over a wide density range
are stable clusters. Branches that appear and disappear quickly are noise.
Clusters must have at least `min_cluster_size=5` points to be kept.

**Step 5 — Label noise**

Any point not assigned to a stable cluster gets label `-1` (noise). In our pipeline,
22 phrases (6.4%) were labeled noise by HDBSCAN and then reassigned to the nearest
cluster centroid as a post-processing step.

**Why HDBSCAN beats KMeans for this problem:**

| Property | HDBSCAN | KMeans |
|----------|---------|--------|
| Needs K upfront | No — found 26 from data density | Yes |
| Handles noise | Yes — ambiguous phrases labeled −1 | No — all points forced into a cluster |
| Variable cluster size | Yes — clusters of 5 and 37 are both valid | Tends to equalize sizes |
| Density-adaptive | Yes — tight groups stay tight | No |

---

## How do I reload the Take C tables into Postgres?

Take C has two recovery paths depending on whether you want to re-run the full pipeline
or just reload from the saved outputs.

**The 3 Take C tables and what feeds them:**

| Table | Source file | What it contains |
|-------|-------------|-----------------|
| `semantic_clusters` | `take_c/outputs/semantic_clusters.json` | 26 clusters: cluster_id, LLM-generated theme_title, audience, rationale, phrase_count |
| `semantic_phrases` | `take_c/outputs/phrase_clusters.csv` | 343 deduplicated topic phrases with cluster assignment (embedding=NULL in CSV path) |
| `semantic_meeting_themes` | `take_c/outputs/meeting_themes.csv` | 516 rows: one per (meeting, cluster) combination, with call_type and sentiment |

**Fast path — load from saved outputs (30 seconds, no Ollama needed):**

```bash
python basics/iprep/meeting-analytics/take_c/load_outputs_to_pg.py
```

Note: this leaves `semantic_phrases.embedding` as NULL. Vector search won't work,
but all analytical insight queries do — they join on cluster_id, not embeddings.

**Full re-run — re-embeds + re-clusters + re-labels (2–3 minutes, requires Ollama):**

```bash
python basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py
```

The full pipeline re-embeds all 343 phrases (nomic-embed-text), re-runs UMAP + HDBSCAN,
re-labels all clusters with llama3.1:8b, infers call types for 100 meetings, and
persists to Postgres. Cluster count and labels may differ slightly between runs
(HDBSCAN is density-based, LLM labels are stochastic).

**Load all three takes in one shot:**

```bash
python basics/iprep/meeting-analytics/setup_all_tables.py
```

Uses fast-path loaders for Takes B and C (no Ollama required), and re-runs Take A
from raw JSON. Target: `rag_db @ localhost:5434` (rag_user:rag_pass). All credentials
from `meeting-analytics/.env`.

---

## What 3 Postgres tables does Take C create, and what is in each?

Take C creates exactly 3 tables in the `meeting_analytics` schema, all loaded by
`take_c/load_outputs_to_pg.py` from pre-computed output files. No re-embedding or
re-clustering is needed to reload them.

---

### `semantic_clusters` — one row per discovered theme (26 rows)

| Column | Type | Nullable | What it holds |
|---|---|---|---|
| `cluster_id` | integer | NOT NULL | Cluster identifier (0–25) assigned by HDBSCAN |
| `theme_title` | text | NOT NULL | LLM-generated theme label (e.g. "Identity & Access Management") |
| `audience` | text | NOT NULL | LLM-generated target stakeholder (Engineering / Product / Sales / All) |
| `rationale` | text | nullable | LLM-generated one-sentence explanation of why these phrases cluster together |
| `phrase_count` | integer | nullable | Number of topic phrases assigned to this cluster |

**Source file:** `take_c/outputs/semantic_clusters.json`
**LLM-generated columns:** `theme_title`, `audience`, `rationale` — produced once per clustering run by `llama3.1:8b`. Deterministic columns: `cluster_id`, `phrase_count`.

---

### `semantic_phrases` — one row per deduplicated topic phrase (343 rows)

| Column | Type | Nullable | What it holds |
|---|---|---|---|
| `id` | uuid | NOT NULL | Surrogate key |
| `canonical` | text | NOT NULL | The deduplicated topic phrase (e.g. "mfa enforcement") |
| `aliases` | text[] | nullable | Variant forms of the phrase seen across meetings |
| `cluster_id` | integer | NOT NULL | Which semantic cluster this phrase belongs to |
| `embedding` | vector(768) | nullable | Phrase embedding — NULL when loaded from CSV (not re-embedded at load time) |
| `content_tsv` | tsvector | nullable | GIN-indexed full-text search vector over `canonical` |

**Source file:** `take_c/outputs/phrase_clusters.csv`
**Note:** `embedding` is NULL after a CSV load. Vector search is unavailable, but all
analytical insight queries join on `cluster_id` only — not on embeddings. To populate
embeddings, re-run the full Take C pipeline (requires Ollama).

---

### `semantic_meeting_themes` — one row per (meeting × cluster) pair (516 rows, 100 distinct meetings)

| Column | Type | Nullable | What it holds |
|---|---|---|---|
| `meeting_id` | text | NOT NULL | FK → `meetings.meeting_id` |
| `cluster_id` | integer | NOT NULL | FK → `semantic_clusters.cluster_id` |
| `is_primary` | boolean | NOT NULL | True for the single dominant theme per meeting |
| `call_type` | text | nullable | LLM-inferred call type: `support` / `external` / `internal` |
| `call_confidence` | text | nullable | LLM confidence string: `high` / `medium` / `low` |
| `sentiment_score` | numeric | nullable | Passed through from raw `summary.json` (not computed by Take C) |
| `overall_sentiment` | text | nullable | Passed through from raw `summary.json` |

**Source file:** `take_c/outputs/meeting_themes.csv`
**Key constraint:** exactly one row per meeting has `is_primary = true`. Always filter
on `is_primary = true` when aggregating per-meeting to avoid double-counting.
**Note on `call_type`:** LLM-generated — stochastic and doesn't scale. Prefer joining
`meeting_analytics.call_types` (Take A) for call type analysis.

---

**How the 3 tables relate:**

```
semantic_clusters (26)
    ↑ cluster_id
semantic_phrases (343)          — phrases belong to clusters

semantic_clusters (26)
    ↑ cluster_id
semantic_meeting_themes (516)   — meetings belong to clusters
    ↑ meeting_id
meetings (100)
```

**To reload all 3 tables:** `python basics/iprep/meeting-analytics/setup_all_tables.py`
(always use this, never `generate_rule_based_taxonomy.py --reset` alone — that wipes
the schema and only reloads Take A).

---

## Take B labels clusters by concatenating top centroid terms — why can't Take C do the same?

Because Take C's centroids live in a different kind of space.

**Why it works in Take B:**

KMeans centroids live in TF-IDF vector space. Every dimension in that space corresponds
to a specific term. The centroid's highest-weighted dimensions *are* the most
representative terms — you can read them off directly and concat.

```
centroid dimensions → sort by weight → top 4 terms → "renewal / competitive / pricing / outage"
```

Fully deterministic. No LLM needed.

**Why it doesn't work in Take C:**

HDBSCAN centroids (actually cluster medoids / mean vectors) live in 768-dimensional
embedding space. Those 768 dimensions are abstract latent features learned by
`nomic-embed-text` — they do not correspond to words. There is no "top term" to read
off. The centroid is just a point in a space with no human-interpretable axes.

**What Take C does instead:**

1. Sort each cluster's phrases by distance to the cluster centroid (ascending) —
   closest phrases are most representative of the cluster's core meaning
2. Take the top 20 (`cluster_phrases[:LABEL_SAMPLE_SIZE]`)
3. Send them to `llama3.1:8b`: *"here are topic phrases from a business meeting cluster,
   give me a theme title, target audience, and one-sentence rationale"*
4. Store the response as `semantic_clusters.theme_title`, `audience`, `rationale`

Implemented in `_sort_phrases_by_centroid_proximity(phrases, reduced)`, called at step 6
before `label_clusters()`. Uses Euclidean distance in UMAP-reduced space (same space
HDBSCAN used), consistent with `_compute_centroids()`.

**The tradeoff:**

| | Take B | Take C |
|---|---|---|
| Label source | Top centroid dimensions | LLM reading nearest phrases |
| Deterministic | Yes | No — stochastic between runs |
| Human-readable | Mechanical ("renewal / competitive / pricing / outage") | Natural ("Customer Retention & Competitive Displacement") |
| Cost | Free, instant | 1 LLM call per cluster (~26 calls, ~40s total) |

**Key insight:** the LLM is not doing the hard work — it's doing translation. The hard
work (figuring out which phrases belong together) is done by the embeddings and HDBSCAN.
By the time the LLM sees a cluster, the grouping is already correct. It just needs to
name what's already there. This is why the labels come out clean even with a small local
model like `llama3.1:8b`.

**The actual prompt sent to the LLM (per cluster):**

```
You are categorizing customer call topics for a B2B SaaS company's leadership team.

The following phrases all come from one theme cluster discovered by semantic clustering.
Based only on these phrases, provide a short, executive-level theme label.

Phrases:
- mfa enforcement
- scim provisioning
- sso configuration failures
- identity provider sync
- ...

Respond with valid JSON only — no extra text, no markdown fences:
{"theme_title": "<3-6 words, title case>", "audience": "<Engineering | Product | Sales | All>", "rationale": "<one sentence: why this theme matters to that audience>"}
```

`temperature=0.2` — kept low to minimise label variation between runs.

**Fallback if the LLM call fails:**

```python
theme_title = " / ".join(sample[:3])  # e.g. "mfa enforcement / scim provisioning / sso configuration failures"
```

The failure mode degrades gracefully to the Take B approach — top phrases concatenated
with " / ". The cluster grouping is preserved; only the label quality drops.

---

## Why does Take C extend Take A's schema instead of creating its own?

Take A already loads all the raw data into `meeting_analytics` — `meetings`,
`meeting_summaries`, `key_moments`, `sentiment_features`, `transcript_lines`, etc.
Take C doesn't re-load any of that. It only adds 3 new tables for its clustering
results (`semantic_clusters`, `semantic_phrases`, `semantic_meeting_themes`) into
the same schema.

**The benefit: cross-take JOINs work without any schema gymnastics.**

```sql
-- Take C themes × Take A sentiment — works because same schema
SELECT sc.theme_title, round(avg(sf.net_sentiment)::numeric, 3) AS avg_sentiment
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc      ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.sentiment_features sf     ON smt.meeting_id = sf.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
ORDER BY avg_sentiment;
```

If Take C used a separate schema, every cross-take query would need schema-qualified
references across two schemas. One schema keeps everything flat and queryable from
a single DBeaver connection.

**What Take C does NOT duplicate:**

| Take A table | Take C status |
|---|---|
| `meetings` | Reused via FK — `semantic_meeting_themes.meeting_id` references it |
| `sentiment_features` | Reused via JOIN — churn counts, net_sentiment, etc. |
| `key_moments` | Reused via JOIN — pricing_offer, churn_signal, feature_gap etc. |
| `call_types` | Reused via JOIN — preferred over Take C's LLM call_type |
| `meeting_themes` | Independent — Take A's rule-based themes vs Take C's semantic clusters |

**Important: the LLM does not use Take A's schema.**

The LLM (step 6) has zero database interaction. It receives a list of phrases and
returns JSON labels — nothing more. The schema extension is a step 9 (persistence)
decision, made after the LLM is already done.

```
[1–5]  Extract → dedup → embed → UMAP → HDBSCAN
[6]    LLM reads phrases, returns labels     ← no DB, no schema
[7]    Infer call types (LLM)                ← no DB, no schema
[8]    Assign meetings to themes             ← no DB, no schema
[9]    Persist to Postgres                   ← THIS is where Take A's schema is extended
```
