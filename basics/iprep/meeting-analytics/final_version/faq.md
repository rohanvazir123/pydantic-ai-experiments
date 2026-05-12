# Final Version — FAQ

## Table of Contents
- [Does the pipeline produce themes?](#does-the-pipeline-produce-themes)
- [Does Final Version use pgvector and tsvector?](#does-final-version-use-pgvector-and-tsvector)
- [What insights/findings can be generated per theme?](#what-insightsfindings-can-be-generated-per-theme)
- [How do we know Final Version produces meaningful themes?](#how-do-we-know-final-version-produces-meaningful-themes)
- [How do we know Final Version won't club unrelated topics into a cluster?](#how-do-we-know-final-version-wont-club-unrelated-topics-into-a-cluster)
- [Did the LLM do a good job labeling clusters? How does it know what to do?](#did-the-llm-do-a-good-job-labeling-clusters-how-does-it-know-what-to-do)
- [Which Final Version outputs were produced by HDBSCAN and which by the LLM?](#which-final-version-outputs-were-produced-by-hdbscan-and-which-by-the-llm)
- [How does UMAP work?](#how-does-umap-work)
- [How does HDBSCAN work?](#how-does-hdbscan-work)
- [How do I reload the Final Version tables into Postgres?](#how-do-i-reload-the-final-version-tables-into-postgres)
- [What 3 Postgres tables does Final Version create, and what is in each?](#what-3-postgres-tables-does-final-version-create-and-what-is-in-each)
- [Why can't we label clusters by concatenating top centroid terms?](#why-cant-we-label-clusters-by-concatenating-top-centroid-terms)
- [Why does semantic_meeting_themes store one row per meeting × cluster instead of a list of meeting_ids per cluster?](#why-does-semantic_meeting_themes-store-one-row-per-meeting--cluster-instead-of-a-list-of-meeting_ids-per-cluster)
- [What are feature gap moments labelled "growing/positive"?](#what-are-feature-gap-moments-labelled-growingpositive)
- [Why aren't phrase embeddings saved to disk for reuse?](#why-arent-phrase-embeddings-saved-to-disk-for-reuse)

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

## Does Final Version use pgvector and tsvector?

Yes — added in `load_output_csvs_to_db.py` (step 9 of the pipeline).

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

**Run commands:**
```bash
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py              # full pipeline
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py --reset-pg   # drop+recreate semantic tables first
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py --skip-pg    # skip Postgres, CSV/JSON only
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py --dry-run    # no Ollama, no Postgres
```

---

## What insights/findings can be generated per theme?

For any theme (e.g. "Customer Retention / Renewal / Commercial Risk"), the following
are available via insight query methods in `load_output_csvs_to_db.py`:

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


---

## How do we know Final Version produces meaningful themes?

Short answer: we don't automatically — unsupervised clustering has no ground truth.
Three practical validation checks, in order of effort:

**1. Silhouette score on phrase embeddings**

Run `silhouette_score` on the UMAP-reduced embeddings with HDBSCAN labels.
Range −1 to 1; above 0.3 is reasonable for short text.

**2. Sense-check the insight queries**

Once insight queries run against real data, ask: do results match intuition?
- Does the "Compliance & Audit" cluster have low sentiment? (audits are stressful)
- Does the "Customer Retention" cluster have the most churn signals?
- Do internal meetings cluster around "Engineering / Planning" themes?

If the insights are counter-intuitive, the clusters are probably wrong.

---

## How do we know Final Version won't club unrelated topics into a cluster?

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

## Which Final Version outputs were produced by HDBSCAN and which by the LLM?

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
covered in `design.md` section 3.4. This entry covers what UMAP actually does.

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

## How do I reload the Final Version tables into Postgres?

Final Version has two recovery paths depending on whether you want to re-run the full pipeline
or just reload from the saved outputs.

**The 3 Final Version tables and what feeds them:**

| Table | Source file | What it contains |
|-------|-------------|-----------------|
| `semantic_clusters` | `final_version/outputs/semantic_clusters.json` | 26 clusters: cluster_id, LLM-generated theme_title, audience, rationale, phrase_count |
| `semantic_phrases` | `final_version/outputs/phrase_clusters.csv` | 343 deduplicated topic phrases with cluster assignment (embedding=NULL in CSV path) |
| `semantic_meeting_themes` | `final_version/outputs/meeting_themes.csv` | 516 rows: one per (meeting, cluster) combination, with call_type and sentiment |

**Fast path — load from saved outputs (30 seconds, no Ollama needed):**

```bash
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py --from-outputs --skip-base-load
```

Note: this leaves `semantic_phrases.embedding` as NULL. Vector search won't work,
but all analytical insight queries do — they join on cluster_id, not embeddings.

**Full re-run — re-embeds + re-clusters + re-labels (2–3 minutes, requires Ollama):**

```bash
python basics/iprep/meeting-analytics/final_version/semantic_clustering.py
```

The full pipeline re-embeds all 343 phrases (nomic-embed-text), re-runs UMAP + HDBSCAN,
re-labels all clusters with llama3.1:8b, infers call types for 100 meetings, and
persists to Postgres. Cluster count and labels may differ slightly between runs
(HDBSCAN is density-based, LLM labels are stochastic).

**Full reset — reload all 9 tables:**

```bash
python basics/iprep/meeting-analytics/final_version/load_raw_jsons_to_db.py --reset
python basics/iprep/meeting-analytics/final_version/load_output_csvs_to_db.py --reset
```

Reloads all 9 tables from scratch — base tables from raw JSON, semantic tables from
`outputs/` CSVs (no Ollama required). Target: `rag_db @ localhost:5434` (rag_user:rag_pass).
All credentials from `meeting-analytics/.env`.

---

## What 3 Postgres tables does Final Version create, and what is in each?

The Final Version creates exactly 3 semantic tables in the `meeting_analytics` schema.
They can be reloaded from pre-computed output files via `--from-outputs` without
re-embedding or re-clustering.

---

### `semantic_clusters` — one row per discovered theme (26 rows)

| Column | Type | Nullable | What it holds |
|---|---|---|---|
| `cluster_id` | integer | NOT NULL | Cluster identifier (0–25) assigned by HDBSCAN |
| `theme_title` | text | NOT NULL | LLM-generated theme label (e.g. "Identity & Access Management") |
| `audience` | text | NOT NULL | LLM-generated target stakeholder (Engineering / Product / Sales / All) |
| `rationale` | text | nullable | LLM-generated one-sentence explanation of why these phrases cluster together |
| `phrase_count` | integer | nullable | Number of topic phrases assigned to this cluster |

**Source file:** `final_version/outputs/semantic_clusters.json`
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

**Source file:** `final_version/outputs/phrase_clusters.csv`
**Note:** `embedding` is NULL after a CSV load. Vector search is unavailable, but all
analytical insight queries join on `cluster_id` only — not on embeddings. To populate
embeddings, re-run the full Final Version pipeline (requires Ollama).

---

### `semantic_meeting_themes` — one row per (meeting × cluster) pair (516 rows, 100 distinct meetings)

| Column | Type | Nullable | What it holds |
|---|---|---|---|
| `meeting_id` | text | NOT NULL | FK → `meetings.meeting_id` |
| `cluster_id` | integer | NOT NULL | FK → `semantic_clusters.cluster_id` |
| `is_primary` | boolean | NOT NULL | True for the single dominant theme per meeting |
| `call_type` | text | nullable | LLM-inferred call type: `support` / `external` / `internal` |
| `call_confidence` | text | nullable | LLM confidence string: `high` / `medium` / `low` |
| `sentiment_score` | numeric | nullable | Passed through from raw `summary.json` (not computed by Final Version) |
| `overall_sentiment` | text | nullable | Passed through from raw `summary.json` |

**Source file:** `final_version/outputs/meeting_themes.csv`
**Key constraint:** exactly one row per meeting has `is_primary = true`. Always filter
on `is_primary = true` when aggregating per-meeting to avoid double-counting.
**Note on `call_type`:** LLM-generated — stochastic and doesn't scale. For higher
consistency, compute call type from `key_moments` signal patterns or `meeting_summaries`
context directly.

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

**To reload all 9 tables:** run `load_raw_jsons_to_db.py --reset` (base tables from
raw JSON), then `load_output_csvs_to_db.py --reset` (semantic tables from `outputs/`).

---

## Why can't we label clusters by concatenating top centroid terms?

Because Final Version's centroids live in a different kind of space than TF-IDF centroids.

**Why it works for TF-IDF/KMeans:**

KMeans centroids in TF-IDF space have one dimension per term. The centroid's highest-weighted
dimensions *are* the most representative terms — you can read them off directly and concat.

```
centroid dimensions → sort by weight → top 4 terms → "renewal / competitive / pricing / outage"
```

Fully deterministic. No LLM needed.

**Why it doesn't work in Final Version:**

HDBSCAN centroids (actually cluster medoids / mean vectors) live in 768-dimensional
embedding space. Those 768 dimensions are abstract latent features learned by
`nomic-embed-text` — they do not correspond to words. There is no "top term" to read
off. The centroid is just a point in a space with no human-interpretable axes.

**What Final Version does instead:**

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

| | TF-IDF/KMeans approach | Final Version (embedding + HDBSCAN) |
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

The failure mode degrades gracefully — top phrases concatenated with " / ". The cluster
grouping is preserved; only the label quality drops.

---

## Why does semantic_meeting_themes store one row per meeting × cluster instead of a list of meeting_ids per cluster?

Short answer: `is_primary` is a junction attribute, and `sentiment`/`call_type` are denormalized here for query convenience.

**Why a flat list wouldn't be enough:**

The obvious simpler design would be a single row per cluster with a `meeting_ids TEXT[]` array. That works for the drill-down case ("give me all meetings in this cluster"), but breaks the analytical queries:

- **`is_primary`** marks which cluster is the dominant theme for each meeting. It belongs to the (meeting, cluster) relationship — not to the meeting alone (one meeting has many clusters) and not to the cluster alone (a cluster touches many meetings). There is no natural home for it in a flat list.
- **Theme × sentiment heatmap** (insight #2) groups by `cluster_id` and aggregates `overall_sentiment`. If sentiment lived only in `meeting_summaries`, every heatmap query would need an extra join back through `meeting_ids[]` unnest. Storing it in the junction row means a direct `GROUP BY cluster_id`.
- **Call type × theme matrix** (insight #6) does the same: `GROUP BY cluster_id, call_type` is a single scan with no unnesting.

**What is genuinely redundant:**

`sentiment` and `call_type` are per-meeting values, not per-(meeting, cluster) values — the same sentiment/call_type appears on every row for a given meeting. This is deliberate denormalization: the duplication is small (516 rows, 100 meetings) and eliminates joins in the most-used queries. A normalized design would store them in `meetings` and join in — equally correct, slightly more verbose SQL.

**Summary:**

| Column | Why it's here |
|--------|--------------|
| `is_primary` | True junction attribute — no other table can own it |
| `sentiment` / `call_type` | Denormalized from meeting level for join-free aggregation |
| `meeting_id` + `cluster_id` | Primary key of the junction |

---

## What are feature gap moments labelled "growing/positive"?

A **feature gap moment** (`moment_type = 'feature_gap'` in `key_moments`) is a transcript moment where a customer explicitly asks for something the product does not currently have.

The **growing/positive** label comes from the *meeting's* overall sentiment — not from the feature request itself. A customer raising a feature gap in a meeting that is overall positive is in growth mode: they like what they have and want more. They are expanding their usage, not at risk of leaving.

**Why the distinction matters (P1 question):**

The same feature request carries completely different priority depending on the account posture:

| Feature gap in a... | Customer posture | Priority |
|---------------------|-----------------|----------|
| Negative-sentiment meeting | Blocked — the gap is causing frustration, potentially threatening renewal | P0 — act now |
| Positive-sentiment meeting | Growing — the gap is a wishlist item for an expanding account | Roadmap — schedule it |

A product team that sees "5 Detect feature requests" without sentiment context will treat all five equally. The P1 chart splits them: requests from at-risk accounts (negative) vs requests from healthy, growing accounts (positive). Same backlog item, different urgency.

**In the data:**

```sql
-- Feature gaps from at-risk accounts (blocked)
WHERE km.moment_type = 'feature_gap'
  AND ms.overall_sentiment IN ('negative', 'very-negative', 'mixed-negative')

-- Feature gaps from growing accounts (wishlist)
WHERE km.moment_type = 'feature_gap'
  AND ms.overall_sentiment IN ('positive', 'very-positive', 'mixed-positive')
```

---

## Why aren't phrase embeddings saved to disk for reuse?

**Short answer:** once HDBSCAN finishes, the embeddings have served their purpose for this pipeline — all downstream work (labeling, insight queries, charts) uses `cluster_id` joins, not vectors. We didn't save them. We should have.

**What happens today:**

The full pipeline (`semantic_clustering.py`) correctly writes embeddings to `semantic_phrases.embedding` in Postgres. But the intermediate output files — `phrase_clusters.csv`, `semantic_clusters.json`, `meeting_themes.csv` — contain no embedding data. When you reload from those files via `--from-outputs` or `load_output_csvs_to_db.py --reset`, `semantic_phrases.embedding` ends up NULL for all 343 rows.

The analytical queries don't care — they join on `cluster_id`. But `SemanticClusterStore.semantic_search_phrases()` and `hybrid_search_phrases()` are dead until the full pipeline is re-run.

**What we should add if this becomes a live search feature:**

Save embeddings to a numpy binary file during the pipeline run, alongside the CSVs:

```python
# In write_outputs() — add after phrase_clusters.csv
import numpy as np
emb_matrix = np.array([p.embedding for p in phrases])  # shape (343, 768)
np.save(output_dir / "phrase_embeddings.npy", emb_matrix)

# Also save an index so load_from_outputs knows which row = which phrase
phrase_index = [p.canonical for p in phrases]
(output_dir / "phrase_index.json").write_text(json.dumps(phrase_index), encoding="utf-8")
```

Then in `load_from_outputs()`:

```python
emb_path = output_dir / "phrase_embeddings.npy"
index_path = output_dir / "phrase_index.json"
if emb_path.exists() and index_path.exists():
    emb_matrix = np.load(emb_path)
    phrase_index = json.loads(index_path.read_text())
    emb_by_canonical = {canonical: emb_matrix[i].tolist() for i, canonical in enumerate(phrase_index)}
else:
    emb_by_canonical = {}

# then when building phrases list:
"embedding": emb_by_canonical.get(row["canonical"].strip()),  # None if not found
```

**Why it wasn't done:**

The pipeline was built around a specific deliverable (analytics for a take-home assignment). Hybrid search was included as a technical showcase, not because any chart or insight required it. The fast-reload path (`--from-outputs`) was designed for DB resets without re-running Ollama — losing embeddings was an acceptable trade-off at the time.

**Files to add if you implement this:**

| File | Size (approx) | Contents |
|------|--------------|----------|
| `outputs/phrase_embeddings.npy` | ~1 MB | 343 × 768 float32 matrix |
| `outputs/phrase_index.json` | ~10 KB | Ordered list of canonical phrase strings (row → phrase mapping) |

Add both to `.gitignore` if you don't want to track large binary files in git.
