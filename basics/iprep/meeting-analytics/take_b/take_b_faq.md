# Take B — FAQ

## Table of Contents
- [Which output file comes from which pipeline step?](#which-output-file-comes-from-which-pipeline-step)
- [How does the full pipeline work — what happens to the TF-IDF matrix?](#how-does-the-full-pipeline-work--what-happens-to-the-tf-idf-matrix)
- [Are the centroid terms sorted? How is the cluster label formed?](#are-the-centroid-terms-sorted-how-is-the-cluster-label-formed)
- [How is K chosen, and is it stable across re-runs?](#how-is-k-chosen-and-is-it-stable-across-re-runs)
- [How many top terms make a good cluster label?](#how-many-top-terms-make-a-good-cluster-label)
- [What does build\_document\_text() do, and why does repeat\_terms exist?](#what-does-build_document_text-do-and-why-does-repeat_terms-exist)
- [How do I load Take B outputs into Postgres?](#how-do-i-load-take-b-outputs-into-postgres)
- [Why didn't Take B use topic phrase embeddings + KMeans instead of TF-IDF?](#why-didnt-take-b-use-topic-phrase-embeddings--kmeans-instead-of-tf-idf)
- [How does Take B generate cluster labels from centroids?](#how-does-take-b-generate-cluster-labels-from-centroids)

---

## Which output file comes from which pipeline step?

All five files are written in a single `write_outputs()` call at the end of the script,
but the *content* of each comes from a different earlier step. The pipeline has no
numbered steps like Take C — here is the logical sequence:

| Step | What happens | Function |
|------|-------------|----------|
| 1 | Load 100 meetings, build flat document text per meeting | `load_meeting_documents()` → `build_document_text()` |
| 2 | Extract participant names for stop-word filtering | `load_participant_names()` |
| 3 | TF-IDF vectorization → 100×2000 matrix | `TfidfVectorizer` + `normalize()` |
| 4 | Auto-k: fit KMeans for k=4–12, score by silhouette, pick best k | `choose_cluster_count()` |
| 5 | Final KMeans clustering with chosen k → cluster label per meeting | `cluster_documents()` |
| 6 | Compute silhouette on final clustering — redundant in `--auto-k` mode (step 4 already has it); only independently useful with `--no-auto-k` | `compute_silhouette_score()` |
| 7 | Extract top-N terms per cluster centroid → cluster label strings | `get_top_terms()` |
| 8 | Build per-cluster summaries with example meetings | `summarize_clusters()` |
| 9 | Write all output files to `outputs/` | `write_outputs()` |

**Output files and which step produced their content:**

| File | Content from step | What it contains |
|------|-------------------|-----------------|
| `meeting_clusters.csv` | Steps 5 + 7 | One row per meeting: `meeting_id`, `cluster_id`, `generated_cluster_label` (top-4 terms), `title`, `overall_sentiment`, `sentiment_score`, `topics` |
| `cluster_terms.csv` | Step 7 | One row per (cluster, rank, term) — all top-N terms per cluster, ranked by centroid weight. This is the evidence behind the generated label. |
| `cluster_summary.json` | Step 8 | Per-cluster dict with label, top terms, example meetings (titles + topics), and meeting count |
| `cluster_metrics.json` | Step 6 (auto-k: redundant recompute; no-auto-k: only silhouette compute) | Run metadata: `k`, `silhouette_score`, `auto_k`, `min/max_clusters`, `max_features`, `random_state` |
| `cluster_scores.csv` | Step 4 | One row per k value tried (4–12): `clusters`, `silhouette_score`. Only written if `--auto-k` was used. |
| `take_b_run.log` | All steps | Terminal output redirected to file — not written by the script itself |

**Quick guide to which file to open first:**

- Spot-check cluster assignments → `meeting_clusters.csv`
- Understand what a cluster is really about → `cluster_terms.csv` (see all 12 terms, not just the 4 in the label)
- Find example meetings per cluster → `cluster_summary.json`
- Check if k selection was sensible → `cluster_scores.csv` (silhouette curve across k=4–12)
- Reproduce the run exactly → `cluster_metrics.json` (has all hyperparameters)

---

## How does the full pipeline work — what happens to the TF-IDF matrix?

Take B turns meeting summaries into clusters in four steps.

**Step 1 — Build a document-term matrix (TF-IDF)**

`build_document_text()` flattens each `summary.json` into a single string.
`TfidfVectorizer` converts all 100 strings into a matrix: 100 rows × 2000 columns.
Each row is one meeting. Each column is one term (word or phrase).
The value in each cell is the TF-IDF weight for that term in that meeting:

- `TF` = how often the term appears in this meeting
- `IDF` = how rare the term is across all meetings (log scale)
- `TF-IDF = TF × IDF` — high weight means the term is distinctive for this meeting

Common words ("the", "and") get near-zero weight. Rare, specific terms ("scim
provisioning", "overage charges") get high weight.

**Step 2 — L2 normalise each row**

`normalize()` scales each row vector to unit length (magnitude = 1).
Without this, a longer meeting would dominate distance calculations just because
it has more words. After normalisation, distances reflect topic similarity, not
meeting length.

**Step 3 — KMeans clustering**

KMeans groups the 100 row-vectors into K clusters in the 2000-dimensional space.

1. Pick K random points as initial centroids.
2. Assign every meeting to its nearest centroid (Euclidean distance).
3. Recalculate each centroid as the mean vector of all meetings assigned to it.
4. Repeat 2–3 until assignments stop changing (or `max_iter` reached).

Result: every meeting gets a cluster ID (0 to K-1).

**Step 4 — Name each cluster from centroid weights**

The centroid of a cluster is the average TF-IDF vector of all meetings in it.
`get_top_terms()` sorts each centroid by weight descending and takes the top N.
That becomes the generated label — e.g. `"billing / overage / dispute"`.
No LLM needed: the label comes directly from the math.

End result for each meeting in the output CSV:
```
meeting_id, cluster_id, generated_cluster_label, sentiment, topics
```

---

## Are the centroid terms sorted? How is the cluster label formed?

Yes — sorted by weight, descending.

`get_top_terms()` does this:
```python
centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
```

`kmeans.cluster_centers_` is a (K × 2000) matrix — one row per cluster, one column
per term, the value being the average TF-IDF weight of that term across all meetings
in the cluster. `argsort()` returns column indices sorted ascending; `[:, ::-1]`
reverses to descending.

So `centroids[0][0]` is the index of the most characteristic term for cluster 0,
`centroids[0][1]` is the second most characteristic, and so on.

The label is formed by joining the first 4 terms with `" / "`:
```
" / ".join(top_terms[:4])  →  "billing / overage / dispute / billing dispute"
```

The full top-N list (default N=12, set by `--top-terms`) is stored in
`cluster_summary.json` and `cluster_terms.csv`. Only 4 are used in the short label.

The choice of 4 is arbitrary — enough to be descriptive, not so many it reads as a
run-on sentence. There is no mathematical basis for it; it is a display decision.

---

## How is K chosen, and is it stable across re-runs?

K is chosen by silhouette score (auto-k mode, the default).

**How it works:**
`score_cluster_counts()` fits KMeans for every k from `--min-clusters` (4) to
`--max-clusters` (12), computes the silhouette score for each, and returns all scores.
`choose_cluster_count()` picks the k with the highest silhouette score.

**Silhouette score** measures how well-separated the clusters are:
for each point, compare its distance to its own cluster vs. the nearest other cluster.
Range −1 to 1. Higher = better separation. Near 0 = clusters overlap heavily.

On re-run with the same data and `random_state=42` (fixed), KMeans is deterministic —
it will always pick the same K and produce the same clusters.
If the data changes (new meetings added), optimal K may shift.

**The silhouette score for this dataset is ~0.03 — very low.** This does not mean the
clusters are wrong. It means the meetings genuinely blend across topics (a renewal
call can also involve an outage; a compliance call can overlap with product roadmap).
Low silhouette in this context reflects the data, not a code problem.
Use the cluster labels and top-topics to judge quality, not the score alone.

---

## How many top terms make a good cluster label?

4 terms are used for the display label. 12 are stored for inspection. Both are
arbitrary choices with different purposes:

| Use | Count | Purpose |
|-----|-------|---------|
| Display label | 4 | Short enough to read at a glance on a slide or CSV |
| Stored top-terms | 12 | Enough to see the full character of the cluster and spot whether unrelated terms snuck in |

There is no formula for the right number. The heuristics are:

- If the first 2 terms already unambiguously name the cluster (`"mfa / identity"`),
  more terms add noise rather than clarity.
- If the first 4 terms are all synonyms (`"backup / recovery / restore / backup recovery"`),
  the cluster is tight — good signal.
- If the top terms span different topics (`"billing / sprint / hipaa"`), the cluster
  is probably impure — investigate with the example meetings in `cluster_summary.json`.

**Tuning levers:**
- `--top-terms N` controls how many terms are stored (default 12)
- The label always uses the first 4 (hardcoded in `summarize_clusters` / `write_outputs`)
- Change the `[:4]` slice there to adjust label length

---

## What does build\_document\_text() do, and why does repeat\_terms exist?

TF-IDF only understands a flat bag of words — it can't work with JSON structure.
`build_document_text()` flattens all useful fields from one `summary.json` into a
single string so TF-IDF can vectorize it.

**The problem `repeat_terms` solves:**

A meeting summary is typically 150–200 words of prose. Topics are short tags like
`"churn risk"` (2 words). Without any boost, the summary dominates the TF-IDF vector
and the curated topic tags barely register.

```python
repeat_terms("churn risk", repeat=4)
# → "churn risk churn risk churn risk churn risk"
```

This inflates the term count so TF-IDF treats the topic as a meaningful signal.

**What the final document text looks like for a billing dispute meeting:**

```
Wayne and the customer discussed invoice discrepancies...   ← summary (once)
billing dispute billing dispute billing dispute ...          ← topics ×4
concern concern concern pricing_offer pricing_offer ...      ← key moment types ×3
Invoice showed overage charges that customer disputes        ← key moment texts (once)
Follow up with finance team                                  ← action items (once)
```

**Why those specific repeat counts (4 and 3):**

There's no mathematical derivation. The judgment is:

```
topics (×4) > key moment types (×3) > prose fields (×1)
```

Topics are the most curated signal. Key moment types (`churn_signal`, `feature_gap`, etc.)
are structured but less specific. Summary prose and action items are already verbose.
If clustering results look off, adjusting these counts is a valid tuning lever.

---

## How do I load Take B outputs into Postgres?

Take B was designed as an analytical validation tool, not a loader. Its primary outputs
are CSV and JSON. A Postgres loader was added to bring it into the shared schema.

**The 3 Take B tables and what feeds them:**

| Table | Source file | What it contains |
|-------|-------------|-----------------|
| `kmeans_clusters` | `cluster_summary.json` + `cluster_metrics.json` | cluster_id, label (top-4 terms), meeting_count, silhouette_score |
| `kmeans_cluster_terms` | `cluster_terms.csv` | top-12 centroid terms per cluster, ranked by TF-IDF weight |
| `kmeans_meeting_clusters` | `meeting_clusters.csv` | one row per meeting → its single hard cluster assignment |

**Load from existing outputs (fast — no re-clustering):**

```bash
python basics/iprep/meeting-analytics/take_b/load_outputs_to_pg.py
```

**Re-run the full clustering pipeline then load:**

```bash
python basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py
python basics/iprep/meeting-analytics/take_b/load_outputs_to_pg.py
```

**Load all three takes in one shot:**

```bash
python basics/iprep/meeting-analytics/setup_all_tables.py
```

Target: `rag_db @ localhost:5434` (rag_user:rag_pass). Credentials from
`meeting-analytics/.env`. Take B adds 3 tables to the 10 from Take A and 3 from Take C
— 16 tables total in `meeting_analytics` schema.

---

## Why didn't Take B use topic phrase embeddings + KMeans instead of TF-IDF?

It would work — and produce better clusters than TF-IDF+KMeans. But it would undermine
the point of having three takes.

**The three takes are a methodological spectrum:**

| Take | Representation | Clustering | Key property |
|---|---|---|---|
| A | Rules / keywords | None | Fully deterministic, auditable |
| B | TF-IDF (bag of words) | KMeans (fixed k) | Statistical, no semantics |
| C | Semantic embeddings | HDBSCAN (data-driven k) | Meaning-aware, no fixed k |

If Take B used embeddings, the only difference between B and C would be the clustering
algorithm — KMeans vs HDBSCAN. That is a narrow comparison. The current design compares
two things simultaneously: representation method (keyword frequency vs semantic meaning)
AND clustering strategy (fixed k vs density-adaptive). That is a richer story.

**The B vs C comparison only has value because the representations differ.**

The headline finding — "TF-IDF lumped outage prevention and incident response into one
cluster; embeddings correctly split them into two" — is a direct argument for semantic
representations over bag-of-words. If B also used embeddings, that argument disappears
and the comparison becomes "KMeans vs HDBSCAN," which is a much weaker insight for a
product/engineering audience.

**What embeddings + KMeans would give you:**

Better clusters than TF-IDF+KMeans, but you'd still need to pick k upfront, and HDBSCAN
would still outperform it by finding cluster count from data density. It sits between B
and C methodologically — useful as an ablation study, but it blurs the narrative rather
than sharpening it.

---

## How does Take B generate cluster labels from centroids?

In KMeans, each cluster centroid is a point in TF-IDF vector space. Every dimension in
that space corresponds to a specific term. So the centroid's highest-weighted dimensions
*are* the most representative terms for that cluster — you can read them off directly.

**Take B label generation — fully deterministic:**

1. After KMeans converges, each centroid is a vector of TF-IDF term weights
2. Sort dimensions by weight descending → top terms emerge naturally
3. Take top 4 terms, concat with " / " → cluster label

Example: centroid with highest weights on `renewal`, `competitive`, `pricing`, `outage`
→ label = `"renewal / competitive / pricing / outage"`

No LLM, no ambiguity, same input always produces the same label. See Take C FAQ for
why this approach is not possible with embedding-based clustering.
