# Take B — TF-IDF / KMeans Clustering: Design Document

## Audit Trail

| Version | Date       | Author | Summary of Changes |
|---------|------------|--------|--------------------|
| v0.1    | 2026-05-09 | rohan  | Initial design doc — written retrospectively from completed implementation |

---

## Table of Contents

1. [Context & Goals](#1-context--goals)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step-by-Step Design](#3-step-by-step-design)
   - [3.1 Document Construction](#31-document-construction)
   - [3.2 Stop-Word Filtering](#32-stop-word-filtering)
   - [3.3 TF-IDF Vectorization](#33-tf-idf-vectorization)
   - [3.4 K Selection](#34-k-selection)
   - [3.5 KMeans Clustering](#35-kmeans-clustering)
   - [3.6 Cluster Labeling](#36-cluster-labeling)
   - [3.7 Output](#37-output)
4. [Decision Log](#4-decision-log)
5. [Data Model](#5-data-model)
6. [CLI Reference](#6-cli-reference)

---

## 1. Context & Goals

**Dataset:** 100 meeting folders, each with `summary.json` (topics, key moments,
action items, sentiment) and `transcript.json` (sentence-level data).

The three goals carried across all takes:

**Goal 1 — Theme assignment:** Assign each meeting one or more business themes
from a discoverable taxonomy.
→ **Take B addresses this.** TF-IDF + KMeans produces 8 topic clusters, each
labelled by its top centroid terms. Meetings get a single cluster assignment.
No hand-crafted rules — themes emerge from term co-occurrence statistics.

**Goal 2 — Call type inference:** Infer the kind of conversation per meeting
(support, sales/renewal, internal).
→ **Take B does not address this.** Call type inference was not in scope for
the TF-IDF approach. Take A handles it with keyword rules; Take C with LLM
classification. Take B's value is in validating the theme structure, not
replicating features already covered.

**Goal 3 — Postgres persistence:** Persist raw and derived fields to a schema
shaped for analytical queries.
→ **Take B does not address this.** Outputs are CSV and JSON only. The schema
is owned by Take A; Take C extends it with semantic tables. Take B is an
analytical tool, not a loader.

**Why Take B at all?**

Take A's themes come from hand-crafted rules — they reflect what the designer
*expected* to find. Take B applies an unsupervised method (TF-IDF/KMeans) to
the same dataset with no prior assumptions. If both methods independently surface
the same groupings, that is strong evidence the themes are real signal in the data
rather than artefacts of the rule design.

Take B also establishes a baseline cluster count (k=8) that Take C can be compared
against — validating whether semantic embedding discovers the same structure at
higher resolution.

**Why not just Take A?**

Take A is brittle to vocabulary variation. A meeting about "post-mortem planning"
and a meeting about "incident review" might not share a single keyword but are
semantically equivalent. TF-IDF catches this through term co-occurrence patterns
that cross individual keyword boundaries.

---

## 2. Pipeline Overview

```
[dataset/]  100 meeting folders
       │
       ▼
[1/7] Load meetings + build flat document text per meeting
       │
       ▼
[2/7] Extract participant names for stop-word filtering
       │
       ▼
[3/7] TF-IDF vectorization → 100×2000 matrix
       │
       ▼
[4/7] Auto-k: fit KMeans for k=4–12, score by silhouette, pick best k
      (skipped if --no-auto-k)
       │
       ▼
[5/7] Final KMeans clustering with chosen k → cluster label per meeting
       │
       ▼
[6/7] Compute silhouette on final clustering
      (redundant in --auto-k mode; only independently useful with --no-auto-k)
       │
       ▼
[7/7] Write all output files to outputs/
```

---

## 3. Step-by-Step Design

### 3.1 Document Construction

TF-IDF requires a flat string per document. `build_document_text()` flattens all
analytically useful fields from `summary.json` into one string per meeting.

**The signal weighting problem:**

A meeting summary is 150–200 words of prose. Topics are 2–3 word tags. Without
any boost, the prose dominates the TF-IDF vector and the curated topic tags barely
register. `repeat_terms()` inflates topic tags by repetition:

```
topics repeated ×4, key moment types repeated ×3, prose fields ×1
```

This ordering reflects a signal quality judgment: topics are the most curated
signal, key moment types are structured but less specific, prose is already verbose.
The multipliers are heuristic — not mathematically derived.

Fields included in document text:

| Field | Source | Repeat |
|-------|--------|--------|
| Summary text | `summary.json summary` | ×1 |
| Topics | `summary.json topics` | ×4 |
| Key moment types | `summary.json keyMoments[].type` | ×3 |
| Key moment texts | `summary.json keyMoments[].text` | ×1 |
| Action items | `summary.json actionItems` | ×1 |

---

### 3.2 Stop-Word Filtering

Person names in meeting text inflate TF-IDF weights for terms that are not
topically meaningful (e.g. "Wayne", "Sarah"). `load_participant_names()` extracts:
- Participant emails (first segment before `@`)
- Speaker names from `transcript.json data[].speaker_name`

These are passed as `extra_stop_words` to `TfidfVectorizer`. Without this, a
cluster might be characterized by a person's name rather than a business topic.

---

### 3.3 TF-IDF Vectorization

`TfidfVectorizer` with `max_features=2000`, `ngram_range=(1,2)` (unigrams and
bigrams), standard English stop words + participant names.

Each of the 100 meetings becomes a row vector of 2000 TF-IDF weights. Rows are
L2-normalized so that meeting length does not bias distance calculations.

---

### 3.4 K Selection

Auto-k mode (default): `score_cluster_counts()` fits KMeans for every k from
`--min-clusters` (4) to `--max-clusters` (12) and scores each by silhouette.
`choose_cluster_count()` picks the highest-scoring k.

**Silhouette score for this dataset: ~0.03 (very low).**

This does not mean the clusters are wrong — it means meetings genuinely blend
across topics. Low silhouette reflects data reality, not a code problem. The
cluster labels and representative meetings are the right quality signal, not
the silhouette score alone.

**Last run result:** k=8, silhouette=0.0322.

---

### 3.5 KMeans Clustering

Standard KMeans with `n_init=20`, `random_state=42` for reproducibility.
Each meeting is assigned to exactly one cluster (hard assignment). The centroid
of each cluster is the mean TF-IDF vector of its member meetings.

---

### 3.6 Cluster Labeling

No LLM. Labels are derived directly from centroid weights.

`get_top_terms()` sorts each cluster's centroid by TF-IDF weight descending.
The top 4 terms are joined with ` / ` to form the display label:
```
"billing / overage / dispute / billing dispute"
```

Top 12 terms per cluster are stored in `cluster_terms.csv` and
`cluster_summary.json` for deeper inspection. The 4-term display label is
a display decision, not a mathematical optimum.

---

### 3.7 Output

All files written by `write_outputs()` in a single call at the end:

| File | Content from step | What it contains |
|------|-------------------|-----------------|
| `meeting_clusters.csv` | Steps 5+6 | One row per meeting: `cluster_id`, `generated_cluster_label`, `sentiment`, `topics` |
| `cluster_terms.csv` | Step 6 | One row per (cluster, rank, term) — all top-12 terms ranked by centroid weight |
| `cluster_summary.json` | Step 6 | Per-cluster label, top terms, example meetings, meeting count |
| `cluster_metrics.json` | Step 6* | Run metadata: k, silhouette, auto_k, hyperparameters |
| `cluster_scores.csv` | Step 4 | Silhouette score for every k tried (4–12); only written in auto-k mode |

*`cluster_metrics.json` silhouette is a recompute in auto-k mode — step 4 already
computed it. Only independently useful in `--no-auto-k` mode.

---

## 4. Decision Log

| Decision | Chosen | Alternatives Considered | Reason |
|----------|--------|------------------------|--------|
| Input unit | Per-meeting flat text | Per-topic phrase | TF-IDF needs document-level term frequencies; phrase-level is Take C's approach |
| Topic boosting | `repeat_terms()` ×4 | TF-IDF subfield weighting | Simple and transparent; easy to tune |
| K selection | Silhouette auto-k (default) | Fixed k, elbow method | Silhouette is objective and reproducible; elbow requires visual judgment |
| Labeling | Top centroid terms | LLM | Free, deterministic, immediately interpretable; LLM labeling is Take C |
| Output | CSV + JSON only | Postgres | Take A owns the schema; Take B is validation, not a loader |
| Call type | Not implemented | Keyword rules (Take A), LLM (Take C) | Out of scope — Take B's purpose is theme validation, not full feature parity |
| Person name filtering | Extract from emails + transcript | Generic name list | Dataset-specific names are more precise than a generic English name corpus |

---

## 5. Data Model

```python
@dataclass
class MeetingDocument:
    meeting_id: str
    title: str
    overall_sentiment: str
    sentiment_score: float
    topics: list[str]
    document_text: str    # flattened + boosted text fed to TF-IDF
```

---

## 6. CLI Reference

```bash
# Default run — auto-k, outputs to take_b/outputs/
python basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py

# Pin k=8 explicitly
python basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py --no-auto-k --clusters 8

# Custom k range for auto-k
python basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py --min-clusters 4 --max-clusters 15

# Write outputs to a different directory (e.g. for k=26 experiment)
python basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py --no-auto-k --clusters 26 \
    --output-dir basics/iprep/meeting-analytics/take_b/outputs_k26
```

**Flags:**

| Flag | Default | Effect |
|------|---------|--------|
| `--dataset` | `../dataset` | Path to meeting folders |
| `--output-dir` | `take_b/outputs/` | Where to write output files |
| `--clusters` | 8 | Fixed k (used with `--no-auto-k`) |
| `--auto-k` | true | Score k=min..max, pick best silhouette |
| `--min-clusters` | 4 | Lower bound for auto-k search |
| `--max-clusters` | 12 | Upper bound for auto-k search |
| `--top-terms` | 12 | Terms stored per cluster (display label always uses first 4) |
| `--examples-per-cluster` | 5 | Example meetings stored in cluster_summary.json |
| `--max-features` | 2000 | TF-IDF vocabulary size |
| `--random-state` | 42 | KMeans seed for reproducibility |
