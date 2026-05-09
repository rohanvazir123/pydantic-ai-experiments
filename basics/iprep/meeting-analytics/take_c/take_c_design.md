# Take C — LLM-Assisted Semantic Clustering: Design Document

## Audit Trail

| Version | Date       | Author | Summary of Changes                             |
|---------|------------|--------|------------------------------------------------|
| v0.1    | 2026-05-08 | rohan  | Initial design — pipeline skeleton, tradeoffs, decision log |
| v0.2    | 2026-05-08 | rohan  | Implementation complete — `take_c_semantic_clustering.py`; dry-run verified: 100 meetings loaded, 600 raw topics → 351 exact-dedup → 343 fuzzy-dedup; deps: rapidfuzz, umap-learn, hdbscan |

---

## Table of Contents

1. [Context & Goals](#1-context--goals)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step-by-Step Design](#3-step-by-step-design)
   - [3.1 Input Representation](#31-input-representation)
   - [3.2 Deduplication](#32-deduplication)
   - [3.3 Embedding Generation](#33-embedding-generation)
   - [3.4 Dimensionality Reduction](#34-dimensionality-reduction)
   - [3.5 Clustering](#35-clustering)
   - [3.6 LLM Cluster Labeling](#36-llm-cluster-labeling)
   - [3.7 Call-Type Inference](#37-call-type-inference)
   - [3.8 Output & Downstream Use](#38-output--downstream-use)
4. [Decision Log (Chosen Approach)](#4-decision-log-chosen-approach)
5. [Data Model](#5-data-model)
6. [Implementation Sketch](#6-implementation-sketch)
7. [Open Questions](#7-open-questions)

---

## 1. Context & Goals

**Dataset:** ~100 meeting `summary.json` files, each containing:
- `summary` (free text)
- `topics` (list of short phrase strings)
- `keyMoments[].type` (churn_signal, concern, technical_issue, feature_gap, praise, pricing_offer)
- `overallSentiment`, `sentimentScore`
- `actionItems`, `meetingId`

**Goal 1 (primary):** Cluster the raw topics (~350 unique phrases from `list_of_topics.txt`) into broader, human-readable **themes** suitable for Engineering, Product, and Sales leadership.

**Goal 1b:** Infer **call type** (customer-support / external-account / internal-engineering) per meeting.

**Why not Take A (rule-based)?** Brittle — misses semantic equivalents (e.g. `pipeline failure` ≡ `detect pipeline failure` ≡ `ingestion pipeline`).

**Why not Take B (TF-IDF + K-Means)?** Requires pre-specifying K; bag-of-words misses phrase semantics; very short phrases kill TF-IDF signal.

**Why Take C?** The topic strings are short, semantically dense, and conceptually overlapping. A vector embedding captures "outage recovery" ≈ "outage remediation" ≈ "post-mortem" without hand-crafted rules.

---

## 2. Pipeline Overview

```
[summary.json × N]
       │
       ▼
[1] Extract & Flatten topics (+ optional: keyMoments, summary snippets)
       │
       ▼
[2] Deduplicate (exact → fuzzy → semantic optional)
       │
       ▼
[3] Embed with local Ollama model (nomic-embed-text)
       │
       ▼
[4] UMAP dimensionality reduction  (768 → 5–10 dims for HDBSCAN)
       │
       ▼
[5] HDBSCAN clustering  (no fixed K, handles noise)
       │
       ▼
[6] LLM labels each cluster  (Ollama, one call per cluster)
       │
       ▼
[7] Assign meetings → themes  (each meeting's topics → majority cluster)
       │
       ▼
[8] Persist to Postgres  (meeting_themes table, reuses Take A schema)
       │
       ▼
[Output] Named themes + per-meeting theme assignment + call-type label
```

---

## 3. Step-by-Step Design

### 3.1 Input Representation

**What text do we embed?**

| Option | Input Text | Pros | Cons |
|--------|-----------|------|------|
| **A** | `topics` phrases only | Clean, short, fast | Ignores semantic richness of summaries |
| **B** | `topics` + `keyMoments[].description` | More signal per item | keyMoments vary in quality |
| **C** | Per-meeting concatenation of summary + topics | Full context | One vector per meeting, not per concept; hard to label clusters meaningfully |
| **D** | `topics` phrases → each phrase is one document | Best cluster granularity | Need meeting→topic fanout; requires re-aggregation at the end |

**Decision:** Option **D** — embed each unique topic phrase independently, then re-aggregate meetings to their assigned cluster IDs.

**Rationale:** Topic phrases are the atomic units of meaning. Meeting-level aggregation happens post-clustering (Step 7), not during embedding.

---

### 3.2 Deduplication

The topic list has near-duplicates: `api rate limiting` / `api rate limits`, `feature gap` / `feature gaps`.

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Exact** | Python set() | Free | Misses plurals, typos |
| **Fuzzy** | rapidfuzz token_sort_ratio ≥ 90 | Catches plurals cheaply | May over-merge distinct concepts |
| **Semantic** | Cosine sim ≥ 0.95 post-embed | Most accurate | Requires embeddings first (chicken-and-egg) |

**Decision:** Two-pass.
1. Exact dedup (set).
2. Fuzzy dedup (rapidfuzz ≥ 90) before embedding — cheap and catches `feature gap`/`feature gaps`.
3. Keep a `canonical → aliases` mapping for traceability.

---

### 3.3 Embedding Generation

| Model | Dims | Speed (CPU) | Quality | Notes |
|-------|------|-------------|---------|-------|
| **nomic-embed-text v1.5** | 768 | Fast | Excellent | Already in Ollama for this project |
| BGE-M3 | 1024 | Medium | Excellent (multilingual) | Overkill for English-only |
| gte-Qwen2-1.5B | 1536 | Slow (needs GPU) | Best | Too heavy for local dev |
| OpenAI text-embedding-3-small | 1536 | API latency | Very good | Requires API key / cost |

**Decision:** `nomic-embed-text:latest` via Ollama — already configured in this project's `.env`, zero marginal cost, 768 dims more than sufficient for ~350 phrases.

**Batching:** Send all phrases in a single `asyncio.gather` across the Ollama embedding endpoint (reuses `rag/ingestion/embedder.py` logic).

---

### 3.4 Dimensionality Reduction

HDBSCAN performance degrades in high-dimensional space (curse of dimensionality). Reducing before clustering is standard practice.

| Method | Dims out | Preserves | Cons |
|--------|----------|-----------|------|
| **UMAP** | 5–15 | Global + local structure | Non-deterministic (fix `random_state`) |
| PCA | 50–100 | Global variance | Loses non-linear structure |
| t-SNE | 2–3 | Local clusters only | Only good for viz, not clustering input |
| None | 768 | Everything | HDBSCAN struggles above ~50 dims |

**Decision:** UMAP with `n_components=10`, `n_neighbors=15`, `min_dist=0.0`, `metric='cosine'`, `random_state=42`.

- `min_dist=0.0` packs cluster members tighter (better for HDBSCAN).
- `metric='cosine'` matches embedding space.
- Also run UMAP at `n_components=2` separately for **visualization only** (no clustering on this).

---

### 3.5 Clustering

| Algorithm | Needs K? | Handles Noise? | Pros | Cons |
|-----------|----------|----------------|------|------|
| **HDBSCAN** | No | Yes (label=-1) | Adapts to density, variable cluster size | Two key hyperparams |
| K-Means | Yes | No | Fast, reproducible | K unknown; forces all points into clusters |
| DBSCAN | No | Yes | Simple | Flat density assumption; sensitive to epsilon |
| Agglomerative | No (with dendrogram cut) | No | Deterministic | Need to choose linkage + cut height |
| Gaussian Mixture | Yes | No | Soft assignments | Needs K |

**Decision:** HDBSCAN.

Key hyperparameters and their tradeoffs:

| Param | Low value effect | High value effect | Starting point |
|-------|-----------------|-------------------|----------------|
| `min_cluster_size` | More clusters, more noise | Fewer, broader clusters | 5 (out of ~350 phrases) |
| `min_samples` | More aggressive clustering | More noise points | 3 |
| `cluster_selection_epsilon` | Tight clusters | Merges nearby clusters | 0.0 (default) |

**Noise handling:** Phrases labeled `-1` (noise) will be assigned to their nearest cluster centroid via cosine similarity as a post-processing step — they are too few to throw away, but HDBSCAN correctly isolates genuinely ambiguous phrases.

---

### 3.6 LLM Cluster Labeling

For each discovered cluster, sample up to 20 representative phrases (by proximity to cluster centroid) and ask a local LLM to assign a leadership-ready theme label.

**Prompt template:**
```
You are categorizing customer call topics for B2B SaaS leadership.

Topics from this group:
{topic_list}

Generate:
1. A short theme title (3-6 words, title case)
2. Which audience it matters most to: Engineering / Product / Sales / All
3. One sentence: why this theme matters to that audience

Return JSON: {"title": "...", "audience": "...", "rationale": "..."}
```

| Labeling strategy | Pros | Cons |
|-------------------|------|------|
| **Single-shot per cluster** | Fast, cheap | May miss nuance |
| Few-shot with examples | More consistent | Needs example curation |
| Iterative (label → review → refine) | Highest quality | Requires human loop |
| Hierarchical (label sub-clusters first) | Natural for large K | Over-engineering at this scale |

**Decision:** Single-shot per cluster. At ~350 unique phrases and likely 8–15 clusters, this is ~10 LLM calls — trivially fast with Ollama.

**Model:** `llama3.1:8b` (already configured). If cluster label quality is poor, bump to `llama3.3:70b` or add a one-shot example in the prompt.

**Structured output:** Use `instructor` or `pydantic_ai` model with a `ClusterLabel` response model to guarantee parseable JSON.

---

### 3.7 Call-Type Inference

Each meeting has a hidden call type (support / external / internal). Two sub-approaches:

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Theme-majority vote** | Map meeting → cluster IDs → look up which clusters correlate with each call type | No extra LLM call | Needs cluster→call-type mapping (manual or auto) |
| **Direct LLM classification** | Feed `summary` text to LLM, ask for call type | Highest accuracy | One LLM call per meeting (N=100 — still cheap) |
| **Keyword rules (Take A fallback)** | Re-use `infer_call_type()` from Take A | Zero cost, already built | Lower accuracy |

**Decision:** Direct LLM classification on the `summary` field. One structured prompt per meeting, 3-way classification (support / external / internal). This is a separate lightweight step from theme clustering and gives clean ground truth for the sentiment analysis dashboard.

---

### 3.8 Output & Downstream Use

Final artifacts written to Postgres (reusing Take A schema, extending `meeting_themes`):

```sql
-- One row per (meeting, theme)
INSERT INTO meeting_themes (meeting_id, theme_id, theme_title, audience, confidence)

-- New table for cluster metadata
CREATE TABLE semantic_clusters (
    cluster_id   INTEGER PRIMARY KEY,
    theme_title  TEXT,
    audience     TEXT,   -- Engineering | Product | Sales | All
    rationale    TEXT,
    phrase_count INTEGER
);
```

For the slide deck / dashboard:
- Heatmap: call type × theme (which themes dominate each call type)
- Sentiment overlay: theme × avg sentimentScore
- Churn signal map: which themes co-occur with `churn_signal` keyMoment type

---

## 4. Decision Log (Chosen Approach)

| Decision | Chosen | Alternatives Considered | Reason |
|----------|--------|------------------------|--------|
| Input unit | Per-topic phrase | Per-meeting text | Topic phrases are atomic; richer cluster resolution |
| Embedding model | nomic-embed-text (Ollama) | BGE-M3, OpenAI | Already in stack; free; 768 dims sufficient |
| Deduplication | Exact + fuzzy (rapidfuzz) | Semantic dedup | Fast; semantic dedup needs embeddings first |
| Dimensionality reduction | UMAP (10 dims) | PCA, none | Preserves non-linear structure; HDBSCAN needs low dims |
| Clustering | HDBSCAN | K-Means, DBSCAN | No fixed K needed; handles noise; density-adaptive |
| Noise reassignment | Nearest centroid | Discard | ~350 phrases is small; don't throw data away |
| Cluster labeling LLM | llama3.1:8b via Ollama | OpenAI GPT-4o | Local, free, already configured |
| Labeling strategy | Single-shot + structured JSON | Few-shot, iterative | Only ~10 clusters; fast iteration preferred |
| Call-type inference | Direct LLM on summary | Theme-majority vote | Cleaner signal; summary text is explicit about context |
| Schema | Extend Take A Postgres tables | Standalone new tables | Reuse existing schema; avoid duplication |

---

## 5. Data Model

```python
from pydantic import BaseModel

class TopicPhrase(BaseModel):
    canonical: str
    aliases: list[str]
    cluster_id: int          # -1 = noise (reassigned post-hoc)
    embedding: list[float]   # 768 dims

class ClusterLabel(BaseModel):
    cluster_id: int
    theme_title: str         # e.g. "Identity & Access Management"
    audience: str            # Engineering | Product | Sales | All
    rationale: str
    representative_phrases: list[str]  # top-20 by centroid proximity

class MeetingThemeAssignment(BaseModel):
    meeting_id: str
    inferred_call_type: str  # support | external | internal
    theme_ids: list[int]     # all clusters present in this meeting's topics
    primary_theme_id: int    # most frequent cluster
```

---

## 6. Implementation Sketch

Target file: `basics/iprep/i1/take_c_semantic_clustering.py`

```
Functions (async throughout):
─────────────────────────────────────────────────────────────
load_summary_records()         → list[MeetingRecord]          # reuse Take A loader
extract_topic_phrases()        → list[TopicPhrase]            # flatten + dedup
embed_phrases()                → list[TopicPhrase]            # Ollama batch embed
reduce_dimensions()            → np.ndarray                   # UMAP 10-dim
cluster_phrases()              → list[TopicPhrase]            # HDBSCAN + noise fix
label_clusters()               → list[ClusterLabel]           # Ollama structured call
assign_meetings_to_themes()    → list[MeetingThemeAssignment] # fanout + majority
infer_call_types()             → dict[str, str]               # meeting_id → call_type
persist_results()              → None                         # write to Postgres

main() orchestrates all steps, prints summary table
```

Dependencies to add:
```
umap-learn
hdbscan
rapidfuzz
```
(No GPU required; all CPU-friendly at this dataset size.)

---

## 7. Open Questions

1. **How many clusters will HDBSCAN find?** Looking at `list_of_topics.txt`, there are visually ~10–14 natural groupings (IAM, Compliance/Audit, Incidents/Outages, Backup/Recovery, Billing, Product Roadmap, Sales/Renewal, Engineering Reliability, Competitive…). HDBSCAN should land in this range with `min_cluster_size=5`.

2. **Is topics-only sufficient, or do we need `keyMoments`?** Initial design uses topics only. If cluster quality is poor, add `keyMoments[].description` as additional phrases per meeting.

3. **Should themes be hierarchical?** E.g. `Compliance` → `HIPAA`, `SOC 2`, `PCI DSS`. At 100 meetings this is optional but could add value for the Product leadership audience. Defer to v0.2 if time allows.

4. **Reuse Take A's Postgres records?** Notes in `notes.txt` flag that Take B re-parsed raw JSON instead of querying Postgres — a code smell. Take C should load from the `meetings` table if Take A has already run, falling back to raw JSON otherwise.

5. **Visualization output?** UMAP at 2 dims → scatter plot colored by cluster → quick sanity check before LLM labeling. Include as a matplotlib/plotly side output in the notebook.
