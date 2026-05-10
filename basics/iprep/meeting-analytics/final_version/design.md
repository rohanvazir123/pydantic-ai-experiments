# Final Version — LLM-Assisted Semantic Clustering: Design Document

## Audit Trail

| Version | Date       | Author | Summary of Changes                             |
|---------|------------|--------|------------------------------------------------|
| v0.1    | 2026-05-08 | rohan  | Initial design — pipeline skeleton, tradeoffs, decision log |
| v0.2    | 2026-05-08 | rohan  | Implementation complete — `semantic_clustering.py`; dry-run verified: 100 meetings loaded, 600 raw topics → 351 exact-dedup → 343 fuzzy-dedup; deps: rapidfuzz, umap-learn, hdbscan |
| v0.3    | 2026-05-09 | rohan  | Full pipeline run complete — 26 clusters found (not 8–15 as estimated); added coherence check step; actual Postgres schema documented; 8 insight queries implemented; open questions resolved; file path corrected after directory rename |
| v0.4    | 2026-05-09 | rohan  | Context & Goals rewritten to use consistent Goal 1/2/3 framing |
| v0.5    | 2026-05-09 | rohan  | Self-contained architecture: `load_raw_jsons_to_db.py` creates all 6 base tables from raw JSON; `semantic_clustering.py` reads from Postgres (not files) so re-clustering works as dataset grows; all 9 tables live in one `meeting_analytics` schema; no Take A/B dependency |

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

**Dataset:** 100 meeting folders, each with `summary.json` (topics, key moments,
action items, sentiment) and `transcript.json` (sentence-level data).

The three goals carried across all takes:

**Goal 1 — Theme assignment:** Assign each meeting one or more business themes
from a discoverable taxonomy.
→ **Final Version addresses this.** 343 deduplicated topic phrases are embedded with
nomic-embed-text, reduced via UMAP, and clustered with HDBSCAN into 26 semantic
clusters. Each cluster receives an LLM-generated label (theme title, target
audience, rationale). Meetings are assigned to themes by mapping their topics to
cluster assignments.

**Goal 2 — Call type inference:** Infer the kind of conversation per meeting
(support, external, internal).
→ **Final Version addresses this.** Step 7 sends each meeting's summary + topics to
llama3.1:8b and receives a structured call type label (support / external /
internal). Stored in `semantic_meeting_themes.call_type`.

**Goal 3 — Postgres persistence:** Persist raw and derived fields to a schema
shaped for analytical queries.
→ **Final Version addresses this end-to-end.** Step 0 (`load_raw_jsons_to_db.py`) creates 6
base tables from raw JSON (`meetings`, `meeting_participants`, `meeting_summaries`,
`key_moments`, `action_items`, `transcript_lines`). Step 9 adds 3 semantic tables
(`semantic_clusters`, `semantic_phrases`, `semantic_meeting_themes`). All 9 tables
live in the single `meeting_analytics` schema. The pipeline is fully self-contained
— no dependency on any other folder or prior run. 8 insight queries join base tables
and semantic tables in the same schema, requiring no cross-schema joins.

**Why not rule-based keyword matching?** Brittle — misses semantic equivalents (e.g. `pipeline failure` ≡ `detect pipeline failure` ≡ `ingestion pipeline`).

**Why not TF-IDF + KMeans?** Requires pre-specifying K; bag-of-words misses phrase semantics; very short phrases kill TF-IDF signal.

**Why Final Version?** The topic strings are short, semantically dense, and conceptually overlapping. A vector embedding captures "outage recovery" ≈ "outage remediation" ≈ "post-mortem" without hand-crafted rules. HDBSCAN finds the natural cluster count without a fixed K. LLM labels make clusters immediately legible to stakeholders without manual review.

---

## 2. Pipeline Overview

```
[dataset/ JSON × N]
       │
       ▼
[0] Load raw JSON → Postgres base tables
    (meetings, key_moments, transcripts, summaries, participants, action_items)
       │
       ▼
[1] Extract & Flatten topics  (reads meeting_analytics.meeting_summaries.topics[])
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
[8] Persist to Postgres  (semantic_clusters, semantic_phrases, semantic_meeting_themes)
       │
       ▼
[Output] Named themes + per-meeting theme assignment + call-type label
         + 8 insight queries joining base tables and semantic tables
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

**Why topic phrases — connection to requirements:**

`req.md` asks to *"build a pipeline that processes the transcripts and categorizes them by **topic or theme**."* The word "topic" is exact — every `summary.json` already contains `topics[]`, a list of pre-extracted atomic business concepts (e.g. `api rate limiting`, `churn signal`, `hipaa compliance`). These were generated during transcript summarization and represent the clearest expression of what each call was about.

Four reasons phrase-level embedding is the right choice here:

1. **Semantic precision.** Each phrase is a single concept. Embedding it independently lets the model capture `outage recovery ≈ outage remediation ≈ post-mortem` without hand-crafted rules — which would be required in a rule-based approach (Take A).

2. **Cluster resolution.** Embedding at phrase level (not meeting level) gives far finer cluster granularity. A meeting about HIPAA and API rate limiting maps to two distinct clusters rather than a blended meeting-level vector that lands in neither.

3. **Clean aggregation path.** The phrase→cluster→theme→meeting path is explicit and auditable: we know exactly which topics pulled a meeting into each theme. This matters for stakeholder trust — an insight like "meetings about API Rate Limiting co-occur 3× more often with churn signals than meetings about Product Roadmap" is traceable to specific phrases.

4. **Repeatability as dataset grows.** New meetings add new topic phrases to `meeting_analytics.meeting_summaries.topics[]` via `load_raw_jsons_to_db.py`. Re-running the pipeline reads from Postgres (Step 1 queries `topics[]`), so the clustering automatically incorporates new data without any filesystem changes.

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

**Actual result:** `min_cluster_size=5`, `min_samples=3` → **26 clusters**, 22 noise phrases (6.4% noise ratio). Noise phrases were reassigned to nearest centroid as designed.

**Noise handling:** Phrases labeled `-1` (noise) are assigned to their nearest cluster centroid via cosine similarity as a post-processing step.

**Coherence check (added during implementation):** After clustering, each cluster's avg pairwise cosine similarity is computed and flagged:
- `tight` ≥ 0.6 — 9 of 26 clusters
- `review` 0.4–0.6 — 17 of 26 clusters
- `LOOSE` < 0.4 — 0 clusters

All 26 clusters are in review+ range; no loose clusters requiring manual inspection.

---

### 3.6 LLM Cluster Labeling

For each discovered cluster, sample up to 20 phrases sorted by distance to the cluster centroid (ascending) and ask a local LLM to assign a leadership-ready theme label. Implemented in `_sort_phrases_by_centroid_proximity(phrases, reduced)`, called at step 6 before `label_clusters()`.

**Actual prompt used:**
```
You are categorizing customer call topics for a B2B SaaS company's leadership team.

The following phrases all come from one theme cluster discovered by semantic clustering.
Based only on these phrases, provide a short, executive-level theme label.

Phrases:
{phrases}

Respond with valid JSON only — no extra text, no markdown fences:
{"theme_title": "<3-6 words, title case>", "audience": "<Engineering | Product | Sales | All>", "rationale": "<one sentence: why this theme matters to that audience>"}
```

| Labeling strategy | Pros | Cons |
|-------------------|------|------|
| **Single-shot per cluster** | Fast, cheap | May miss nuance |
| Few-shot with examples | More consistent | Needs example curation |
| Iterative (label → review → refine) | Highest quality | Requires human loop |
| Hierarchical (label sub-clusters first) | Natural for large K | Over-engineering at this scale |

**Decision:** Single-shot per cluster. With 26 clusters this was 26 LLM calls — completed in ~141s total (dominated by embedding, not labeling). All 26 produced valid labels; no fallback was triggered.

**Model:** `llama3.1:8b` (already configured), temperature=0.2 for consistency. If label quality degrades on re-run, bump to `llama3.3:70b` or add a one-shot example.

**Structured output:** Implemented with regex JSON extraction (`_extract_json()`), not `instructor` or `pydantic_ai`. The response strips markdown fences, finds the first `{...}` block, and parses it. 3 retry attempts per cluster; falls back to a `"phrase1 / phrase2 / phrase3"` title if all retries fail (no fallbacks were needed in the actual run).

---

### 3.7 Call-Type Inference

Each meeting has a hidden call type (support / external / internal). Two sub-approaches:

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Theme-majority vote** | Map meeting → cluster IDs → look up which clusters correlate with each call type | No extra LLM call | Needs cluster→call-type mapping (manual or auto) |
| **Direct LLM classification** | Feed `summary` text to LLM, ask for call type | Highest accuracy | One LLM call per meeting (N=100 — still cheap) |
| **Keyword rules** | Match summary keywords to call-type vocabulary | Zero cost, no LLM needed | Lower accuracy |

**Decision:** Direct LLM classification on the `summary` field. One structured prompt per meeting, 3-way classification (support / external / internal). This is a separate lightweight step from theme clustering and gives clean ground truth for the sentiment analysis dashboard.

---

### 3.8 Output & Downstream Use

Final artifacts written to Postgres (`meeting_analytics` schema, three new tables):

```sql
-- Cluster metadata (one row per cluster)
CREATE TABLE meeting_analytics.semantic_clusters (
    cluster_id   INTEGER PRIMARY KEY,
    theme_title  TEXT,
    audience     TEXT,   -- Engineering | Product | Sales | All
    rationale    TEXT
);

-- Embedded phrases (one row per canonical topic phrase)
CREATE TABLE meeting_analytics.semantic_phrases (
    id            SERIAL PRIMARY KEY,
    canonical     TEXT,
    cluster_id    INTEGER REFERENCES semantic_clusters(cluster_id),
    embedding     vector(768),     -- IVFFlat index for cosine search
    content_tsv   tsvector         -- GIN index for full-text search
);

-- Meeting-to-theme assignments (one row per meeting × theme)
CREATE TABLE meeting_analytics.semantic_meeting_themes (
    meeting_id   TEXT,
    cluster_id   INTEGER REFERENCES semantic_clusters(cluster_id),
    is_primary   BOOLEAN,
    call_type    TEXT,   -- support | external | internal
    sentiment    TEXT
);
```

**8 insight query methods** (all run automatically after persist, implemented in `load_output_csvs_to_db.py`):

| Method | What it shows |
|--------|--------------|
| `insight_theme_sentiment()` | Avg sentiment score per theme |
| `insight_churn_by_theme()` | Churn signal count per theme |
| `insight_call_type_theme_matrix()` | Meeting count by call type × theme |
| `insight_feature_gap_themes()` | Feature gap signal count per theme |
| `insight_sentiment_distribution_by_theme()` | Sentiment bucket breakdown per theme |
| `insight_signal_counts_by_theme()` | All keyMoment signal types per theme |
| `insight_theme_cooccurrence()` | Which themes appear together in the same meeting |
| `insight_high_risk_meetings()` | Meetings with churn signals + negative sentiment |

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
| Schema | Single `meeting_analytics` schema (all 9 tables) | Separate schemas | Self-contained; insight queries join base + semantic tables without cross-schema joins; new meetings added via `load_raw_jsons_to_db.py`, clustering re-reads from DB |

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
    representative_phrases: list[str]  # top-20 sorted by centroid proximity (ascending distance)

class MeetingThemeAssignment(BaseModel):
    meeting_id: str
    inferred_call_type: str  # support | external | internal
    theme_ids: list[int]     # all clusters present in this meeting's topics
    primary_theme_id: int    # most frequent cluster
```

---

## 6. Implementation Sketch

Target file: `basics/iprep/meeting-analytics/final_version/semantic_clustering.py`

```
Functions (async throughout):
─────────────────────────────────────────────────────────────
load_records_from_db()         → list[MeetingRecord]          # queries meeting_analytics.meeting_summaries
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

## 7. Open Questions — Resolved

1. **How many clusters will HDBSCAN find?**
   → **26 clusters** with `min_cluster_size=5`, `min_samples=3`. Estimate of 8–15 was too conservative; the semantic embedding space supports finer resolution than TF-IDF k=8.

2. **Is topics-only sufficient, or do we need `keyMoments`?**
   → Topics-only was sufficient. All 26 clusters have coherence ≥ 0.45; no cluster required `keyMoments` augmentation.

3. **Should themes be hierarchical?**
   → Not needed at this scale. The 26 clusters are already specific enough (e.g. HIPAA Compliance and Reporting vs Audit Readiness vs Compliance and Governance as distinct clusters). Hierarchy deferred — would add value only in a live product with growing data.

4. **Reuse existing Postgres records?**
   → Yes. Step 0 (`load_raw_jsons_to_db.py`) loads raw JSON into 6 base tables on first run. Step 1 then reads from `meeting_analytics.meeting_summaries.topics[]` — not from disk. Re-running the pipeline on a growing dataset only requires re-running `load_raw_jsons_to_db.py` to insert new meetings; the clustering reads all records from the DB automatically. Use `--skip-base-load` to skip Step 0 when the DB is already populated, or `--reset-db` to rebuild from scratch.

5. **Visualization output?**
   → 2-dim UMAP is computed but scatter plot rendering is deferred to the Jupyter notebook deliverable (next step).
