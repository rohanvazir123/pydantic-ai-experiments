# Data Model — Meeting Analytics
Schema: `meeting_analytics` @ `localhost:5434` / `rag_db`

---

## Entity Relationship Overview

```
meetings (100)
  │
  ├── meeting_participants (311)     1 meeting → many participants
  ├── meeting_summaries (100)        1:1 — sentiment, topics[], products[]
  ├── key_moments (402)              1 meeting → many moments (8 types)
  ├── action_items (397)             1 meeting → many action items
  └── transcript_lines (4313)        1 meeting → many transcript lines

semantic_clusters (26)              one row per discovered theme
  │
  ├── semantic_phrases (343)         topic phrases → clusters
  └── semantic_meeting_themes (516)  junction: meetings ↔ clusters
        │
        └── action_items_by_theme   VIEW: action items + primary theme
```

---

## Base Tables

### `meetings` — 100 rows
One row per meeting.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT PK | e.g. `"meeting_001"` |
| `title` | TEXT | Human-readable meeting title |
| `organizer_email` | TEXT | Who called the meeting |
| `duration_minutes` | NUMERIC | Call length |
| `start_time` | TIMESTAMPTZ | When the call happened |

---

### `meeting_participants` — 311 rows
One row per (meeting, participant email) pair.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT FK → meetings | |
| `email` | TEXT | Participant email |

PK: `(meeting_id, email)`

---

### `meeting_summaries` — 100 rows
One row per meeting. AI-generated summary fields + extracted signals.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT PK FK → meetings | |
| `summary_text` | TEXT | Full AI-generated summary |
| `overall_sentiment` | TEXT | See values below |
| `sentiment_score` | NUMERIC | 1.0–5.0 (1=most negative) |
| `topics` | TEXT[] | Raw topic phrases from AI summary (600 total across 100 meetings, deduped to 343) |
| `products` | TEXT[] | AegisCloud products mentioned — extracted at load time |

**`overall_sentiment` values:** `negative` · `very-negative` · `mixed-negative` · `neutral` · `mixed-positive` · `positive` · `very-positive`

**`products` values:** `Detect` · `Protect` · `Comply` · `Identity`
Coverage: Comply 59, Detect 59, Protect 24, Identity 23, untagged 8.
Query pattern: `WHERE 'Detect' = ANY(products)`

---

### `key_moments` — 402 rows
One row per notable moment extracted from a meeting. The primary signal table.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT FK → meetings | |
| `moment_index` | INTEGER | Order within meeting |
| `moment_type` | TEXT | See 8 types below |
| `text` | TEXT | Verbatim quote or description |
| `speaker` | TEXT | Who said it |
| `time_seconds` | NUMERIC | Position in the call |

PK: `(meeting_id, moment_index)`
Index: `(meeting_id, moment_type)` — use this for signal filtering.

**`moment_type` values and what they mean:**

| Type | Signal | Who cares |
|------|--------|-----------|
| `churn_signal` | Customer expressing intent to leave or evaluate alternatives | Sales, CSMs |
| `concern` | Frustration, complaint, or worry raised | Support, Product |
| `feature_gap` | Customer asking for something we don't have | Product |
| `technical_issue` | Reported bug, outage, or performance problem | Engineering |
| `praise` | Positive feedback about the product | Product, Marketing |
| `pricing_offer` | Pricing discussion, discount, or commercial negotiation | Sales |
| `positive_pivot` | Conversation turning from negative to positive | Support, Engineering |

---

### `action_items` — 397 rows
One row per action item extracted from a meeting.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT FK → meetings | |
| `item_index` | INTEGER | Order within meeting |
| `owner` | TEXT | Person responsible |
| `text` | TEXT | What needs to be done |

PK: `(meeting_id, item_index)`

---

### `transcript_lines` — 4313 rows
One row per sentence in the meeting transcript.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT FK → meetings | |
| `line_index` | INTEGER | Order within transcript |
| `speaker` | TEXT | Speaker name |
| `sentence` | TEXT | Spoken sentence |
| `sentiment_type` | TEXT | Per-sentence sentiment |
| `time_seconds` | NUMERIC | Position in the call |

PK: `(meeting_id, line_index)`

---

## Semantic Tables

### `semantic_clusters` — 26 rows
One row per theme cluster discovered by HDBSCAN. All label fields are LLM-generated.

| Column | Type | Notes |
|--------|------|-------|
| `cluster_id` | INTEGER PK | 0–25, assigned by HDBSCAN |
| `theme_title` | TEXT | LLM-generated, e.g. "Identity & Access Management" |
| `audience` | TEXT | LLM-generated: Engineering · Product · Sales · All |
| `rationale` | TEXT | LLM-generated one-sentence explanation |
| `phrase_count` | INTEGER | How many topic phrases map to this cluster |

---

### `semantic_phrases` — 343 rows
One row per deduplicated topic phrase. The atomic input to clustering.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID PK | |
| `canonical` | TEXT UNIQUE | Deduplicated phrase, e.g. "mfa enforcement" |
| `aliases` | TEXT[] | Variant forms seen across meetings |
| `cluster_id` | INTEGER FK → semantic_clusters | Which theme this phrase belongs to |
| `embedding` | vector(768) | NULL after CSV load; populated by full pipeline run |
| `content_tsv` | tsvector | GENERATED from canonical — GIN indexed |

Indexes: IVFFlat on `embedding` (cosine), GIN on `content_tsv`, B-tree on `cluster_id`.

---

### `semantic_meeting_themes` — 516 rows
Junction table: one row per (meeting, cluster) pair. The core analytical table.

| Column | Type | Notes |
|--------|------|-------|
| `meeting_id` | TEXT | FK → meetings |
| `cluster_id` | INTEGER | FK → semantic_clusters |
| `is_primary` | BOOLEAN | True for exactly **one** row per meeting — the dominant theme |
| `call_type` | TEXT | LLM-inferred: `support` · `external` · `internal` |
| `call_confidence` | TEXT | LLM confidence: `high` · `medium` · `low` |
| `sentiment_score` | NUMERIC | Denormalized from `meeting_summaries.sentiment_score` |
| `overall_sentiment` | TEXT | Denormalized from `meeting_summaries.overall_sentiment` |
| `products` | TEXT[] | Denormalized from `meeting_summaries.products` |

PK: `(meeting_id, cluster_id)`

**`is_primary` explained:** A meeting touches multiple clusters. `is_primary = true` marks the cluster where the majority of that meeting's topic phrases landed. Filter on `is_primary = true` for per-meeting aggregations to avoid double-counting. Drop the filter for multi-label analysis (co-occurrence, compound problems).

**Why denormalize `sentiment`, `call_type`, `products`:** All three are per-meeting attributes carried into the junction so that product × theme, call type × theme, and sentiment × theme queries need no additional joins.

---

## View

### `action_items_by_theme` — 397 rows
Pre-joined view: every action item with its primary theme and audience.

```sql
SELECT
    ai.meeting_id,
    ai.owner,
    ai.text        AS action_item,
    smt.cluster_id,
    sc.theme_title,
    sc.audience
FROM action_items ai
JOIN semantic_meeting_themes smt ON ai.meeting_id = smt.meeting_id AND smt.is_primary = true
JOIN semantic_clusters sc        ON smt.cluster_id = sc.cluster_id
```

Use this for: action item volume by theme, owner workload by theme, follow-through accountability.

---

## Common Join Paths

**Theme + signal counts (no extra join needed):**
```sql
SELECT sc.theme_title, count(*) AS signals
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc USING (cluster_id)
JOIN meeting_analytics.key_moments km USING (meeting_id)
WHERE smt.is_primary = true
  AND km.moment_type = 'churn_signal'
GROUP BY sc.theme_title;
```

**Product × theme (products already on junction):**
```sql
WHERE 'Detect' = ANY(smt.products)
  AND smt.is_primary = true
```

**Product × signal type:**
```sql
FROM meeting_analytics.meeting_summaries ms
JOIN meeting_analytics.key_moments km USING (meeting_id)
WHERE 'Detect' = ANY(ms.products)
  AND km.moment_type = 'technical_issue'
```

**High-risk accounts (churn + negative sentiment):**
```sql
WHERE smt.is_primary = true
  AND smt.sentiment_score < 3.0
  AND km.moment_type = 'churn_signal'
```

---

## Reference Counts

| Table | Rows | Loaded by |
|-------|------|-----------|
| `meetings` | 100 | `load_raw_jsons_to_db.py` |
| `meeting_participants` | 311 | `load_raw_jsons_to_db.py` |
| `meeting_summaries` | 100 | `load_raw_jsons_to_db.py` |
| `key_moments` | 402 | `load_raw_jsons_to_db.py` |
| `action_items` | 397 | `load_raw_jsons_to_db.py` |
| `transcript_lines` | 4313 | `load_raw_jsons_to_db.py` |
| `semantic_clusters` | 26 | `load_output_csvs_to_db.py` |
| `semantic_phrases` | 343 | `load_output_csvs_to_db.py` |
| `semantic_meeting_themes` | 516 | `load_output_csvs_to_db.py` |
| `action_items_by_theme` | 397 | view (no load needed) |
