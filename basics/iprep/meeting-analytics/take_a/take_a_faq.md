# Take A — FAQ

## Table of Contents
- [What is schema\_dump.sql?](#what-is-schema_dumpsql)
- [Why no LLM in Take A?](#why-no-llm-in-take-a)
- [How does theme scoring work?](#how-does-theme-scoring-work)
- [How is call type inferred?](#how-is-call-type-inferred)
- [What does the confidence score in call\_types mean?](#what-does-the-confidence-score-in-call_types-mean)
- [What files are loaded vs discarded, and why?](#what-files-are-loaded-vs-discarded-and-why)
- [What are the 10 Postgres tables and what is in each?](#what-are-the-10-postgres-tables-and-what-is-in-each)
- [How do I reload the Take A tables from scratch?](#how-do-i-reload-the-take-a-tables-from-scratch)
- [What is sentiment\_features and how is net\_sentiment computed?](#what-is-sentiment_features-and-how-is-net_sentiment-computed)
- [How does the external taxonomy.json work?](#how-does-the-external-taxonomyjson-work)
- [Why does concern not get a direct theme boost?](#why-does-concern-not-get-a-direct-theme-boost)

---

## What is schema\_dump.sql?

An aside — a set of DBeaver-style introspection queries for the `meeting_analytics`
schema created by Take A. Not part of any pipeline. Useful for browsing the live
Postgres schema:

- List all tables in `meeting_analytics`
- Describe columns (type, nullability, defaults)
- Show constraints (PKs, FKs)
- Pull the distinct topic list from `summary_topics`

---

## Why no LLM in Take A?

The dataset already contains structured analytical fields that an LLM would normally
be asked to extract from raw text:

| Field | Location | What it provides |
|-------|----------|-----------------|
| `topics` | `summary.json` | What the meeting is about — curated topic tags |
| `keyMoments[].type` | `summary.json` | Business signals: `churn_signal`, `concern`, `technical_issue`, `feature_gap`, `praise`, `pricing_offer` |
| `overallSentiment` + `sentimentScore` | `summary.json` | Meeting-level sentiment |
| `sentimentType` | `transcript.json data[]` | Per-sentence sentiment evidence |

Because those fields already exist, the problem is normalization — mapping granular
topic tags into stakeholder-useful themes — not open-ended text understanding.

A deterministic ruleset is the better first choice because:
- **Auditable**: every theme assignment can be traced to a specific keyword match
- **Rerunnable**: same input always produces the same output
- **Debuggable in SQL**: `WHERE theme = 'X'` shows you exactly which meetings matched
- **Explainable in Q&A**: "this meeting was tagged Compliance because its topics included 'soc 2' and 'audit'"

Take B and Take C serve as independent validation of what Take A found — their value
comes partly from the comparison.

---

## How does theme scoring work?

`infer_themes()` scores each of the 8 themes per meeting across three input sources:

**Source 1 — Topic keyword matching**

Each `summary.json` topic is normalized (lowercased, whitespace collapsed) and
matched against `THEME_KEYWORDS`. For every keyword that appears as a substring,
that theme's score increments by 1. Multiple topics can match the same theme.

Example: topic `"soc 2 audit"` matches keywords `"audit"` and `"soc 2"` in
`"Compliance / Audit / Security Assurance"` → score +2.

**Source 2 — Key-moment type boosts**

| Signal | Theme boosted | Amount |
|--------|--------------|--------|
| `technical_issue` | Reliability / Incidents / Outages | +1 |
| `technical_issue` | Support / Customer Escalation | +1 |
| `churn_signal` | Customer Retention / Commercial Risk | +2 |
| `feature_gap` | Product Feedback / Feature Gaps / Roadmap | +2 |
| `concern` | *(no direct boost — see source 3)* | — |

`churn_signal` and `feature_gap` get a double boost because they are specific,
high-value signals. `concern` is intentionally broad.

**Source 3 — Summary scan for `concern`**

When `concern` appears in key moments, the full `summary` text is scanned for
theme keywords. This provides extra evidence when a meeting has a broadly negative
tone but the topics alone don't spell out the theme.

**Primary theme selection:**

The highest-scoring theme is marked `is_primary = true` in `meeting_themes`.
All nonzero-scoring themes are stored (one row each), allowing a meeting to belong
to multiple themes. The `evidence_count` column stores the raw score.

---

## How is call type inferred?

`infer_call_type()` is a separate classifier from theme inference. Themes answer
"what is this meeting about?" Call type answers "what kind of conversation is this?"

**Step 1 — Combined text scan**

Topics and summary text are concatenated and scored against `CALL_TYPE_HINTS`
(5 call types, each with a keyword list). Same substring matching as theme scoring.

**Step 2 — Theme-based nudges**

Weak signal from already-computed theme scores nudges likely call types:

| Theme with nonzero score | Call type nudged | Amount |
|--------------------------|-----------------|--------|
| Customer Retention / Commercial Risk | `sales_or_renewal` | +1 |
| Support / Customer Escalation | `support_escalation` | +1 |
| Internal Engineering / Planning | `internal_planning` | +1 |
| Reliability (without "customer" in combined text) | `internal_incident` | +1 |

The "without customer" guard on Reliability prevents outage meetings that involve
a customer from being miscategorised as internal incident reviews.

**The 5 call types:**

| Call type | What it represents |
|-----------|--------------------|
| `support_escalation` | Customer-facing support or bug escalation |
| `sales_or_renewal` | Renewal, pricing, contract, QBR, or competitive discussion |
| `internal_incident` | Post-mortem, incident review, root cause analysis |
| `internal_planning` | Sprint planning, architecture review, capacity planning |
| `external_customer` | Onboarding, product demo, customer feedback session |

If no keywords match at all, the call type is `"unknown"` with confidence 0.0.

**Important: 5-type taxonomy is self-invented — req.md defines 3 types.**

The brief (`req.md`) explicitly defines only **three** call types: `support`, `external`, `internal`.
Take A's 5-type taxonomy (`support_escalation`, `sales_or_renewal`, `internal_incident`,
`internal_planning`, `external_customer`) was added as a finer-grained elaboration but deviates
from the assignment spec. The raw dataset JSON files do not contain any `call_type` field.

The planned fix is a deterministic 3-type classifier that maps directly to req.md's taxonomy.
Until then, use the 5-type table for all queries — or collapse to 3 in Python before charting:
- `support_escalation` → `support`
- `sales_or_renewal` → `external`
- `external_customer` → `external`
- `internal_incident` → `internal`
- `internal_planning` → `internal`

---

## What does the confidence score in call\_types mean?

```python
confidence = min(0.95, 0.45 + (score * 0.15))
```

It is a lightweight heuristic, not a calibrated probability.

| Winning score | Confidence |
|---------------|-----------|
| 1 | 0.60 |
| 2 | 0.75 |
| 3 | 0.90 |
| 4+ | 0.95 (capped) |

The floor of 0.45 means even a single keyword match produces a reasonable-looking
number. The practical use is to flag weak classifications: anything below ~0.60
(score = 1) deserves a quick manual review. It is intentionally not designed to
be precise — just transparent enough to be useful in Q&A.

---

## What files are loaded vs discarded, and why?

| File | Loaded? | Reason |
|------|---------|--------|
| `meeting-info.json` | Yes | Meeting metadata, participants, timestamps |
| `summary.json` | Yes | Topics, key moments, action items, sentiment — all core analytical fields |
| `transcript.json` | Yes | Sentence-level sentiment evidence |
| `speakers.json` | No | Redundant — speaker names and IDs already in transcript lines |
| `events.json` | No | Join/leave telemetry — not needed for theme or sentiment analysis |
| `speaker-meta.json` | No | Redundant with transcript_lines speaker_id/speaker_name |

The design principle: load what the assignment questions require, not every source
file. The `--dry-run` output prints each file type with its load/discard decision
so the reasoning is visible without reading the code.

---

## What are the 10 Postgres tables and what is in each?

All tables live in the `meeting_analytics` schema. Natural keys throughout
(`meeting_id` is the text folder name from the dataset).

| Table | One row per | Key columns |
|-------|-------------|-------------|
| `meetings` | Meeting | `meeting_id`, `title`, `organizer_email`, `start_time`, `duration_minutes` |
| `meeting_participants` | Meeting × participant | `meeting_id`, `email`, `participant_role` (`all_email` / `invitee`) |
| `meeting_summaries` | Meeting | `summary` text, `overall_sentiment`, `sentiment_score` |
| `summary_topics` | Meeting × topic | `topic` (original), `normalized_topic` (lowercased, trimmed) |
| `action_items` | Meeting × action item | `owner` (extracted from "Name: action" prefix), `action_item` text |
| `key_moments` | Meeting × moment | `time_seconds`, `speaker_name`, `moment_type`, `text` |
| `transcript_lines` | Meeting × sentence | `speaker_name`, `sentiment_type`, `start_seconds`, `sentence` |
| `meeting_themes` | Meeting × theme | `theme`, `evidence_count`, `is_primary` |
| `call_types` | Meeting | `call_type`, `confidence` |
| `sentiment_features` | Meeting | Aggregated sentiment ratios + signal counts (see below) |

The schema is shaped around analytical questions (themes, sentiment trends, signal
counts) rather than mirroring JSON file structure.

---

## What is sentiment\_features and how is net\_sentiment computed?

`sentiment_features` is a derived table aggregated from `transcript_lines`. It gives
a continuous, transcript-grounded sentiment signal per meeting:

```
positive_ratio  = positive_lines  / total_lines
negative_ratio  = negative_lines  / total_lines
net_sentiment   = positive_ratio  - negative_ratio
```

Range of `net_sentiment`: −1.0 (all negative) to +1.0 (all positive).

It also stores key-moment signal counts from `summary.json`:
`concern_count`, `churn_signal_count`, `technical_issue_count`,
`feature_gap_count`, `praise_count`, `action_item_count`.

**Why bother when summary.json already has sentimentScore?**

`sentimentScore` comes from the dataset provider's AI summary model. `net_sentiment`
is computed directly from transcript line counts. The two can diverge — a meeting
where most transcript lines are neutral but one highly-negative key moment drives
a low `sentimentScore` will show different values in the two columns. Having both
lets you interrogate that divergence.

---

## How does the external taxonomy.json work?

By default, Take A uses the built-in `THEME_KEYWORDS` and `CALL_TYPE_HINTS` maps
hard-coded in the script. If `taxonomy.json` exists (or is passed via `--taxonomy`),
`load_taxonomy()` reads it instead.

**Expected format:**

```json
{
  "themes": [
    {
      "theme_name": "Reliability / Incidents / Outages",
      "keyword_hints": ["outage", "incident", "sla", "failure"]
    }
  ],
  "call_types": [
    {
      "call_type": "support_escalation",
      "keyword_hints": ["support", "ticket", "escalation"]
    }
  ]
}
```

**What `load_taxonomy()` does to keywords:**
- Lowercases and trims whitespace (`clean_topic()`)
- Removes duplicates
- Sorts alphabetically for stable run-to-run behavior
- Validates that at least one theme and one call type have usable keywords — raises
  `ValueError` if the file loads but is effectively empty

**Use case:** an expert can refine the taxonomy between runs (add keywords, rename
themes, add a new call type) without touching the source code. The pipeline behavior
is identical regardless of whether it reads from the built-in map or the JSON file.

---

## Why does concern not get a direct theme boost?

`concern` is the broadest signal type in the dataset — it appears across almost
every call type and topic area. Boosting "Reliability" or "Support" every time
`concern` appears would inflate those themes for meetings that are actually about
billing disputes, product gaps, or compliance deadlines.

Instead, when `concern` is present, the full `summary` text is scanned for theme
keywords. This lets the actual content of the concern determine which theme gets
the boost, rather than making an assumption about what concerns are typically about.

Example: a meeting with `concern` + summary text mentioning "audit deadline" and
"soc 2" will correctly boost "Compliance / Audit / Security Assurance". Without
the summary scan, it would boost nothing useful.

---

## How do I reload the Take A tables from scratch?

Take A's source of truth is the raw dataset (100 meeting folders). There are no
intermediate output files — the script reads JSON and writes directly to Postgres.

**Full reset (drop schema + recreate + reload):**

```bash
python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py --reset
```

This drops the entire `meeting_analytics` schema and rebuilds it from scratch,
including all 10 tables. **This will also drop Take B and Take C tables** — run
`setup_all_tables.py` instead if you want all three takes reloaded together.

**Reload all three takes in one shot:**

```bash
python basics/iprep/meeting-analytics/setup_all_tables.py
```

Target: `rag_db @ localhost:5434` (rag_user:rag_pass). Credentials read from
`meeting-analytics/.env`. Output: 16 tables in `meeting_analytics` schema.

**What each Take A table comes from:**

| Table | Source in dataset |
|-------|------------------|
| `meetings` | `meeting-info.json` |
| `meeting_participants` | `meeting-info.json participants` |
| `meeting_summaries` | `summary.json summary + sentiment` |
| `summary_topics` | `summary.json topics` |
| `action_items` | `summary.json actionItems` |
| `key_moments` | `summary.json keyMoments` |
| `transcript_lines` | `transcript.json data[]` |
| `meeting_themes` | derived by `infer_themes()` |
| `call_types` | derived by `infer_call_type()` |
| `sentiment_features` | aggregated from `transcript_lines` |
