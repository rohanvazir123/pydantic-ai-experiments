# Take A — Rule-Based Taxonomy: Design Document

## Audit Trail

| Version | Date       | Author | Summary of Changes |
|---------|------------|--------|--------------------|
| v0.1    | 2026-05-09 | rohan  | Initial design doc — written retrospectively from completed implementation |

---

## Table of Contents

1. [Context & Goals](#1-context--goals)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step-by-Step Design](#3-step-by-step-design)
   - [3.1 Dataset Loading](#31-dataset-loading)
   - [3.2 Taxonomy Source](#32-taxonomy-source)
   - [3.3 Theme Inference](#33-theme-inference)
   - [3.4 Call-Type Inference](#34-call-type-inference)
   - [3.5 Sentiment Features](#35-sentiment-features)
   - [3.6 Schema Design](#36-schema-design)
4. [Decision Log](#4-decision-log)
5. [Data Model](#5-data-model)
6. [CLI Reference](#6-cli-reference)

---

## 1. Context & Goals

**Dataset:** 100 meeting folders, each containing:
- `meeting-info.json` — meeting metadata (title, organizer, participants, timestamps)
- `summary.json` — AI-extracted summary, topics, key moments, action items, sentiment
- `transcript.json` — sentence-level transcript with per-line sentiment
- `speakers.json`, `events.json`, `speaker-meta.json` — discarded (see 3.1)

**Goal 1 (primary):** Assign each meeting one or more business **themes** from a
fixed taxonomy (e.g. "Reliability / Incidents / Outages") using deterministic keyword rules.

**Goal 2:** Infer a **call type** per meeting (support escalation, sales/renewal,
internal incident review, internal planning, external customer).

**Goal 3:** Persist all raw and derived fields to Postgres in a schema shaped for
analytical queries — not a mirror of the JSON file boundaries.

**Why not start with an LLM or ML?**

The dataset already contains structured analytical fields that an LLM would normally
be asked to extract from raw text:
- `summary.json topics` — what the meeting is about
- `summary.json keyMoments[].type` — business signals (`churn_signal`, `concern`,
  `technical_issue`, `feature_gap`, `praise`, `pricing_offer`)
- `overallSentiment` + `sentimentScore` — meeting-level sentiment
- `transcript.json data[].sentimentType` — sentence-level sentiment evidence

Because those fields already exist, the problem is normalization: mapping granular topic
tags into stakeholder-useful themes. A deterministic ruleset is a better first choice
because it is auditable, cheap to rerun, easy to debug in SQL, and simple to explain
during Q&A.

Take B and Take C exist to show what ML and LLM approaches find independently —
their value comes partly from being compared against Take A's hand-crafted baseline.

---

## 2. Pipeline Overview

```
[dataset/]  100 meeting folders
       │
       ▼
[1] Load records — meeting-info.json + summary.json + transcript.json
       │
       ▼
[2] Load taxonomy — taxonomy.json (if exists) or built-in THEME_KEYWORDS fallback
       │
       ▼
[3] Create / reset Postgres schema (meeting_analytics) + 10 tables
       │
       ▼
[4] For each meeting:
       ├── Insert meetings + meeting_participants
       ├── Insert meeting_summaries + summary_topics + action_items + key_moments
       ├── Insert transcript_lines (sentence-level sentiment)
       ├── infer_themes()  → insert meeting_themes
       ├── infer_call_type() → insert call_types
       └── aggregate sentiment/signal counts → insert sentiment_features
       │
       ▼
[5] Print row counts + starter DBeaver queries
```

---

## 3. Step-by-Step Design

### 3.1 Dataset Loading

**What is loaded:**

| File | Loaded | Reason |
|------|--------|--------|
| `meeting-info.json` | Yes | Meeting metadata, participants, timestamps |
| `summary.json` | Yes | Topics, key moments, action items, sentiment — core analytical fields |
| `transcript.json` | Yes | Sentence-level sentiment evidence |
| `speakers.json` | No | Redundant with transcript_lines speaker fields |
| `events.json` | No | Join/leave telemetry — not needed for analysis |
| `speaker-meta.json` | No | Redundant with transcript_lines speaker_id/speaker_name |

**Why discard three files?**

The design principle is to load what the assignment asks for, not to mirror every
source file. `speakers.json` and `speaker-meta.json` duplicate speaker information
already present in transcript lines. `events.json` contains join/leave telemetry
that has no bearing on topics, themes, or sentiment analysis.

Discarding them keeps the schema lean and the insert logic simple. The `--dry-run`
output prints every file type seen and whether it was loaded or discarded, so the
decision is transparent and auditable.

---

### 3.2 Taxonomy Source

Take A supports two taxonomy sources, both deterministic:

**Option A — Built-in fallback (default)**

`THEME_KEYWORDS` and `CALL_TYPE_HINTS` are hand-reviewed keyword maps built from
the observed topic vocabulary across all 100 meetings. 8 themes, 5 call types.

**Option B — External `taxonomy.json`**

If `basics/iprep/meeting-analytics/taxonomy.json` exists (or is passed via `--taxonomy`),
`load_taxonomy()` reads it instead of the fallback. Expected shape:

```json
{
  "themes": [
    { "theme_name": "Reliability / Incidents / Outages", "keyword_hints": ["outage", "incident", "sla"] }
  ],
  "call_types": [
    { "call_type": "support_escalation", "keyword_hints": ["support", "ticket", "escalation"] }
  ]
}
```

This enables a human expert to refine the taxonomy between runs without editing source
code. The downstream classifier treats built-in and external taxonomy identically.

**The 8 built-in themes:**

| Theme | Sample keywords |
|-------|----------------|
| Reliability / Incidents / Outages | outage, incident, sla, failure, post-mortem |
| Compliance / Audit / Security Assurance | compliance, audit, soc 2, hipaa, pci, gdpr |
| Identity / Access / Security Controls | identity, mfa, sso, scim, saml, ldap, provisioning |
| Product Feedback / Feature Gaps / Roadmap | feature, roadmap, product, launch, feedback |
| Customer Retention / Renewal / Commercial Risk | renewal, churn, pricing, billing, competitive |
| Support / Customer Escalation | support, escalation, ticket, workaround, bug |
| Implementation / Onboarding / Adoption | onboarding, deployment, migration, integration |
| Internal Engineering / Planning / Execution | sprint, qa, technical debt, architecture, planning |

**The 5 built-in call types:**

| Call type | Sample keywords |
|-----------|----------------|
| `support_escalation` | support, ticket, escalation, bug, response time |
| `sales_or_renewal` | renewal, pricing, contract, billing, upsell, qbr |
| `internal_incident` | post-mortem, incident review, root cause, remediation |
| `internal_planning` | sprint, qa, roadmap planning, architecture, design review |
| `external_customer` | onboarding, adoption, customer feedback, product demo |

---

### 3.3 Theme Inference

`infer_themes()` scores each theme by counting keyword matches across three input
sources, with structured business-signal boosts:

**Step 1 — Topic keyword matching**

Each `summary.json` topic is matched against `THEME_KEYWORDS`. For every keyword
that appears as a substring in the normalized topic, that theme's score increments
by 1. Multiple topics can match the same theme; scores accumulate.

**Step 2 — Key-moment type boosts**

Structured `keyMoments[].type` values add deterministic boosts:

| Signal | Boost |
|--------|-------|
| `technical_issue` | Reliability +1, Support +1 |
| `churn_signal` | Customer Retention / Commercial Risk +2 |
| `feature_gap` | Product Feedback / Roadmap +2 |
| `concern` | No direct boost — triggers summary scan (step 3) |

`churn_signal` and `feature_gap` get a double boost because they are specific,
high-value signals. `concern` is intentionally broad (it appears in many context
types) so it does not map to one theme directly.

**Step 3 — Summary scan for `concern`**

When `concern` appears in key moments, the full `summary` text is scanned for
theme keywords. This provides extra evidence when a meeting has a broadly negative
tone but the topics themselves don't spell out the theme.

**Primary theme selection:** The highest-scoring theme is marked `is_primary = true`
in `meeting_themes`. All nonzero-scoring themes are stored (one row each), allowing
a meeting to belong to multiple themes.

---

### 3.4 Call-Type Inference

`infer_call_type()` is a separate classifier that answers "what kind of conversation
is this?" — distinct from the themes, which answer "what is it about?".

**Step 1 — Combined text scan**

Topics and summary text are concatenated and scored against `CALL_TYPE_HINTS`.

**Step 2 — Theme-based nudges**

Weak signal from themes nudges likely call types:

| Theme present | Call type nudged |
|--------------|-----------------|
| Customer Retention / Commercial Risk | `sales_or_renewal` +1 |
| Support / Customer Escalation | `support_escalation` +1 |
| Internal Engineering / Planning | `internal_planning` +1 |
| Reliability (without "customer" in text) | `internal_incident` +1 |

The "without customer" guard on Reliability prevents outage meetings that involve
a customer from being miscategorised as internal incident reviews.

**Confidence score:**

```python
confidence = min(0.95, 0.45 + (score * 0.15))
```

This is a lightweight heuristic, not a calibrated probability. Meetings with a
winning score of 1 get confidence ~0.60; score of 3 gets ~0.90. Anything below
~0.55 deserves review. It is not designed to be precise — just to flag weak
classifications.

---

### 3.5 Sentiment Features

`sentiment_features` is a derived table aggregated per meeting from transcript lines:

- `positive_ratio` = positive lines / total lines
- `negative_ratio` = negative lines / total lines
- `net_sentiment` = positive_ratio − negative_ratio

This gives a continuous sentiment signal that can be trended over time and
compared across themes and call types. It is complementary to `sentimentScore`
(which comes from the summary provider) — the two can diverge if the AI summary
weights certain moments differently.

Signal counts from `keyMoments[].type` are also stored here:
`concern_count`, `churn_signal_count`, `technical_issue_count`, `feature_gap_count`,
`praise_count`, `action_item_count`.

---

### 3.6 Schema Design

10 tables in `meeting_analytics` schema, designed around analytical questions
rather than JSON file boundaries:

```
meetings                 — meeting metadata (1 row per meeting)
meeting_participants     — participant emails with role (all_email / invitee)
meeting_summaries        — AI-extracted summary text + sentiment score
summary_topics           — normalized topic tags (1 row per meeting × topic)
action_items             — action items with extracted owner
key_moments              — timestamped business signals (churn, concern, etc.)
transcript_lines         — sentence-level transcript + per-line sentiment
meeting_themes           — derived themes with evidence count + is_primary flag
call_types               — inferred call type + confidence score
sentiment_features       — aggregated sentiment ratios + signal counts
```

Indexes on `meeting_id` for all child tables to support fast per-meeting queries.
Natural keys throughout (no serial IDs) — `meeting_id` is the text folder name.

---

## 4. Decision Log

| Decision | Chosen | Alternatives Considered | Reason |
|----------|--------|------------------------|--------|
| Classification approach | Deterministic keyword rules | LLM classification, ML | Dataset already has structured topic/signal fields; rules are auditable and rerunnable |
| Taxonomy source | Built-in + optional external JSON | Hard-coded only | Allows expert refinement without code changes |
| Theme scoring unit | Per topic tag | Per summary sentence | Topics are already curated; topic-level scoring is more precise |
| Multi-label themes | Yes — all nonzero themes stored | Single-label only | A meeting can genuinely span multiple themes |
| `concern` handling | Triggers summary scan | Boost a generic theme | `concern` is too broad to map to one theme; summary scan finds the actual theme |
| Discarded files | speakers, events, speaker-meta | Load everything | Redundant or irrelevant to assignment questions |
| Confidence formula | Linear heuristic on winning score | Calibrated probability | Cheap, transparent, sufficient to flag weak classifications |
| Schema design | 10 tables by analytical concern | Mirror JSON file boundaries | Assignment asks for themes, sentiment trends, and signals — not raw JSON storage |
| Async | asyncpg for all Postgres I/O | psycopg2 sync | Consistent with RAG project patterns; future-proof for concurrent inserts |

---

## 5. Data Model

```python
@dataclass
class MeetingRecord:
    meeting_id: str
    meeting_info: dict      # from meeting-info.json
    summary: dict           # from summary.json
    transcript_lines: list  # from transcript.json data[]

@dataclass
class Taxonomy:
    theme_keywords: dict[str, list[str]]    # theme → keywords
    call_type_hints: dict[str, list[str]]   # call_type → keywords
```

---

## 6. CLI Reference

```bash
# Inspect dataset + taxonomy, no Postgres connection
python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py --dry-run

# Drop + recreate schema, then load all 100 meetings
python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py --reset

# Use external taxonomy file
python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py \
    --taxonomy basics/iprep/meeting-analytics/taxonomy.json --reset

# Load without resetting (truncates tables, keeps schema)
python basics/iprep/meeting-analytics/take_a/generate_rule_based_taxonomy.py
```

**Flags:**

| Flag | Default | Effect |
|------|---------|--------|
| `--dataset` | `../dataset` | Path to meeting folders |
| `--schema` | `meeting_analytics` | Postgres schema name |
| `--taxonomy` | `taxonomy.json` (if exists) | External taxonomy JSON path |
| `--reset` | false | Drop + recreate schema before loading |
| `--dry-run` | false | Print schema plan only, no Postgres connection |
