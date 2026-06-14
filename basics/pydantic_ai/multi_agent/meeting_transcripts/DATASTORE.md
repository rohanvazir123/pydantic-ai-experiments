# Datastore Reference

Complete reference for all data stores used by the meeting transcript pipeline:
PostgreSQL schema, file-based stores, and checkpoint layout.

## Table of Contents

1. [Overview](#1-overview)
2. [Entity Diagram](#2-entity-diagram)
3. [PostgreSQL Database](#3-postgresql-database)
   - 3.1 [Connection](#31-connection)
   - 3.2 [Schema: `public`](#32-schema-public)
   - 3.3 [Table: `meetings`](#33-table-meetings)
   - 3.4 [Table: `meeting_insights`](#34-table-meeting_insights)
   - 3.5 [Table: `action_items`](#35-table-action_items)
   - 3.6 [Indexes](#36-indexes)
   - 3.7 [Constraints and Check Rules](#37-constraints-and-check-rules)
   - 3.8 [Full DDL](#38-full-ddl)
4. [File-Based Stores](#4-file-based-stores)
   - 4.1 [Pipeline Checkpoints](#41-pipeline-checkpoints)
   - 4.2 [Audit Log](#42-audit-log)
   - 4.3 [History (Memory)](#43-history-memory)
5. [Dataset Layout](#5-dataset-layout)
6. [Key SQL Queries](#6-key-sql-queries)

---

## 1. Overview

| Store | Type | Purpose | Managed by |
|-------|------|---------|-----------|
| PostgreSQL `meetings` | RDBMS table | Core meeting record | `ingestion.py` |
| PostgreSQL `meeting_insights` | RDBMS table | Sentiment shifts, pain points, competitor mentions | `ingestion.py` |
| PostgreSQL `action_items` | RDBMS table | Validated action items per meeting | `ingestion.py` |
| `.pipeline_checkpoints/` | Local filesystem | Stage-level JSON checkpoints (resume on failure) | `pipeline.py` |
| `audit.jsonl` | JSONL file | Per-stage audit trail with latency and token counts | `pipeline.py` |
| `~/.meeting_pipeline/history.json` | JSON file | Cross-session record of processed meetings | `pipeline.py` |
| `dataset/<id>/` | Read-only directory | Raw meeting JSON from the transcript provider | input only |

---

## 2. Entity Diagram

```
┌──────────────────────────────────┐
│            meetings              │
├──────────────────────────────────┤
│ id           VARCHAR(64)  PK     │
│ title        TEXT         NN     │
│ processed_at TIMESTAMPTZ  NN     │
│ participants TEXT[]        NN     │
│ meeting_date TIMESTAMPTZ  NULL   │
└──────────────┬───────────────────┘
               │ 1
               │ has many
               │ N
       ┌───────┴────────────────────┐
       │                            │
       ▼                            ▼
┌──────────────────────────────┐  ┌────────────────────────────────┐
│       meeting_insights       │  │          action_items          │
├──────────────────────────────┤  ├────────────────────────────────┤
│ id          SERIAL     PK    │  │ id          SERIAL      PK     │
│ meeting_id  VARCHAR(64) FK──►│  │ meeting_id  VARCHAR(64) FK──►  │
│ insight_type TEXT      NN    │  │ owner       TEXT        NN     │
│   CHECK IN ('sentiment_shift'│  │ action      TEXT        NN     │
│            'pain_point'      │  │ deadline    TEXT        NULL   │
│            'competitor')     │  │ verdict     TEXT        NN     │
│ speaker     TEXT       NULL  │  │   CHECK IN ('valid','invalid') │
│ content     TEXT       NN    │  │ reason      TEXT        NULL   │
│ polarity    TEXT       NULL  │  │ created_at  TIMESTAMPTZ NN     │
│ created_at  TIMESTAMPTZ NN   │  └────────────────────────────────┘
└──────────────────────────────┘

  insight_type values:
    sentiment_shift  → speaker + content + polarity populated
    pain_point       → only content populated
    competitor       → only content populated

  verdict values:
    valid   → action item was clearly agreed upon
    invalid → item was rejected, withdrawn, or hypothetical
```

---

## 3. PostgreSQL Database

### 3.1 Connection

| Parameter | Value |
|-----------|-------|
| Driver | `asyncpg` |
| DSN env var | `DATABASE_URL` |
| DSN format | `postgresql://user:pass@host:port/dbname` |
| SSL | Controlled by DSN `?sslmode=require` |
| Schema | `public` (default) |

The `DATABASE_URL` is also read by the existing RAG system in `rag/config/settings.py`.
The meeting pipeline tables live in the same database and schema.

### 3.2 Schema: `public`

All tables use the PostgreSQL default `public` schema. No custom schema is created.

### 3.3 Table: `meetings`

Primary record for each processed meeting.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `VARCHAR(64)` | NOT NULL | — | Meeting ID from the dataset (e.g. `01KQ03B0303900521BB089CA`) |
| `title` | `TEXT` | NOT NULL | — | Meeting title as extracted by the PreProcessing agent |
| `processed_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | UTC timestamp when the pipeline completed |
| `participants` | `TEXT[]` | NOT NULL | `'{}'` | Array of speaker names extracted by the PreProcessing agent |
| `meeting_date` | `TIMESTAMPTZ` | NULL | — | Actual meeting start time (from `meeting-info.json`), if available |

**Primary Key**: `id`

**Upsert behaviour**: `ingestion.py` uses `INSERT ... ON CONFLICT (id) DO UPDATE` so
re-running the pipeline on the same meeting ID overwrites all fields cleanly.

### 3.4 Table: `meeting_insights`

One row per extracted insight item (sentiment shift, pain point, or competitor mention).

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `SERIAL` | NOT NULL | auto | Surrogate PK |
| `meeting_id` | `VARCHAR(64)` | NOT NULL | — | FK → `meetings.id` |
| `insight_type` | `TEXT` | NOT NULL | — | `'sentiment_shift'`, `'pain_point'`, or `'competitor'` |
| `speaker` | `TEXT` | NULL | — | Speaker name (only for `sentiment_shift`) |
| `content` | `TEXT` | NOT NULL | — | The extracted insight text |
| `polarity` | `TEXT` | NULL | — | `'positive'`, `'negative'`, `'neutral'`, or `'mixed'` (only for `sentiment_shift`) |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Row insertion timestamp |

**Re-ingestion**: Before re-inserting, `ingestion.py` deletes all existing rows for the
`meeting_id` in a single transaction with the upsert, so the table always reflects the
latest pipeline run.

### 3.5 Table: `action_items`

One row per validated action item extracted from a meeting.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `SERIAL` | NOT NULL | auto | Surrogate PK |
| `meeting_id` | `VARCHAR(64)` | NOT NULL | — | FK → `meetings.id` |
| `owner` | `TEXT` | NOT NULL | — | The person responsible for the action |
| `action` | `TEXT` | NOT NULL | — | Verb-centric description of the task |
| `deadline` | `TEXT` | NULL | — | Natural-language deadline or `'Unspecified'` |
| `verdict` | `TEXT` | NOT NULL | — | `'valid'` or `'invalid'` (enforced by CHECK) |
| `reason` | `TEXT` | NULL | — | One-sentence justification from the Validation agent |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Row insertion timestamp |

**Note**: Both `valid` and `invalid` items are stored. Query with `WHERE verdict = 'valid'`
to retrieve agreed-upon actions.

### 3.6 Indexes

| Index name | Table | Column(s) | Type | Purpose |
|-----------|-------|-----------|------|---------|
| `meetings_pkey` | `meetings` | `id` | B-tree (PK) | Lookups by meeting ID |
| `ix_mi_meeting` | `meeting_insights` | `meeting_id` | B-tree | All insights for a meeting |
| `ix_mi_type` | `meeting_insights` | `insight_type` | B-tree | Filter by insight type |
| `ix_ai_meeting` | `action_items` | `meeting_id` | B-tree | All items for a meeting |
| `ix_ai_owner` | `action_items` | `owner` | B-tree | Items by owner (Q2, Q14) |
| `ix_ai_verdict` | `action_items` | `verdict` | B-tree | Filter valid/invalid items |

### 3.7 Constraints and Check Rules

| Table | Constraint | Expression |
|-------|-----------|-----------|
| `meetings` | PRIMARY KEY | `id` |
| `meeting_insights` | FOREIGN KEY | `meeting_id → meetings(id) ON DELETE CASCADE` |
| `meeting_insights` | CHECK | `insight_type IN ('sentiment_shift', 'pain_point', 'competitor')` |
| `action_items` | FOREIGN KEY | `meeting_id → meetings(id) ON DELETE CASCADE` |
| `action_items` | CHECK | `verdict IN ('valid', 'invalid')` |

### 3.8 Full DDL

```sql
-- Run this once to initialise the schema (also in ingestion.py:init_schema)

CREATE TABLE IF NOT EXISTS meetings (
    id             VARCHAR(64)  PRIMARY KEY,
    title          TEXT         NOT NULL,
    processed_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    participants   TEXT[]       NOT NULL DEFAULT '{}',
    meeting_date   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS meeting_insights (
    id           SERIAL       PRIMARY KEY,
    meeting_id   VARCHAR(64)  NOT NULL
                              REFERENCES meetings(id) ON DELETE CASCADE,
    insight_type TEXT         NOT NULL
                              CHECK (insight_type IN (
                                  'sentiment_shift', 'pain_point', 'competitor'
                              )),
    speaker      TEXT,
    content      TEXT         NOT NULL,
    polarity     TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_mi_meeting ON meeting_insights (meeting_id);
CREATE INDEX IF NOT EXISTS ix_mi_type    ON meeting_insights (insight_type);

CREATE TABLE IF NOT EXISTS action_items (
    id          SERIAL       PRIMARY KEY,
    meeting_id  VARCHAR(64)  NOT NULL
                             REFERENCES meetings(id) ON DELETE CASCADE,
    owner       TEXT         NOT NULL,
    action      TEXT         NOT NULL,
    deadline    TEXT,
    verdict     TEXT         NOT NULL CHECK (verdict IN ('valid', 'invalid')),
    reason      TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_ai_meeting ON action_items (meeting_id);
CREATE INDEX IF NOT EXISTS ix_ai_owner   ON action_items (owner);
CREATE INDEX IF NOT EXISTS ix_ai_verdict ON action_items (verdict);
```

---

## 4. File-Based Stores

### 4.1 Pipeline Checkpoints

**Location**: `.pipeline_checkpoints/<meeting_id>/<stage>.json`

**Purpose**: Allow resuming a failed pipeline run without re-running completed stages.

**Layout**:
```
.pipeline_checkpoints/
└── 01KQ03B0303900521BB089CA/
    ├── preprocessing.json   ← CleanTranscript (meeting_title, participants, turns[])
    ├── extraction.json      ← Insight (sentiment_shifts[], pain_points[], competitor_mentions[])
    ├── commitments.json     ← CommitmentsOutput (action_items[])
    ├── validation.json      ← ValidationResult (items[])
    └── audit.jsonl          ← one JSON line per stage execution (see §4.2)
```

**Format**: Each `.json` file is the Pydantic model serialised with `.model_dump_json(indent=2)`.

**Lifecycle**:
- Written after each successful stage (`save_checkpoint`)
- Read at stage start; if found, stage is skipped (`load_checkpoint`)
- Cleared by `--force` flag (pipeline ignores existing checkpoints)
- Coroutines are `.close()`d when a checkpoint is restored to prevent `ResourceWarning`

### 4.2 Audit Log

**Location**: `.pipeline_checkpoints/<meeting_id>/audit.jsonl`

**Purpose**: Immutable per-meeting audit trail. One JSON line per stage execution.

**Schema** (one line = one `AuditEntry`):

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | 8-char hex correlation ID for this pipeline run |
| `stage` | `str` | `preprocessing`, `extraction`, `commitments`, `validation` |
| `meeting_id` | `str` | Meeting ID |
| `started_at` | `ISO-8601` | Stage start timestamp (UTC) |
| `completed_at` | `ISO-8601` | Stage end timestamp (UTC) |
| `duration_s` | `float` | Wall-clock seconds |
| `input_tokens` | `int?` | LLM input tokens consumed (null if skipped) |
| `output_tokens` | `int?` | LLM output tokens produced (null if skipped) |
| `status` | `enum` | `"success"`, `"error"`, or `"skipped"` |
| `error` | `str?` | Error message if `status == "error"` |

**Example line**:
```json
{"run_id":"603a50fd","stage":"preprocessing","meeting_id":"01KQ03B0303900521BB089CA","started_at":"2026-06-06T00:08:03Z","completed_at":"2026-06-06T00:09:51Z","duration_s":108.1,"input_tokens":2290,"output_tokens":1956,"status":"success","error":null}
```

### 4.3 History (Memory)

**Location**: `~/.meeting_pipeline/history.json`

**Purpose**: Cross-session record of every meeting that has been successfully processed.
Used by `watcher.py` to skip already-ingested meetings.

**Schema**:
```json
{
  "01KQ03B0303900521BB089CA": {
    "title": "Detect Outage - Remediation Plan Review",
    "processed_at": "2026-06-06T01:41:29.090030+00:00",
    "valid_action_items": 2
  },
  "01KQ0C1280EDA4E70AAD7C35": {
    "title": "Support Case #9279 - Summit Trust Billing Inquiry",
    "processed_at": "2026-06-06T01:07:24.090030+00:00",
    "valid_action_items": 4
  }
}
```

---

## 5. Dataset Layout

Raw input data — **read-only**, never written by the pipeline.

```
dataset/
└── <meeting_id>/          ← one directory per meeting
    ├── meeting-info.json  ← MeetingInfo: id, title, organizer, start/end times
    ├── transcript.json    ← {"data": [TranscriptEntry, ...]}
    ├── speakers.json      ← speaker metadata with timestamps
    ├── speaker-meta.json  ← speaker names and roles
    ├── events.json        ← meeting events timeline
    └── summary.json       ← ground-truth summary + action items (evaluation gold)
```

### `meeting-info.json` fields used by pipeline

| Field | Maps to | Notes |
|-------|---------|-------|
| `meetingId` | `MeetingInfo.meeting_id` | Primary key for all stores |
| `title` | `MeetingInfo.title` | Passed to preprocessing agent |
| `startTime` | `MeetingInfo.start_time` | Stored as `meetings.meeting_date` |

### `transcript.json` fields used by pipeline

| Field | Maps to | Notes |
|-------|---------|-------|
| `data[].sentence` | `TranscriptEntry.sentence` | The spoken text |
| `data[].speaker_name` | `TranscriptEntry.speaker_name` | Already resolved to full name |
| `data[].time` | `TranscriptEntry.time` | Seconds from meeting start |
| `data[].endTime` | `TranscriptEntry.end_time` | Alias field |
| `data[].sentimentType` | `TranscriptEntry.sentiment_type` | Provider-supplied, not used by agents |

---

## 6. Key SQL Queries

### All valid action items with deadlines (Q2 — Head of Engineering)
```sql
SELECT ai.owner, ai.action, ai.deadline, m.title
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND ai.deadline <> 'Unspecified'
ORDER BY ai.owner, ai.deadline;
```

### Hallucination rate per meeting (Q5)
```sql
SELECT m.title,
       COUNT(*) FILTER (WHERE ai.verdict = 'valid')   AS valid_items,
       COUNT(*) FILTER (WHERE ai.verdict = 'invalid') AS invalid_items,
       ROUND(
           COUNT(*) FILTER (WHERE ai.verdict = 'invalid')::numeric /
           NULLIF(COUNT(*), 0) * 100, 1
       ) AS hallucination_pct
FROM meetings m
JOIN action_items ai ON ai.meeting_id = m.id
GROUP BY m.id, m.title
ORDER BY hallucination_pct DESC;
```

### Sentiment polarity breakdown by speaker (Q10)
```sql
SELECT mi.speaker,
       COUNT(*) FILTER (WHERE mi.polarity = 'negative') AS negative,
       COUNT(*) FILTER (WHERE mi.polarity = 'mixed')    AS mixed,
       COUNT(*) FILTER (WHERE mi.polarity = 'positive') AS positive,
       COUNT(*) FILTER (WHERE mi.polarity = 'neutral')  AS neutral
FROM meeting_insights mi
WHERE mi.insight_type = 'sentiment_shift'
  AND mi.speaker IS NOT NULL
GROUP BY mi.speaker
ORDER BY negative DESC, mixed DESC;
```

### Customer-facing commitments (Q11 — Customer Support)
```sql
SELECT ai.owner, ai.action, ai.deadline, m.title
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND (ai.action ILIKE '%customer%' OR ai.action ILIKE '%communication%'
       OR ai.action ILIKE '%email%' OR ai.action ILIKE '%notify%')
ORDER BY m.meeting_date DESC NULLS LAST;
```

### Meetings with no valid action items (Q15)
```sql
SELECT m.id, m.title, m.meeting_date
FROM meetings m
WHERE NOT EXISTS (
    SELECT 1 FROM action_items ai
    WHERE ai.meeting_id = m.id AND ai.verdict = 'valid'
)
ORDER BY m.meeting_date DESC NULLS LAST;
```
