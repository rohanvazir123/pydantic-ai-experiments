# Transcript Intelligence — Task List

## Status Legend
- `[x]` Done
- `[-]` In progress
- `[ ]` Not started

---

## Phase 1: Data Ingestion

- [x] Parse dataset directory structure (`<meeting_id>/` with 6 JSON files each)
- [x] Load raw JSONs into PostgreSQL (`raw_files` table — payload as JSONB)
- [x] Flatten `meeting-info.json` → `meetings` table
- [x] Flatten `summary.json` → `summaries`, `summary_topics`, `action_items` tables
- [x] Flatten `transcript.json` → `transcript_lines` view (per-line with speaker, sentiment, timing)
- [x] Flatten `speakers.json` → `speaker_turns` view (turn-level speaker/timestamp)
- [x] Flatten `events.json` → `participant_events` view (join/leave events with timestamps)
- [x] Flatten `speaker-meta.json` → `speaker_meta` view (speaker_id → name mapping)

---

## Phase 2: Categorization Pipeline (Task 1)

- [ ] Decide approach: LLM-based / clustering / hybrid
- [ ] Build topic categorization pipeline over transcripts
- [ ] Store category labels per meeting (new table or column on `meetings`)
- [ ] Document category taxonomy + examples per category
- [ ] Evaluate category quality (spot-check, coverage)

---

## Phase 3: Sentiment Analysis (Task 2)

- [ ] Aggregate sentiment by call type (support / external / internal)
- [ ] Identify sentiment trends over time
- [ ] Surface outliers / anomalies worth calling out
- [ ] Write narrative: what the trends mean, who should care

---

## Phase 4: Bonus Insights (Task 3)

- [ ] Brainstorm 2–3 additional insight ideas with stakeholder mapping
- [ ] Decide which to implement vs. describe
- [ ] Implement chosen insights
- [ ] Write up the rest with reasoning

---

## Phase 5: Deliverables

- [ ] Slide deck (insights-first, 30-min presentation)
- [ ] Notebook / code repo (clean, commented)
- [ ] Video demo (5–10 min screen recording)

---

## Schema Reference

**PostgreSQL schema:** `iprep_meeting-analytics`

### Table (real)

**`raw_files`** — PRIMARY KEY `(meeting_id, file_type)`

| Column | Type | Nullable |
|---|---|---|
| meeting_id | text | NOT NULL |
| file_type | text | NOT NULL |
| payload | jsonb | NOT NULL |
| source_path | text | NOT NULL |
| loaded_at | timestamptz | NOT NULL DEFAULT now() |

Indexes: `raw_files_file_type_idx` B-tree `(file_type)`, `raw_files_payload_gin_idx` GIN `(payload)`

### Views (derived from `raw_files`)

**`meetings`** — source `file_type = 'meeting-info'`

| Column | Type |
|---|---|
| meeting_id | text |
| title | text |
| organizer_email | text |
| host | text |
| start_time | timestamptz |
| end_time | timestamptz |
| duration_minutes | numeric |
| all_emails | jsonb |
| invitees | jsonb |

**`summaries`** — source `file_type = 'summary'`

| Column | Type |
|---|---|
| meeting_id | text |
| summary | text |
| overall_sentiment | text |
| sentiment_score | numeric |
| topics | jsonb |
| action_items | jsonb |
| key_moments | jsonb |

**`summary_topics`** — source `file_type = 'summary'`

| Column | Type |
|---|---|
| meeting_id | text |
| topic | text |

**`action_items`** — source `file_type = 'summary'`

| Column | Type |
|---|---|
| meeting_id | text |
| action_index | integer |
| action_item | text |

**`transcript_lines`** — source `file_type = 'transcript'`

| Column | Type |
|---|---|
| meeting_id | text |
| line_index | integer |
| speaker_name | text |
| speaker_id | text |
| sentiment_type | text |
| start_seconds | numeric |
| end_seconds | numeric |
| confidence | numeric |
| sentence | text |

**`speaker_turns`** — source `file_type = 'speakers'`

| Column | Type |
|---|---|
| meeting_id | text |
| turn_index | integer |
| speaker_name | text |
| start_seconds | numeric |
| end_seconds | numeric |

**`participant_events`** — source `file_type = 'events'`

| Column | Type |
|---|---|
| meeting_id | text |
| event_index | integer |
| participant_name | text |
| event_type | text |
| seconds_from_start | numeric |
| event_time | timestamptz |

**`speaker_meta`** — source `file_type = 'speaker-meta'`

| Column | Type |
|---|---|
| meeting_id | text |
| speaker_id | integer |
| speaker_name | text |

### Source Files per Meeting
```
dataset/
  <meeting_id>/
    meeting-info.json   → iprep_meeting-analytics.meetings (view)
    transcript.json     → iprep_meeting-analytics.transcript_lines (view)
    summary.json        → iprep_meeting-analytics.summaries, summary_topics, action_items (views)
    speakers.json       → iprep_meeting-analytics.speaker_turns (view)
    events.json         → iprep_meeting-analytics.participant_events (view)
    speaker-meta.json   → iprep_meeting-analytics.speaker_meta (view)
```

---

## Local DB Topology

| Port | DB | Purpose |
|---|---|---|
| 5432 | postgres | iprep_meeting-analytics schema lives here (this project) |
| 5433 | legal_graph | Apache AGE knowledge graph (RAG project) |
| 5434 | rag_db | RAG pgvector docs/chunks/embeddings |

MCP PostgreSQL tool → `localhost:5432/postgres`

---

## Notes / Decisions Log

- Schema: `iprep_meeting-analytics` (all 6 tables live here, on port 5432)
- `speakers.json`, `events.json`, `speaker-meta.json` in `raw_files` only — no dedicated typed tables yet
