# Session Context — Meeting Analytics
Last updated: 2026-05-09 (session 10)

## How to reload this session
Tell Claude: "Read basics/iprep/meeting-analytics/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment — see `req.md` for the full brief.
100 meeting folders in `dataset/`. Three approaches to theme classification:
  - **Take A**: rule-based keyword taxonomy → DONE
  - **Take B**: TF-IDF + KMeans clustering → DONE
  - **Final Version**: semantic embedding + HDBSCAN + LLM labeling → DONE

---

## Postgres — single source of truth
**Connection:** `localhost:5434` / database `rag_db` / user `rag_user` / password `rag_pass`
**Schema:** `meeting_analytics`
**Credentials file:** `basics/iprep/meeting-analytics/.env`
**DBeaver:** connect to localhost:5434 (Docker container with pgvector)

> Port 5432 = local Windows Postgres — separate instance, not used for this project.
> Port 5433 = Apache AGE Docker (knowledge graph project — separate, unrelated).
> Port 5434 = Docker pgvector — canonical DB for this project. Use this for everything.

**WARNING:** MCP postgresql tool connects to port 5432, not 5434. Never use it to verify
schema changes — always use DBeaver or `psql` pointed at port 5434.

### Schema overview

All tables live in the `meeting_analytics` schema. `final_version` is **self-contained**:
`load_raw_jsons_to_db.py` creates 6 base tables from raw JSON and `load_output_csvs_to_db.py`
creates 3 semantic tables — no dependency on Take A or B. Take A and Take B optionally add
their own tables on top; the 6 base table names overlap but inserts are idempotent.

**Final Version only (9 tables + 1 view):** run `load_raw_jsons_to_db.py --reset` then `load_output_csvs_to_db.py --reset`.
**All takes:** also run `take_a/generate_rule_based_taxonomy.py` and `take_b/load_outputs_to_pg.py`.

---

### Tables — verified row counts

**Final Version base tables (6) — `final_version/load_raw_jsons_to_db.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `meetings` | 100 | meeting_id, title, organizer_email, duration_minutes, start_time |
| `meeting_participants` | 311 | meeting_id, email (unique allEmails pairs from meeting-info.json) |
| `meeting_summaries` | 100 | summary_text, overall_sentiment, sentiment_score, **topics TEXT[]** |
| `key_moments` | 402 | moment_type (8 types), text, speaker, time_seconds |
| `action_items` | 397 | meeting_id, owner, text |
| `transcript_lines` | 4313 | speaker, sentence, sentiment_type, time_seconds |

**Final Version semantic tables (3) + 1 view — `final_version/load_output_csvs_to_db.py`**

| Table / View | Rows | What it holds |
|--------------|------|---------------|
| `semantic_clusters` | 26 | theme_title (LLM), audience (LLM), rationale (LLM), phrase_count |
| `semantic_phrases` | 343 | canonical phrase, cluster_id, embedding=NULL (CSV path), tsvector GIN |
| `semantic_meeting_themes` | 516 | meeting_id, cluster_id, is_primary, call_type (LLM), sentiment |
| `action_items_by_theme` *(view)* | 397 | action_items JOIN semantic_meeting_themes (primary) JOIN semantic_clusters — action item text + owner + theme_title + audience |

26 clusters from 343 deduplicated topic phrases. 22 noise phrases (6.4%) reassigned to nearest centroid.

**Take A additional tables (4 + 1 view) — `take_a/generate_rule_based_taxonomy.py`**

> Take A also creates the 6 base tables (same names, same schema). The 4 below are Take A-only.

| Table | Rows | What it holds |
|-------|------|---------------|
| `summary_topics` | 600 | one row per meeting × topic tag |
| `meeting_themes` | 466 | theme, evidence_count, is_primary — rule-based, 8 themes |
| `call_types` | 100 | call_type (5 values), confidence — rule-based |
| `sentiment_features` | 100 | net_sentiment, positive/negative ratio, all 8 signal counts |
| `action_items_by_theme` *(view)* | — | action_items JOIN meeting_themes WHERE is_primary |

**Take B (3 tables) — `take_b/load_outputs_to_pg.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `kmeans_clusters` | 8 | cluster_id, label (top-4 centroid terms), meeting_count, silhouette_score |
| `kmeans_cluster_terms` | 96 | cluster_id, rank, term — top-12 centroid terms per cluster |
| `kmeans_meeting_clusters` | 100 | meeting_id → cluster_id (single hard assignment) |

Take B clusters (k=8, silhouette=0.0376):
```
0  renewal / competitive / pricing / outage          (18 meetings)
1  outage / incident / failure / communication       (26 meetings)
2  billing / overage / dispute / billing dispute     ( 6 meetings)
3  planning / sprint / launch / pdf                  ( 7 meetings)
4  hipaa / compliance / reporting / framework        (17 meetings)
5  backup / performance / response time / support    ( 8 meetings)
6  mfa / identity / management / sso                 (12 meetings)
7  backup / hybrid / recovery / onboarding           ( 6 meetings)
```

---

### Setup scripts

| Script | What it does |
|--------|-------------|
| `final_version/load_raw_jsons_to_db.py` | Standalone: raw JSON → 6 base tables. `--reset` drops entire schema first. |
| `final_version/load_output_csvs_to_db.py` | Standalone: outputs/ CSVs/JSON → 3 semantic tables. `--reset` drops semantic tables first. |
| `final_version/semantic_clustering.py` | Full pipeline (steps 0–9). `--from-outputs` skips embedding and loads from outputs/ instead. `--reset-db` wipes everything. |
| `final_version/verify.py` | Checks all 9 Final Version tables (+ skips Take A/B if absent). Sources `sql/01_verify_tables.sql`. |
| `take_a/generate_rule_based_taxonomy.py --reset` | Drop + recreate schema, load Take A tables + view. **WARNING: wipes all tables** including base and semantic. |
| `take_b/load_outputs_to_pg.py` | Load 3 Take B tables from outputs/ CSVs. |

### Docker — persistent volume
pgvector container (`rag_pgvector`) runs with named volume `pydantic-ai-experiments_pgvector_data`.
Always start via `docker compose up -d pgvector` from the repo root — not from Docker Desktop UI.
`docker compose down` preserves data. `docker compose down -v` wipes it.

---

## Call type — NOT being fixed
Two inconsistent taxonomies exist:
- **Take A**: 5 types (`support_escalation`, `sales_or_renewal`, `internal_incident`, `internal_planning`, `external_customer`)
- **Final Version**: 3 types (`support`, `external`, `internal`) — LLM-generated

Decision: not fixing this. For notebook queries, use Take A's `call_types` table (5 types) or collapse to 3 in Python before charting:
- `support_escalation` → `support`
- `sales_or_renewal`, `external_customer` → `external`
- `internal_incident`, `internal_planning` → `internal`

---

## Open design question (lower priority)

### LLM-generated cluster labels — scalability
`semantic_clusters.theme_title`, `audience`, `rationale` are generated by llama3.1:8b (26 calls, once per clustering run). Labels change between runs (stochastic). Centroid-proximity sort implemented — LLM receives the 20 phrases closest to each cluster centroid.

**Remaining open question:** keep LLM labels as one-time annotation vs replace with deterministic top-N phrase concat (Take B style). Decision pending — not blocking notebook work.

---

## Insight catalogue — 10 insights identified

All 10 SQL queries in `sql/02_insight_queries.sql`. All verified against rag_db @ localhost:5434.

| # | Question | Stakeholder | Key finding |
|---|----------|-------------|-------------|
| 1 | Theme volume and evidence strength | Leadership | Reliability touches 83 meetings; Compliance has highest evidence density |
| 2 | Theme × sentiment heatmap | Support, Engineering | Reliability is the only theme with majority-negative profile |
| 3 | Net sentiment by theme (transcript-grounded) | CX leadership | Reliability −0.29, Compliance +0.51 — largest gap in the dataset |
| 4 | Churn signal density by theme | Sales, CSMs | Reliability: 1.04 churn signals/meeting — higher than Customer Retention (0.71) |
| 5 | Call type distribution | Operations | 47% support escalation, 27% sales/renewal, 12% internal incident |
| 6 | Call type × theme matrix | Support + Sales leads | 12 of 47 support escalations are Reliability-primary — outages drive support volume |
| 7 | High-risk meeting watchlist | CSMs, AEs | Meetings with churn_signal ≥ 1 AND negative sentiment — named accounts, actionable |
| 8 | Reliability-to-commercial bleed | Revenue leadership | Outage meetings routinely span into renewal discussions |
| 9 | Feature gap prioritisation | Product managers | Reliability gaps raised under duress (net_sentiment −0.20) vs Compliance gaps constructively (+0.57) |
| 10 | Theme co-occurrence | Product + Engineering | "Compliance + Product Expansion" co-occur in 33 meetings |

---

## Stakeholder questions — DONE

16 additional questions written and verified in `sql/03_stakeholder_questions.sql`. All run clean against rag_db @ localhost:5434.

| ID | Question | Stakeholder |
|----|----------|-------------|
| S1 | Pricing signal concentration — which themes/call types pair with pricing_offer moments? | Sales |
| S2 | Repeat organizers in high-risk meetings — which contacts keep appearing? | Sales, CSMs |
| S3 | Churn signal text samples — verbatim quotes | Sales, CSMs |
| S4 | Action item volume by theme — which themes create the most follow-up? | Support |
| S5 | Meeting duration by theme / call type — are reliability meetings longer? | Support, Ops |
| S6 | Technical issue concentration by theme | Support, Engineering |
| S7 | Key moment type breakdown — distribution of all 8 types | All |
| E1 | Technical issue text samples grouped by theme | Engineering |
| E2 | Positive pivot signals — which themes recover mid-call? | Engineering |
| P1 | Feature gap text samples — verbatim customer asks | Product |
| P2 | Top summary topic tags — product area frequency | Product |
| P3 | Praise signal concentration — where are we winning? | Product |
| P4 | Final Version cluster size + dominant call type + avg sentiment | Product, Engineering |
| O1 | Participant count vs meeting outcome — larger meetings → worse sentiment? | Ops |
| O2 | Action item ownership — which owners are overloaded? | Ops |
| O3 | Take B vs Final Version cross-tab pivot for all 100 meetings | All |

---

## SQL scripts (DBeaver)
Connect DBeaver to: `localhost:5434` / `rag_db` / `rag_user` / `rag_pass`

| File | Purpose |
|------|---------|
| `sql/01_verify_tables.sql` | Row count checks + spot checks (Final Version + optional Take A/B) |
| `sql/02_insight_queries.sql` | 10 insight queries with stakeholder notes |
| `sql/03_stakeholder_questions.sql` | 16 stakeholder questions — all verified |

---

## Next steps

### Deliverables (from req.md — all three required):

**a) Jupyter notebook** — START HERE
- **Final Version only** — keep it simple and demo-able
- Connect to `rag_db @ localhost:5434` for all queries
- Kernel: `pydantic_ai_agents` conda env
- Sections:
  1. DB connection + schema overview
  2. Final Version clusters — theme titles, sizes, audience breakdown
  3. UMAP 2-dim scatter (`final_version/outputs/viz_coords.csv` — confirmed present)
  4. Key insight charts from `sql/02_insight_queries.sql` and `sql/03_stakeholder_questions.sql`
  5. High-risk meeting watchlist (I7) — most actionable output for leadership

**b) Slide deck** — 30-min presentation to product + engineering leadership
- Lead with insights, not code
- Headline: Reliability is the #1 churn driver (1.04 signals/meeting)

**c) Video demo** — 5-10 min screen recording with narration

---

## Key design decisions (don't revisit without good reason)
- Embed topic phrases (not full meetings) — finer cluster resolution; see `final_version/design.md` §3.1 for rationale tied to req.md
- HDBSCAN not KMeans — no fixed K, density-adaptive, found 26 naturally
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama (768 dims) — local, free, sufficient
- Final Version is fully self-contained — `load_raw_jsons_to_db.py` + `load_output_csvs_to_db.py` create all 9 tables; no dependency on Take A or B
- LLM is translation, not intelligence — embeddings + HDBSCAN cluster; LLM only names the result
- Human inspection required after every schema/data change — verify via `final_version/verify.py` or DBeaver @ port 5434, not via MCP tool

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── req.md / req.pdf             source of truth for deliverables
├── .env                         PG credentials (rag_db @ localhost:5434)
├── notes.txt                    running project notes
├── dataset/                     100 meeting folders, each with meeting-info.json,
│                                summary.json, transcript.json, events.json,
│                                speaker-meta.json, speakers.json
├── sql/
│   ├── 01_verify_tables.sql     Row count + spot checks
│   ├── 02_insight_queries.sql   10 insight queries with stakeholder notes
│   └── 03_stakeholder_questions.sql  16 stakeholder questions — all verified
├── take_a/
│   ├── generate_rule_based_taxonomy.py   main Take A script (--reset to rebuild)
│   ├── load_dataset_to_postgres.py       raw JSON → Postgres loader
│   ├── take_a_design.md
│   └── take_a_faq.md
├── take_b/
│   ├── cluster_taxonomy_v2.py            TF-IDF + KMeans pipeline
│   ├── load_outputs_to_pg.py             Load Take B outputs → 3 Postgres tables
│   ├── take_b_design.md
│   ├── take_b_faq.md
│   └── outputs/                          meeting_clusters.csv, cluster_summary.json,
│                                         cluster_terms.csv, cluster_metrics.json,
│                                         cluster_scores.csv, take_b_run.log
└── final_version/
    ├── semantic_clustering.py            full pipeline (steps 0–9); --from-outputs to skip embedding
    ├── load_raw_jsons_to_db.py           standalone: raw JSON → 6 base Postgres tables
    ├── load_output_csvs_to_db.py         standalone: outputs/ CSVs/JSON → 3 semantic tables + action_items_by_theme view + SemanticClusterStore
    ├── verify.py                         standalone: checks all 9 tables; sources sql/01_verify_tables.sql
    ├── design.md                         design doc v0.5
    ├── faq.md                            FAQ
    └── outputs/                          meeting_themes.csv, phrase_clusters.csv,
                                          semantic_clusters.json, cluster_metrics.json,
                                          viz_coords.csv, run.log
```

---

## Environment
- Python 3.13, conda env: `pydantic_ai_agents` (already active — no `conda run` needed)
- Ollama: `http://localhost:11434/v1`
- Embedding model: `nomic-embed-text:latest` (768 dims)
- LLM model: `llama3.1:8b`
- Postgres: `meeting_analytics` schema in `rag_db` Docker container (same DB as RAG project)
- Deps installed: `rapidfuzz`, `umap-learn`, `hdbscan`, `scikit-learn`, `pandas`, `asyncpg`, `pgvector`
