# Session Context — Meeting Analytics
Last updated: 2026-05-09 (session 8)

## How to reload this session
Tell Claude: "Read basics/iprep/meeting-analytics/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment — see `req.md` for the full brief.
100 meeting folders in `dataset/`. Three approaches to theme classification:
  - **Take A**: rule-based keyword taxonomy → DONE
  - **Take B**: TF-IDF + KMeans clustering → DONE
  - **Take C**: semantic embedding + HDBSCAN + LLM labeling → DONE

---

## Postgres — single source of truth
**Connection:** `localhost:5434` / database `rag_db` / user `rag_user` / password `rag_pass`
**Schema:** `meeting_analytics`
**Credentials file:** `basics/iprep/meeting-analytics/.env`
**DBeaver:** connect to localhost:5434 (Docker container with pgvector)

> Port 5432 = local Windows Postgres (postgres:postgres) — has only the 10 Take A tables, no pgvector.
> Port 5433 = Apache AGE Docker (knowledge graph project — separate, unrelated).
> Port 5434 = Docker pgvector — canonical DB for this project. Use this for everything.

**WARNING:** MCP postgresql tool connects to port 5432, not 5434. Never use it to verify
schema changes — always use DBeaver or `psql` pointed at port 5434.

### All 16 tables + 1 view — verified row counts

**Take A (10 tables) — from raw dataset JSON via `generate_rule_based_taxonomy.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `meetings` | 100 | meeting_id, title, organizer_email, duration_minutes |
| `meeting_participants` | 622 | meeting_id, email, participant_role |
| `meeting_summaries` | 100 | summary text, overall_sentiment, sentiment_score |
| `summary_topics` | 600 | one row per meeting × topic tag |
| `action_items` | 397 | owner, action_item text |
| `key_moments` | 402 | moment_type (8 types: churn_signal / concern / technical_issue / feature_gap / praise / pricing_offer / positive_pivot / action_item), text |
| `transcript_lines` | 4313 | speaker, sentiment_type, sentence |
| `meeting_themes` | 466 | theme, evidence_count, is_primary — rule-based, 8 themes |
| `call_types` | 100 | call_type (5 values), confidence — rule-based |
| `sentiment_features` | 100 | net_sentiment, positive/negative ratio, all 8 signal counts (incl. positive_pivot_count, pricing_offer_count added session 8) |

**Take A view (auto-created by setup)**

| View | What it holds |
|------|---------------|
| `action_items_by_theme` | action_items JOIN meeting_themes WHERE is_primary — theme-level action item breakdown |

**Take B (3 tables) — from `take_b/outputs/` CSVs via `take_b/load_outputs_to_pg.py`**

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

**Take C (3 tables) — from `take_c/outputs/` via `take_c/load_outputs_to_pg.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `semantic_clusters` | 26 | cluster_id, theme_title (LLM), audience (LLM), rationale (LLM), phrase_count |
| `semantic_phrases` | 343 | canonical phrase, cluster_id, embedding=NULL (CSV path), tsvector GIN |
| `semantic_meeting_themes` | 516 | meeting_id, cluster_id, is_primary, call_type (LLM), sentiment |

Take C: 26 clusters from 343 deduplicated topic phrases. 22 noise phrases (6.4%) reassigned to nearest centroid.

### Setup scripts

| Script | What it does |
|--------|-------------|
| `setup_all_tables.py` | One-shot: Take A --reset + Take B loader + Take C loader → 16 tables + view. **Always use this.** |
| `take_a/generate_rule_based_taxonomy.py --reset` | Drop + recreate schema, load 10 Take A tables + view. WARNING: wipes Take B and C tables too. |
| `take_b/load_outputs_to_pg.py` | Load 3 Take B tables from outputs/ CSVs — no re-clustering needed |
| `take_c/load_outputs_to_pg.py` | Load 3 Take C tables from outputs/ JSON+CSVs — no re-embedding needed |
| `export_all_to_csv.py` | Snapshot all 16 tables to `outputs/csv/` — run after any pipeline change |

### Docker — persistent volume
pgvector container (`rag_pgvector`) now runs with a named volume `pydantic-ai-experiments_pgvector_data`.
Always start via `docker compose up -d pgvector` from the repo root — not from Docker Desktop UI.
`docker compose down` preserves data. `docker compose down -v` wipes it.

---

## Call type — NOT being fixed
Two inconsistent taxonomies exist:
- **Take A**: 5 types (`support_escalation`, `sales_or_renewal`, `internal_incident`, `internal_planning`, `external_customer`)
- **Take C**: 3 types (`support`, `external`, `internal`) — LLM-generated

Decision: not fixing this. For notebook queries, use Take A's `call_types` table (5 types) or collapse to 3 in Python before charting:
- `support_escalation` → `support`
- `sales_or_renewal`, `external_customer` → `external`
- `internal_incident`, `internal_planning` → `internal`

---

## Open design question (lower priority)

### LLM-generated cluster labels in Take C — scalability
`semantic_clusters.theme_title`, `audience`, `rationale` are generated by llama3.1:8b (26 calls, once per clustering run). Labels change between runs (stochastic). LLM samples first 20 phrases in cluster order (not centroid-proximity sorted — not implemented).

**Options:**
- Replace with deterministic top-N phrase labels (same approach as Take B centroid terms)
- Keep as one-time annotation — accept that re-clustering requires re-labeling
- Decision pending — not blocking notebook work

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
| 10 | Take C theme co-occurrence | Product + Engineering | "Compliance + Product Expansion" co-occur in 33 meetings |

---

## Stakeholder questions — DONE

16 additional questions written and verified in `sql/03_stakeholder_questions.sql`. All run clean against rag_db @ localhost:5434.

| ID | Question | Stakeholder |
|----|----------|-------------|
| S1 | Pricing signal concentration — which themes/call types pair with pricing_offer moments? | Sales |
| S2 | Repeat organizers in high-risk meetings — which contacts keep appearing? | Sales, CSMs |
| S3 | Churn signal text samples — verbatim quotes | Sales, CSMs |
| S4 | Action item volume by theme — which themes create the most follow-up? | Support (also: `action_items_by_theme` view) |
| S5 | Meeting duration by theme / call type — are reliability meetings longer? | Support, Ops |
| S6 | Technical issue concentration by theme | Support, Engineering |
| S7 | Key moment type breakdown — distribution of all 8 types | All |
| E1 | Technical issue text samples grouped by theme | Engineering |
| E2 | Positive pivot signals — which themes recover mid-call? | Engineering |
| P1 | Feature gap text samples — verbatim customer asks | Product |
| P2 | Top summary topic tags — product area frequency | Product |
| P3 | Praise signal concentration — where are we winning? | Product |
| P4 | Take C cluster size + dominant call type + avg sentiment | Product, Engineering |
| O1 | Participant count vs meeting outcome — larger meetings → worse sentiment? | Ops |
| O2 | Action item ownership — which owners are overloaded? | Ops |
| O3 | Take B vs Take C cross-tab pivot for all 100 meetings | All |

---

## SQL scripts (DBeaver)
Connect DBeaver to: `localhost:5434` / `rag_db` / `rag_user` / `rag_pass`

| File | Purpose |
|------|---------|
| `sql/01_verify_tables.sql` | Row count checks + spot checks for all 16 tables |
| `sql/02_insight_queries.sql` | All 10 insight queries with comments |
| `sql/03_stakeholder_questions.sql` | 16 stakeholder questions — all verified |

---

## Take B vs Take C cross-validation — COMPLETE
Script: `compare_b_vs_c.py` (run from repo root)
All 100 meetings matched. Agreement proxy 37% — reflects resolution difference (8 B clusters vs 26 C clusters), not disagreement.

Key findings:
- B-2 billing and B-5 backup confirmed as real cohesive topics by both methods
- B-1 outage/incident: Take C correctly splits into Outage Prevention (C-09) vs Incident Response (C-08)
- B-4 compliance: Take C separates HIPAA, Audit Readiness, and Compliance Governance
- B-3 planning: Take C splits product launch from engineering sprint meetings

---

## Next steps

### Deliverables (from req.md — all three required):

**a) Jupyter notebook** — START HERE
- Connect to `rag_db @ localhost:5434` for all queries
- Cover: Take A (rule-based), Take B (TF-IDF/KMeans), Take C (semantic clustering), insight queries
- Turn the 10 insight queries + key stakeholder questions into charts (heatmap, bar charts, scatter)
- Include UMAP 2-dim scatter (`take_c/outputs/viz_coords.csv` — confirmed present)
- Include B vs C cross-tab comparison (O3)
- Kernel: `pydantic_ai_agents` conda env

**b) Slide deck** — 30-min presentation to product + engineering leadership
- Lead with insights, not code
- Headline: Reliability is the #1 churn driver (1.04 signals/meeting)
- B vs C cross-validation confirms themes are real signal, not artefacts

**c) Video demo** — 5-10 min screen recording with narration

---

## Key design decisions (don't revisit without good reason)
- Embed topic phrases (not full meetings) in Take C — finer cluster resolution
- HDBSCAN not KMeans in Take C — no fixed K, density-adaptive, found 26 naturally
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama (768 dims) — local, free, sufficient
- Extend Take A's schema — don't create duplicate tables
- CSV outputs exist for all three takes — always run `export_all_to_csv.py` after any pipeline change
- LLM in Take C is translation, not intelligence — embeddings + HDBSCAN do the clustering; LLM just names the result
- Human inspection required after every schema/data change — verify in DBeaver @ port 5434, not via MCP tool

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── req.md / req.pdf           source of truth for deliverables
├── compare_b_vs_c.py          Take B vs Take C cross-validation script
├── setup_all_tables.py        One-shot: reload all 16 tables + view
├── export_all_to_csv.py       Snapshot all 16 Postgres tables to outputs/csv/
├── .env                       PG credentials (rag_db @ localhost:5434)
├── notes.txt                  running project notes
├── dataset/                   100 meeting folders, each with meeting-info.json,
│                              summary.json, transcript.json, events.json,
│                              speaker-meta.json, speakers.json
├── sql/
│   ├── 01_verify_tables.sql   Row count + spot checks for all 16 tables
│   ├── 02_insight_queries.sql 10 insight queries with stakeholder notes
│   └── 03_stakeholder_questions.sql  16 stakeholder questions — all verified
├── take_a/
│   ├── generate_rule_based_taxonomy.py   main Take A script (--reset to rebuild)
│   ├── load_dataset_to_postgres.py       raw JSON → Postgres loader (used by main script)
│   ├── take_a_design.md                  design doc v0.1
│   └── take_a_faq.md                     FAQ — 15 entries
├── take_b/
│   ├── cluster_taxonomy_v2.py            TF-IDF + KMeans pipeline ([1/7]–[7/7] labelled)
│   ├── load_outputs_to_pg.py             Load Take B outputs → 3 Postgres tables
│   ├── take_b_design.md                  design doc v0.1
│   ├── take_b_faq.md                     FAQ — 9 entries
│   └── outputs/                          meeting_clusters.csv, cluster_summary.json,
│                                         cluster_terms.csv, cluster_metrics.json,
│                                         cluster_scores.csv, take_b_run.log
├── take_c/
│   ├── take_c_semantic_clustering.py     main Take C pipeline (9 steps)
│   ├── take_c_pg_store.py                Postgres store (SemanticClusterStore)
│   ├── load_outputs_to_pg.py             Load Take C outputs → 3 Postgres tables
│   ├── take_c_design.md                  design doc v0.4
│   ├── take_c_faq.md                     FAQ — 16 entries
│   └── outputs/                          meeting_themes.csv, phrase_clusters.csv,
│                                         semantic_clusters.json, cluster_metrics.json,
│                                         viz_coords.csv, take_c_run.log
└── outputs/
    └── csv/                              Flat-file backup of all 16 Postgres tables
```

---

## Environment
- Python 3.13, conda env: `pydantic_ai_agents` (already active — no `conda run` needed)
- Ollama: `http://localhost:11434/v1`
- Embedding model: `nomic-embed-text:latest` (768 dims)
- LLM model: `llama3.1:8b`
- Postgres: `meeting_analytics` schema in `rag_db` Docker container (same DB as RAG project)
- Deps installed: `rapidfuzz`, `umap-learn`, `hdbscan`, `scikit-learn`, `pandas`, `asyncpg`, `pgvector`
