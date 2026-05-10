# Transcript Intelligence — Task List

## Status Legend
- `[x]` Done
- `[-]` In progress
- `[ ]` Not started

---

## Phase 1: Data Ingestion

- [x] Parse dataset directory structure (`<meeting_id>/` with 6 JSON files each)
- [x] Flatten all JSON files → 6 real Postgres tables (not views) in `meeting_analytics` schema
  - `meetings` (100 rows)
  - `meeting_participants` (311 rows)
  - `meeting_summaries` (100 rows) — includes `products TEXT[]` column (Comply/Detect/Protect/Identity)
  - `key_moments` (402 rows) — 8 moment types
  - `action_items` (397 rows)
  - `transcript_lines` (4313 rows)
- [x] Script: `final_version/load_raw_jsons_to_db.py` (standalone, `--reset` flag)

---

## Phase 2: Categorization Pipeline (Task 1)

All three approaches implemented and loaded into Postgres.

### Take A — Rule-based keyword taxonomy
- [x] Keyword taxonomy → 8 themes, 5 call types
- [x] Tables: `meeting_themes`, `call_types`, `sentiment_features`, `summary_topics`, `action_items_by_theme` view
- [x] Script: `take_a/generate_rule_based_taxonomy.py`

### Take B — TF-IDF + KMeans clustering
- [x] TF-IDF vectorization over topic phrases, KMeans k=8 (silhouette=0.0376)
- [x] Tables: `kmeans_clusters`, `kmeans_cluster_terms`, `kmeans_meeting_clusters`
- [x] Script: `take_b/cluster_taxonomy_v2.py` + `take_b/load_outputs_to_pg.py`

### Final Version — Semantic embedding + HDBSCAN + LLM labeling
- [x] 343 deduplicated topic phrases embedded (nomic-embed-text 768 dims via Ollama)
- [x] UMAP 10-dim reduction → HDBSCAN → 26 clusters (6.4% noise reassigned to nearest centroid)
- [x] LLM labeling: `llama3.1:8b` → `theme_title`, `audience`, `rationale` per cluster
- [x] Call type inference: LLM per meeting → `support` / `external` / `internal`
- [x] Tables: `semantic_clusters` (26), `semantic_phrases` (343), `semantic_meeting_themes` (516)
- [x] View: `action_items_by_theme` (action items joined to primary semantic theme)
- [x] Script: `final_version/semantic_clustering.py` + `final_version/load_output_csvs_to_db.py`

---

## Phase 3: Sentiment Analysis (Task 2)

- [x] Transcript-grounded net sentiment per meeting (`sentiment_features` table via Take A)
- [x] Sentiment by theme — heatmap query (I2), net sentiment by theme (I3)
- [x] Churn signal density by theme (I4) — Reliability 1.04/meeting vs Customer Retention 0.71
- [x] High-risk meeting watchlist (I7) — churn ≥ 1 AND negative sentiment
- [x] Sentiment by call type available via Take A `call_types` + `sentiment_features`

---

## Phase 4: Bonus Insights (Task 3)

- [x] 10 core insight queries — `sql/02_insight_queries.sql` (all verified @ localhost:5434)
- [x] 16 stakeholder questions — `sql/03_stakeholder_questions.sql` (all verified @ localhost:5434)
- [x] NL question index — `nl_questions.md`
- [x] Insight catalogue + 9-slide deck structure — `INSIGHTS_GUIDE.md`

Key findings:
- Reliability is the #1 churn driver (1.04 churn signals/meeting)
- Reliability is the only theme with majority-negative sentiment profile
- 12 of 47 support escalations are Reliability-primary — outages drive support volume
- "Compliance + Product Expansion" co-occur in 33 meetings

---

## Phase 5: Deliverables

- [ ] **Jupyter notebook** — connect to `rag_db @ localhost:5434`, kernel `pydantic_ai_agents`
  - [ ] DB connection + schema overview
  - [ ] Final Version clusters — theme titles, sizes, audience breakdown
  - [ ] UMAP 2-dim scatter (`final_version/outputs/viz_coords.csv`)
  - [ ] Key insight charts (I1–I10, selected stakeholder Qs)
  - [ ] High-risk meeting watchlist (I7)
- [x] **Slide deck structure** — 9-slide sequence in `INSIGHTS_GUIDE.md`
- [ ] **Slide deck content** — charts + narrative in presentation tool
- [ ] **Video demo** — 5–10 min screen recording with narration

---

## DB Topology

| Port | DB | Purpose |
|------|----|---------|
| 5432 | postgres | local Windows Postgres — not used for this project |
| 5433 | rag_db | Apache AGE Docker — knowledge graph project (unrelated) |
| **5434** | **rag_db** | **Docker pgvector — canonical DB for this project** |

- Schema: `meeting_analytics` (all 9 tables + 1 view)
- Credentials: `basics/iprep/meeting-analytics/.env`
- MCP postgresql tool connects to 5432 — **never use it to verify schema changes**; use DBeaver or `psql` @ 5434

---

## Key Scripts

| Script | What it does |
|--------|-------------|
| `final_version/load_raw_jsons_to_db.py --reset` | Drop + recreate all 9 tables from raw JSON + outputs/ |
| `final_version/load_output_csvs_to_db.py --reset` | Reload 3 semantic tables from outputs/ (no Ollama needed) |
| `final_version/verify.py` | Check all 9 tables have correct row counts |
| `take_a/generate_rule_based_taxonomy.py --reset` | Rebuild Take A tables (WARNING: wipes entire schema) |
| `take_b/load_outputs_to_pg.py` | Load 3 Take B tables |

## Key Docs

| File | Purpose |
|------|---------|
| `SESSION_CONTEXT.md` | Full project state — reload this each session |
| `INSIGHTS_GUIDE.md` | Insight catalogue, all SQL, 9-slide deck structure — start here for notebook |
| `nl_questions.md` | All 26 NL questions indexed by stakeholder |
| `req.md` | Source of truth for deliverables |
| `final_version/design.md` | Pipeline design decisions |
| `final_version/faq.md` | FAQ on clustering approach |
| `sql/02_insight_queries.sql` | 10 insight queries |
| `sql/03_stakeholder_questions.sql` | 16 stakeholder questions |
