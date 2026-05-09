# Session Context — Meeting Analytics
Last updated: 2026-05-09 (session 6)

## How to reload this session
Tell Claude: "Read basics/iprep/meeting-analytics/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment (see basics/iprep/meeting-analytics/req.md).
100 meeting summary JSONs in basics/iprep/meeting-analytics/dataset/.
Three approaches to clustering meeting topics into themes:
  - Take A: rule-based (generate_rule_based_taxonomy.py) — DONE, ran against Postgres
  - Take B: TF-IDF + KMeans (cluster_taxonomy_v2.py) — DONE, validated, log saved
  - Take C: LLM-assisted semantic clustering — DONE, full run complete, validated

---

## Current state

### Housekeeping completed across sessions (session 6 additions)
- take_c_design.md updated to v0.4 — Context & Goals rewritten with Goal 1/2/3 framing matching Take A and Take B
- Postgres consolidated to ONE target: rag_db @ localhost:5434 (rag_user:rag_pass) for all three takes
- Port inventory confirmed: 5432=local Windows Postgres (postgres:postgres, 10 Take A tables only, no pgvector), 5433=Apache AGE Docker (KG project, separate), 5434=Docker pgvector (rag_user:rag_pass, all 16 tables)
- take_b/load_outputs_to_pg.py written — loads 3 Take B tables from outputs/ CSVs (no re-clustering needed)
- take_c/load_outputs_to_pg.py written — loads 3 Take C tables from outputs/ JSON+CSVs (no re-embedding needed)
- setup_all_tables.py written — master one-shot script: Take A --reset + Take B loader + Take C loader → 16 tables
- export_all_to_csv.py written — exports all 16 Postgres tables to outputs/csv/ as flat-file backup
- All three FAQs updated with "How do I reload tables into Postgres?" entries
- Insight catalogue built: 10 insights with SQL, stakeholder, and reading for each
- Background Take C re-run completed (exit code 0, 144s, 26 clusters again)

### Housekeeping completed across sessions
- Directory renamed: basics/iprep/i1 → basics/iprep/meeting-analytics
- Folder re-org: scripts split into take_a/, take_b/, take_c/ subdirs
- Postgres schema renamed: iprep_i1_functional → meeting_analytics (done in DBeaver)
- All Neon DB references removed; local .env only (rag_db @ localhost:5434)
- All standalone print() calls replaced with print("\n...") across all scripts
- faq.txt split into per-take markdown files with TOC (take_a_faq.md, take_b_faq.md, take_c_faq.md)
- take_c_design.md updated to v0.3 — actual run results, corrected schema, resolved open questions
- compare_b_vs_c.py written and run — cross-validation complete
- take_a_design.md written (v0.1) — full retrospective design doc
- take_a_faq.md expanded to 10 entries (was 1)
- take_b_faq.md expanded to 6 entries — output→step mapping added, step labels added to source
- take_c_faq.md expanded to 12 entries — UMAP/HDBSCAN mechanics, LLM vs HDBSCAN output split
- Pipeline step labels [1/7]–[7/7] added to cluster_taxonomy_v2.py main()

### Take B — complete and validated
File: basics/iprep/meeting-analytics/take_b/cluster_taxonomy_v2.py
- auto-k on by default; use --no-auto-k --clusters N to pin k
- Person name filtering: participant emails + transcript speaker_names extracted
  and passed as extra stop words to TfidfVectorizer
- Last run: k=8, silhouette=0.0322, all cluster labels are clean business terms
- Output log: basics/iprep/meeting-analytics/take_b/outputs/take_b_run.log
- Outputs: meeting_clusters.csv, cluster_summary.json, cluster_terms.csv,
           cluster_metrics.json, cluster_scores.csv

Take B clusters (k=8):
  0  renewal / competitive / pricing / outage
  1  outage / incident / failure / communication
  2  billing / overage / dispute
  3  planning / sprint / launch
  4  hipaa / compliance / reporting / framework
  5  backup / performance / response time / support response
  6  mfa / identity / management / sso
  7  backup / hybrid / recovery / onboarding

### Take C pipeline — COMPLETE, all insights firing, run log generated
File: basics/iprep/meeting-analytics/take_c/take_c_semantic_clustering.py
Steps:
  [1/9] Load 100 meeting records from raw JSON
  [2/9] Extract + deduplicate topics: 600 raw -> 351 exact-dedup -> 343 fuzzy-dedup
  [3/9] Embed phrases with nomic-embed-text via Ollama
  [4/9] UMAP dimensionality reduction (10-dim for clustering, 2-dim for viz)
  [5/9] HDBSCAN clustering + noise reassignment to nearest centroid
         + coherence check: avg pairwise cosine similarity per cluster
         flags: tight (>=0.6) / review (0.4-0.6) / LOOSE (<0.4)
  [6/9] LLM labels each cluster (llama3.1:8b, 26 calls, structured JSON)
  [7/9] LLM infers call type per meeting (support / external / internal)
  [8/9] Assign meetings to themes, write CSV/JSON outputs to take_c/outputs/
  [9/9] Persist to Postgres (meeting_analytics schema) + print insight queries

Result: 26 clusters, 22 noise phrases (6.4%), 9 tight / 17 review / 0 loose.
All 26 clusters got clean LLM labels — zero fallbacks triggered.

### Postgres — all 16 tables in meeting_analytics @ rag_db localhost:5434
DB: rag_db @ localhost:5434 (rag_user:rag_pass) — pgvector installed
Credentials in: basics/iprep/meeting-analytics/.env

Take A tables (10): meetings, meeting_participants, meeting_summaries, summary_topics,
  action_items, key_moments, transcript_lines, meeting_themes, call_types, sentiment_features

Take B tables (3):
  kmeans_clusters         — cluster_id, label (top-4 centroid terms), meeting_count, silhouette_score
  kmeans_cluster_terms    — cluster_id, rank, term (top-12 per cluster)
  kmeans_meeting_clusters — meeting_id → cluster_id (hard single assignment)

Take C tables (3):
  semantic_clusters        — cluster_id, theme_title, audience, rationale, phrase_count
  semantic_phrases         — canonical text + vector(768) + tsvector GIN (embedding=NULL in CSV path)
  semantic_meeting_themes  — meeting_id, cluster_id, is_primary, call_type, call_confidence, sentiment

Setup scripts:
  setup_all_tables.py       — one-shot: Take A --reset + Take B loader + Take C loader (16 tables)
  take_b/load_outputs_to_pg.py — load Take B from outputs/ CSVs only (no re-clustering)
  take_c/load_outputs_to_pg.py — load Take C from outputs/ JSON+CSVs only (no re-embedding)
  export_all_to_csv.py      — snapshot all 16 tables to outputs/csv/ (run after any pipeline change)

### Take B vs Take C cross-validation — COMPLETE
Script: basics/iprep/meeting-analytics/compare_b_vs_c.py (run from repo root)
All 100 meetings matched. Agreement proxy 37% — intentionally low (see take_c_faq.md).

Key findings:
  - B-2 billing and B-5 backup confirmed as real cohesive topics by both methods
  - B-1 outage/incident: Take C correctly splits into Outage Prevention (C-09) vs
    Incident Response (C-08) — two distinct workflows TF-IDF conflated
  - B-4 compliance: Take C separates HIPAA, Audit Readiness, and Compliance Governance
  - B-3 planning: Take C correctly splits product launch from engineering sprint meetings
  - DO NOT re-run Take B at k=26 — meaningless comparison (see take_c_faq.md)

---

## Next steps

1. DELIVERABLES (from req.md — all three required)

   a) Jupyter notebook (technical reference material) — START HERE
      - Connect to rag_db @ localhost:5434 for all queries
      - Cover: data loading, Take A, Take B, Take C, insight queries, key findings
      - 10 insight queries catalogued (see session 6 notes) — turn into charts
      - Include UMAP 2-dim scatter plot (viz_coords.csv in take_c/outputs/ — check if still there)
      - Include B vs C cross-tab comparison (compare_b_vs_c.py output)
      - Kernel: pydantic_ai_agents conda env

   b) Slide deck (30-min presentation to product + engineering leadership)
      - Lead with insights, not code
      - Key charts from notebook: theme x sentiment heatmap, churn by theme, call type distribution
      - Headline finding: Reliability is the #1 churn driver (1.04 churn signals/meeting)
      - B vs C cross-validation confirms themes are real signal, not artefacts

   c) Video demo (5-10 min screen recording with narration)

---

## Key design decisions (don't revisit without good reason)
- Embed topic phrases (not meetings) in Take C — finer cluster resolution
- HDBSCAN not KMeans in Take C — no fixed K needed, density-adaptive
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama — 768 dims sufficient
- llama3.1:8b for labeling — local, free, structured JSON output
- Extend Take A's meeting_analytics schema — don't duplicate tables
- Copy from rag/storage/vector_store/postgres.py, never import from RAG
- LLM phrase sampling: first N in natural order (centroid sort not implemented — didn't matter)

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── req.md / req.pdf           source of truth for deliverables
├── compare_b_vs_c.py          Take B vs Take C cross-validation script
├── setup_all_tables.py        One-shot: reload all 16 tables (Take A --reset + B + C loaders)
├── export_all_to_csv.py       Snapshot all 16 Postgres tables to outputs/csv/
├── notes.txt                  running project notes
├── schema_dump.sql            DBeaver introspection queries
├── list_of_topics.txt         351 unique topics extracted across 100 meetings
├── .env                       PG credentials (rag_db @ localhost:5434)
├── dataset/                   100 meeting folders, each with summary.json
├── take_a/
│   ├── generate_rule_based_taxonomy.py   main Take A script
│   ├── load_dataset_to_postgres.py       raw JSON → Postgres loader
│   ├── take_a_design.md                  design doc (v0.1) — pipeline, schema, decisions
│   └── take_a_faq.md                     FAQ with TOC (11 entries)
├── take_b/
│   ├── cluster_taxonomy_v2.py            TF-IDF + KMeans (auto-k default, [1/7]–[7/7] steps labelled)
│   ├── load_outputs_to_pg.py             Load Take B outputs → 3 Postgres tables (no re-clustering)
│   ├── take_b_faq.md                     FAQ with TOC (7 entries)
│   └── outputs/                          meeting_clusters.csv, cluster_summary.json,
│                                         cluster_terms.csv, take_b_run.log, ...
├── outputs/csv/                           Flat-file backup of all 16 Postgres tables
└── take_c/
    ├── take_c_semantic_clustering.py     main Take C pipeline (9 steps)
    ├── take_c_pg_store.py                Postgres store (SemanticClusterStore)
    ├── load_outputs_to_pg.py             Load Take C outputs → 3 Postgres tables (no re-embedding)
    ├── take_c_design.md                  design decisions — v0.4, Goal 1/2/3 framing added
    ├── take_c_faq.md                     FAQ with TOC (13 entries)
    └── outputs/                          meeting_themes.csv, phrase_clusters.csv,
                                          semantic_clusters.json, cluster_metrics.json,
                                          take_c_run.log
```

---

## Environment
- Python 3.13, conda env: pydantic_ai_agents (already active — no conda run needed)
- Ollama: http://localhost:11434/v1
- Embedding model: nomic-embed-text:latest (768 dims)
- LLM model: llama3.1:8b
- Postgres: meeting_analytics schema (same DB as RAG project)
- Deps installed: rapidfuzz, umap-learn, hdbscan, scikit-learn, pandas
