# Session Context — Meeting Analytics
Last updated: 2026-05-09 (session 3)

## How to reload this session
Tell Claude: "Read basics/iprep/meeting-analytics/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment (see basics/iprep/meeting-analytics/req.md).
100 meeting summary JSONs in basics/iprep/meeting-analytics/dataset/.
Three approaches to clustering meeting topics into themes:
  - Take A: rule-based (generate_rule_based_taxonomy.py) — DONE, ran against Postgres
  - Take B: TF-IDF + KMeans (cluster_taxonomy_v2.py) — DONE, validated, log saved
  - Take C: LLM-assisted semantic clustering — NEXT, implemented, not yet run end-to-end

---

## Current state

### Housekeeping completed across sessions
- Directory renamed: basics/iprep/meeting-analytics → basics/iprep/meeting-analytics
- Folder re-org: scripts split into take_a/, take_b/, take_c/ subdirs
- Postgres schema renamed: iprep_meeting-analytics_functional → meeting_analytics (done in DBeaver)
- All Neon DB references removed; local .env only (rag_db @ localhost:5434)
- All standalone print() calls replaced with print("\n...") across all scripts
- take_c_run.log (131KB) generated — mirrors take_b_run.log format

### Take B — complete and validated
File: basics/iprep/meeting-analytics/cluster_taxonomy_v2.py
- auto-k on by default; use --no-auto-k --clusters N to pin k
- Person name filtering: participant emails + transcript speaker_names extracted
  and passed as extra stop words to TfidfVectorizer
- Last run: k=8, silhouette=0.0322, all cluster labels are clean business terms
- Output log: basics/iprep/meeting-analytics/cluster_work/take_b_run.log
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
File: basics/iprep/meeting-analytics/take_c_semantic_clustering.py
Steps:
  [1/9] Load 100 meeting records from raw JSON
  [2/9] Extract + deduplicate topics: 600 raw -> 351 exact-dedup -> 343 fuzzy-dedup
  [3/9] Embed phrases with nomic-embed-text via Ollama
  [4/9] UMAP dimensionality reduction (10-dim for clustering, 2-dim for viz)
  [5/9] HDBSCAN clustering + noise reassignment to nearest centroid
         + coherence check: avg pairwise cosine similarity per cluster
         flags: tight (>=0.6) / review (0.4-0.6) / LOOSE (<0.4)
  [6/9] LLM labels each cluster (llama3.1:8b, ~10 calls, structured JSON)
  [7/9] LLM infers call type per meeting (support / external / internal)
  [8/9] Assign meetings to themes, write CSV/JSON outputs to cluster_work_c/
  [9/9] Persist to Postgres (meeting_analytics schema) + print insight queries

### Postgres store — fully implemented and working
File: basics/iprep/meeting-analytics/take_c_pg_store.py
DB: rag_db @ localhost:5434 (rag_user:rag_pass) — pgvector installed, meeting_analytics schema
Tables in meeting_analytics schema:
  semantic_clusters        — cluster_id, theme_title, audience, rationale
  semantic_phrases         — canonical text + vector(768) IVFFlat + tsvector GIN
  semantic_meeting_themes  — meeting_id, cluster_id, is_primary, call_type, sentiment

8 insight query methods (all implemented, run automatically after persist):
  insight_theme_sentiment()
  insight_churn_by_theme()
  insight_call_type_theme_matrix()
  insight_feature_gap_themes()
  insight_sentiment_distribution_by_theme()
  insight_signal_counts_by_theme()
  insight_theme_cooccurrence()
  insight_high_risk_meetings()

---

## Next steps

1. COMPARE TAKE B vs TAKE C
   Goal: validate that two independent methods discover the same themes.
   Take B clusters are already in cluster_work/meeting_clusters.csv (meeting_id → cluster_id).
   Take C output is in cluster_work_c/meeting_themes.csv.
   Comparison: for each meeting, do B and C agree on the theme? Where do they diverge?
   If agreement is high (~80%+), the themes are real — not artefacts of the method.
   Specific question: does Take C split any of Take B's 8 clusters further?
   (e.g. Take B has one "backup" cluster; does Take C separate backup-performance
    from backup-recovery?)

2. Validate Take C cluster quality
   - Check coherence scores in cluster_work_c/cluster_metrics.json
   - Scan phrase_clusters.csv per cluster
   - Compare Take C cluster names vs Take B's 8 labels above

3. DELIVERABLES (from req.md — all three required)

   a) Slide deck (30-min presentation to product + engineering leadership)
      - Lead with insights, not code
      - Show all 3 approaches + progression reasoning
      - Key charts: theme x sentiment heatmap, churn by theme, call type distribution
      - At least 2-3 bonus insight ideas beyond the required tasks

   b) Jupyter notebook (technical reference material)
      - Must be clean and runnable end-to-end
      - Cover: data loading, Take A (rule-based), Take B (TF-IDF/KMeans),
        Take C (semantic clustering), insight queries, key findings
      - Kernel: pydantic_ai_agents conda env
      - Explain key decisions in markdown cells

   c) Video demo (5-10 min screen recording with narration)
      - Show pipeline running
      - Walk through outputs and insights/visualizations

---

## Key design decisions (don't revisit without good reason)
- Embed topic phrases (not meetings) in Take C — finer cluster resolution
- HDBSCAN not KMeans in Take C — no fixed K needed, density-adaptive
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama — 768 dims sufficient
- llama3.1:8b for labeling — local, free, structured JSON output
- Extend Take A's meeting_analytics schema — don't duplicate tables
- Copy from rag/storage/vector_store/postgres.py, never import from RAG

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── req.md / req.pdf       source of truth for deliverables
├── faq.txt                all Q&A across all takes
├── notes.txt              running project notes
├── schema_dump.sql        DBeaver introspection queries
├── list_of_topics.txt     351 unique topics extracted across 100 meetings
├── .env                   PG credentials (rag_db @ localhost:5434)
├── dataset/               100 meeting folders, each with summary.json
├── take_a/
│   ├── generate_rule_based_taxonomy.py   main Take A script
│   ├── load_dataset_to_postgres.py       raw JSON → Postgres loader
│   ├── export_taxonomy_prompt_inputs.py  exports topics for LLM taxonomy review
│   └── taxonomy_work/                    taxonomy_input.json + taxonomy_prompt.md
├── take_b/
│   ├── cluster_taxonomy_v2.py            TF-IDF + KMeans (auto-k default)
│   └── outputs/                          meeting_clusters.csv, cluster_summary.json,
│                                         cluster_terms.csv, take_b_run.log, ...
└── take_c/
    ├── take_c_semantic_clustering.py     main Take C pipeline (9 steps)
    ├── take_c_pg_store.py                Postgres store (SemanticClusterStore)
    ├── take_c_design.md                  design decisions and tradeoff notes
    └── outputs/                          meeting_themes.csv, phrase_clusters.csv,
                                          semantic_clusters.json, take_c_run.log (131KB)
```

---

## Environment
- Python 3.13, conda env: pydantic_ai_agents (already active — no conda run needed)
- Ollama: http://localhost:11434/v1
- Embedding model: nomic-embed-text:latest (768 dims)
- LLM model: llama3.1:8b
- Postgres: meeting_analytics schema (same DB as RAG project)
- Deps installed: rapidfuzz, umap-learn, hdbscan, scikit-learn, pandas
