# Session Context — Meeting Analytics
Last updated: 2026-05-09

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

### Housekeeping done this session
- Directory renamed: basics/iprep/i1 → basics/iprep/meeting-analytics
- Class renamed: IprepPhraseStore → SemanticClusterStore
- Schema constant renamed: iprep_i1_functional → meeting_analytics (in all .py files)
- cluster_taxonomy.py (Take B v1) deleted
- All stale paths in faq.txt and notes.txt updated

⚠️  POSTGRES SCHEMA RENAME STILL NEEDED (Take A already ran):
    ALTER SCHEMA iprep_i1_functional RENAME TO meeting_analytics;
    Run this in DBeaver or psql before running Take A or Take C again.

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

### Take C pipeline — fully implemented, not yet run
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

### Postgres store — fully implemented
File: basics/iprep/meeting-analytics/take_c_pg_store.py
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

1. RENAME POSTGRES SCHEMA (before anything else)
   ALTER SCHEMA iprep_i1_functional RENAME TO meeting_analytics;

2. RUN TAKE C end-to-end
   Command: python basics/iprep/meeting-analytics/take_c_semantic_clustering.py --reset-pg --skip-viz
   Requires: ollama serve + nomic-embed-text:latest + llama3.1:8b pulled

3. COMPARE TAKE B vs TAKE C
   Goal: validate that two independent methods discover the same themes.
   Take B clusters are already in cluster_work/meeting_clusters.csv (meeting_id → cluster_id).
   After Take C runs, meeting_themes.csv has the same.
   Comparison: for each meeting, do B and C agree on the theme? Where do they diverge?
   If agreement is high (~80%+), the themes are real — not artefacts of the method.
   Specific question: does Take C split any of Take B's 8 clusters further?
   (e.g. Take B has one "backup" cluster; does Take C separate backup-performance
    from backup-recovery?)

4. Validate Take C cluster quality
   - Check coherence scores in cluster_work_c/cluster_metrics.json
   - Scan phrase_clusters.csv per cluster
   - Compare Take C cluster names vs Take B's 8 labels above

5. Slide deck
   - Lead with insights, not code
   - Show all 3 approaches + progression reasoning
   - Key charts: theme x sentiment heatmap, churn by theme, call type distribution

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

| File | Purpose |
|------|---------|
| req.md | Source of truth for deliverables |
| notes.txt | Running project notes |
| faq.txt | All Q&A across all takes |
| take_c_semantic_clustering.py | Take C main pipeline (9 steps) |
| take_c_pg_store.py | Postgres store (SemanticClusterStore) |
| generate_rule_based_taxonomy.py | Take A — rule-based |
| cluster_taxonomy_v2.py | Take B — TF-IDF + KMeans (auto-k default) |
| schema_dump.sql | DBeaver introspection queries (aside only) |
| list_of_topics.txt | All 351 unique topics extracted across 100 meetings |
| dataset/ | 100 meeting folders, each with summary.json |
| cluster_work/ | Take B outputs + take_b_run.log |
| cluster_work_c/ | Take C outputs (created on first run) |

---

## Environment
- Python 3.13, conda env: pydantic_ai_agents (already active — no conda run needed)
- Ollama: http://localhost:11434/v1
- Embedding model: nomic-embed-text:latest (768 dims)
- LLM model: llama3.1:8b
- Postgres: meeting_analytics schema (same DB as RAG project)
- Deps installed: rapidfuzz, umap-learn, hdbscan, scikit-learn, pandas
