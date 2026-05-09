# Session Context — Take C Semantic Clustering
Last updated: 2026-05-09

## How to reload this session
Tell Claude: "Read basics/iprep/i1/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment (see basics/iprep/i1/req.md).
100 meeting summary JSONs in basics/iprep/i1/dataset/.
Three approaches to clustering meeting topics into themes:
  - Take A: rule-based (generate_rule_based_taxonomy.py) — DONE, runs against Postgres
  - Take B: TF-IDF + K-Means (cluster_taxonomy_v2.py) — implemented, not yet validated
  - Take C: LLM-assisted semantic clustering — MAIN FOCUS, implemented, not yet run end-to-end

---

## Current state (as of session end)

### Take C pipeline — fully implemented, dry-run verified
File: basics/iprep/i1/take_c_semantic_clustering.py
Steps:
  [1/9] Load 100 meeting records from raw JSON
  [2/9] Extract + deduplicate topics: 600 raw -> 351 exact-dedup -> 343 fuzzy-dedup
  [3/9] Embed phrases with nomic-embed-text via Ollama
  [4/9] UMAP dimensionality reduction (10-dim for clustering, 2-dim for viz)
  [5/9] HDBSCAN clustering + noise reassignment to nearest centroid
         + coherence check: avg pairwise cosine similarity per cluster (sklearn)
         flags: tight (>=0.6) / review (0.4-0.6) / LOOSE (<0.4)
  [6/9] LLM labels each cluster (llama3.1:8b, ~10 calls, structured JSON)
  [7/9] LLM infers call type per meeting (support / external / internal)
  [8/9] Assign meetings to themes, write CSV/JSON outputs to cluster_work_c/
  [9/9] Persist to Postgres (iprep_i1_functional schema) + print insight queries

### Postgres store — fully implemented
File: basics/iprep/i1/take_c_pg_store.py
Adapted from rag/storage/vector_store/postgres.py (no RAG imports).
Tables in iprep_i1_functional schema:
  semantic_clusters        — cluster_id, theme_title, audience, rationale
  semantic_phrases         — canonical text + vector(768) IVFFlat + tsvector GIN
  semantic_meeting_themes  — meeting_id, cluster_id, is_primary, call_type, sentiment

Hybrid search: semantic_search_phrases / text_search_phrases / hybrid_search_phrases (RRF k=60)

Insight query methods (8 total):
  insight_theme_sentiment()                 avg sentiment per theme
  insight_churn_by_theme()                  churn_signal count + per_meeting rate
  insight_call_type_theme_matrix()          call_type x theme matrix
  insight_feature_gap_themes()              feature_gap count per theme
  insight_sentiment_distribution_by_theme() count of each sentiment category per theme
  insight_signal_counts_by_theme()          all signal types pivoted per theme
  insight_theme_cooccurrence()              theme pairs that appear together most
  insight_high_risk_meetings()              churn signals + low sentiment meetings

Note: signal-count queries join Take A's key_moments table.
      Run Take A first if key_moments is empty.

---

## What was NOT done yet (next steps)

1. ACTUALLY RUN the full pipeline end-to-end
   Command: python basics/iprep/i1/take_c_semantic_clustering.py --reset-pg --skip-viz
   Requires: ollama serve, nomic-embed-text and llama3.1:8b pulled
   Then review themes output and check coherence scores

2. Validate cluster quality
   - Check coherence scores in cluster_metrics.json (tight >=0.6, review 0.4-0.6, LOOSE <0.4)
   - Scan phrase_clusters.csv per cluster — do phrases belong together?
   - Compare Take C clusters vs Take A THEME_KEYWORDS (are same groupings discovered?)

3. Run Take A first (if key_moments not yet populated)
   Command: python basics/iprep/i1/generate_rule_based_taxonomy.py --reset
   This populates key_moments, sentiment_features — needed for churn/feature_gap insights

4. Trend analysis pipeline (paused — foundation in place)
   - Test cases for the 8 insight query methods
   - Temporal trend query (theme frequency / sentiment drift over time using meeting timestamps)
   - Thin reporting layer (CSV export or print) for the slide deck

5. Take B validation
   - Get cluster_taxonomy_v2.py working end-to-end
   - Compare its clusters against Take C's — agreement = evidence both are right

6. Slide deck
   - Lead with insights, not code
   - Show all 3 approaches + progression reasoning
   - Key charts: theme x sentiment heatmap, churn by theme, call type distribution

---

## Key design decisions made (don't revisit without good reason)

- Embed topic phrases (not meetings) — gives finer cluster resolution
- HDBSCAN not K-Means — no fixed K needed, density-adaptive
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama — already in stack, 768 dims sufficient
- llama3.1:8b for labeling — local, free, structured JSON output
- Direct LLM on summary text for call-type inference (not theme-vote heuristic)
- Extend Take A's iprep_i1_functional schema — don't duplicate tables
- Copy from rag/storage/vector_store/postgres.py, never import from RAG

---

## File map

| File | Purpose |
|------|---------|
| req.md | Source of truth for deliverables |
| notes.txt | Running project notes across all takes |
| take_c_design.md | Pipeline design + tradeoffs + audit trail |
| take_c_faq.txt | All Q&A from session — validation, insights, pgvector |
| take_c_semantic_clustering.py | Main pipeline (9 steps) |
| take_c_pg_store.py | Postgres store (pgvector + tsvector + insight queries) |
| generate_rule_based_taxonomy.py | Take A — rule-based |
| cluster_taxonomy_v2.py | Take B — TF-IDF + K-Means (keep, delete v1) |
| cluster_taxonomy.py | Take B v1 — DELETE this |
| schema_dump.sql | DBeaver introspection queries — not a pipeline, aside only |
| list_of_topics.txt | All 351 unique topics extracted across 100 meetings |
| dataset/ | 100 meeting folders, each with summary.json |
| cluster_work_c/ | Take C outputs (created on first run) |

---

## Environment
- Python 3.13, conda env: pydantic_ai_agents
- Ollama: http://localhost:11434/v1
- Embedding model: nomic-embed-text:latest (768 dims)
- LLM model: llama3.1:8b
- Postgres: iprep_i1_functional schema (same DB as RAG project)
- New deps installed: rapidfuzz, umap-learn, hdbscan
