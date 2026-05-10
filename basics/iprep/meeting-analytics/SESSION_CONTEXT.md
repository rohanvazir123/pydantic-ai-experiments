# Session Context — Meeting Analytics
Last updated: 2026-05-10 (session 13)

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
creates 3 semantic tables — no dependency on Take A or B.

**Final Version only (9 tables + 1 view):** run `load_raw_jsons_to_db.py --reset` then `load_output_csvs_to_db.py --reset`.

---

### Tables — verified row counts

**Final Version base tables (6) — `final_version/load_raw_jsons_to_db.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `meetings` | 100 | meeting_id, title, organizer_email, duration_minutes, start_time |
| `meeting_participants` | 311 | meeting_id, email |
| `meeting_summaries` | 100 | summary_text, overall_sentiment, sentiment_score, topics TEXT[], **products TEXT[]** |
| `key_moments` | 402 | moment_type (8 types), text, speaker, time_seconds |
| `action_items` | 397 | meeting_id, owner, text |
| `transcript_lines` | 4313 | speaker, sentence, sentiment_type, time_seconds |

**Final Version semantic tables (3) + 1 view — `final_version/load_output_csvs_to_db.py`**

| Table / View | Rows | What it holds |
|--------------|------|---------------|
| `semantic_clusters` | 26 | theme_title (LLM), audience (LLM), rationale (LLM), phrase_count |
| `semantic_phrases` | 343 | canonical phrase, cluster_id, tsvector GIN |
| `semantic_meeting_themes` | 516 | meeting_id, cluster_id, is_primary, call_type (LLM), sentiment, products TEXT[] |
| `action_items_by_theme` *(view)* | 397 | action_items JOIN semantic_meeting_themes (primary) JOIN semantic_clusters — columns: meeting_id, owner, action_item, cluster_id, theme_title, audience |

26 clusters from 343 deduplicated topic phrases. 22 noise phrases (6.4%) reassigned to nearest centroid.

---

### Setup scripts

| Script | What it does |
|--------|-------------|
| `final_version/load_raw_jsons_to_db.py` | Standalone: raw JSON → 6 base tables. `--reset` drops entire schema first. |
| `final_version/load_output_csvs_to_db.py` | Standalone: outputs/ CSVs/JSON → 3 semantic tables. `--reset` drops semantic tables first. |
| `final_version/generate_charts.py` | **Main pipeline**: DB → CSVs → PNGs. Default exports CSVs then generates PNGs. `--no-export` skips DB step. |
| `final_version/verify.py` | Checks all 9 Final Version tables. |

### Docker — persistent volume
pgvector container (`rag_pgvector`) runs with named volume `pydantic-ai-experiments_pgvector_data`.
Always start via `docker compose up -d pgvector` from the repo root.
`docker compose down` preserves data. `docker compose down -v` wipes it.

---

## Chart pipeline — COMPLETE

```
DB (localhost:5434)
  └─ generate_charts.py (default: exports CSVs then PNGs)
       ├─ final_version/outputs/chart_data/   17 CSVs
       └─ final_version/outputs/charts/       10 PNGs
```

**Run from terminal (use full conda path — conda activate doesn't work in PS without initialization):**
```powershell
& "C:\Users\rohan\anaconda3\envs\pydantic_ai_agents\python.exe" final_version/generate_charts.py
& "C:\Users\rohan\anaconda3\envs\pydantic_ai_agents\python.exe" final_version/generate_charts.py --no-export
```

**Charts produced (in order):**

| File | Title | Stakeholder tag |
|------|-------|----------------|
| `00_dataset_overview.png` | Call type + product breakdown | All Leadership |
| `01_cluster_table.png` | 26 themes table | All Leadership |
| `02_sentiment_by_calltype.png` | Sentiment distribution + avg score | All Leadership |
| `03_theme_sentiment_heatmap.png` | Theme × sentiment heatmap | All Leadership |
| `04_churn_density.png` | Churn signals per meeting by theme | Sales & CS |
| `05_product_signals.png` | Tech issues + churn signals by product | Engineering · Product |
| `06_positive_signals.png` | Praise by product + Comply external sentiment | Product · Marketing · Sales & CS |
| `07_detect_external_impact.png` | E3/R4 — Detect outage contamination | Engineering (CTO) · Sales |
| `08_feature_gaps_by_product.png` | P1 — Feature gaps × sentiment bucket | Product (CPO) |
| `09_action_item_owners.png` | S3 — Owner × theme (left) + Owner × product (right) | Operations · Engineering · CS · Sales |

All charts have a stakeholder badge (top-right corner) added via `_tag(fig, "label")`.

**Known gotcha:** `generate_charts.py` print statements use ASCII only (`-` not `─`, `->` not `→`) — Windows cp1252 can't encode box-drawing chars.

---

## Notebook — `meeting_analytics.ipynb` — COMPLETE

**Location:** `basics/iprep/meeting-analytics/meeting_analytics.ipynb`
**Kernel:** `pydantic_ai_agents` conda env
**Runs standalone, top to bottom, no errors.**

### Sections

| Cell ID | Section | Notes |
|---------|---------|-------|
| `cell_01_setup` | DB connection + path setup | `q()` helper for async DB queries |
| `625a87b0` | Optional DB regen header | Markdown only |
| `4efc9898` | Step 1 — base tables | subprocess `load_raw_jsons_to_db.py --reset` |
| `7ec071f3` | Step 2 — semantic tables | subprocess `load_output_csvs_to_db.py --reset` |
| `cell_03_overview` | Dataset overview charts | Call types + products |
| `cell_05_umap` | UMAP 2D scatter | `final_version/outputs/viz_coords.csv` |
| `cell_06_cluster_table` | 26 themes table | styled DataFrame |
| `cell_10_sentiment_calltype` | Sentiment by call type | stacked bars + avg score |
| `cell_12_heatmap` | Theme × sentiment heatmap | `.astype(int)` before sns.heatmap — required |
| `cell_14_churn` | Churn density by theme | |
| `cell_15_watchlist` | High-risk account watchlist | 38 meetings |
| `cell_16_product` | Tech issues + churn by product | |
| `cell_17_positive` | Positive signals (Comply) | |
| `622eeaab` | Leadership questions header | Markdown: E3/R4, P1, S3 |
| `4f330762` | E3/R4 — Detect external contamination | stacked bar + pie |
| `12ef1817` | P1 — Feature gaps × sentiment | grouped bar |
| `bf958936` | S3 — Action item owners | **Two-panel**: owner×theme (left) + owner×product (right) |
| `d13fdd8c` | Export header | Markdown |
| `5c69f112` | Export cell | subprocess `generate_charts.py` |

### Key implementation notes
- `matplotlib.use("Agg")` only when `"ipykernel" not in sys.modules` — safe in both Jupyter and standalone
- `pivot_table(fill_value=0)` returns float dtype → `.astype(int)` required before `fmt="d"` in seaborn heatmap
- S3 cell uses `action_items_by_theme` view (owner × theme) + `action_items` JOIN `meeting_summaries` (owner × product); top 12 owners by total, sorted ascending so highest appears at top of barh
- Export cell uses `sys.executable` so it picks up the active conda env Python automatically

---

## Deliverables status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Notebook | **DONE** | Runs top to bottom; all 7 sections complete |
| Chart pipeline | **DONE** | 10 PNGs in `final_version/outputs/charts/` |
| Video demo | **TODO** | 5–10 min screen recording, narrate the "so what" |
| Slide deck | **TODO** | 9-slide structure in `INSIGHTS_GUIDE.md`; screenshots from notebook |

---

## Next steps

1. **Video** — record screen walkthrough of notebook top to bottom; narrate findings not code
2. **Slide deck** — pull screenshots from notebook; 9-slide structure in `INSIGHTS_GUIDE.md`

---

## Key design decisions (don't revisit without good reason)
- Embed topic phrases (not full meetings) — finer cluster resolution
- HDBSCAN not KMeans — no fixed K, density-adaptive, found 26 naturally
- UMAP 10-dim before HDBSCAN, separate 2-dim for viz only
- nomic-embed-text via Ollama (768 dims) — local, free, sufficient
- Final Version is fully self-contained
- LLM is translation, not intelligence — embeddings + HDBSCAN cluster; LLM only names the result
- Human inspection required after every schema/data change — verify via `final_version/verify.py` or DBeaver @ port 5434

---

## Environment
- Python 3.13, conda env: `pydantic_ai_agents`
- **Full path required in PowerShell:** `C:\Users\rohan\anaconda3\envs\pydantic_ai_agents\python.exe`
- **VS Code interpreter:** set to the above path in `settings.json`
- Ollama: `http://localhost:11434/v1` · model: `llama3.1:8b` · embeddings: `nomic-embed-text:latest`
- Postgres: `meeting_analytics` schema in `rag_db` Docker container @ port 5434

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── INSIGHTS_GUIDE.md            insight catalogue, SQL, 9-slide deck structure
├── req.md / req.pdf             source of truth for deliverables
├── .env                         PG credentials
├── meeting_analytics.ipynb      MAIN NOTEBOOK — complete, runs standalone
├── dataset/                     100 meeting folders
├── sql/
│   ├── 01_verify_tables.sql
│   ├── 02_insight_queries.sql
│   └── 03_stakeholder_questions.sql
└── final_version/
    ├── generate_charts.py        DB→CSV→PNG pipeline (10 charts)
    ├── export_chart_data.py      superseded; logic folded into generate_charts.py
    ├── load_raw_jsons_to_db.py   raw JSON → 6 base tables
    ├── load_output_csvs_to_db.py outputs/ CSVs → 3 semantic tables + view
    ├── verify.py                 checks all 9 tables
    ├── nl_questions.md           20 stakeholder questions
    ├── faq.md                    FAQ incl. feature gap / growing-positive explanation
    ├── design.md                 design doc
    └── outputs/
        ├── chart_data/           17 CSVs (one per chart dataset)
        ├── charts/               10 PNGs (generated by generate_charts.py)
        ├── viz_coords.csv        UMAP 2D coords for scatter plot
        ├── meeting_themes.csv    clustering output
        ├── phrase_clusters.csv   phrase→cluster assignments
        └── semantic_clusters.json  cluster metadata with LLM labels
```
