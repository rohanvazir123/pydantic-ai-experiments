# Session Context — Meeting Analytics
Last updated: 2026-05-10 (session 14)

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

## Machines

| Machine | User | OS | Role |
|---------|------|----|------|
| Rohan's PC | `rohan` | Windows | Original dev machine (conda env, PowerShell) |
| Pramada's Mac | `ppotnis` | macOS (Darwin 25.4) | Current machine — all session 14 work done here |

**Session 14 was entirely on Pramada's Mac.** The Windows conda paths in earlier sessions no longer apply.

---

## Postgres — single source of truth
**Connection:** `localhost:5434` / database `rag_db` / user `rag_user` / password `rag_pass`
**Schema:** `meeting_analytics`
**Credentials file:** `basics/iprep/meeting-analytics/.env`

> Port 5432 = local system Postgres — not used for this project.
> Port 5433 = Apache AGE Docker (separate, unrelated).
> Port 5434 = Docker pgvector — canonical DB for this project.

**WARNING:** MCP postgresql tool connects to port 5432, not 5434. Never use it — always use pgcli or DBeaver at port 5434.

### Tables — verified row counts

**Base tables (6) — `final_version/load_raw_jsons_to_db.py`**

| Table | Rows | What it holds |
|-------|------|---------------|
| `meetings` | 100 | meeting_id, title, organizer_email, duration_minutes, start_time |
| `meeting_participants` | 311 | meeting_id, email |
| `meeting_summaries` | 100 | summary_text, overall_sentiment, sentiment_score, topics TEXT[], products TEXT[] |
| `key_moments` | 402 | moment_type (8 types), text, speaker, time_seconds |
| `action_items` | 397 | meeting_id, owner, text |
| `transcript_lines` | 4313 | speaker, sentence, sentiment_type, time_seconds |

**Semantic tables (3) + 1 view — `final_version/load_output_csvs_to_db.py`**

| Table / View | Rows | What it holds |
|--------------|------|---------------|
| `semantic_clusters` | 26 | theme_title (LLM), audience (LLM), rationale (LLM), phrase_count |
| `semantic_phrases` | 343 | canonical phrase, cluster_id, tsvector GIN |
| `semantic_meeting_themes` | 516 | meeting_id, cluster_id, is_primary, call_type, overall_sentiment, products TEXT[] |
| `action_items_by_theme` *(view)* | 397 | meeting_id, owner, action_item, cluster_id, theme_title, audience |

**Rebuild from scratch:**
```
load_raw_jsons_to_db.py --reset   # drops schema, creates 6 base tables
load_output_csvs_to_db.py --reset  # drops semantic tables, creates 3 + view
```

### Docker
```
docker compose up -d pgvector     # always start this way (named volume = data persists)
docker compose down               # preserves data
docker compose down -v            # WIPES data
```

---

## Environment — Pramada's Mac

**Python:** 3.14.2
**Venv:** `pydantic-ai` venv at `/Users/ppotnis/Documents/rovaz/pydantic-ai-experiments/pydantic-ai/`
- This venv is shared with the main RAG project — do NOT install packages that downgrade or conflict with existing ones
- Check `pip list` before installing anything new

**Packages installed for meeting-analytics (in addition to RAG project packages):**
- `asyncpg`, `pgvector`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nest-asyncio`
- `pgcli`, `psycopg-binary` (psycopg-binary bundles libpq — required on Mac without admin rights)
- `openai`, `rapidfuzz`, `pydantic`, `python-dotenv`

**No admin rights on this machine** — cannot run `sudo`. Xcode license not accepted, so `brew install` is blocked.

---

## Chart pipeline — COMPLETE

```
DB (localhost:5434)
  └─ generate_charts.py  (default: exports CSVs then PNGs)
       ├─ final_version/outputs/chart_data/   17 CSVs
       └─ final_version/outputs/charts/       10 PNGs
```

**Run from `basics/iprep/meeting-analytics/`:**
```bash
python final_version/generate_charts.py
```

**Charts produced:**

| File | Title | Stakeholder departments |
|------|-------|------------------------|
| `00_dataset_overview.png` | Call type + product breakdown | All Leadership |
| `01_cluster_table.png` | 26 themes table | All Leadership |
| `02_sentiment_by_calltype.png` | Sentiment distribution + avg score | All Leadership |
| `03_theme_sentiment_heatmap.png` | Theme × sentiment heatmap | All Leadership |
| `04_churn_density.png` | Churn signals per meeting by theme | Sales · CS |
| `05_product_signals.png` | Tech issues + churn signals by product | Engineering · Product |
| `06_positive_signals.png` | Praise by product + Comply external sentiment | Product · Marketing · Sales & CS |
| `07_detect_external_impact.png` | E3/R4 — Detect outage contamination | Engineering (CTO) · Sales |
| `08_feature_gaps_by_product.png` | P1 — Feature gaps × sentiment bucket | Product (CPO) |
| `09_action_item_owners.png` | S3 — Action items by theme + by dept × product | Operations · Engineering · CS · Sales |

---

## Database browsers

### pgcli
**Connect (use explicit flags — URL form has a parsing bug in pgcli 4.x):**
```bash
PGPASSWORD=rag_pass pgcli -h localhost -p 5434 -U rag_user -d rag_db
```

### DBeaver
- Installed at: `~/Applications/DBeaver.app` (no admin rights — dragged from DMG)
- JDBC driver: `~/Downloads/postgresql-jdbc.jar` (downloaded manually — Maven blocked on corporate network)
- Class name set manually: `org.postgresql.Driver` (Find Class doesn't work)
- Connection: host `localhost`, port `5434`, database `rag_db`, user `rag_user`, password `rag_pass`

---

## Git

**Remote:** `https://github.com/rohanvazir123/pydantic-ai-experiments.git`
**Push access:** via Rohan's PAT embedded in remote URL
```bash
git remote set-url origin https://rohanvazir123:<token>@github.com/rohanvazir123/pydantic-ai-experiments.git
git push origin main
```
- Token is tied to Rohan's GitHub account (not this machine)
- `ppotnis_paypal` does not have direct push access — must use Rohan's PAT

---

## Notebook — `final_version/meeting_analytics.ipynb`

**Status: NOT YET RUN this session — needs a full top-to-bottom run before video.**
**Kernel:** registered as "Python (meeting-analytics)" pointing to the pydantic-ai venv

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
| `bf958936` | S3 — Action item owners | Left: by theme (no owner names) · Right: dept × product stacked bars |
| `d13fdd8c` | Export header | Markdown |
| `5c69f112` | Export cell | subprocess `generate_charts.py` |

### Launch Jupyter
```bash
cd basics/iprep/meeting-analytics
jupyter notebook final_version/meeting_analytics.ipynb
# Select kernel: Python (meeting-analytics)
```

---

## Deliverables status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Notebook | **READY — needs run** | All sections coded; run top-to-bottom before video |
| Chart pipeline | **DONE** | 10 PNGs in `final_version/outputs/charts/` |
| Video demo | **TODO** | 5–10 min; narrate findings not code; record after notebook run |
| Slide deck | **TODO** | 9-slide structure in `INSIGHTS_GUIDE.md`; screenshots from notebook |

---

## Immediate next steps

1. **Run notebook top to bottom** — kernel: Python (meeting-analytics); fix any errors before recording
2. **Record video** — 5–10 min screen recording, narrate the "so what" for each chart
3. **Build slides** — 9-slide structure from `INSIGHTS_GUIDE.md`; pull screenshots from notebook

---

## Key design decisions (don't revisit)
- Embed topic phrases (not full meetings) — finer cluster resolution
- HDBSCAN not KMeans — no fixed K, density-adaptive, found 26 naturally
- UMAP 10-dim before HDBSCAN; separate 2-dim for viz only
- Final Version is fully self-contained
- LLM is translation not intelligence — HDBSCAN clusters; LLM only names
- Human inspection required after schema/data changes — verify via pgcli or DBeaver @ port 5434, not MCP tool

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── INSIGHTS_GUIDE.md            insight catalogue, SQL, 9-slide deck structure
├── README.md                    full setup instructions (Mac/Linux + Windows)
├── req.md / req.pdf             source of truth for deliverables
├── .env                         PG credentials (PG_* vars + DATABASE_URL)
├── .env.example                 template
├── requirements.txt             pinned Python dependencies (includes pgcli)
├── dataset/                     100 meeting folders
├── sql/
│   ├── 01_verify_tables.sql
│   ├── 02_insight_queries.sql
│   └── 03_stakeholder_questions.sql
└── final_version/
    ├── meeting_analytics.ipynb   MAIN NOTEBOOK — run this for the video
    ├── generate_charts.py        DB->CSV->PNG pipeline (17 CSVs, 10 PNGs)
    ├── load_raw_jsons_to_db.py   raw JSON -> 6 base tables
    ├── load_output_csvs_to_db.py outputs/ CSVs -> 3 semantic tables + view
    ├── verify.py                 checks all 9 tables
    ├── INSIGHTS_GUIDE.md
    └── outputs/
        ├── chart_data/           17 CSVs
        ├── charts/               10 PNGs (with colored dept pills)
        ├── viz_coords.csv        UMAP 2D coords (DO NOT DELETE — source data)
        ├── meeting_themes.csv    clustering output (DO NOT DELETE — source data)
        ├── phrase_clusters.csv   phrase->cluster assignments (DO NOT DELETE — source data)
        └── semantic_clusters.json (DO NOT DELETE — source data)
```
