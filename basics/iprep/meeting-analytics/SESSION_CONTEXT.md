# Session Context — Meeting Analytics
Last updated: 2026-05-10 (session 14)

## How to reload this session
Tell Claude: "Read basics/iprep/meeting-analytics/SESSION_CONTEXT.md and pick up where we left off."

---

## What we are building
Transcript Intelligence take-home assignment — see `req.md` for the full brief.
100 meeting folders in `dataset/`. Three approaches to theme classification:
  - **Approach 1**: rule-based keyword taxonomy → DONE
  - **Approach 2**: TF-IDF + KMeans clustering → DONE
  - **Final Version**: semantic embedding + HDBSCAN + LLM labeling → DONE

---

## Machine

macOS (Darwin 25.4)

---

## Postgres — single source of truth
**Connection:** `localhost:5434` / database `rag_db` / user `rag_user` / password `rag_pass`
**Schema:** `meeting_analytics`
**Credentials file:** `basics/iprep/meeting-analytics/.env`

> Port 5434 = Docker pgvector — canonical DB for this project.
> MCP postgresql tool connects to port 5432 — never use it.

### Tables — verified row counts

| Table | Rows |
|-------|------|
| `meetings` | 100 |
| `meeting_participants` | 311 |
| `meeting_summaries` | 100 |
| `key_moments` | 402 |
| `action_items` | 397 |
| `transcript_lines` | 4313 |
| `semantic_clusters` | 26 |
| `semantic_phrases` | 343 |
| `semantic_meeting_themes` | 516 |
| `action_items_by_theme` *(view)* | 397 |

**Rebuild from scratch:**
```bash
python final_version/load_raw_jsons_to_db.py --reset
python final_version/load_output_csvs_to_db.py --reset
```

### Docker
```bash
docker compose up -d pgvector   # always use this — named volume preserves data
docker compose down             # preserves data
docker compose down -v          # WIPES data
```

Container runtime: **Rancher Desktop** (not Docker Desktop)
```bash
open "/Applications/Rancher Desktop.app"   # must be running before docker commands
```

---

## Environment

**Python:** 3.14.2
**Venv:** `pydantic-ai` venv at `pydantic-ai/bin/python`
- Shared with the main RAG project — do NOT downgrade or conflict with existing packages
- Check `pip list` before installing anything new

**Key packages installed for meeting-analytics:**
`asyncpg`, `pgvector`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nest-asyncio`,
`pgcli`, `psycopg`, `psycopg-binary`, `openai`, `rapidfuzz`, `pydantic`, `python-dotenv`,
`python-pptx`, `pypandoc`

**No admin rights** — `sudo` and `brew install` are blocked. Use `pip install` for everything.

**Git push fix:** `git config --global http.postBuffer 524288000` — required to push large commits.

---

## Database browsers

### pgcli
```bash
PGPASSWORD=rag_pass pgcli -h localhost -p 5434 -U rag_user -d rag_db
```
Note: URL form (`pgcli postgresql://...`) has a parsing bug — use explicit flags.

### DBeaver
- Installed at: `~/Applications/DBeaver.app`
- JDBC driver: `~/Downloads/postgresql-jdbc.jar` (downloaded manually — Maven blocked)
- Class name set manually: `org.postgresql.Driver`
- Connection: host `localhost`, port `5434`, database `rag_db`, user `rag_user`, password `rag_pass`

---

## Git

**Remote:** `https://github.com/rohanvazir123/pydantic-ai-experiments.git`
**Push:** via PAT embedded in remote URL
```bash
git remote set-url origin https://rohanvazir123:<token>@github.com/rohanvazir123/pydantic-ai-experiments.git
git push origin main
```
**Commit author:** always use `--author="Ro Vaz <rohanvazir123@users.noreply.github.com>"` for meeting-analytics commits.
**HTTP buffer:** `git config --global http.postBuffer 524288000` — needed for large pushes.

### .gitignore — do NOT track these
```
basics/iprep/meeting-analytics/final_version/meeting_analytics.html
basics/iprep/meeting-analytics/final_version/meeting_analytics.pptx
basics/iprep/meeting-analytics/final_version/outputs_notebook/
```
These are generated files — rebuild locally with the notebook or `build_pptx.py`.

---

## CLI pipeline (no Jupyter needed)

```bash
cd basics/iprep/meeting-analytics

python final_version/load_raw_jsons_to_db.py --reset    # load 6 base tables
python final_version/load_output_csvs_to_db.py --reset  # load 3 semantic tables
python final_version/verify.py                          # 11/11 checks
python final_version/generate_charts.py                 # 10 PNGs to outputs/charts/
```

---

## Notebook — `final_version/meeting_analytics.ipynb`

**Status: COMPLETE — self-contained, runs top to bottom**
**Kernel:** registered as "Python (meeting-analytics)" → `pydantic-ai` venv

### What it does
1. Connects to DB, defines `q()`, `save()`, `label_bars()`, `stakeholder()`, `chart_header()` helpers
2. Defines helpers then loads all raw data into Postgres inline (no subprocess calls)
3. Verifies all 9 tables
4. Shows Chart Summary table with clickable links
5. Renders 11 charts across sections 3–7, saving PNGs to `outputs_notebook/`

### Key design decisions in notebook
- Uses `psycopg` (sync) not `asyncpg` — avoids Python 3.14 asyncio.timeout issues
- `outputs_notebook/` is wiped on every setup cell run — charts always fresh
- `chart_header(audience, title)` renders HTML header before table outputs
- `stakeholder(ax, label)` stamps dark pill on chart figures
- All chart text uses ASCII only (no `—`, `·`) — matplotlib font compatibility

### Launch Jupyter
```bash
cd basics/iprep/meeting-analytics
jupyter notebook final_version/meeting_analytics.ipynb
# Select kernel: Python (meeting-analytics)
```

### Smoke test (headless)
```bash
jupyter nbconvert --to notebook --execute final_version/meeting_analytics.ipynb \
  --output-dir final_version/ --ExecutePreprocessor.timeout=180
```

---

## Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Notebook | **DONE** | `final_version/meeting_analytics.ipynb` |
| Chart pipeline (CLI) | **DONE** | `final_version/generate_charts.py` → `outputs/charts/` |
| Charts (notebook) | **DONE** | `final_version/outputs_notebook/` (local only, gitignored) |
| PPTX slide deck | **DONE** | `final_version/meeting_analytics.pptx` (local only, gitignored) |
| HTML export | **DONE** | `final_version/meeting_analytics.html` (local only, gitignored) |
| Narration guide | **DONE** | `final_version/narration.md` |
| Video demo | **TODO** | Record screen walkthrough of notebook using narration.md |

---

## Immediate next steps

1. **Record video** — open Jupyter, run cells one at a time with Shift+Enter, narrate from `narration.md`
2. **Submit** — video + notebook + PPTX

---

## Key design decisions (don't revisit)
- Embed topic phrases (not full meetings) — finer cluster resolution
- HDBSCAN not KMeans — no fixed K, density-adaptive, found 26 naturally
- UMAP 10-dim before HDBSCAN; separate 2-dim for viz only
- LLM is translation not intelligence — HDBSCAN clusters; LLM only names
- Two-layer DB schema — base tables from JSON, semantic tables from clustering outputs
- `psycopg` sync not asyncpg — Python 3.14 compatibility
- Human inspection required after data changes — verify via pgcli or DBeaver @ port 5434, not MCP tool

---

## File map

```
meeting-analytics/
├── SESSION_CONTEXT.md
├── README.md                    full setup instructions
├── req.md / req.pdf             assignment brief
├── .env                         PG credentials (PG_* vars + DATABASE_URL)
├── .env.example                 template
├── requirements.txt             pinned Python dependencies
└── final_version/
    ├── meeting_analytics.ipynb   MAIN NOTEBOOK
    ├── meeting_analytics.pptx    SLIDE DECK (local only — gitignored)
    ├── meeting_analytics.html    HTML export (local only — gitignored)
    ├── build_pptx.py             script to regenerate the PPTX
    ├── narration.md              slide-by-slide speaking notes
    ├── generate_charts.py        DB -> CSV -> PNG pipeline
    ├── load_raw_jsons_to_db.py   raw JSON -> 6 base tables
    ├── load_output_csvs_to_db.py outputs/ CSVs -> 3 semantic tables + view
    ├── verify.py                 checks all 9 tables (11 assertions)
    ├── INSIGHTS_GUIDE.md         full insight catalogue
    ├── data_model.png            DBeaver ER diagram
    └── outputs/
        ├── charts/               10 PNGs (CLI pipeline output)
        ├── chart_data/           17 CSVs
        ├── outputs_notebook/     notebook chart PNGs (local only — gitignored)
        ├── viz_coords.csv        UMAP 2D coords (DO NOT DELETE)
        ├── meeting_themes.csv    clustering output (DO NOT DELETE)
        ├── phrase_clusters.csv   phrase->cluster assignments (DO NOT DELETE)
        └── semantic_clusters.json (DO NOT DELETE)
```
