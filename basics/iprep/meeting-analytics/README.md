# Meeting Analytics

> **New session / new machine?** Read the [Project Status](#project-status) section first — it tells you exactly where we are and what to do next. Then follow [Environment Setup](#environment-setup) to get running.

Transcript Intelligence take-home assignment. 100 meeting folders → theme classification → charts → video demo.

---

## Project Status

Last updated: 2026-05-10

### What this is

Take-home assignment brief: `req.md` / `req.pdf`. The goal is to classify themes across 100 meeting transcripts and surface actionable insights for different stakeholders.

Three approaches were built (all complete):

| Approach | Method | Status |
|----------|--------|--------|
| Take A | Rule-based keyword taxonomy | Done |
| Take B | TF-IDF + KMeans clustering | Done |
| **Final Version** | Semantic embeddings + HDBSCAN + LLM labeling | **Done — this is the deliverable** |

### Deliverables status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Chart pipeline | **Done** | 10 PNGs in `final_version/outputs/charts/` |
| Notebook | **Ready — needs a full run** | All cells coded; run top-to-bottom before recording |
| Video demo | **TODO** | 5–10 min screen recording; narrate findings not code |
| Slide deck | **TODO** | 9-slide structure in `INSIGHTS_GUIDE.md`; screenshots from notebook |

### Immediate next steps

1. **Run the notebook top-to-bottom** — open `final_version/meeting_analytics.ipynb`, select the venv kernel, run all cells. Fix any errors before recording.
2. **Record the video** — 5–10 min screen recording of the notebook. Narrate the "so what" for each chart, not the code.
3. **Build the slides** — 9-slide structure is in `INSIGHTS_GUIDE.md`. Pull screenshots from the notebook run.

---

## Key Design Decisions (do not revisit)

- **Embed topic phrases, not full meetings** — finer cluster resolution
- **HDBSCAN not KMeans** — no fixed K, density-adaptive, found 26 clusters naturally
- **UMAP 10-dim before HDBSCAN; separate 2-dim for viz only**
- **LLM is translation, not intelligence** — HDBSCAN clusters; LLM only names them
- **Final Version is fully self-contained** — don't mix with Take A / Take B code

---

## Chart Pipeline

Run from `basics/iprep/meeting-analytics/`:

```bash
source venv/bin/activate
python final_version/generate_charts.py
```

Outputs to `final_version/outputs/charts/` (10 PNGs) and `final_version/outputs/chart_data/` (17 CSVs).

| File | Title | For |
|------|-------|-----|
| `00_dataset_overview.png` | Call type + product breakdown | All Leadership |
| `01_cluster_table.png` | 26 themes table | All Leadership |
| `02_sentiment_by_calltype.png` | Sentiment distribution + avg score | All Leadership |
| `03_theme_sentiment_heatmap.png` | Theme × sentiment heatmap | All Leadership |
| `04_churn_density.png` | Churn signals per meeting by theme | Sales · CS |
| `05_product_signals.png` | Tech issues + churn signals by product | Engineering · Product |
| `06_positive_signals.png` | Praise by product + Comply sentiment | Product · Marketing · Sales · CS |
| `07_detect_external_impact.png` | Detect outage contamination (E3/R4) | Engineering (CTO) · Sales |
| `08_feature_gaps_by_product.png` | Feature gaps × sentiment bucket (P1) | Product (CPO) |
| `09_action_item_owners.png` | Action items by theme + dept × product (S3) | Operations · Engineering · CS · Sales |

**Stakeholder pills:** each chart has a colored `For:` label top-left. Colors: Engineering=dark blue, Product=dark green, Sales=orange, CS=purple, Marketing=teal, Operations=brown, Leadership=charcoal.

**Known gotchas in `generate_charts.py`:**
- Print statements use ASCII only (`-` not `─`) — Windows cp1252 can't encode box-drawing chars
- `matplotlib.use("Agg")` only when `"ipykernel" not in sys.modules`
- `pivot_table(fill_value=0)` returns float → `.astype(int)` required before `fmt="d"` in sns.heatmap

---

## Notebook

**File:** `final_version/meeting_analytics.ipynb`  
**Kernel:** venv (select `venv/bin/python` in Jupyter)  
**Status:** All cells coded — needs a full top-to-bottom run before recording.

### Cell map

| Cell ID | Section | Notes |
|---------|---------|-------|
| `cell_01_setup` | DB connection + `q()` helper | Must run first |
| `625a87b0` | Optional DB regen header | Markdown only |
| `4efc9898` | Step 1 — load base tables | subprocess: `load_raw_jsons_to_db.py --reset` |
| `7ec071f3` | Step 2 — load semantic tables | subprocess: `load_output_csvs_to_db.py --reset` |
| `cell_03_overview` | Dataset overview | Call types + products |
| `cell_05_umap` | UMAP 2D scatter | reads `final_version/outputs/viz_coords.csv` |
| `cell_06_cluster_table` | 26 themes table | styled DataFrame |
| `cell_10_sentiment_calltype` | Sentiment by call type | stacked bars + avg score |
| `cell_12_heatmap` | Theme × sentiment heatmap | `.astype(int)` before `sns.heatmap` — required |
| `cell_14_churn` | Churn density by theme | |
| `cell_15_watchlist` | High-risk account watchlist | 38 meetings |
| `cell_16_product` | Tech issues + churn by product | |
| `cell_17_positive` | Positive signals (Comply) | |
| `622eeaab` | Leadership questions header | Markdown: E3/R4, P1, S3 |
| `4f330762` | E3/R4 — Detect external contamination | stacked bar + pie |
| `12ef1817` | P1 — Feature gaps × sentiment | grouped bar |
| `bf958936` | S3 — Action item owners | Left: by theme · Right: dept × product |
| `d13fdd8c` | Export header | Markdown |
| `5c69f112` | Export cell | subprocess: `generate_charts.py` |

---

## Database Schema

**Connection:** `postgresql://rag_user:rag_pass@localhost:5434/rag_db`  
**Schema:** `meeting_analytics`

### Base tables — loaded by `load_raw_jsons_to_db.py`

| Table | Rows | Key columns |
|-------|------|-------------|
| `meetings` | 100 | meeting_id, title, organizer_email, duration_minutes, start_time |
| `meeting_participants` | 311 | meeting_id, email |
| `meeting_summaries` | 100 | summary_text, overall_sentiment, sentiment_score, topics TEXT[], products TEXT[] |
| `key_moments` | 402 | moment_type (8 types), text, speaker, time_seconds |
| `action_items` | 397 | meeting_id, owner, text |
| `transcript_lines` | 4313 | speaker, sentence, sentiment_type, time_seconds |

### Semantic tables — loaded by `load_output_csvs_to_db.py`

| Table / View | Rows | Key columns |
|--------------|------|-------------|
| `semantic_clusters` | 26 | theme_title (LLM), audience (LLM), rationale (LLM), phrase_count |
| `semantic_phrases` | 343 | canonical phrase, cluster_id, tsvector GIN |
| `semantic_meeting_themes` | 516 | meeting_id, cluster_id, is_primary, call_type, overall_sentiment, products TEXT[] |
| `action_items_by_theme` *(view)* | 397 | meeting_id, owner, action_item, cluster_id, theme_title, audience |

### Rebuild from scratch

```bash
python final_version/load_raw_jsons_to_db.py --reset    # drops schema, creates 6 base tables
python final_version/load_output_csvs_to_db.py --reset  # drops semantic tables, creates 3 + view
```

---

## File Map

```
meeting-analytics/
├── README.md                    this file — start here
├── INSIGHTS_GUIDE.md            insight catalogue, SQL, 9-slide deck structure
├── SESSION_CONTEXT.md           detailed session notes (Windows dev machine)
├── req.md / req.pdf             assignment brief — source of truth for deliverables
├── .env / .env.example          DB + Ollama credentials
├── requirements.txt             pinned Python dependencies
├── setup.sh                     one-shot setup (Mac / Linux)
├── setup.ps1                    one-shot setup (Windows)
├── dataset/                     100 meeting JSON folders (raw input)
├── sql/
│   ├── 01_verify_tables.sql
│   ├── 02_insight_queries.sql
│   └── 03_stakeholder_questions.sql
└── final_version/
    ├── meeting_analytics.ipynb   MAIN NOTEBOOK — run this for the video
    ├── generate_charts.py        DB → CSV → PNG pipeline
    ├── load_raw_jsons_to_db.py   raw JSON → 6 base tables
    ├── load_output_csvs_to_db.py outputs/ CSVs → 3 semantic tables + view
    ├── verify.py                 checks all 9 tables
    ├── semantic_clustering.py    HDBSCAN pipeline (already run — outputs committed)
    ├── nl_questions.md           20 stakeholder questions
    ├── faq.md                    FAQ incl. feature gap / growing-positive explanation
    ├── design.md                 design doc
    └── outputs/
        ├── charts/               10 PNGs (final deliverable)
        ├── chart_data/           17 CSVs
        ├── viz_coords.csv        UMAP 2D coords
        ├── meeting_themes.csv    clustering output
        ├── phrase_clusters.csv   phrase → cluster assignments
        └── semantic_clusters.json
```

---

---

# Environment Setup

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | `brew install python@3.13` |
| Docker Desktop | latest | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |
| Git | any | `brew install git` |

---

## 1. Clone the repo

```bash
git clone https://github.com/rohanvazir123/pydantic-ai-experiments.git
cd pydantic-ai-experiments
```

---

## 2. Run setup (Mac / Linux)

```bash
bash basics/iprep/meeting-analytics/setup.sh
```

This script:
- Creates a Python venv at `meeting-analytics/venv/`
- Installs all pinned dependencies from `requirements.txt`
- Starts the pgvector Docker container on port 5434
- Pulls the two Ollama models (`llama3.1:8b`, `nomic-embed-text:latest`)

**Windows:** use `setup.ps1` instead:
```powershell
cd basics\iprep\meeting-analytics
.\setup.ps1
```

---

## 3. Docker containers

The `docker-compose.yml` at the **repo root** defines three services. For this project you only need `pgvector`.

| Container | Port | Used for | Required? |
|-----------|------|----------|-----------|
| `pgvector` (`rag_pgvector`) | 5434 | Main database — stores meetings, clusters, embeddings | **Yes** |
| `age` (`rag_age`) | 5433 | Apache AGE knowledge graph (separate RAG project) | No |
| `age-viewer` (`rag_age_viewer`) | 3001 | Browser UI for the AGE graph | No |

### Start only what you need

```bash
# From the repo root
docker compose up -d pgvector

# Verify it's healthy
docker ps | grep rag_pgvector
# Expected: "healthy" in STATUS column
```

### Stop / wipe

```bash
docker compose down          # stop, data preserved
docker compose down -v       # WARNING: wipes all data
```

---

## 4. Ollama setup

Ollama runs the LLM and embedding models locally. On Mac it installs as a menu-bar app that auto-starts on login.

### Install

```bash
# Option A — GUI app (recommended for Mac)
# Download from https://ollama.com/download and drag to Applications

# Option B — Homebrew
brew install ollama
```

### Start the server

If you installed the GUI app, Ollama starts automatically and runs in the menu bar. If you used Homebrew:

```bash
ollama serve   # starts server on http://localhost:11434
```

Verify it's running:
```bash
curl http://localhost:11434   # should return: "Ollama is running"
```

### Pull the required models

```bash
ollama pull llama3.1:8b              # LLM — used for cluster labeling (~4.7 GB)
ollama pull nomic-embed-text:latest  # embeddings — 768-dim (~274 MB)
```

This only needs to be done once. Models are cached in `~/.ollama/models/`.

### Verify models are available

```bash
ollama list
# Expected output includes:
#   llama3.1:8b
#   nomic-embed-text:latest
```

### Note on the setup script

`setup.sh` pulls both models automatically if Ollama is already installed. If Ollama wasn't installed when you ran the script, install it now and pull manually with the commands above.

---

## 5. PostgreSQL setup


The database runs **inside the Docker container** — you do not install PostgreSQL separately.

| Setting | Value |
|---------|-------|
| PostgreSQL version | 17 (image: `pgvector/pgvector:pg17`) |
| Extensions | `pgvector` pre-installed |
| Host | `localhost` |
| Port | `5434` |
| Database | `rag_db` |
| Username | `rag_user` |
| Password | `rag_pass` |
| Schema | `meeting_analytics` (created by load scripts) |
| Connection string | `postgresql://rag_user:rag_pass@localhost:5434/rag_db` |

Docker auto-creates the database and user on first start. No manual SQL needed.

### Verify connection

```bash
# psql client on Mac (client tools only — no server needed)
brew install libpq
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

psql "postgresql://rag_user:rag_pass@localhost:5434/rag_db"
```

### Configure .env

```bash
cd basics/iprep/meeting-analytics
cp .env.example .env   # defaults work as-is with the Docker container
```

```env
DATABASE_URL=postgresql://rag_user:rag_pass@localhost:5434/rag_db

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_DIMENSION=768
```

---

## 6. Database browser (optional)

Useful for verifying data after loading and running ad-hoc queries.

### pgcli (recommended — pip installable)

`pgcli` is a terminal PostgreSQL client with autocomplete, syntax highlighting, and pretty-printed output.

```bash
pip install pgcli psycopg-binary   # psycopg-binary bundles libpq — required on Mac without admin rights
```

Connect (use explicit flags — the connection URL form has a parsing bug in pgcli 4.x):

```bash
PGPASSWORD=rag_pass pgcli -h localhost -p 5434 -U rag_user -d rag_db
```

Useful pgcli commands once connected:

```sql
-- List all tables in the schema
\dt meeting_analytics.*

-- Row counts
SELECT schemaname, relname, n_live_tup
FROM pg_stat_user_tables
WHERE schemaname = 'meeting_analytics'
ORDER BY relname;

-- Browse the 26 themes
SELECT cluster_id, theme_title, audience, phrase_count
FROM meeting_analytics.semantic_clusters
ORDER BY phrase_count DESC;

-- Meetings per theme
SELECT sc.theme_title, COUNT(*) AS meetings
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc USING (cluster_id)
WHERE smt.is_primary
GROUP BY sc.theme_title
ORDER BY meetings DESC;
```

Type `\q` to quit.

### DBeaver (alternative — GUI)

Download **DBeaver Community** (free) from [dbeaver.io](https://dbeaver.io/download/).

**Install (no admin rights):** mount the `.dmg` and drag `DBeaver.app` to `~/Applications` (not `/Applications`).

```bash
# Or let Claude do it:
hdiutil attach ~/Downloads/dbeaver-ce-*.dmg -nobrowse
cp -R "/Volumes/DBeaver Community 1/DBeaver.app" ~/Applications/
hdiutil detach "/Volumes/DBeaver Community 1"

open ~/Applications/DBeaver.app
```

**Create connection:**

1. **Database → New Database Connection → PostgreSQL → Next**
2. Fill in:

   | Field | Value |
   |-------|-------|
   | Host | `localhost` |
   | Port | `5434` |
   | Database | `rag_db` |
   | Username | `rag_user` |
   | Password | `rag_pass` |

3. Click **Test Connection**. If Maven can't resolve the driver (common on corporate networks), add it manually:
   - Download the JAR: `curl -L -o ~/Downloads/postgresql-jdbc.jar "https://jdbc.postgresql.org/download/postgresql-42.7.5.jar"`
   - In the error dialog click **Edit Driver** → **Libraries** tab → **Add File** → select `~/Downloads/postgresql-jdbc.jar`
   - Go to the **Settings** tab → set **Class Name** to `org.postgresql.Driver` (type it manually — Find Class may not work)
   - Click **OK** → **Test Connection** again
4. Click **Finish**.
5. Browse: **rag_db → Schemas → meeting_analytics → Tables**

> **Note:** Docker must be running (`docker compose up -d pgvector`) before connecting with either tool.

---

## 7. Run without Jupyter (CLI only)

Everything can be run end-to-end from the terminal. The notebook is only needed for the video recording.

```bash
cd basics/iprep/meeting-analytics
source venv/bin/activate

# Step 1 — load raw meeting JSONs into 6 base tables
python final_version/load_raw_jsons_to_db.py --reset

# Step 2 — load pre-computed clustering outputs into 3 semantic tables
python final_version/load_output_csvs_to_db.py --reset

# Step 3 — verify all 9 tables have correct row counts
python final_version/verify.py

# Step 4 — export chart CSVs from DB then generate 10 PNGs
python final_version/generate_charts.py

# Open the charts folder on Mac
open final_version/outputs/charts/
```

---

## 7a. Regenerating outputs

The `final_version/outputs/` directory has three layers, each produced by a different script:

```
final_version/outputs/
├── semantic_clusters.json    ─┐
├── meeting_themes.csv         │  semantic_clustering.py
├── phrase_clusters.csv        │  (slow — needs Ollama, ~10 min)
├── viz_coords.csv             │  already committed, rarely needed
├── cluster_metrics.json      ─┘
│
├── chart_data/               ─┐  generate_charts.py (default)
│   └── *.csv  (17 files)      │  pulls fresh from DB (~5 sec)
│                             ─┘
└── charts/                   ─┐  generate_charts.py
    └── *.png  (10 files)       │  reads chart_data/ CSVs (~10 sec)
                               ─┘
```

### Regenerate chart CSVs + PNGs (most common)

Run this any time you change the DB or want fresh charts:

```bash
cd basics/iprep/meeting-analytics
source venv/bin/activate
python final_version/generate_charts.py
```

### Regenerate PNGs only (CSVs unchanged)

If you only changed chart styling in `generate_charts.py`:

```bash
python final_version/generate_charts.py --no-export
```

### Regenerate clustering outputs (rare)

Only needed if you change the clustering algorithm or parameters in `semantic_clustering.py`. Requires Ollama running and takes ~10 minutes (embedding + HDBSCAN + LLM calls).

```bash
# Make sure Ollama is running first
curl http://localhost:11434   # should return "Ollama is running"

python final_version/semantic_clustering.py
```

This overwrites `semantic_clusters.json`, `meeting_themes.csv`, `phrase_clusters.csv`, `viz_coords.csv`, and `cluster_metrics.json`. After running, reload the semantic tables into the DB:

```bash
python final_version/load_output_csvs_to_db.py --reset
python final_version/verify.py
python final_version/generate_charts.py
```

 > [!WARNING]
> **Re-running `semantic_clustering.py` is destructive and irreversible.**
> HDBSCAN is stochastic — every run produces different cluster IDs and boundaries.
> LLM labels will also differ. All downstream files (`meeting_themes.csv`, `phrase_clusters.csv`,
> `viz_coords.csv`) and the DB semantic tables will be overwritten and out of sync with any
> prior analysis. Only do this if you intentionally want to redo the clustering from scratch.

---

## 8. Load data into the database

### `load_raw_jsons_to_db.py` — 6 base tables

Reads the 100 meeting folders in `dataset/` and writes the base tables into the `meeting_analytics` schema.

```bash
# Full reset (drop schema + recreate) — use this on a fresh machine
python final_version/load_raw_jsons_to_db.py --reset

# Incremental — keeps existing rows, only inserts new ones
python final_version/load_raw_jsons_to_db.py

# Custom dataset path
python final_version/load_raw_jsons_to_db.py --dataset path/to/dataset --reset
```

**Expected output:** 6 tables — `meetings` (100), `meeting_participants` (311), `meeting_summaries` (100), `key_moments` (402), `action_items` (397), `transcript_lines` (4313).

### `load_output_csvs_to_db.py` — 3 semantic tables

Reads the pre-computed clustering output files from `final_version/outputs/` and writes the semantic tables. These outputs are already committed — you do not need to re-run `semantic_clustering.py`.

```bash
# Full reset — use this on a fresh machine
python final_version/load_output_csvs_to_db.py --reset

# Incremental — idempotent (ON CONFLICT DO UPDATE)
python final_version/load_output_csvs_to_db.py

# Custom outputs directory
python final_version/load_output_csvs_to_db.py --output-dir path/to/outputs --reset
```

**Expected output:** 3 tables — `semantic_clusters` (26), `semantic_phrases` (343), `semantic_meeting_themes` (516) — plus the `action_items_by_theme` view.

### `verify.py` — check all 9 tables

Connects to the DB and checks exact row counts for all tables and the view. Run after loading to confirm everything landed correctly.

```bash
python final_version/verify.py
```

Expected output:
```
* PASS  meetings                        100
* PASS  meeting_participants            311
* PASS  meeting_summaries              100
* PASS  key_moments                    402
* PASS  action_items                   397
* PASS  transcript_lines              4313
* PASS  semantic_clusters              26
* PASS  semantic_phrases              343
* PASS  semantic_meeting_themes        516
* PASS  semantic primary themes (one per meeting)  100
* PASS  action_items_by_theme view    397
ALL PASS  --  11 passed, 0 failed, 0 skipped
```

---

## 9. Generate charts

Pulls data from the DB, exports 17 CSVs, then renders 10 PNGs with stakeholder labels.

```bash
# Default — export fresh CSVs from DB, then generate PNGs
python final_version/generate_charts.py

# Skip DB export — regenerate PNGs from existing CSVs (faster)
python final_version/generate_charts.py --no-export

# Custom paths
python final_version/generate_charts.py \
  --input-dir path/to/chart_data \
  --output-dir path/to/charts

# Custom DB connection
python final_version/generate_charts.py --dsn postgresql://user:pass@host:port/db
```

**Outputs:**
- `final_version/outputs/chart_data/` — 17 CSVs (one per query)
- `final_version/outputs/charts/` — 10 PNGs

```bash
# View on Mac
open final_version/outputs/charts/

# View a specific chart
open final_version/outputs/charts/03_theme_sentiment_heatmap.png
```

---

## 10. Run the notebook

### Install Jupyter and register the venv kernel

```bash
# Activate the venv first (skip if already active)
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1

# Install Jupyter inside the venv
pip install jupyter ipykernel

# Register the venv as a named Jupyter kernel
python -m ipykernel install --user --name meeting-analytics --display-name "Python (meeting-analytics)"
```

You only need to do this once. After registration the kernel persists across Jupyter sessions.

### Launch Jupyter

```bash
# From the meeting-analytics/ directory
jupyter notebook final_version/meeting_analytics.ipynb
```

Jupyter will open in your browser. In the top-right corner select the kernel:
**Kernel → Change kernel → Python (meeting-analytics)**

If the kernel doesn't appear, restart Jupyter after running the `ipykernel install` step above.

### Run the notebook

Run all cells top-to-bottom: **Cell → Run All** (or Shift+Enter through each cell).

**Expected flow:**
1. `cell_01_setup` — connects to DB, defines `q()` helper. If this fails, check Docker is running and `.env` is configured.
2. Steps 1 & 2 cells — reload the database from scratch (optional; skip if data is already loaded and verified).
3. Analysis cells — generate all charts inline.
4. Export cell (`5c69f112`) — runs `generate_charts.py` as a subprocess, writes PNGs to `final_version/outputs/charts/`.

**Gotchas:**
- The heatmap cell (`cell_12_heatmap`) requires `.astype(int)` before `sns.heatmap(fmt="d")` — already in the code, just don't remove it.
- `matplotlib.use("Agg")` is guarded with `if "ipykernel" not in sys.modules` — the guard is there so it doesn't interfere with inline rendering in the notebook.
- If a cell fails mid-run, fix the error and re-run from that cell only — you don't need to restart from the top unless the DB connection cell fails.

---

## Troubleshooting

**Container not healthy**
```bash
docker ps -a | grep pgvector
docker logs rag_pgvector
```

**Ollama not responding**
```bash
ollama serve   # Mac: runs in background after install; re-run if it stopped
```

**venv not active**
```bash
which python   # should show venv/bin/python
source venv/bin/activate
```

**Port 5434 in use**
```bash
lsof -i :5434
```
