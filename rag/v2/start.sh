#!/usr/bin/env bash
# RAG v2 — launch API + frontend with pre-flight checks and sanity Q&A
# Usage:  bash start.sh
set -euo pipefail
cd "$(dirname "$0")"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✗${RESET}  $1"; exit 1; }
step() { echo -e "\n${BOLD}$1${RESET}"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   RAG v2 — Start                     ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  Logs →  tail -f /tmp/rag-api.log   (API)"
echo -e "          tail -f /tmp/rag-ui.log    (frontend)"
echo -e "          docker compose logs -f      (postgres / redis)"
echo ""

# ── 1. .env ───────────────────────────────────────────────────────────────────
step "1. Environment"
if [ ! -f ".env" ]; then
  cp .env.example .env
  warn ".env not found — copied from .env.example (fill in values then re-run)"
fi
ok ".env present (loaded by Python via dotenv)"

# ── 2. JWT keys ───────────────────────────────────────────────────────────────
step "2. JWT keys"
if [ ! -f "infra/keys/private.pem" ] || [ ! -f "infra/keys/public.pem" ]; then
  warn "RSA keys missing — generating now"
  mkdir -p infra/keys
  openssl genrsa -out infra/keys/private.pem 2048 2>/dev/null
  openssl rsa -in infra/keys/private.pem -pubout -out infra/keys/public.pem 2>/dev/null
fi
ok "JWT keys present"

# ── 3. Docker services ────────────────────────────────────────────────────────
step "3. Docker services"
docker compose up -d postgres redis >/dev/null 2>&1
ok "postgres + redis up"

# ── 4. Python venv ────────────────────────────────────────────────────────────
step "4. Python venv"
if [ ! -d ".venv" ]; then
  warn ".venv missing — running uv sync"
fi
uv sync --extra all 2>&1 | tail -3
uv run python -m spacy info en_core_web_sm >/dev/null 2>&1 || \
  uv run python -m spacy download en_core_web_sm 2>&1 | tail -2
ok "venv ready"

# ── 5. DB schemas ─────────────────────────────────────────────────────────────
step "5. Database"

# Wait for PostgreSQL to be ready (up to 15 s) before querying.
# docker compose up -d returns immediately; the container may still be starting.
PG_READY=0
echo -n "  Waiting for postgres"
for i in $(seq 1 15); do
  if docker compose exec -T postgres pg_isready -q 2>/dev/null; then
    echo ""; PG_READY=1; break
  fi
  echo -n "."; sleep 1
done
if [ "$PG_READY" -eq 0 ]; then
  echo ""
  warn "PostgreSQL not ready after 15 s — check: docker compose logs postgres"
fi

CHUNK_COUNT=$(uv run python - 2>/dev/null <<'PYEOF' | tail -1
import asyncio, asyncpg, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)
async def run():
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"], timeout=5)
        n = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        await conn.close()
        print(n)
    except Exception:
        print(0)
asyncio.run(run())
PYEOF
) || CHUNK_COUNT=0

if [ "${CHUNK_COUNT:-0}" -eq 0 ]; then
  warn "No chunks in DB — ingest sample docs with:"
  echo  "          make seed                   (runs in foreground, ~2 min)"
  echo  "          tail -f /tmp/rag-seed.log   (progress — if running in bg)"
else
  ok "${CHUNK_COUNT} chunks in DB"
fi

# ── 6. Kill stale processes ───────────────────────────────────────────────────
step "6. Clearing ports"
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:7200 | xargs kill -9 2>/dev/null || true
ok "Ports 8001 and 7200 clear"

# ── 7. Start API ──────────────────────────────────────────────────────────────
step "7. Starting API (:8001)"
uv run uvicorn knowledge.api.app:app --port 8001 --reload \
  > /tmp/rag-api.log 2>&1 &
echo $! > /tmp/rag-api.pid

echo -n "  Waiting for API"
API_UP=0
for i in $(seq 1 45); do
  if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    echo ""; API_UP=1; break
  fi
  echo -n "."; sleep 1
done
if [ "$API_UP" -eq 0 ]; then
  echo ""
  fail "API did not start — run: tail -50 /tmp/rag-api.log"
fi
ok "API running"

# ── 8. Start frontend ─────────────────────────────────────────────────────────
step "8. Starting frontend (:7200)"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
(cd frontend && PORT=7200 npm run dev > /tmp/rag-ui.log 2>&1) &
echo $! > /tmp/rag-ui.pid

echo -n "  Waiting for UI"
UI_UP=0
for i in $(seq 1 45); do
  if curl -sf http://localhost:7200 >/dev/null 2>&1; then
    echo ""; UI_UP=1; break
  fi
  echo -n "."; sleep 1
done
if [ "$UI_UP" -eq 0 ]; then
  echo ""
  warn "UI did not respond on :7200 — run: tail -50 /tmp/rag-ui.log"
fi

# ── 9. Sanity check ───────────────────────────────────────────────────────────
step "9. Sanity check"
uv run python - <<'PYEOF' 2>/dev/null
import asyncio, os
from dotenv import load_dotenv
load_dotenv(".env", override=False)

QUERY      = "What does NeuralFlow AI do?"
CORPUS_ID  = "default"
TENANT_ID  = "default"
OK   = "\033[0;32m✓\033[0m"
WARN = "\033[0;33m⚠\033[0m"
FAIL = "\033[0;31m✗\033[0m"

async def run() -> None:
    from knowledge.config.settings import load_settings
    from knowledge.ingestion.embedder import Embedder
    from knowledge.store.vector import PostgresHybridStore

    settings = load_settings()
    vs       = PostgresHybridStore(settings=settings)
    embedder = Embedder(settings=settings)
    await vs.initialize()

    # ── a. tsvector (BM25 full-text) ─────────────────────────────────────────
    try:
        rows = await vs.text_search(QUERY, CORPUS_ID, TENANT_ID, k=3)
        if rows:
            top = rows[0]["content"][:80].replace("\n", " ")
            print(f"  {OK} tsvector  — {len(rows)} hit(s)  top: \"{top}…\"")
        else:
            print(f"  {WARN}  tsvector  — 0 hits (corpus empty? try: make seed)")
    except Exception as e:
        print(f"  {FAIL} tsvector  — {e}")

    # ── b. pgvector (ANN cosine) ──────────────────────────────────────────────
    try:
        embedding = await embedder.embed(QUERY)
        rows = await vs.semantic_search(embedding, CORPUS_ID, TENANT_ID, k=3)
        if rows:
            top_score = rows[0]["score"]
            print(f"  {OK} pgvector  — {len(rows)} hit(s)  top score: {top_score:.4f}")
        else:
            print(f"  {WARN}  pgvector  — 0 hits (embeddings missing? try: make seed)")
    except Exception as e:
        print(f"  {FAIL} pgvector  — {e}")

    # ── c. hybrid (RRF) via API ───────────────────────────────────────────────
    try:
        import httpx
        from knowledge.api.routes.auth import make_dev_token
        token = make_dev_token(ttl=300)
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "http://localhost:8001/api/v2/search",
                json={"query": QUERY, "corpus_ids": [CORPUS_ID]},
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code == 200:
            results = r.json().get("data", {}).get("results", [])
            if results:
                print(f"  {OK} hybrid    — {len(results)} hit(s) via API (RRF)")
            else:
                print(f"  {WARN}  hybrid    — 0 hits via API")
        else:
            print(f"  {WARN}  hybrid    — API returned HTTP {r.status_code}")
    except Exception as e:
        print(f"  {FAIL} hybrid    — {e}")

    await vs.close()

asyncio.run(run())
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ✓ RAG v2 is running${RESET}"
echo -e "  UI   →  http://localhost:7200"
echo -e "  API  →  http://localhost:8001/health"
echo -e "  Logs →  tail -f /tmp/rag-api.log"
echo -e "  Stop →  kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open http://localhost:7200 2>/dev/null || true
