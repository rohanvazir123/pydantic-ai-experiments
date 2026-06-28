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
ok "venv ready"

# ── 5. DB schemas ─────────────────────────────────────────────────────────────
step "5. Database"
CHUNK_COUNT=$(uv run python - 2>/dev/null <<'PYEOF' | tail -1
import asyncio, asyncpg, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)
async def run():
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        n = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        await conn.close()
        print(n)
    except Exception:
        print(0)
asyncio.run(run())
PYEOF
) || CHUNK_COUNT=0

if [ "${CHUNK_COUNT:-0}" -eq 0 ]; then
  warn "No chunks in DB — applying schemas and seeding sample documents"
  uv run python - <<'PYEOF' 2>&1 | grep -v "^$"
import asyncio, asyncpg, os, glob
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)
async def run():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    for f in sorted(glob.glob("schema/*.sql")):
        await conn.execute(open(f).read())
    await conn.close()
asyncio.run(run())
PYEOF
  uv run python scripts/seed.py --force 2>&1 | tail -10
  CHUNK_COUNT=$(uv run python - 2>/dev/null <<'PYEOF' | tail -1
import asyncio, asyncpg, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)
async def run():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    n = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    await conn.close()
    print(n)
asyncio.run(run())
PYEOF
  ) || CHUNK_COUNT=0
  ok "${CHUNK_COUNT} chunks seeded"
else
  ok "${CHUNK_COUNT} chunks already in DB"
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

# ── 9. Sanity Q&A ─────────────────────────────────────────────────────────────
step "9. Sanity check"
SANITY=$(uv run python - <<'PYEOF' 2>/dev/null
import asyncio, os, sys

async def run():
    try:
        import httpx
        from knowledge.api.routes.auth import make_dev_token
        token = make_dev_token(ttl=300)
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "http://localhost:8001/api/v2/search",
                json={"query": "What does NeuralFlow AI do?", "corpus_ids": ["default"]},
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code == 200:
            data = r.json()
            results = data.get("data", {}).get("results", [])
            print(f"OK:{len(results)}")
        else:
            print(f"FAIL:{r.status_code}")
    except Exception as e:
        print(f"ERR:{e}")

asyncio.run(run())
PYEOF
) || SANITY="ERR:sanity-script-failed"

case "$SANITY" in
  OK:0)
    warn "Search returned 0 results — corpus may be empty; try: make seed" ;;
  OK:*)
    COUNT="${SANITY#OK:}"
    ok "Search sanity passed — got ${COUNT} result(s) for test query" ;;
  FAIL:*)
    warn "Search returned HTTP ${SANITY#FAIL:} — check API logs" ;;
  ERR:*)
    warn "Sanity check error: ${SANITY#ERR:}" ;;
  *)
    warn "Unexpected sanity output: ${SANITY}" ;;
esac

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ✓ RAG v2 is running${RESET}"
echo -e "  UI   →  http://localhost:7200"
echo -e "  API  →  http://localhost:8001/health"
echo -e "  Logs →  tail -f /tmp/rag-api.log"
echo -e "  Stop →  kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open http://localhost:7200 2>/dev/null || true
