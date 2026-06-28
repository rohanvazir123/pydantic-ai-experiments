#!/usr/bin/env bash
# RAG v2 — launch API + frontend
# Usage:  bash start.sh
set -euo pipefail
cd "$(dirname "$0")"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; exit 1; }

# ── Pre-flight ────────────────────────────────────────────────────────────────

echo -e "\n${BOLD}Pre-flight checks${RESET}"

[ -f .env ]               || fail ".env missing — cp .env.example .env and fill in values"
ok ".env found"

docker info >/dev/null 2>&1 || fail "Docker not running — start Docker Desktop first"
ok "Docker running"

command -v uv >/dev/null 2>&1 || fail "uv not found — curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv found"

[ -d .venv ] || { warn ".venv missing — running uv sync..."; uv sync --extra ingestion --extra observability --extra reranker --extra audio; }
ok ".venv ready"

[ -f infra/keys/private.pem ] || {
  warn "JWT keys missing — generating..."
  mkdir -p infra/keys
  openssl genrsa -out infra/keys/private.pem 2048 2>/dev/null
  openssl rsa -in infra/keys/private.pem -pubout -out infra/keys/public.pem 2>/dev/null
}
ok "JWT keys ready"

# ── Docker services ───────────────────────────────────────────────────────────

echo -e "\n${BOLD}Starting Docker services${RESET}"
docker compose up -d postgres redis >/dev/null 2>&1

PG_PORT=$(grep    "^DATABASE_URL" .env | sed 's/.*localhost:\([0-9]*\)\/.*/\1/')
REDIS_PORT=$(grep "^REDIS_URL"    .env | sed 's/.*localhost:\([0-9]*\).*/\1/')

echo -n "  Waiting for postgres (:${PG_PORT})"
for i in $(seq 1 20); do
  nc -z localhost "$PG_PORT" 2>/dev/null && { echo ""; ok "Postgres ready on :${PG_PORT}"; break; }
  echo -n "."; sleep 1
  [ "$i" -eq 20 ] && { echo ""; fail "Postgres not reachable on :${PG_PORT} — docker-compose port: $(docker compose port postgres 5432 2>/dev/null), .env port: ${PG_PORT}"; }
done

echo -n "  Waiting for Redis (:${REDIS_PORT})"
for i in $(seq 1 20); do
  nc -z localhost "$REDIS_PORT" 2>/dev/null && { echo ""; ok "Redis ready on :${REDIS_PORT}"; break; }
  echo -n "."; sleep 1
  [ "$i" -eq 20 ] && { echo ""; fail "Redis not reachable on :${REDIS_PORT} — docker-compose port: $(docker compose port redis 6379 2>/dev/null), .env port: ${REDIS_PORT}"; }
done

# ── DB schemas + seed ─────────────────────────────────────────────────────────

echo -e "\n${BOLD}Checking database${RESET}"
CHUNK_COUNT=$(uv run python -c "
import asyncio, asyncpg, os
from dotenv import load_dotenv; load_dotenv()
async def run():
    try:
        conn = await asyncpg.connect(os.environ['DATABASE_URL'])
        n = await conn.fetchval('SELECT COUNT(*) FROM chunks')
        await conn.close()
        print(n)
    except Exception:
        print(0)
asyncio.run(run())
" 2>/dev/null | tail -1)

if [ "${CHUNK_COUNT:-0}" -eq 0 ]; then
  warn "No chunks found — applying schemas and seeding..."
  uv run python -c "
import asyncio, asyncpg, os, glob
from dotenv import load_dotenv; load_dotenv()
async def run():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])
    for f in sorted(glob.glob('schema/*.sql')):
        print(f'  Applying {f}...')
        await conn.execute(open(f).read())
    await conn.close()
asyncio.run(run())
"
  uv run python scripts/seed.py
else
  ok "${CHUNK_COUNT} chunks in DB"
fi

# ── Ollama ────────────────────────────────────────────────────────────────────

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama not running — starting..."
  ollama serve >/dev/null 2>&1 &
  sleep 3
  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && ok "Ollama started" || warn "Ollama still not responding — LLM calls will fail"
else
  ok "Ollama running"
fi

# ── Start services ────────────────────────────────────────────────────────────

lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:7200 | xargs kill -9 2>/dev/null || true
sleep 1

echo -e "\n${BOLD}Starting API on :8001${RESET}"
uv run uvicorn knowledge.api.app:app --port 8001 --reload > /tmp/rag-api.log 2>&1 &
echo $! > /tmp/rag-api.pid

echo -e "${BOLD}Starting frontend on :7200${RESET}"
(cd frontend && \
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}" && \
  [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh" && \
  PORT=7200 npm run dev > /tmp/rag-ui.log 2>&1) &
echo $! > /tmp/rag-ui.pid

echo -n "  Waiting for API"
for i in $(seq 1 40); do
  curl -sf http://localhost:8001/health >/dev/null 2>&1 && { echo ""; ok "API ready  →  http://localhost:8001/health"; break; }
  echo -n "."; sleep 1
  [ "$i" -eq 40 ] && { echo ""; echo "API failed:"; tail -20 /tmp/rag-api.log; exit 1; }
done

echo -n "  Waiting for UI"
for i in $(seq 1 40); do
  curl -sf http://localhost:7200 >/dev/null 2>&1 && { echo ""; ok "UI ready   →  http://localhost:7200"; break; }
  echo -n "."; sleep 1
  [ "$i" -eq 40 ] && { echo ""; echo "UI failed:"; tail -20 /tmp/rag-ui.log; exit 1; }
done

# ── Done ──────────────────────────────────────────────────────────────────────

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI   →  http://localhost:7200"
echo -e "  API  →  http://localhost:8001/health"
echo -e "  Logs →  tail -f /tmp/rag-api.log | tail -f /tmp/rag-ui.log"
echo -e "  Stop →  kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open http://localhost:7200 2>/dev/null || xdg-open http://localhost:7200 2>/dev/null || true
