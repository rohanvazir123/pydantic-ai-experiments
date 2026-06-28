#!/usr/bin/env bash
# RAG v2 — launch API + frontend
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; }

API_PORT=7100
UI_PORT=7200

# ── Pre-flight checks ────────────────────────────────────────────────────────

echo -e "\n${BOLD}Pre-flight checks${RESET}"

# 1. .env exists
if [ ! -f .env ]; then
  fail ".env not found — run: cp .env.example .env and fill in values"
  exit 1
fi
ok ".env found"

# 2. Docker running
if ! docker info >/dev/null 2>&1; then
  fail "Docker is not running — start Docker Desktop first"
  exit 1
fi
ok "Docker running"

# 3. uv installed
if ! command -v uv >/dev/null 2>&1; then
  fail "uv not found — run: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
ok "uv found"

# 4. .venv exists
if [ ! -d .venv ]; then
  warn ".venv missing — running uv sync..."
  uv sync --extra ingestion --extra observability --extra reranker
fi
ok ".venv ready"

# 5. JWT keys
if [ ! -f infra/keys/private.pem ] || [ ! -f infra/keys/public.pem ]; then
  warn "JWT keys missing — generating..."
  mkdir -p infra/keys
  openssl genrsa -out infra/keys/private.pem 2048 2>/dev/null
  openssl rsa -in infra/keys/private.pem -pubout -out infra/keys/public.pem 2>/dev/null
fi
ok "JWT keys ready"

# ── Start Docker services ────────────────────────────────────────────────────

echo -e "\n${BOLD}Starting Docker services${RESET}"
docker compose up -d postgres redis >/dev/null 2>&1

# Extract ports from .env
PG_PORT=$(grep "^DATABASE_URL" .env | sed 's/.*localhost:\([0-9]*\)\/.*/\1/')
REDIS_PORT=$(grep "^REDIS_URL" .env | sed 's/.*localhost:\([0-9]*\).*/\1/')

# Wait for postgres
echo -n "  Waiting for postgres (:${PG_PORT})"
for i in $(seq 1 20); do
  if nc -z localhost "$PG_PORT" 2>/dev/null; then
    echo ""; ok "Postgres ready on :${PG_PORT}"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 20 ]; then
    echo ""
    fail "Postgres not reachable on :${PG_PORT} — check docker-compose ports vs DATABASE_URL in .env"
    echo "  docker-compose ports: $(docker compose port postgres 5432 2>/dev/null || echo 'unknown')"
    echo "  .env DATABASE_URL port: ${PG_PORT}"
    exit 1
  fi
done

# Wait for redis
echo -n "  Waiting for Redis (:${REDIS_PORT})"
for i in $(seq 1 20); do
  if nc -z localhost "$REDIS_PORT" 2>/dev/null; then
    echo ""; ok "Redis ready on :${REDIS_PORT}"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 20 ]; then
    echo ""
    fail "Redis not reachable on :${REDIS_PORT} — check docker-compose ports vs REDIS_URL in .env"
    echo "  docker-compose ports: $(docker compose port redis 6379 2>/dev/null || echo 'unknown')"
    echo "  .env REDIS_URL port: ${REDIS_PORT}"
    exit 1
  fi
done

# ── Ollama ───────────────────────────────────────────────────────────────────

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama not running — starting..."
  ollama serve >/dev/null 2>&1 &
  sleep 3
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama still not responding — LLM calls will fail (non-fatal)"
  else
    ok "Ollama started"
  fi
else
  ok "Ollama running"
fi

# ── Kill stale processes ─────────────────────────────────────────────────────

lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$UI_PORT  | xargs kill -9 2>/dev/null || true
sleep 1

# ── Start API ────────────────────────────────────────────────────────────────

echo -e "\n${BOLD}Starting API on :${API_PORT}${RESET}"
uv run uvicorn knowledge.api.app:app --host 0.0.0.0 --port "$API_PORT" \
  > /tmp/rag-api.log 2>&1 &
echo $! > /tmp/rag-api.pid

echo -n "  Waiting for API"
for i in $(seq 1 40); do
  if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    echo ""; ok "API ready  →  http://localhost:${API_PORT}/health"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 40 ]; then
    echo ""
    fail "API failed to start. Last 20 lines of log:"
    tail -20 /tmp/rag-api.log
    exit 1
  fi
done

# ── Start frontend ───────────────────────────────────────────────────────────

echo -e "${BOLD}Starting frontend on :${UI_PORT}${RESET}"
(cd frontend && \
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}" && \
  [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh" && \
  nvm use --silent 2>/dev/null || true && \
  PORT=$UI_PORT npm run dev > /tmp/rag-ui.log 2>&1) &
echo $! > /tmp/rag-ui.pid

echo -n "  Waiting for UI"
for i in $(seq 1 40); do
  if curl -sf "http://localhost:${UI_PORT}" >/dev/null 2>&1; then
    echo ""; ok "UI ready   →  http://localhost:${UI_PORT}"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 40 ]; then
    echo ""
    fail "UI failed to start. Last 20 lines of log:"
    tail -20 /tmp/rag-ui.log
    exit 1
  fi
done

# ── Done ─────────────────────────────────────────────────────────────────────

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI   →  http://localhost:${UI_PORT}"
echo -e "  API  →  http://localhost:${API_PORT}/health"
echo -e "  Logs →  tail -f /tmp/rag-api.log"
echo -e "  Stop →  kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open "http://localhost:${UI_PORT}" 2>/dev/null || \
  xdg-open "http://localhost:${UI_PORT}" 2>/dev/null || true
