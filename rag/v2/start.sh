#!/usr/bin/env bash
# RAG v2 — launch API + frontend
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }

API_PORT=7100
UI_PORT=7200

docker compose up -d postgres redis >/dev/null 2>&1
ok "Docker services running"

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama not running — starting..."
  ollama serve >/dev/null 2>&1 &
  sleep 3
fi

lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$UI_PORT  | xargs kill -9 2>/dev/null || true
sleep 1

echo -e "\n${BOLD}Starting API on :${API_PORT}${RESET}"
uv run uvicorn knowledge.api.app:app --host 0.0.0.0 --port "$API_PORT" \
  > /tmp/rag-api.log 2>&1 &
echo $! > /tmp/rag-api.pid

echo -e "${BOLD}Starting frontend on :${UI_PORT}${RESET}"
# Load nvm so the subshell gets Node 20 (required by Vite 8)
(cd frontend && \
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}" && \
  [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh" && \
  nvm use --silent 2>/dev/null || true && \
  PORT=$UI_PORT npm run dev > /tmp/rag-ui.log 2>&1) &
echo $! > /tmp/rag-ui.pid

echo -n "  Waiting for API"
for i in $(seq 1 40); do
  if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    echo ""; ok "API ready  →  http://localhost:${API_PORT}/health"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 40 ]; then
    echo ""; echo "API failed. Log:"; tail -20 /tmp/rag-api.log; exit 1
  fi
done

echo -n "  Waiting for UI"
for i in $(seq 1 40); do
  if curl -sf "http://localhost:${UI_PORT}" >/dev/null 2>&1; then
    echo ""; ok "UI ready   →  http://localhost:${UI_PORT}"; break
  fi
  echo -n "."; sleep 1
  if [ "$i" -eq 40 ]; then
    echo ""; echo "UI failed. Log:"; tail -20 /tmp/rag-ui.log; exit 1
  fi
done

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI   →  http://localhost:${UI_PORT}"
echo -e "  API  →  http://localhost:${API_PORT}/health"
echo -e "  Logs →  tail -f /tmp/rag-api.log"
echo -e "  Stop →  kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open "http://localhost:${UI_PORT}" 2>/dev/null || \
  xdg-open "http://localhost:${UI_PORT}" 2>/dev/null || true
