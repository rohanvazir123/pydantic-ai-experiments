#!/usr/bin/env bash
# RAG v2 — launch API + frontend
# Usage:  bash start.sh
set -euxo pipefail
cd "$(dirname "$0")"

BOLD='\033[1m'; GREEN='\033[0;32m'; RESET='\033[0m'
ok() { echo -e "  ${GREEN}✓${RESET} $1"; }


# Ensure Docker services are up
docker compose up -d postgres redis >/dev/null 2>&1
ok "Docker services running"

# Ensure Ollama is running
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  ollama serve >/dev/null 2>&1 &
  sleep 3
  ok "Ollama started"
fi

# Kill any previous instances on these ports
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:7200 | xargs kill -9 2>/dev/null || true

# Start API
echo -e "\n${BOLD}Starting API on :8001${RESET}"
uv run uvicorn knowledge.api.app:app --port 8001 --reload \
  > /tmp/rag-api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/rag-api.pid

# Start frontend
echo -e "${BOLD}Starting frontend on :7200${RESET}"
(cd frontend && \
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}" && \
  [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh" && \
  PORT=7200 npm run dev > /tmp/rag-ui.log 2>&1) &
UI_PID=$!
echo $UI_PID > /tmp/rag-ui.pid

# Wait for API to be ready
echo -n "  Waiting for API"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    echo ""; ok "API ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 30 ] && { echo ""; echo "API failed to start:"; tail -20 /tmp/rag-api.log; exit 1; }
done

# Wait for Next.js
echo -n "  Waiting for UI"
for i in $(seq 1 30); do
  if curl -sf http://localhost:7200 >/dev/null 2>&1; then
    echo ""; ok "UI ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 30 ] && { echo ""; echo "UI failed to start — tail /tmp/rag-ui.log"; exit 1; }
done

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI  →  http://localhost:7200"
echo -e "  API →  http://localhost:8001/health"
echo -e "  Logs → tail -f /tmp/rag-api.log"
echo -e "  Stop → kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

# Open browser (macOS)
open http://localhost:7200 2>/dev/null || \
  xdg-open http://localhost:7200 2>/dev/null || \
  echo "  Open http://localhost:7200 in your browser"
