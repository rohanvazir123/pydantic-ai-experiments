#!/usr/bin/env bash
# RAG v2 — launch API + frontend
set -euxo pipefail

# Always run from the script's own directory (rag/v2/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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
lsof -ti:7100 | xargs kill -9 2>/dev/null || true
lsof -ti:7200 | xargs kill -9 2>/dev/null || true

# Start API (must run from rag/v2/ for 'knowledge' package to be found)
echo -e "\n${BOLD}Starting API on :7100${RESET}"
uv run uvicorn knowledge.api.app:app --port 7100 \
  > /tmp/rag-api.log 2>&1 &
echo $! > /tmp/rag-api.pid

# Start frontend
echo -e "${BOLD}Starting frontend on :7200${RESET}"
(cd "$SCRIPT_DIR/frontend" && PORT=7200 npm run dev > /tmp/rag-ui.log 2>&1) &
echo $! > /tmp/rag-ui.pid

# Wait for API
echo -n "  Waiting for API"
for i in $(seq 1 40); do
  if curl -sf http://localhost:7100/health >/dev/null 2>&1; then
    echo ""; ok "API ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 40 ] && { echo ""; echo "API failed — check: tail /tmp/rag-api.log"; exit 1; }
done

# Wait for frontend
echo -n "  Waiting for UI"
for i in $(seq 1 40); do
  if curl -sf http://localhost:7200 >/dev/null 2>&1; then
    echo ""; ok "UI ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 40 ] && { echo ""; echo "UI failed — check: tail /tmp/rag-ui.log"; exit 1; }
done

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI  →  http://localhost:7200"
echo -e "  API →  http://localhost:7100/health"
echo -e "  Logs → tail -f /tmp/rag-api.log"
echo -e "  Stop → kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

open http://localhost:7200 2>/dev/null || \
  xdg-open http://localhost:7200 2>/dev/null || \
  echo "  Open http://localhost:7200 in your browser"
