#!/usr/bin/env bash
# Pull all Ollama models required by RAG v2.
# Usage: bash scripts/pull_models.sh
set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✗${RESET}  $1"; }

# Models listed smallest-first so the system is usable as quickly as possible.
# Parallel arrays (avoid declare -A which breaks with set -u on some bash versions).
MODELS=(
  "nomic-embed-text:latest"
  "qwen2.5:0.5b"
  "llama3.2:3b"
)
ROLES=(
  "Embeddings — required for ingest + search (~270 MB)"
  "Nano tier  — query routing + content policy (~400 MB)"
  "Small tier — chat responses (~2.0 GB)"
)

# ── Ensure Ollama is running ──────────────────────────────────────────────────
echo -e "\n${BOLD}==> Checking Ollama${RESET}"
if ! command -v ollama >/dev/null 2>&1; then
  if [ -f /Applications/Ollama.app/Contents/Resources/bin/ollama ]; then
    export PATH="/Applications/Ollama.app/Contents/Resources/bin:$PATH"
  else
    fail "ollama not found — install from https://ollama.com"
    exit 1
  fi
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  warn "Ollama not running — starting it..."
  ollama serve >/dev/null 2>&1 &
  echo -n "  Waiting for Ollama"
  for i in $(seq 1 15); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && { echo ""; break; }
    echo -n "."; sleep 1
    if [ "$i" -eq 15 ]; then
      echo ""
      fail "Ollama did not start — run 'ollama serve' manually"
      exit 1
    fi
  done
fi
ok "Ollama running"

# ── Pull models ───────────────────────────────────────────────────────────────
echo -e "\n${BOLD}==> Pulling ${#MODELS[@]} models${RESET}"

TOTAL=${#MODELS[@]}
for i in $(seq 0 $((TOTAL - 1))); do
  model="${MODELS[$i]}"
  role="${ROLES[$i]}"
  echo -e "\n  ${BOLD}[$((i+1))/$TOTAL]${RESET} $model — $role"
  ollama pull "$model"
  ok "$model"
done

# ── Verify ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}==> Verifying${RESET}"
MISSING=0
for model in "${MODELS[@]}"; do
  base="${model%%:*}"
  if ollama list | grep -q "$base"; then
    ok "$model present"
  else
    fail "$model NOT found after pull"
    MISSING=1
  fi
done

[ "$MISSING" -eq 1 ] && { fail "One or more models missing — re-run this script"; exit 1; }

echo -e "\n${BOLD}${GREEN}All models ready.${RESET}"
echo "  Run 'make start' or 'bash start.sh' to launch the API and UI."
