#!/usr/bin/env bash
# Pull all Ollama models required by RAG v2.
# Usage: bash scripts/pull_models.sh
set -euxo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✗${RESET}  $1"; }

# ── Models ────────────────────────────────────────────────────────────────────
# Listed smallest-first so the system is usable as quickly as possible.

declare -A MODEL_ROLES=(
  ["nomic-embed-text:latest"]="Embeddings — required for ingest + search (~270 MB)"
  ["qwen2.5:0.5b"]="Nano tier — query routing + content policy (~400 MB)"
  ["llama3.2:3b"]="Small tier — chat responses (~2.0 GB)"
)

# Ordered pull sequence (smallest first)
MODELS=(
  "nomic-embed-text:latest"
  "qwen2.5:0.5b"
  "llama3.2:3b"
)

# ── Ensure Ollama is running ──────────────────────────────────────────────────
echo -e "\n${BOLD}==> Checking Ollama${RESET}"
if ! command -v ollama >/dev/null 2>&1; then
  # macOS app bundle fallback
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
    [ "$i" -eq 15 ] && { echo ""; fail "Ollama did not start — run 'ollama serve' manually"; exit 1; }
  done
fi
ok "Ollama running"

# ── Pull models ───────────────────────────────────────────────────────────────
echo -e "\n${BOLD}==> Pulling models (${#MODELS[@]} total)${RESET}"

TOTAL=${#MODELS[@]}
N=0
for model in "${MODELS[@]}"; do
  N=$((N + 1))
  role="${MODEL_ROLES[$model]}"
  echo -e "\n  ${BOLD}[$N/$TOTAL]${RESET} $model — $role"
  ollama pull "$model"
  ok "$model"
done

# ── Verify ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}==> Verifying${RESET}"
MISSING=0
for model in "${MODELS[@]}"; do
  if ollama list | grep -q "$(echo "$model" | cut -d: -f1)"; then
    ok "$model present"
  else
    fail "$model NOT found after pull"
    MISSING=1
  fi
done

[ "$MISSING" -eq 1 ] && { fail "One or more models missing — re-run this script"; exit 1; }

echo -e "\n${BOLD}${GREEN}All models ready.${RESET}"
echo "  Run 'bash start.sh' to launch the API and UI."
