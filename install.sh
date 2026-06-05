#!/usr/bin/env bash
# RAG Agent — one-shot developer install (Linux / macOS)
#
# Usage:
#   ./install.sh                # non-interactive: installs everything silently
#   ./install.sh --interactive  # step-by-step prompts at each stage
#   ./install.sh ingestion      # install a specific extra, non-interactive
#   ./install.sh --interactive ingestion
set -euo pipefail

# ── argument parsing ──────────────────────────────────────────────────────────
INTERACTIVE=false
EXTRAS="all"
for arg in "$@"; do
  case "$arg" in
    --interactive|-i) INTERACTIVE=true ;;
    *) EXTRAS="$arg" ;;
  esac
done

# ── colours ───────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  GREEN="\033[0;32m"; YELLOW="\033[1;33m"; CYAN="\033[0;36m"; RESET="\033[0m"
else
  GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
step() { echo -e "${CYAN}▶${RESET} $*"; }

# Prompt helper (only used in interactive mode)
# ask "Question text" → returns 0 for yes, 1 for no
ask() {
  local prompt="$1" reply
  printf "%b" "${CYAN}?${RESET} $prompt [Y/n] "
  read -r reply
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ── uv ────────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  step "Installing uv..."
  if command -v curl &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget &>/dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "ERROR: neither curl nor wget found. Install one and retry." >&2; exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
  [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# Fallback: search known install paths if uv still not in PATH
if ! command -v uv &>/dev/null; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$candidate" ]; then
      export PATH="$(dirname "$candidate"):$PATH"; break
    fi
  done
fi
command -v uv &>/dev/null || { echo "ERROR: uv not found after install. Open a new terminal and re-run." >&2; exit 1; }
ok "uv $(uv --version)"

# ── .env ──────────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.sample .env
  ok "Created .env from .env.sample — edit it before running"
else
  ok ".env already exists"
fi

# ── extras selection (interactive) ───────────────────────────────────────────
if [ "$INTERACTIVE" = true ] && [ "$EXTRAS" = "all" ]; then
  echo ""
  echo "Available extras:"
  echo "  ingestion     Docling document pipeline"
  echo "  audio         Whisper ASR (also needs FFmpeg in PATH)"
  echo "  reranker      CrossEncoder reranking (sentence-transformers)"
  echo "  ui            Streamlit chat interface"
  echo "  observability Langfuse tracing"
  echo "  mcp           MCP server"
  echo "  mem0          User-memory layer"
  echo "  nl2sql        NL-to-SQL query parsing"
  echo "  all           Everything (recommended)"
  echo ""
  printf "%b" "${CYAN}?${RESET} Which extras? [all] "
  read -r input
  EXTRAS="${input:-all}"
fi

# ── Python packages ───────────────────────────────────────────────────────────
step "Installing rag-agent[$EXTRAS]..."
uv sync --extra "$EXTRAS"
ok "Python environment ready (.venv/)"

# ── Docker / PostgreSQL ───────────────────────────────────────────────────────
echo ""
DO_DOCKER=true
if [ "$INTERACTIVE" = true ]; then
  ask "Start pgvector container (requires Docker)?" || DO_DOCKER=false
fi

if [ "$DO_DOCKER" = true ]; then
  if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    step "Starting PostgreSQL + pgvector (port 5434)..."
    docker compose up -d pgvector
    ok "pgvector running on localhost:5434"
  else
    if ! command -v docker &>/dev/null; then
      warn "Docker not found — install Docker Desktop and run: docker compose up -d pgvector"
    else
      warn "Docker is installed but not running — start Docker Desktop, then: docker compose up -d pgvector"
    fi
  fi
fi

# ── Ollama models ─────────────────────────────────────────────────────────────
echo ""
DO_OLLAMA=true
if [ "$INTERACTIVE" = true ]; then
  ask "Pull Ollama models now (requires Ollama running)?" || DO_OLLAMA=false
fi

if [ "$DO_OLLAMA" = true ]; then
  if command -v ollama &>/dev/null && ollama list &>/dev/null 2>&1; then
    step "Pulling Ollama models..."
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text:latest
    ok "Ollama models ready"
  else
    if ! command -v ollama &>/dev/null; then
      warn "Ollama not found — install from https://ollama.com and run:"
      warn "  ollama pull llama3.1:8b && ollama pull nomic-embed-text"
    else
      warn "Ollama is not running — start it with 'ollama serve', then pull models:"
      warn "  ollama pull llama3.1:8b && ollama pull nomic-embed-text"
    fi
  fi
fi

# ── Reranker model (CrossEncoder) ─────────────────────────────────────────────
echo ""
DO_RERANKER=false
# Auto-download in non-interactive mode only if sentence-transformers is installed
if uv run python -c "import sentence_transformers" &>/dev/null 2>&1; then
  if [ "$INTERACTIVE" = true ]; then
    ask "Pre-download BAAI/bge-reranker-base cross-encoder model (~1.1 GB)?" && DO_RERANKER=true
  else
    DO_RERANKER=true
  fi
fi

if [ "$DO_RERANKER" = true ]; then
  step "Pre-downloading cross-encoder model (BAAI/bge-reranker-base)..."
  uv run python -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-base')
" && ok "Cross-encoder model cached (~/.cache/huggingface/)" \
  || warn "Cross-encoder pre-download failed — it will download on first use"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  RAG Agent install complete"
echo "══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Edit .env               — set DATABASE_URL, LLM_*, EMBEDDING_*"
echo "  2. ollama serve            — start Ollama (separate terminal)"
echo "  3. ollama pull llama3.1:8b && ollama pull nomic-embed-text"
echo "  4. uv run python -m rag.main --validate"
echo "  5. uv run python -m rag.main --ingest --documents rag/documents"
echo "  6. uv run pytest rag/tests/core/ -v"
echo ""
echo "API server:   uv run uvicorn rag.api.app:app --reload"
echo "Streamlit UI: uv run streamlit run rag/app/streamlit/streamlit_app.py"
echo ""
echo "Activate venv to drop 'uv run':  source .venv/bin/activate"
