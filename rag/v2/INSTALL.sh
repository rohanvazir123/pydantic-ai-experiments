#!/usr/bin/env bash
# RAG v2 — local dev setup
# Run from the rag/v2/ directory: bash INSTALL.sh
set -euo pipefail

# ── Prerequisites ─────────────────────────────────────────────────────────────
# Required before running this script:
#
#   python3 >= 3.13   https://www.python.org/downloads/
#   Docker Desktop    https://www.docker.com/products/docker-desktop/
#                     Must be running (not just installed)
#   Ollama            https://ollama.com  (macOS: drag to /Applications)
#   openssl           Pre-installed on macOS; Linux: sudo apt install openssl
#
# uv (Python package manager) is installed automatically if missing.
# -----------------------------------------------------------------------------
MISSING=0

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found — install from https://www.python.org/downloads/"
  MISSING=1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found — install Docker Desktop from https://www.docker.com/products/docker-desktop/"
  MISSING=1
elif ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed but not running — start Docker Desktop first"
  MISSING=1
fi

# Ollama: check PATH, then fall back to macOS app bundle location
if ! command -v ollama >/dev/null 2>&1; then
  if [ -f /Applications/Ollama.app/Contents/MacOS/Ollama ]; then
    export PATH="/Applications/Ollama.app/Contents/MacOS:$PATH"
  else
    echo "ERROR: ollama not found — install from https://ollama.com"
    MISSING=1
  fi
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl not found — macOS: brew install openssl / Linux: sudo apt install openssl"
  MISSING=1
fi

[ "$MISSING" -eq 1 ] && exit 1

# ── 1. Install uv ────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin — source bashrc to pick it up
  export PATH="$HOME/.local/bin:$PATH"
  source ~/.bashrc 2>/dev/null || true
fi
echo "==> uv: $(uv --version)"

# ── 2. Create venv and install Python deps ───────────────────────────────────
echo "==> Creating virtual environment (.venv)..."
uv venv --clear .venv
echo "==> Installing Python dependencies..."
uv sync --extra all

# ── 2. Environment ────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "==> Copying .env.example → .env"
  cp .env.example .env
  echo "    Edit .env if you need non-default DB/Redis/LLM settings."
else
  echo "==> .env already exists, skipping copy."
fi

# ── 3. JWT RSA keys (required for auth) ──────────────────────────────────────
KEY_DIR="infra/keys"
JWE_DIR="$KEY_DIR/jwe"
if [ ! -f "$KEY_DIR/private.pem" ]; then
  echo "==> Generating RSA key pair for JWT auth..."
  mkdir -p "$KEY_DIR"
  openssl genrsa -out "$KEY_DIR/private.pem" 2048 2>/dev/null
  openssl rsa -in "$KEY_DIR/private.pem" -pubout -out "$KEY_DIR/public.pem" 2>/dev/null
  echo "    Keys written to $KEY_DIR/"
else
  echo "==> JWT keys already exist, skipping."
fi
if [ ! -d "$JWE_DIR" ]; then
  echo "==> Creating JWE keys directory..."
  mkdir -p "$JWE_DIR"
fi

# ── 4. Start infrastructure ───────────────────────────────────────────────────
# Ollama runs natively (not in Docker) — the Docker service requires Nvidia GPU
# drivers which are unavailable on Mac and most dev machines.
echo "==> Starting Docker services (postgres, age, redis)..."
docker compose up -d postgres age redis
echo "    Waiting 10 s for services to become healthy..."
sleep 10
docker compose ps

# ── 5. Pull Ollama models ─────────────────────────────────────────────────────
echo "==> Pulling Ollama models (this may take a while)..."
make pull-models PATH="$PATH"

# ── 6. Migrate + seed ─────────────────────────────────────────────────────────
echo "==> Running migrations and seeding sample data..."
make seed

# ── 7. Unit tests ─────────────────────────────────────────────────────────────
echo "==> Running unit tests..."
make test-unit

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "==> Setup complete."
echo ""
echo "Activate the venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Start the API:"
echo "  uv run uvicorn knowledge.api.app:app --reload --port 8001"
echo ""
echo "Health check:"
echo "  curl http://localhost:8001/health"
echo ""
echo "For auth-gated endpoints, import the Postman collection:"
echo "  postman/RAG_v2.postman_collection.json"
echo "  postman/RAG_v2_local.postman_environment.json"
