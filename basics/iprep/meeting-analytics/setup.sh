#!/usr/bin/env bash
# Meeting Analytics — one-shot setup script (macOS / Linux)
#
# Run from anywhere inside the repo:
#   bash basics/iprep/meeting-analytics/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="$SCRIPT_DIR/venv"

step() { printf "\n[%s] %s\n" "$1" "$2"; }

# ── 1. Python ────────────────────────────────────────────────────────────────
step "1/6" "Checking Python..."
if ! command -v python3 &>/dev/null; then
  echo "  ERROR: python3 not found. Install with: brew install python@3.13"
  exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" \
  || { echo "  ERROR: Python 3.10+ required (found $PY_VER)"; exit 1; }
echo "  OK — Python $PY_VER"

# ── 2. Virtual environment ───────────────────────────────────────────────────
step "2/6" "Setting up venv..."
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "  Created $VENV"
else
  echo "  Already exists — skipping"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ── 3. Python dependencies ───────────────────────────────────────────────────
step "3/6" "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "  Done"

# ── 4. Jupyter kernel ────────────────────────────────────────────────────────
step "4/6" "Installing Jupyter and registering venv kernel..."
pip install --quiet jupyter ipykernel
python3 -m ipykernel install --user \
  --name meeting-analytics \
  --display-name "Python (meeting-analytics)"
echo "  Kernel registered: Python (meeting-analytics)"

# ── 5. Docker + pgvector ─────────────────────────────────────────────────────
step "5/6" "Starting pgvector Docker container..."
if ! command -v docker &>/dev/null; then
  echo "  WARNING: Docker not found."
  echo "  Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/"
  echo "  Then re-run this script or run manually: docker compose up -d pgvector"
else
  (cd "$REPO_ROOT" && docker compose up -d pgvector)
  echo "  pgvector running on localhost:5434"
fi

# ── 6. Ollama models ─────────────────────────────────────────────────────────
step "6/6" "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "  WARNING: Ollama not found."
  echo "  Install from: https://ollama.com/download"
  echo "  Then run:"
  echo "    ollama pull llama3.1:8b"
  echo "    ollama pull nomic-embed-text:latest"
else
  echo "  Pulling llama3.1:8b..."
  ollama pull llama3.1:8b
  echo "  Pulling nomic-embed-text:latest..."
  ollama pull nomic-embed-text:latest
  echo "  Models ready"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
printf "\n==============================\n"
printf "Setup complete. Next steps:\n\n"
printf "  cd basics/iprep/meeting-analytics\n"
printf "  source venv/bin/activate\n"
printf "  cp .env.example .env\n\n"
printf "  # Load data into DB:\n"
printf "  python final_version/load_raw_jsons_to_db.py --reset\n"
printf "  python final_version/load_output_csvs_to_db.py --reset\n"
printf "  python final_version/verify.py\n\n"
printf "  # Run the notebook:\n"
printf "  jupyter notebook final_version/meeting_analytics.ipynb\n"
printf "  # Select kernel: Python (meeting-analytics)\n"
printf "==============================\n\n"
