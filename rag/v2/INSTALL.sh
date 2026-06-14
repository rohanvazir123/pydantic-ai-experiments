#!/usr/bin/env bash
# RAG v2 — one-shot install + launch
# Usage:  bash INSTALL.sh
# Run from the rag/v2/ directory.
set -euxo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

step()  { echo -e "\n${BOLD}==> $1${RESET}"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail()  { echo -e "  ${RED}✗${RESET}  $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# 0. Must run from rag/v2/
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "pyproject.toml" ] || [ ! -d "knowledge" ]; then
  fail "Run this script from the rag/v2/ directory:  cd rag/v2 && bash INSTALL.sh"
  exit 1
fi

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║   RAG v2 — Install & Launch          ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Prerequisite checks
# ─────────────────────────────────────────────────────────────────────────────
step "Checking prerequisites"
MISSING=0

# Python 3.13+
if command -v python3 >/dev/null 2>&1; then
  PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  ok "python3 $PY_VER"
else
  fail "python3 not found — https://www.python.org/downloads/"
  MISSING=1
fi

# Docker
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker running"
  else
    fail "Docker installed but not running — start Docker Desktop first"
    MISSING=1
  fi
else
  fail "docker not found — https://www.docker.com/products/docker-desktop/"
  MISSING=1
fi

# Node.js 18+
if command -v node >/dev/null 2>&1; then
  NODE_VER=$(node --version)
  ok "node $NODE_VER"
else
  fail "node not found — https://nodejs.org  (install v20 LTS)"
  MISSING=1
fi

# npm
if command -v npm >/dev/null 2>&1; then
  ok "npm $(npm --version)"
else
  fail "npm not found (comes with Node.js)"
  MISSING=1
fi

# Ollama — check PATH, then macOS app bundle
if ! command -v ollama >/dev/null 2>&1; then
  if [ -f /Applications/Ollama.app/Contents/MacOS/Ollama ]; then
    export PATH="/Applications/Ollama.app/Contents/Resources/bin:$PATH"
    ok "ollama (from /Applications)"
  else
    fail "ollama not found — https://ollama.com"
    MISSING=1
  fi
else
  ok "ollama $(ollama --version 2>/dev/null || echo '')"
fi

# openssl
if command -v openssl >/dev/null 2>&1; then
  ok "openssl"
else
  fail "openssl not found — brew install openssl"
  MISSING=1
fi

[ "$MISSING" -eq 1 ] && { echo -e "\n${RED}Fix the above and re-run.${RESET}"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 2. Install uv
# ─────────────────────────────────────────────────────────────────────────────
step "Installing uv (Python package manager)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Python environment + dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "Installing Python dependencies"
uv venv --python 3.13 --clear .venv 2>/dev/null || uv venv --clear .venv
uv sync --extra ingestion --extra observability
ok "Python deps installed"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Environment file — MUST come before Docker so compose picks up passwords
# ─────────────────────────────────────────────────────────────────────────────
step "Setting up environment"
if [ ! -f .env ]; then
  cp .env.example .env
  ok ".env created from .env.example"
  warn "Review .env if you need custom DB/Redis/LLM settings"
else
  ok ".env already exists"
fi
# Docker Compose needs POSTGRES_PASSWORD / AGE_DB_PASSWORD in the shell env.
# Extract them from .env directly (avoids xargs mangling complex JSON values).
# pydantic-settings reads .env automatically — no need to export everything.
set +x
POSTGRES_PASSWORD="$(grep '^POSTGRES_PASSWORD=' .env 2>/dev/null | cut -d= -f2- | tr -d "'\"" || true)"
AGE_DB_PASSWORD="$(grep '^AGE_DB_PASSWORD=' .env 2>/dev/null | cut -d= -f2- | tr -d "'\"" || true)"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-changeme}"
export AGE_DB_PASSWORD="${AGE_DB_PASSWORD:-changeme}"
set -x

# ─────────────────────────────────────────────────────────────────────────────
# 5. JWT RSA keys
# ─────────────────────────────────────────────────────────────────────────────
step "Generating JWT keys"
KEY_DIR="infra/keys"
if [ ! -f "$KEY_DIR/private.pem" ]; then
  mkdir -p "$KEY_DIR"
  openssl genrsa -out "$KEY_DIR/private.pem" 2048 2>/dev/null
  openssl rsa -in "$KEY_DIR/private.pem" -pubout -out "$KEY_DIR/public.pem" 2>/dev/null
  ok "RSA key pair written to $KEY_DIR/"
else
  ok "JWT keys already exist"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Frontend dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "Installing frontend dependencies (npm)"
(cd frontend && npm install --no-fund --loglevel=error)
ok "Frontend deps installed"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Docker services
# ─────────────────────────────────────────────────────────────────────────────
step "Starting Docker services (postgres, redis)"
docker compose up -d postgres redis
echo -n "  Waiting for services"
for i in $(seq 1 20); do
  if docker compose exec -T postgres pg_isready -U ragv2 >/dev/null 2>&1 && \
     docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo ""
    ok "postgres + redis healthy"
    break
  fi
  echo -n "."
  sleep 2
  if [ "$i" -eq 20 ]; then
    echo ""
    fail "Services did not become healthy in 40s — check: docker compose ps"
    exit 1
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 8. Ollama — ensure running, pull models
# ─────────────────────────────────────────────────────────────────────────────
step "Pulling Ollama models"
bash scripts/pull_models.sh

# ─────────────────────────────────────────────────────────────────────────────
# 9. Database migrations + seed
# ─────────────────────────────────────────────────────────────────────────────
step "Running database migrations"
# .env already exported above
uv run python - <<'PYEOF'
import asyncio, asyncpg, glob, os, sys
async def main():
    url = os.environ.get("DATABASE_URL", "postgresql://ragv2:changeme@localhost:5432/ragv2")
    conn = await asyncpg.connect(url, timeout=10)
    files = sorted(glob.glob("schema/*.sql"))
    for path in files:
        await conn.execute(open(path).read())
        print(f"  applied {path}")
    await conn.close()
    print(f"  {len(files)} migrations applied")
asyncio.run(main())
PYEOF
ok "Migrations complete"

step "Seeding sample documents"
uv run python scripts/seed.py
ok "Seed complete"

# ─────────────────────────────────────────────────────────────────────────────
# 10. Smoke tests
# ─────────────────────────────────────────────────────────────────────────────
step "Running smoke tests"
uv run pytest tests/unit/test_smoke.py -q --tb=short 2>&1 | tail -3
ok "Smoke tests passed"

# ─────────────────────────────────────────────────────────────────────────────
# 11. Write start.sh for future launches
# ─────────────────────────────────────────────────────────────────────────────
cat > start.sh << 'STARTEOF'
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
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start API
echo -e "\n${BOLD}Starting API on :8001${RESET}"
uv run uvicorn knowledge.api.app:app --port 8001 --reload \
  > /tmp/rag-api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/rag-api.pid

# Start frontend
echo -e "${BOLD}Starting frontend on :3000${RESET}"
(cd frontend && npm run dev > /tmp/rag-ui.log 2>&1) &
UI_PID=$!
echo $UI_PID > /tmp/rag-ui.pid

# Wait for API to be ready
echo -n "  Waiting for API"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    echo ""; ok "API ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 30 ] && { echo ""; echo "API failed to start — tail /tmp/rag-api.log"; exit 1; }
done

# Wait for Next.js
echo -n "  Waiting for UI"
for i in $(seq 1 30); do
  if curl -sf http://localhost:3000 >/dev/null 2>&1; then
    echo ""; ok "UI ready"; break
  fi
  echo -n "."; sleep 1
  [ "$i" -eq 30 ] && { echo ""; echo "UI failed to start — tail /tmp/rag-ui.log"; exit 1; }
done

echo -e "\n${BOLD}  ✓ Everything running${RESET}"
echo -e "  UI  →  http://localhost:3000"
echo -e "  API →  http://localhost:8001/health"
echo -e "  Logs → tail -f /tmp/rag-api.log"
echo -e "  Stop → kill \$(cat /tmp/rag-api.pid) \$(cat /tmp/rag-ui.pid)"
echo ""

# Open browser (macOS)
open http://localhost:3000 2>/dev/null || \
  xdg-open http://localhost:3000 2>/dev/null || \
  echo "  Open http://localhost:3000 in your browser"
STARTEOF
chmod +x start.sh
ok "start.sh created"

# ─────────────────────────────────────────────────────────────────────────────
# 12. Launch
# ─────────────────────────────────────────────────────────────────────────────
step "Launching RAG v2"
bash start.sh
