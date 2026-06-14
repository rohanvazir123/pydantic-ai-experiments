#!/usr/bin/env pwsh
# RAG v2 — local dev setup (Windows / macOS / Linux via PowerShell 7.1+)
#
# Requires PowerShell 7.1+ (pwsh). The built-in Windows "powershell" (5.x) will
# fail with parse errors — install pwsh from https://aka.ms/powershell
#
# Usage (from rag/v2/):
#   pwsh -ExecutionPolicy Bypass -File INSTALL.ps1
#
# Windows: if script execution is blocked, run once in an elevated pwsh:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Or bypass for this session:
#   pwsh -ExecutionPolicy Bypass -File INSTALL.ps1
#
# Prerequisites — install these before running:
#
#   Python 3.13+    https://www.python.org/downloads/
#                   Windows: check "Add python.exe to PATH" during install
#   Docker Desktop  https://www.docker.com/products/docker-desktop/
#                   Must be running (not just installed)
#   Ollama          https://ollama.com
#                   Windows: run the .exe installer
#   OpenSSL         Bundled with Git for Windows — https://git-scm.com/downloads
#                   Or: winget install ShiningLight.OpenSSL
#
# uv (Python package manager) is installed automatically if missing.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$IsWin = $IsWindows -or ($env:OS -eq "Windows_NT")

function step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function err($msg)  { Write-Host "ERROR: $msg" -ForegroundColor Red }

# ── Prerequisites check ───────────────────────────────────────────────────────
$missing = $false

if (-not (Get-Command python3 -ErrorAction SilentlyContinue) -and
    -not (Get-Command python  -ErrorAction SilentlyContinue)) {
    err "python not found — install from https://www.python.org/downloads/"
    $missing = $true
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    err "docker not found — install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    $missing = $true
} else {
    try { docker info 2>$null | Out-Null }
    catch {
        err "Docker is installed but not running — start Docker Desktop first"
        $missing = $true
    }
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    err "ollama not found — install from https://ollama.com"
    $missing = $true
}

if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
    err "openssl not found — install Git for Windows (https://git-scm.com/downloads) or: winget install ShiningLight.OpenSSL"
    $missing = $true
}

if ($missing) { exit 1 }

# ── 1. Install uv ────────────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    step "uv not found, installing..."
    if ($IsWin) {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    } else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        $env:PATH = "$env:HOME/.local/bin:$env:PATH"
    }
    # fallback: search known install paths
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        $candidates = @(
            "$env:USERPROFILE\.local\bin\uv.exe",
            "$env:USERPROFILE\.cargo\bin\uv.exe",
            "$env:HOME/.local/bin/uv"
        )
        $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($found) {
            $env:PATH = "$(Split-Path $found);$env:PATH"
        } else {
            err "uv not found after install — open a new terminal and re-run."
            exit 1
        }
    }
}
ok "uv $(uv --version)"

# ── 2. Create venv and install Python deps ───────────────────────────────────
step "Creating virtual environment (.venv)..."
uv venv --clear .venv
step "Installing Python dependencies..."
uv sync --extra all

# ── 3. Environment ────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    step "Copying .env.example -> .env"
    Copy-Item ".env.example" ".env"
    ok "Edit .env if you need non-default DB/Redis/LLM settings."
} else {
    ok ".env already exists, skipping."
}

# ── 4. JWT RSA keys (required for auth) ──────────────────────────────────────
$keyDir = "infra/keys"
$jweDir = "$keyDir/jwe"
if (-not (Test-Path "$keyDir/private.pem")) {
    step "Generating RSA key pair for JWT auth..."
    New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
    openssl genrsa -out "$keyDir/private.pem" 2048 2>$null
    openssl rsa -in "$keyDir/private.pem" -pubout -out "$keyDir/public.pem" 2>$null
    ok "Keys written to $keyDir/"
} else {
    ok "JWT keys already exist, skipping."
}
if (-not (Test-Path $jweDir)) {
    New-Item -ItemType Directory -Force -Path $jweDir | Out-Null
}

# ── 5. Start infrastructure ───────────────────────────────────────────────────
# Ollama runs natively (not in Docker) — the Docker service requires Nvidia GPU
# drivers which are unavailable on most dev machines.
step "Starting Docker services (postgres, age, redis)..."
docker compose up -d postgres age redis
Write-Host "    Waiting 10 s for services to become healthy..."
Start-Sleep -Seconds 10
docker compose ps

# ── 6. Pull Ollama models ─────────────────────────────────────────────────────
step "Pulling Ollama models (this may take a while)..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text:latest
ollama pull qwen2.5:0.5b
ollama pull llama3.1:70b

# ── 7. Migrate + seed ─────────────────────────────────────────────────────────
step "Running migrations..."
$env:DATABASE_URL = if ($env:DATABASE_URL) { $env:DATABASE_URL } else {
    (Get-Content ".env" | Where-Object { $_ -match "^DATABASE_URL=" }) -replace "^DATABASE_URL=", ""
}
Get-ChildItem "schema/*.sql" | Sort-Object Name | ForEach-Object {
    Write-Host "    Running $($_.Name) ..."
    psql $env:DATABASE_URL -f $_.FullName
}
step "Seeding default tenant, corpus, and sample documents..."
uv run python scripts/seed.py

# ── 8. Unit tests ─────────────────────────────────────────────────────────────
step "Running unit tests..."
uv run pytest tests/unit/ -v

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host ""
if ($IsWin) {
    Write-Host "Activate the venv:"
    Write-Host "  .venv\Scripts\Activate.ps1"
} else {
    Write-Host "Activate the venv:"
    Write-Host "  source .venv/bin/activate"
}
Write-Host ""
Write-Host "Start the API:"
Write-Host "  uv run uvicorn knowledge.api.app:app --reload --port 8001"
Write-Host ""
Write-Host "Health check:"
Write-Host "  curl http://localhost:8001/health"
Write-Host ""
Write-Host "For auth-gated endpoints, import the Postman collection:"
Write-Host "  postman/RAG_v2.postman_collection.json"
Write-Host "  postman/RAG_v2_local.postman_environment.json"
