#!/usr/bin/env pwsh
# RAG Agent — one-shot developer install (Windows / macOS / Linux via PowerShell 7+)
#
# Usage:
#   .\install.ps1                        # non-interactive: installs everything silently
#   .\install.ps1 -Interactive           # step-by-step prompts at each stage
#   .\install.ps1 ingestion              # install a specific extra, non-interactive
#   .\install.ps1 ingestion -Interactive
#
# Windows: if script execution is blocked, run once in an elevated PowerShell:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Or bypass for this session:
#   powershell -ExecutionPolicy Bypass -File install.ps1
[CmdletBinding()]
param(
    [string]$Extras = "all",
    [switch]$Interactive
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── helpers ───────────────────────────────────────────────────────────────────
function ok($msg)   { Write-Host "✓ $msg" -ForegroundColor Green }
function warn($msg) { Write-Host "⚠ $msg" -ForegroundColor Yellow }
function step($msg) { Write-Host "▶ $msg" -ForegroundColor Cyan }

function ask($prompt) {
    $reply = Read-Host "? $prompt [Y/n]"
    return ($reply -eq "" -or $reply -match "^[Yy]")
}

$IsWin = $IsWindows -or $env:OS -eq "Windows_NT"

# ── uv ────────────────────────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    step "Installing uv..."
    if ($IsWin) {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    } else {
        # macOS / Linux running pwsh
        if (Get-Command curl -ErrorAction SilentlyContinue) {
            curl -LsSf https://astral.sh/uv/install.sh | sh
        } elseif (Get-Command wget -ErrorAction SilentlyContinue) {
            wget -qO- https://astral.sh/uv/install.sh | sh
        } else {
            throw "Neither curl nor wget found. Install one and retry."
        }
        $env:PATH = "$env:HOME/.local/bin:$env:PATH"
    }
}

# Fallback: search known install paths if uv still not in PATH
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
        Write-Host "ERROR: uv not found after install. Open a new terminal and re-run." -ForegroundColor Red
        exit 1
    }
}
ok "uv $(uv --version)"

# ── .env ──────────────────────────────────────────────────────────────────────
if (-not (Test-Path .env)) {
    Copy-Item .env.sample .env
    ok "Created .env from .env.sample — edit it before running"
} else {
    ok ".env already exists"
}

# ── extras selection (interactive) ───────────────────────────────────────────
if ($Interactive -and $Extras -eq "all") {
    Write-Host ""
    Write-Host "Available extras:"
    Write-Host "  ingestion     Docling document pipeline"
    Write-Host "  audio         Whisper ASR (also needs FFmpeg in PATH)"
    Write-Host "  reranker      CrossEncoder reranking (sentence-transformers)"
    Write-Host "  ui            Streamlit chat interface"
    Write-Host "  observability Langfuse tracing"
    Write-Host "  mcp           MCP server"
    Write-Host "  mem0          User-memory layer"
    Write-Host "  nl2sql        NL-to-SQL query parsing"
    Write-Host "  all           Everything (recommended)"
    Write-Host ""
    $input = Read-Host "? Which extras? [all]"
    if ($input) { $Extras = $input }
}

# ── Python packages ───────────────────────────────────────────────────────────
step "Installing rag-agent[$Extras]..."
uv sync --extra $Extras
ok "Python environment ready (.venv/)"

# ── Docker / PostgreSQL ───────────────────────────────────────────────────────
Write-Host ""
$doDocker = $true
if ($Interactive) { $doDocker = ask "Start pgvector container (requires Docker)?" }

if ($doDocker) {
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($dockerCmd) {
        $dockerRunning = $false
        try { docker info 2>$null | Out-Null; $dockerRunning = $true } catch {}
        if ($dockerRunning) {
            step "Starting PostgreSQL + pgvector (port 5434)..."
            docker compose up -d pgvector
            ok "pgvector running on localhost:5434"
        } else {
            warn "Docker is installed but not running — start Docker Desktop, then: docker compose up -d pgvector"
        }
    } else {
        warn "Docker not found — install Docker Desktop and run: docker compose up -d pgvector"
    }
}

# ── Ollama models ─────────────────────────────────────────────────────────────
Write-Host ""
$doOllama = $true
if ($Interactive) { $doOllama = ask "Pull Ollama models now (requires Ollama running)?" }

if ($doOllama) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        $ollamaRunning = $false
        try { ollama list 2>$null | Out-Null; $ollamaRunning = $true } catch {}
        if ($ollamaRunning) {
            step "Pulling Ollama models..."
            ollama pull llama3.1:8b
            ollama pull nomic-embed-text:latest
            ok "Ollama models ready"
        } else {
            warn "Ollama is not running — start it with 'ollama serve', then:"
            warn "  ollama pull llama3.1:8b; ollama pull nomic-embed-text"
        }
    } else {
        warn "Ollama not found — install from https://ollama.com, then:"
        warn "  ollama pull llama3.1:8b; ollama pull nomic-embed-text"
    }
}

# ── Reranker model (CrossEncoder) ─────────────────────────────────────────────
Write-Host ""
$stAvailable = $false
try { uv run python -c "import sentence_transformers" 2>$null | Out-Null; $stAvailable = $true } catch {}

$doReranker = $false
if ($stAvailable) {
    if ($Interactive) {
        $doReranker = ask "Pre-download BAAI/bge-reranker-base cross-encoder model (~1.1 GB)?"
    } else {
        $doReranker = $true
    }
}

if ($doReranker) {
    step "Pre-downloading cross-encoder model (BAAI/bge-reranker-base)..."
    try {
        uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
        ok "Cross-encoder model cached (~/.cache/huggingface/)"
    } catch {
        warn "Cross-encoder pre-download failed — it will download on first use"
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════"
Write-Host "  RAG Agent install complete"
Write-Host "══════════════════════════════════════════════════"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit .env               — set DATABASE_URL, LLM_*, EMBEDDING_*"
Write-Host "  2. ollama serve            — start Ollama (separate terminal)"
if ($IsWin) {
    Write-Host "  3. ollama pull llama3.1:8b; ollama pull nomic-embed-text"
} else {
    Write-Host "  3. ollama pull llama3.1:8b && ollama pull nomic-embed-text"
}
Write-Host "  4. uv run python -m rag.main --validate"
Write-Host "  5. uv run python -m rag.main --ingest --documents rag/documents"
Write-Host "  6. uv run pytest rag/tests/core/ -v"
Write-Host ""
Write-Host "API server:   uv run uvicorn rag.api.app:app --reload"
Write-Host "Streamlit UI: uv run streamlit run rag/app/streamlit/streamlit_app.py"
Write-Host ""
if ($IsWin) {
    Write-Host "Activate venv to drop 'uv run':  .venv\Scripts\Activate.ps1"
} else {
    Write-Host "Activate venv to drop 'uv run':  source .venv/bin/activate"
}
