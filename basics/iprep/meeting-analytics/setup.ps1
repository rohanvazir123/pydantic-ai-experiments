# Meeting Analytics — one-shot setup script (Windows PowerShell)
# Run from the meeting-analytics/ directory:
#   cd basics/iprep/meeting-analytics
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── 1. Python version check ──────────────────────────────────────────────────
Write-Host "`n[1/5] Checking Python version..."
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found. Install Python 3.10+."; exit 1 }
Write-Host "      $pyVersion"
$parts = ($pyVersion -replace "Python ","").Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Write-Error "Python 3.10+ required (found $pyVersion)"; exit 1
}

# ── 2. Virtual environment ───────────────────────────────────────────────────
Write-Host "`n[2/5] Setting up virtual environment..."
$venvPath = Join-Path $ScriptDir "venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
    Write-Host "      Created venv at $venvPath"
} else {
    Write-Host "      venv already exists — skipping creation"
}
$pip = Join-Path $venvPath "Scripts\pip.exe"
$python = Join-Path $venvPath "Scripts\python.exe"

# ── 3. Install dependencies ──────────────────────────────────────────────────
Write-Host "`n[3/5] Installing Python dependencies..."
& $pip install --upgrade pip --quiet
& $pip install -r (Join-Path $ScriptDir "requirements.txt")
Write-Host "      Done."

# ── 4. Docker + pgvector container ──────────────────────────────────────────
Write-Host "`n[4/5] Starting pgvector Docker container..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warning "Docker not found. Install Docker Desktop, then re-run this script."
} else {
    # docker-compose.yml lives two levels up from meeting-analytics/
    $composeDir = Resolve-Path (Join-Path $ScriptDir "../../..")
    Push-Location $composeDir
    docker compose up -d pgvector
    Pop-Location
    Write-Host "      pgvector running on localhost:5434"
}

# ── 5. Ollama models ─────────────────────────────────────────────────────────
Write-Host "`n[5/5] Checking Ollama..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama not found. Install from https://ollama.com, then run:"
    Write-Warning "  ollama pull llama3.1:8b"
    Write-Warning "  ollama pull nomic-embed-text:latest"
} else {
    Write-Host "      Pulling llama3.1:8b ..."
    ollama pull llama3.1:8b
    Write-Host "      Pulling nomic-embed-text:latest ..."
    ollama pull nomic-embed-text:latest
    Write-Host "      Models ready."
}

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host @"

Setup complete. Next steps:
  1. Copy .env.example to .env and fill in your DB credentials
  2. Activate the venv:  .\venv\Scripts\Activate.ps1
  3. Load data into DB:  python final_version/load_raw_jsons_to_db.py --reset
                         python final_version/load_output_csvs_to_db.py --reset
  4. Run charts:         python final_version/generate_charts.py
"@
