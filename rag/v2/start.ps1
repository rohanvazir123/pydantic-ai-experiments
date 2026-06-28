#!/usr/bin/env pwsh
# RAG v2 — launch API + frontend (Windows)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function ok($msg)   { Write-Host "  $msg" -ForegroundColor Green }
function warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }

$API_PORT = 7100
$UI_PORT  = 7200

# Load .env into process environment — overrides any system-level env vars (e.g. DATABASE_URL
# pointing to a cloud database) so local Docker URLs are used for development.
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match "^\s*[A-Z_][A-Z0-9_]*=" -and $_ -notmatch "^\s*#" } | ForEach-Object {
        $key, $val = $_ -split "=", 2
        Set-Item -Path "Env:$($key.Trim())" -Value $val.Trim()
    }
}

docker compose up -d postgres 2>$null | Out-Null
ok "Docker services running"

# Redis — expose on host port 7500 (docker-compose redis has no host port mapping)
$redisUp = $false
try { $t = [System.Net.Sockets.TcpClient]::new("localhost", 7500); $t.Close(); $redisUp = $true } catch {}
if (-not $redisUp) {
    docker rm -f rag-redis 2>$null | Out-Null
    docker run -d --name rag-redis -p 7500:6379 redis:7-alpine | Out-Null
    ok "Redis started on :7500"
} else {
    ok "Redis already up on :7500"
}

# Start Ollama if not running
$ollamaUp = $false
try { Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 2 | Out-Null; $ollamaUp = $true } catch {}
if (-not $ollamaUp) {
    warn "Ollama not running — starting..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# Kill anything already on our ports
foreach ($port in @($API_PORT, $UI_PORT)) {
    netstat -ano 2>$null | Select-String ":$port\s" | ForEach-Object {
        $pid_ = ($_ -split '\s+')[-1]
        try { Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue } catch {}
    }
}
Start-Sleep -Seconds 1

# Start API
Write-Host "`nStarting API on :$API_PORT" -ForegroundColor Cyan
$apiLog = "$env:TEMP\rag-api.log"
$apiProc = Start-Process pwsh -ArgumentList @(
    "-NoProfile", "-Command",
    "Set-Location '$PSScriptRoot'; uv run uvicorn knowledge.api.app:app --host 0.0.0.0 --port $API_PORT"
) -RedirectStandardOutput $apiLog -RedirectStandardError "$env:TEMP\rag-api-err.log" -WindowStyle Hidden -PassThru

# Start frontend
Write-Host "Starting frontend on :$UI_PORT" -ForegroundColor Cyan
$uiLog = "$env:TEMP\rag-ui.log"
$uiProc = Start-Process pwsh -ArgumentList @(
    "-NoProfile", "-Command",
    "Set-Location '$PSScriptRoot\frontend'; `$env:PORT='$UI_PORT'; npm run dev"
) -RedirectStandardOutput $uiLog -RedirectStandardError "$env:TEMP\rag-ui-err.log" -WindowStyle Hidden -PassThru

# Wait for API
Write-Host -NoNewline "  Waiting for API"
$ready = $false
for ($i = 1; $i -le 40; $i++) {
    $up = $false
    try { Invoke-RestMethod "http://localhost:$API_PORT/health" -TimeoutSec 1 | Out-Null; $up = $true } catch {}
    if ($up) { Write-Host ""; ok "API ready  →  http://localhost:$API_PORT/health"; $ready = $true; break }
    Write-Host -NoNewline "."; Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host ""
    Write-Host "API failed to start. Log:" -ForegroundColor Red
    Get-Content $apiLog -Tail 20 -ErrorAction SilentlyContinue
    Get-Content "$env:TEMP\rag-api-err.log" -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}

# Wait for UI
Write-Host -NoNewline "  Waiting for UI"
$ready = $false
for ($i = 1; $i -le 40; $i++) {
    $up = $false
    try { Invoke-WebRequest "http://localhost:$UI_PORT" -TimeoutSec 1 -UseBasicParsing | Out-Null; $up = $true } catch {}
    if ($up) { Write-Host ""; ok "UI ready   →  http://localhost:$UI_PORT"; $ready = $true; break }
    Write-Host -NoNewline "."; Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host ""
    Write-Host "UI failed to start. Log:" -ForegroundColor Red
    Get-Content $uiLog -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host "  Everything running" -ForegroundColor Green
Write-Host "  UI   ->  http://localhost:$UI_PORT"
Write-Host "  API  ->  http://localhost:$API_PORT/health"
Write-Host "  Logs ->  Get-Content $apiLog -Wait"
Write-Host "  Stop ->  Stop-Process -Id $($apiProc.Id),$($uiProc.Id)"
Write-Host ""

Start-Process "http://localhost:$UI_PORT"
