#!/usr/bin/env pwsh
# Windows: if script execution is blocked, run once in an elevated pwsh:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Or bypass for this session:
#   pwsh -ExecutionPolicy Bypass -File install.ps1
Set-Location "$PSScriptRoot/rag/v2"
& pwsh -ExecutionPolicy Bypass -File INSTALL.ps1
