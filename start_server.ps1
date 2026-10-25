$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$Python = if (Test-Path '.\.venv\Scripts\python.exe') {
    '.\.venv\Scripts\python.exe'
} elseif (Test-Path '.\.venv\bin\python') {
    '.\.venv\bin\python'
} else {
    'python'
}

# Auto-optimize and sync videos if any new files were added
if (Test-Path '.\tools\apply_faststart.ps1') {
    try {
        & powershell -ExecutionPolicy Bypass -File .\tools\apply_faststart.ps1 | Out-Null
    } catch {
        # Non-fatal if video sync has minor warning
    }
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  GamiSenpai - Multi-Origami Web Server  " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting server at: http://127.0.0.1:8000" -ForegroundColor Green

& $Python -m uvicorn app.app:app --host 127.0.0.1 --port 8000 --ws none --log-level info
