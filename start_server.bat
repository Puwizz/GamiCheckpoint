@echo off
cd /d "%~dp0"
echo =========================================
echo   GamiSenpai - Multi-Origami Web Server
echo =========================================

if exist tools\apply_faststart.ps1 (
    powershell -ExecutionPolicy Bypass -File tools\apply_faststart.ps1 >nul 2>&1
)

echo Starting server at: http://127.0.0.1:8000
echo.

if exist .venv\Scripts\python.exe (
    call .venv\Scripts\python.exe -m uvicorn app.app:app --host 127.0.0.1 --port 8000 --ws none --log-level info
) else (
    python -m uvicorn app.app:app --host 127.0.0.1 --port 8000 --ws none --log-level info
)
