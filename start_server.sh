#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="python3"
fi

echo "Starting GamiSenpai server on http://127.0.0.1:8000 ..."
exec "$PYTHON" -m uvicorn app.app:app --host 127.0.0.1 --port 8000 --log-level warning
