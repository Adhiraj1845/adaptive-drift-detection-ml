@echo off
echo ─────────────────────────────────────────────
echo  Drift Detection Dashboard
echo ─────────────────────────────────────────────
echo.
echo [1/2] Starting API server (port 8000)...
start "Drift API"      cmd /k "python api.py"
timeout /t 2 /nobreak >nul
echo [2/2] Starting frontend (port 5173)...
start "Drift Frontend" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul
echo.
echo  Open: http://localhost:5173
echo.
start http://localhost:5173
