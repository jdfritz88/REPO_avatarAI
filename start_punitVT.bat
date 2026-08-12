@echo off
REM Starts the full AI Avatar System stack: Redis, Postgres, backend, frontend.
REM Each service opens in its own window so you can see its logs / stop it individually.

set ROOT=%~dp0

echo Starting Redis...
start "Redis" cmd /k "cd /d "%ROOT%redis-portable" && redis-server.exe redis.conf"

echo Making sure Postgres service is running...
net start postgresql-x64-15 >nul 2>&1

echo Starting backend (FastAPI)...
start "Backend" cmd /k "cd /d "%ROOT%backend" && venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo Starting frontend (Next.js)...
start "Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo.
echo All services starting in their own windows.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Waiting a few seconds before opening the browser...
timeout /t 8 /nobreak >nul
start "" "http://localhost:3000"
