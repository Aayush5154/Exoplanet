@echo off
echo Starting Exoplanet Prediction Application...
echo.
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && call venv\Scripts\activate && python app.py"
echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul
echo.
echo Starting Frontend...
start "Frontend" cmd /k "cd frontend && npm start"
echo.
echo Both servers are starting! Check the opened windows.
echo Backend: http://127.0.0.1:5000
echo Frontend: http://localhost:3000
pause


