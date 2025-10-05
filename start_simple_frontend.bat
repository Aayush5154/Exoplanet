@echo off
echo Starting Exoplanet Prediction (Backend + simple_frontend.html)...
echo.
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && call venv\Scripts\activate && python app.py"
echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul
echo.
echo Opening simple_frontend.html in your default browser...
start "" "%~dp0simple_frontend.html"
echo.
echo Backend: http://127.0.0.1:5000
echo If the page doesn't connect, wait a few seconds and refresh.
pause


