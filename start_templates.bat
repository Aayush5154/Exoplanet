@echo off
echo Starting Flask backend (templates UI)...
start "Backend Server" cmd /k "cd backend && call venv\Scripts\activate && python app.py"
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul
echo Opening http://127.0.0.1:5000/ in your default browser...
start "" http://127.0.0.1:5000/
echo If the page is blank, wait a moment and refresh.
pause


