@echo off
echo ============================================================
echo  TRaNKSP - TRaNK Scenario Planner
echo ============================================================

if not exist venv (
    echo ERROR: Virtual environment not found. Run setup_environment.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist .env (
    echo ERROR: .env file not found. Run setup_environment.bat first.
    pause
    exit /b 1
)

echo Checking API keys...
python check_keys.py
echo.

echo Starting TRaNKSP server...
echo Dashboard: http://localhost:8765
echo.

REM Launch browser after 8 second delay (gives server time to fully start + warm Yahoo session)
start "" cmd /c "timeout /t 8 /nobreak >nul && start http://localhost:8765"

uvicorn dashboard.app:app --host 0.0.0.0 --port 8765 --reload
