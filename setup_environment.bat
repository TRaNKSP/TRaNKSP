@echo off
echo ============================================================
echo  TRaNKSP — TRaNK Scenario Planner Environment Setup
echo ============================================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.11+ first.
    pause & exit /b 1
)

echo Removing old virtual environment (clean install)...
if exist venv rmdir /s /q venv

echo Creating fresh virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing dependencies...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Dependency installation failed. See output above.
    pause & exit /b 1
)

if not exist .env (
    echo.
    echo Creating .env template...
    (
        echo ANTHROPIC_API_KEY=your_anthropic_key_here
        echo TAVILY_API_KEY=your_tavily_key_here
        echo MASSIVE_API_KEY=your_massive_key_here
        echo FINNHUB_API_KEY=your_finnhub_key_here
    ) > .env
    echo IMPORTANT: Edit .env and add your API keys before starting!
)

if not exist data mkdir data
if not exist logs mkdir logs

echo Running database migration...
python migrate_db.py

echo.
echo ============================================================
echo  Setup complete! Run start_program.bat to launch TRaNKSP.
echo ============================================================
pause
