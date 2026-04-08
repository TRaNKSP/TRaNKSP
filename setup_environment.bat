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
        echo # ── REQUIRED ──────────────────────────────────────────────────────────
        echo ANTHROPIC_API_KEY=your_anthropic_key_here
        echo TAVILY_API_KEY=your_tavily_key_here
        echo MASSIVE_API_KEY=your_massive_key_here
        echo.
        echo # ── OPTIONAL: Additional LLM providers for multi-model consensus ──────
        echo # Each adds a parallel voice to squeeze candidate selection.
        echo # Leave blank to skip that provider — Claude always runs.
        echo.
        echo # OpenAI GPT-4o-mini  →  https://platform.openai.com/api-keys
        echo OPENAI_API_KEY=
        echo.
        echo # xAI Grok-2          →  https://console.x.ai/  (use XAI_API_KEY)
        echo XAI_API_KEY=
        echo.
        echo # Google Gemini       →  https://aistudio.google.com/app/apikey  (use GOOGLE_API_KEY)
        echo GOOGLE_API_KEY=
        echo.
        echo # ── OPTIONAL: Financial Datasets  ─────────────────────────────────────
        echo # Paid API for fundamentals enrichment (income stmt, balance sheet, cash flow,
        echo # real-time prices, company news). Supplements Yahoo Finance SI data.
        echo # Sign up: https://financialdatasets.ai  — add paid API key below.
        echo # On free tier set FD_CALL_DELAY=13 in .env to respect 5 req/min limit.
        echo FINANCIAL_DATASETS_API_KEY=
        echo FD_CALL_DELAY=1
        echo.
        echo # ── OPTIONAL: Short interest fallback ─────────────────────────────────
        echo # Finnhub free tier (60 req/min)  →  https://finnhub.io/dashboard
        echo FINNHUB_API_KEY=
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
