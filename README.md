# TRaNKSP — TRaNK Scenario Planner
**Short Squeeze Lifecycle Intelligence Platform**

A fully agentic short squeeze detection, tracking, and P&L modelling platform built with Python 3.11, FastAPI, LangGraph, and Claude Sonnet.

---

## Architecture Overview

```
TRaNKSP/
├── agents/squeeze/
│   ├── state.py           # LangGraph TypedDict state definitions
│   ├── tools.py           # 5 ReAct tools (search_news, SEC, short data, earnings, competitors)
│   ├── rag.py             # ChromaDB RAG (squeeze_news, squeeze_filings, squeeze_lifecycle)
│   ├── memory.py          # Per-ticker SQLite thesis history (max 5 generations)
│   ├── react_agent.py     # LangGraph ReAct agent (langgraph.prebuilt.create_react_agent)
│   ├── chains.py          # MapReduce synthesis + bearish thesis + score explanation
│   ├── nodes.py           # 3-stage pipeline node functions
│   ├── graph.py           # LangGraph compiled screener pipeline
│   ├── universe_builder.py# Live internet universe builder (Finviz, Yahoo, Reddit WSB, StockAnalysis)
│   ├── lifecycle_tracker.py # Daily lifecycle evaluation with 3 bearish triggers
│   ├── options_analyzer.py  # yfinance options chain + Black-Scholes pricing
│   ├── scenario_calculator.py # 3-scenario P&L (Stock, Calls, Puts, Combined A+C)
│   ├── prompts.py         # All LLM prompt templates
│   └── output_schema.py   # Pydantic output models (ThesisOutput, BearishThesisOutput, etc.)
├── dashboard/
│   ├── app.py             # FastAPI backend (all endpoints + SSE + APScheduler)
│   └── index.html         # Vanilla HTML/CSS/JS frontend (6 tabs)
├── data/                  # SQLite DB + ChromaDB (auto-created)
├── logs/                  # Rotating log file (auto-created)
├── migrate_db.py          # Idempotent database migration
├── requirements.txt
├── setup_environment.bat  # First-time setup
└── start_program.bat      # Launch server
```

---

## Quick Start

### 1. First-Time Setup
```
setup_environment.bat
```
This creates the virtual environment, installs dependencies, creates `.env`, and runs the database migration.

### 2. Add API Keys
Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

### 3. Launch
```
start_program.bat
```
Open: **http://localhost:8765**

---

## Feature Reference

### Tab 1 — Short Squeeze
- **Universe**: Populated from Intake tab
- **Run Screen**: Triggers the 3-stage LangGraph pipeline:
  1. **Screen** (parallel yfinance): scores all universe tickers, filters by market cap and min score
  2. **Enrich** (ReAct agent): searches news, SEC filings, short data, earnings, competitors per ticker
  3. **Thesis** (MapReduce): Haiku maps chunks → Sonnet reduces to structured `ThesisOutput`
- **Cards**: Ticker, score ring, 6 metrics, catalyst chips, thesis setup preview
- **Thesis button**: Full structured thesis (setup, trigger, mechanics, risk, confidence)
- **Explain button**: On-demand LLM score explanation via Haiku
- **Scenario button**: Auto-populates Scenario Planner
- **Re-eval button**: Manual lifecycle re-evaluation

### Tab 2 — Scenario Planner
3 scenarios calculated using Black-Scholes:
- **A**: Long stock (capital / entry = shares)
- **B**: ATM call options at detection (estimated entry, estimated or actual exit)
- **C**: ATM-ish put options at squeeze peak (estimated entry, estimated or actual exit)
- **Combined A+C**: Total P&L of long + puts

User can override option premiums with actual market prices.

### Tab 3 — Intake
- **Build from Web**: Parallel async scrape of Finviz, Yahoo, StockAnalysis, Reddit WSB
- **Add Manual**: Add any ticker by symbol
- **Remove**: Deactivate any ticker (soft delete)

### Tab 4 — History
Archive of closed squeeze cycles (SQUEEZE_COMPLETE, REVERSAL, FAILED, STALE) with full lifecycle metrics.

### Tab 5 — Logs
Rotating file handler at `logs/trankSP.log`. Quick filters: ReAct, MapReduce, RAG, Memory, Thesis, Errors. Auto-refreshes every 5 seconds.

### Tab 6 — Settings
| Setting | Default | Description |
|---|---|---|
| capital_per_scenario | $15,000 | Capital per scenario calculation |
| portfolio_size | 10 | Max tickers enriched per run |
| min_score_threshold | 40 | Minimum composite score 0–100 |
| mc_floor | $250M | Minimum market cap |
| put_oi_threshold | 500 | Liquid puts OI threshold |
| auto_eval_hours | 72 | Hours before bearish eval triggers |

---

## Scoring Algorithm

| Factor | Weight | Scoring |
|---|---|---|
| Short Float % | 35% | ≥50%→35, ≥30%→28, ≥20%→21, ≥10%→14, else→7 |
| Days to Cover | 25% | ≥10→25, ≥7→20, ≥5→15, ≥3→10, else→5 |
| Volume Ratio | 20% | ≥5x→20, ≥3x→16, ≥2x→12, ≥1x→8, else→4 |
| Float Size (M) | 10% | ≤5→10, ≤10→8, ≤20→6, ≤50→4, else→2 |
| SI Trend | 10% | RISING→10, FLAT→5, FALLING→0 |

---

## Lifecycle Statuses

| Status | Meaning |
|---|---|
| ACTIVE | Detected, monitoring |
| SQUEEZE_FIRING | Price up ≥20% from entry |
| SQUEEZE_COMPLETE | SI dropped ≥50% from detection |
| REVERSAL | Price down ≥30% from peak |
| FAILED | Price down ≥20% from entry, SI unchanged |
| STALE | No updates |

---

## Bearish Trigger Logic (runs after auto_eval_hours)
1. **squeeze_complete**: SI dropped ≥50% from detection level
2. **post_spike_reversal**: Price dropped ≥30% from peak price
3. **squeeze_failed**: Price dropped ≥20% from entry AND SI unchanged (±10%)

---

## Confirmed Working Import Paths
- `create_react_agent`: `langgraph.prebuilt`
- `ChatAnthropic`: `langchain_anthropic`
- `SQLChatMessageHistory` pattern: custom SQLite implementation in `memory.py`

---

## Database Tables

| Table | Purpose |
|---|---|
| `squeeze_universe` | Live ticker universe |
| `squeeze_results` | Screener run results |
| `squeeze_lifecycle` | Daily lifecycle snapshots |
| `squeeze_options` | Options chain snapshots |
| `squeeze_scenarios` | P&L calculations |
| `squeeze_thesis_details` | Structured ThesisOutput per generation |
| `langchain_chat_history` | Per-ticker thesis history (max 5) |
| `settings` | Key-value user config |
| `agent_logs` | Agent activity log |

---

## ChromaDB Collections

| Collection | Purpose |
|---|---|
| `squeeze_news` | News chunks per ticker (RAG for thesis) |
| `squeeze_filings` | SEC filing chunks per ticker |
| `squeeze_lifecycle` | Daily lifecycle snapshots (semantic recall for bearish thesis) |
