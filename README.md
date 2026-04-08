# TRaNKSP — TRaNK Scenario Planner
**Short Squeeze Lifecycle Intelligence Platform**

A fully agentic short squeeze detection, tracking, and P&L modelling platform built with Python 3.11, FastAPI, LangGraph, and Claude Sonnet.

---

## Architecture Overview

```
TRaNKSP/
├── agents/squeeze/
│   ├── state.py                   # LangGraph TypedDict state definitions
│   ├── tools.py                   # 5 ReAct tools (search_news, SEC, short data, earnings, competitors)
│   ├── rag.py                     # ChromaDB RAG (squeeze_news, squeeze_filings, squeeze_lifecycle)
│   ├── memory.py                  # Per-ticker SQLite thesis history (max 5 generations)
│   ├── react_agent.py             # LangGraph ReAct agent (langgraph.prebuilt.create_react_agent)
│   ├── chains.py                  # MapReduce synthesis + bearish thesis + score explanation
│   ├── nodes.py                   # 3-stage pipeline node functions
│   ├── graph.py                   # LangGraph compiled screener pipeline
│   ├── universe_builder.py        # Live internet universe builder (Finviz, Yahoo, Reddit WSB, StockAnalysis)
│   ├── multi_llm_client.py        # Multi-LLM universe builder (Claude + Grok + OpenAI + Gemini)
│   ├── financial_datasets_client.py # Financial Datasets API integration (fundamentals enrichment)
│   ├── lifecycle_tracker.py       # Daily lifecycle evaluation with 3 bearish triggers
│   ├── options_analyzer.py        # yfinance options chain + Black-Scholes pricing
│   ├── scenario_calculator.py     # 3-scenario P&L (Stock, Calls, Puts, Combined A+C)
│   ├── prompts.py                 # All LLM prompt templates
│   └── output_schema.py           # Pydantic output models (ThesisOutput, BearishThesisOutput, etc.)
├── dashboard/
│   ├── app.py                     # FastAPI backend (all endpoints + SSE + APScheduler)
│   └── index.html                 # Vanilla HTML/CSS/JS frontend (6 tabs + new controls)
├── data/                          # SQLite DB + ChromaDB (auto-created)
├── logs/                          # Rotating log file (auto-created)
├── migrate_db.py                  # Idempotent database migration
├── requirements.txt
├── setup_environment.bat          # First-time setup
└── start_program.bat              # Launch server
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
# Required
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
MASSIVE_API_KEY=your_polygon_key...

# Optional — Multi-LLM consensus (each adds a parallel voice)
XAI_API_KEY=             # Grok-3
OPENAI_API_KEY=          # GPT-4.1-mini
GOOGLE_API_KEY=          # Gemini 2.5 Flash

# Optional — Financial Datasets fundamentals enrichment
FINANCIAL_DATASETS_API_KEY=
FD_CALL_DELAY=1          # Set to 13 if on free tier (5 req/min)

# Optional — Short interest fallback
FINNHUB_API_KEY=
```

### 3. Launch
```
start_program.bat
```
Open: **http://localhost:8765**

---

## LLM Model Versions (current as of April 2026)

| Provider | Model String | Role | Notes |
|---|---|---|---|
| **Anthropic Claude** | `claude-haiku-4-5-20251001` | Universe building + thesis | Required |
| **xAI Grok** | `grok-3` | Parallel universe voice | `grok-3` auto-aliases to latest stable |
| **OpenAI** | `gpt-4.1-mini` | Parallel universe voice | Upgraded from gpt-4o-mini — better & cheaper |
| **Google Gemini** | `gemini-2.5-flash` | Parallel universe voice | Uses `google-genai` SDK with 60s timeout |

> **Gemini note:** The old `google-generativeai` SDK has a known bug where `gemini-2.5-flash` hangs
> indefinitely instead of raising a timeout error when overloaded. TRaNKSP uses the new `google-genai`
> SDK with an explicit `asyncio.wait_for(..., timeout=60.0)` to prevent this.

---

## Feature Reference

### Tab 1 — Short Squeeze

#### Buttons (top right of tab)
| Button | Action |
|---|---|
| **↻ Refresh SI** | Re-fetch short interest from Yahoo for all visible cards |
| **💲 Check Prices** | Fetch latest prices from Yahoo → Massive for all DB tickers, streams progress live |
| **🤖 LLM Price Check** | Ask enabled LLMs for price + SI estimates. If ≥3 LLMs agree within 1%, accept Claude's value and tag with consensus note |
| **⚙ LLMs** | Toggle LLM selector panel — choose which providers to use for this run |
| **▶ Run Screen** | Trigger the full 3-stage LangGraph pipeline |

#### LLM Selector Panel
Checkboxes for Claude / Grok / OpenAI / Gemini. Uncheck any provider to skip it for that run. Useful when a provider is timing out or you want to save cost.

#### Enhanced Status Monitor (during Run Screen)
Replaces the plain progress bar with a live detail panel showing:
- **Identified by:** Claude-25 · Grok-24 · OpenAI-22 · Gemini-25 → Total: 96
- **Consensus:** 🔥 4-LLM: 8 · ⭐ 3-LLM: 14 · 💎 2-LLM: 22 · ⚪ 1-LLM: 52
- **Processing:** 34/96 · Avg time/ticker: ~43s · ETA to finish: ~27 min
- **Started:** 07:35:12 UTC · Est end: 08:02:00 UTC

#### LLM Price/SI Consensus Logic
When **🤖 LLM Price Check** is clicked:
1. Each enabled LLM is asked for price + short interest % + DTC per ticker
2. For each field, if ≥3 LLMs agree within 1%, Claude's value is accepted
3. DB is updated with a consensus note: `"Price updated based on LLM consensus: Claude, Grok, Gemini (all within 1% of $15.42)"`
4. Streams per-ticker progress with per-LLM breakdown visible in status panel

#### Screen Pipeline
- **Universe**: Multi-LLM build (Claude + enabled providers in parallel, consensus-ranked)
- **Run Screen**: Triggers the 3-stage LangGraph pipeline:
  1. **Screen** (parallel): Yahoo Finance → Massive → Financial Datasets → Finnhub waterfall per ticker
  2. **Enrich** (ReAct agent): news, SEC filings, Financial Datasets fundamentals per ticker
  3. **Thesis** (MapReduce): Haiku maps chunks → Sonnet reduces to structured `ThesisOutput`
- **Cards**: Ticker, score ring, 6 metrics, catalyst chips, thesis setup preview
- **Thesis button**: Full structured thesis (setup, trigger, mechanics, risk, confidence)
- **Explain button**: On-demand LLM score explanation via Haiku
- **Scenario button**: Auto-populates Scenario Planner
- **Re-eval button**: Manual lifecycle re-evaluation

### Tab 2 — Scenario Planner
3 scenarios calculated using Black-Scholes:
- **A**: Long stock (capital / entry = shares)
- **B**: ATM call options at detection
- **C**: ATM-ish put options at squeeze peak
- **Combined A+C**: Total P&L of long + puts

User can override option premiums with actual market prices.

### Tab 3 — Intake
- **Build from Web**: Multi-LLM universe build (Claude + Grok + OpenAI + Gemini in parallel, consensus-ranked by agreement level)
- **Add Manual**: Add any ticker by symbol
- **Remove**: Deactivate any ticker (soft delete)
- Consensus badges: 🔥🔥🔥🔥 4-LLM · ⭐⭐⭐ 3-LLM · 💎💎 2-LLM · ⚪ 1-LLM

### Tab 4 — History
Archive of closed squeeze cycles (SQUEEZE_COMPLETE, REVERSAL, FAILED, STALE) with full lifecycle metrics.

### Tab 5 — Logs
Rotating file handler at `logs/trankSP.log`. Quick filters: ReAct, MapReduce, RAG, Memory, Thesis, Errors. Auto-refreshes every 5 seconds.

### Tab 6 — LLM Consensus
Cross-LLM comparison table showing which providers picked each ticker, with consensus score and catalyst.

### Tab 7 — Settings
| Setting | Default | Description |
|---|---|---|
| capital_per_scenario | $15,000 | Capital per scenario calculation |
| portfolio_size | 10 | Max tickers enriched per run |
| min_score_threshold | 40 | Minimum composite score 0–100 |
| mc_floor | $250M | Minimum market cap |
| put_oi_threshold | 500 | Liquid puts OI threshold |
| auto_eval_hours | 72 | Hours before bearish eval triggers |

---

## Data Source Waterfall (per ticker, Screen stage)

| Priority | Source | Data | Condition |
|---|---|---|---|
| 1 | Yahoo Finance quoteSummary | Price + SI % + DTC + float + market cap | Always tried first |
| 2 | Massive (Polygon) daily aggs | Price + volume ratio | If Yahoo returns no price |
| 3 | Financial Datasets snapshot | Price | If Yahoo + Massive both fail AND FD key set |
| 4 | Finnhub short interest | SI % + DTC | If Yahoo returns no SI AND FINNHUB_API_KEY set |
| 5 | Massive reference | Market cap + float | Fallback for missing fundamentals |

### Financial Datasets Enrichment (Enrich stage)
When `FINANCIAL_DATASETS_API_KEY` is set, each enriched ticker also receives:
- **Income statement**: Revenue trend (GROWING/FLAT/DECLINING), net income, gross profit
- **Balance sheet**: Cash, total debt, debt/equity ratio, debt load (HIGH/MODERATE/LOW)
- **Cash flow**: Free cash flow, cash burn status (BURNING/GENERATING)
- **Company news**: Up to 8 recent headlines fed into thesis context

All of this is injected into the `adaptive_context` that Claude uses for thesis generation.

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

## Database Tables

| Table | Purpose |
|---|---|
| `squeeze_universe` | Live ticker universe with multi-LLM consensus columns |
| `squeeze_results` | Screener run results (includes `price_consensus_note`, `si_consensus_note`) |
| `squeeze_lifecycle` | Daily lifecycle snapshots |
| `squeeze_options` | Options chain snapshots |
| `squeeze_scenarios` | P&L calculations |
| `squeeze_thesis_details` | Structured ThesisOutput per generation |
| `langchain_chat_history` | Per-ticker thesis history (max 5) |
| `llm_daily_suggestions` | Per-LLM ticker suggestions (powers LLM Consensus tab) |
| `settings` | Key-value user config |
| `agent_logs` | Agent activity log |
| `screen_runs` | Run history with delta stats |
| `squeeze_predictions` | Layer 1 outcome tracking |
| `lessons_learned` | Learning engine cycle notes |

---

## ChromaDB Collections

| Collection | Purpose |
|---|---|
| `squeeze_news` | News chunks per ticker (RAG for thesis) |
| `squeeze_filings` | SEC filing chunks per ticker |
| `squeeze_lifecycle` | Daily lifecycle snapshots (semantic recall for bearish thesis) |

---

## API Endpoints

### Screen
| Endpoint | Method | Description |
|---|---|---|
| `/api/screen/stream` | GET SSE | Full pipeline with per-LLM counts + timing |
| `/api/results` | GET | Latest run results |
| `/api/results/{ticker}` | GET | Latest result for specific ticker |
| `/api/runs/detail/{run_id}` | GET | Run details |

### Prices
| Endpoint | Method | Description |
|---|---|---|
| `/api/prices/check/stream` | GET SSE | Bulk price refresh (Yahoo → Massive) with live progress |
| `/api/prices/llm_consensus/stream` | GET SSE | LLM price+SI consensus with per-ticker streaming |
| `/api/prices/llm_consensus` | POST | Non-streaming fallback (API compatibility) |

### Universe
| Endpoint | Method | Description |
|---|---|---|
| `/api/universe/build` | POST | Multi-LLM universe build (accepts `enabled_providers` list) |
| `/api/universe/multi_llm` | GET | Universe with consensus badges |
| `/api/universe` | GET | Raw universe list |
| `/api/universe/add` | POST | Add ticker manually |
| `/api/universe/{ticker}` | DELETE | Deactivate ticker |

### Lifecycle, Scenarios, Predictions
| Endpoint | Method | Description |
|---|---|---|
| `/api/lifecycle` | GET | All lifecycle snapshots |
| `/api/lifecycle/{ticker}` | GET | Ticker history |
| `/api/lifecycle/evaluate` | POST | Manual re-evaluation |
| `/api/scenarios/calculate` | POST | Calculate P&L scenarios |
| `/api/scenarios/{ticker}` | GET | Saved scenarios |
| `/api/predictions` | GET | Prediction history |
| `/api/predictions/calibration` | GET | Calibration stats |
| `/api/predictions/evaluate-now` | POST | Trigger outcome evaluation |

---

## Known Issues & Fixes Applied

| Issue | Fix |
|---|---|
| `cannot access local variable 'os'` in nodes.py | Removed `import os` inside `_screen_one_ticker()` — was shadowing module-level import |
| Gemini 504 timeout / indefinite hang | Switched to `google-genai` SDK + `asyncio.wait_for(..., timeout=60.0)` |
| `google-generativeai` SDK deprecated | Replaced with `google-genai>=1.0.0` in requirements.txt |
| Yahoo 429 rate limiting | 2s inter-call delay + 30s back-off + crumb-free fallback |
| Yahoo 401 crumb expiry | Auto re-warm session on 401 |

---

## Confirmed Working Import Paths
- `create_react_agent`: `langgraph.prebuilt`
- `ChatAnthropic`: `langchain_anthropic`
- `SQLChatMessageHistory` pattern: custom SQLite implementation in `memory.py`
- Google Gemini: `from google import genai as google_genai` (new SDK)
