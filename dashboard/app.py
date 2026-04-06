"""
TRaNKSP — FastAPI Application
All REST endpoints + SSE streaming + APScheduler for daily lifecycle checks
"""

import os
import uuid
import json
import sqlite3
import asyncio
import logging
import logging.handlers
from typing import Any, Dict, List, Optional
from datetime import datetime, date
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Silence ChromaDB telemetry noise in logs
import os as _os
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
_os.environ.setdefault("CHROMA_TELEMETRY", "False")

# ── Logging Setup ─────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logger = logging.getLogger("tranksp")
logger.setLevel(logging.DEBUG)

# Rotating file handler
fh = logging.handlers.RotatingFileHandler(
    "logs/trankSP.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

# ── DB path ───────────────────────────────────────────────────────────────────
DB_PATH = os.path.join("data", "tranksp.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_settings_dict() -> Dict[str, str]:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT key, value FROM settings")
    rows = dict(c.fetchall())
    conn.close()
    return rows


# ── APScheduler ───────────────────────────────────────────────────────────────
scheduler = None

def setup_scheduler():
    global scheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from agents.squeeze.lifecycle_tracker import run_daily_lifecycle_check
    
    scheduler = AsyncIOScheduler(timezone="America/Chicago")
    scheduler.add_job(
        run_daily_lifecycle_check,
        CronTrigger(hour=8, minute=30),
        id="daily_lifecycle",
        replace_existing=True
    )

    from agents.squeeze.learning_engine import run_daily_outcome_check
    scheduler.add_job(
        run_daily_outcome_check,
        CronTrigger(hour=8, minute=45),
        id="daily_outcome_check",
        replace_existing=True
    )

    from agents.squeeze.prediction_tracker import daily_outcome_update
    scheduler.add_job(
        lambda: asyncio.create_task(daily_outcome_update()),
        CronTrigger(hour=9, minute=0),
        id="daily_outcome_update",
        replace_existing=True
    )
    scheduler.start()
    logger.info("[Scheduler] APScheduler started. Daily lifecycle check at 8:30 AM CT.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run migration on startup
    import migrate_db
    migrate_db.run_migrations()
    setup_scheduler()
    logger.info("[TRaNKSP] Server started.")
    # Pre-warm Yahoo session and acquire crumb at startup
    from agents.squeeze.yahoo_quote import _SESSION as _yq_session
    await asyncio.to_thread(_yq_session.warm_up)
    yield
    if scheduler:
        scheduler.shutdown()
    logger.info("[TRaNKSP] Server stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="TRaNKSP", version="1.0.0", lifespan=lifespan)

# Serve frontend
dashboard_dir = os.path.join(os.path.dirname(__file__))
app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(dashboard_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Inline SVG favicon — lightning bolt in TRaNKSP cyan
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#090c10"/>'
        '<polygon points="18,2 8,18 15,18 14,30 24,14 17,14" fill="#00e5ff"/>'
        '</svg>'
    )
    from fastapi.responses import Response
    return Response(content=svg.encode(), media_type="image/svg+xml")


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    tickers: Optional[List[str]] = None  # if None, use full universe

class ScenarioRequest(BaseModel):
    ticker: str
    capital: Optional[float] = None
    entry_price: float
    bullish_exit: float
    bearish_target: float
    peak_price: Optional[float] = None
    call_strike_override: Optional[float] = None
    put_strike_override:  Optional[float] = None
    call_entry_override: Optional[float] = None
    call_exit_override: Optional[float] = None
    put_entry_override: Optional[float] = None
    put_exit_override: Optional[float] = None

class SettingsUpdate(BaseModel):
    settings: Dict[str, str]

class ExplainRequest(BaseModel):
    ticker: str
    score: float
    short_float: float
    days_to_cover: float
    volume_ratio: float
    float_shares: float
    si_trend: str

class LifecycleManualRequest(BaseModel):
    ticker: str

class UniverseAddRequest(BaseModel):
    ticker: str
    source: str = "manual"

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return get_settings_dict()


@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    conn = get_db()
    c = conn.cursor()
    for key, value in req.settings.items():
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
    logger.info(f"[Settings] Updated: {list(req.settings.keys())}")
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE (Intake tab)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/universe")
def get_universe():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM squeeze_universe ORDER BY added_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.post("/api/universe/build")
async def build_universe():
    """Trigger live universe build from all internet sources."""
    from agents.squeeze.universe_builder import build_universe as _build
    
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT ticker FROM squeeze_universe WHERE active=1")
    existing = [r[0] for r in c.fetchall()]
    conn.close()
    
    candidates = await _build(existing_tickers=existing)
    
    conn = get_db()
    c = conn.cursor()
    added = 0
    for cand in candidates:
        try:
            c.execute("""
                INSERT OR IGNORE INTO squeeze_universe (ticker, source)
                VALUES (?, ?)
            """, (cand["ticker"], cand.get("source", "web")))
            if c.rowcount > 0:
                added += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    
    logger.info(f"[Universe] Added {added} new tickers. Total candidates: {len(candidates)}")
    return {"added": added, "total_found": len(candidates), "candidates": candidates}


@app.post("/api/universe/add")
def add_to_universe(req: UniverseAddRequest):
    ticker = req.ticker.upper().strip()
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO squeeze_universe (ticker, source) VALUES (?, ?)",
        (ticker, req.source)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "ticker": ticker}


@app.delete("/api/universe/{ticker}")
def remove_from_universe(ticker: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE squeeze_universe SET active=0 WHERE ticker=?", (ticker.upper(),))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER (Short Squeeze tab) — SSE streaming
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/screen/stream")
async def screen_stream(request: Request):
    """SSE endpoint: streams screener progress + results."""
    from agents.squeeze.graph import run_screener_pipeline

    async def event_generator():
        try:
            settings = get_settings_dict()
            
            from agents.squeeze.claude_universe import get_claude_universe, get_tickers_only

            run_id = str(uuid.uuid4())[:8]

            # ── Step 0: Ask Claude for today's top squeeze candidates ──────────
            universe_count = int(settings.get("universe_size", 30))
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'universe', 'message': f'Asking Claude to identify top {universe_count} short squeeze candidates for today...'})}\n\n"

            claude_candidates = await get_claude_universe(count=universe_count)
            claude_tickers    = get_tickers_only(claude_candidates)

            # Store Claude's context so thesis generation can reference it
            settings["_claude_candidates"] = {c["ticker"]: c for c in claude_candidates}

            # Merge with any manually pinned tickers from the universe DB
            conn = get_db()
            c_db = conn.cursor()
            c_db.execute("SELECT ticker FROM squeeze_universe WHERE active=1")
            pinned = [r[0] for r in c_db.fetchall()]
            conn.close()

            tickers = claude_tickers.copy()
            for t in pinned:
                if t not in tickers:
                    tickers.append(t)

            if not tickers:
                from agents.squeeze.universe_builder import SEED_TICKERS
                tickers = SEED_TICKERS

            logger.info(f"[Screen] Universe: {len(claude_tickers)} from Claude + {len(pinned)} pinned = {len(tickers)} total")

            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id, 'ticker_count': len(tickers), 'claude_universe': claude_tickers[:10]})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'screen', 'message': f'Claude identified {len(claude_tickers)} candidates. Validating live data for each...'})}\n\n"
            
            result = await run_screener_pipeline(run_id, tickers, settings)
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'enrich', 'message': 'Enriching candidates...'})}\n\n"
            
            # ── Save results to DB + compute delta stats ──────────────────────────
            conn = get_db()
            c = conn.cursor()

            # Get previous run stats for delta calculation
            prev_run = c.execute(
                "SELECT run_id, total_in_db FROM screen_runs ORDER BY run_at DESC LIMIT 1"
            ).fetchone()
            prev_run_id = prev_run[0] if prev_run else None
            prev_total = prev_run[1] if prev_run else 0

            # Get tickers already in squeeze_results from ANY prior run
            existing_tickers_in_db = set(
                r[0] for r in c.execute("SELECT DISTINCT ticker FROM squeeze_results").fetchall()
            )

            new_tickers = 0
            updated_tickers = 0

            for r in result["results"]:
                ticker = r["ticker"]
                thesis = r.get("thesis") or {}

                # Track new vs updated
                if ticker in existing_tickers_in_db:
                    updated_tickers += 1
                else:
                    new_tickers += 1

                # Save screener result (always insert — each run is a new record)
                c.execute("""
                    INSERT INTO squeeze_results
                        (run_id, ticker, score, short_float, days_to_cover, float_shares,
                         price, market_cap, volume_ratio, si_trend, has_options, phase)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, ticker, r.get("score"), r.get("short_float"),
                    r.get("days_to_cover"), r.get("float_shares"), r.get("price"),
                    r.get("market_cap"), r.get("volume_ratio"), r.get("si_trend"),
                    int(r.get("has_options", False)), r.get("phase", "DETECTION")
                ))

                # Save thesis
                if thesis:
                    c.execute("""
                        INSERT INTO squeeze_thesis_details
                            (ticker, run_id, phase, setup, trigger, mechanics, risk,
                             catalyst_types, confidence, time_horizon)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        ticker, run_id, "BULLISH",
                        thesis.get("setup"), thesis.get("trigger"),
                        thesis.get("mechanics"), thesis.get("risk"),
                        json.dumps(thesis.get("catalyst_types", [])),
                        thesis.get("confidence"), thesis.get("time_horizon")
                    ))

                # Init lifecycle if ticker is brand new
                lc_existing = c.execute(
                    "SELECT id FROM squeeze_lifecycle WHERE ticker=?", (ticker,)
                ).fetchone()
                if not lc_existing:
                    c.execute("""
                        INSERT OR IGNORE INTO squeeze_lifecycle
                            (ticker, snapshot_date, status, entry_price, current_price, short_interest)
                        VALUES (?,?,?,?,?,?)
                    """, (
                        ticker, date.today().isoformat(), "ACTIVE",
                        r.get("price"), r.get("price"), r.get("short_float")
                    ))
                    from agents.squeeze.rag import store_lifecycle_snapshot
                    snap_text = (
                        f"DETECTION: {ticker} | Date: {date.today().isoformat()}\n"
                        f"Price: ${r.get('price', 0):.2f} | SI: {r.get('short_float', 0):.1f}% | "
                        f"DTC: {r.get('days_to_cover', 0):.1f} | Score: {r.get('score', 0):.0f}"
                    )
                    store_lifecycle_snapshot(ticker, snap_text, date.today().isoformat(), "ACTIVE")

            # Total unique tickers now in DB
            total_in_db = c.execute(
                "SELECT COUNT(DISTINCT ticker) FROM squeeze_results"
            ).fetchone()[0]

            # Save run stats
            run_at_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("""
                INSERT OR REPLACE INTO screen_runs
                    (run_id, run_at, tickers_screened, candidates_found,
                     new_tickers, updated_tickers, total_in_db, prev_run_id, prev_total)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                run_id, run_at_str, len(tickers), len(result["results"]),
                new_tickers, updated_tickers, total_in_db, prev_run_id, prev_total
            ))

            conn.commit()
            conn.close()

            # Build run summary for UI
            run_summary = {
                "run_id": run_id,
                "run_at": run_at_str,
                "tickers_screened": len(tickers),
                "candidates_found": len(result["results"]),
                "new_tickers": new_tickers,
                "updated_tickers": updated_tickers,
                "total_in_db": total_in_db,
                "prev_total": prev_total,
                "delta": total_in_db - prev_total,
            }

            logger.info(
                f"[Screen] Run {run_id}: {len(result['results'])} candidates "
                f"({new_tickers} new, {updated_tickers} updated). "
                f"Total in DB: {total_in_db} (+{total_in_db - prev_total} vs last run)"
            )

            yield f"data: {json.dumps({'type': 'results', 'results': result['results'], 'run_id': run_id, 'run_summary': run_summary})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'count': len(result['results']), 'run_summary': run_summary})}\n\n"

        except Exception as e:
            logger.error(f"[Screen Stream] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/screen/runs")
def get_screen_runs(limit: int = 20):
    """Get run history with delta stats for the Run History banner."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT run_id, run_at, tickers_screened, candidates_found,
               new_tickers, updated_tickers, total_in_db, prev_total,
               (total_in_db - prev_total) as delta
        FROM screen_runs
        ORDER BY run_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(zip(
        ["run_id","run_at","tickers_screened","candidates_found",
         "new_tickers","updated_tickers","total_in_db","prev_total","delta"],
        r
    )) for r in c.fetchall()]
    conn.close()
    return rows


@app.get("/api/results")
def get_results(limit: int = 50):
    """Get most recent screener results."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT sr.*, std.setup, std.trigger, std.confidence, std.time_horizon,
               std.catalyst_types, std.mechanics, std.risk
        FROM squeeze_results sr
        LEFT JOIN squeeze_thesis_details std ON std.ticker = sr.ticker 
            AND std.run_id = sr.run_id
        WHERE sr.run_id = (SELECT run_id FROM squeeze_results ORDER BY screened_at DESC LIMIT 1)
        ORDER BY sr.score DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.get("/api/results/{ticker}")
def get_ticker_result(ticker: str):
    """Get latest result for a specific ticker."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT sr.*, std.setup, std.trigger, std.confidence, std.time_horizon,
               std.catalyst_types, std.mechanics, std.risk
        FROM squeeze_results sr
        LEFT JOIN squeeze_thesis_details std ON std.ticker = sr.ticker AND std.run_id = sr.run_id
        WHERE sr.ticker = ?
        ORDER BY sr.screened_at DESC LIMIT 1
    """, (ticker.upper(),))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Ticker not found")
    return dict(row)


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/lifecycle")
def get_lifecycle_all():
    """Get latest lifecycle snapshot for all tracked tickers."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT sl.*
        FROM squeeze_lifecycle sl
        WHERE sl.snapshot_date = (
            SELECT MAX(sl2.snapshot_date) FROM squeeze_lifecycle sl2 WHERE sl2.ticker = sl.ticker
        )
        ORDER BY sl.ticker
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.get("/api/lifecycle/{ticker}")
def get_lifecycle_ticker(ticker: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_lifecycle WHERE ticker=?
        ORDER BY snapshot_date DESC LIMIT 30
    """, (ticker.upper(),))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.post("/api/lifecycle/evaluate")
async def manual_evaluate(req: LifecycleManualRequest):
    """Manual trigger for lifecycle re-evaluation."""
    from agents.squeeze.lifecycle_tracker import evaluate_ticker_lifecycle
    
    ticker = req.ticker.upper()
    conn = get_db()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_lifecycle WHERE ticker=?
        ORDER BY snapshot_date DESC LIMIT 1
    """, (ticker,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"{ticker} not in lifecycle tracking")
    
    result = await evaluate_ticker_lifecycle(ticker, dict(row))
    logger.info(f"[Lifecycle] Manual re-eval for {ticker}: {result['snapshot']['status']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO PLANNER
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/scenarios/calculate")
def calculate_scenarios(req: ScenarioRequest):
    from agents.squeeze.scenario_calculator import calculate_all_scenarios
    
    settings = get_settings_dict()
    capital = req.capital or float(settings.get("capital_per_scenario", 15000))
    
    result = calculate_all_scenarios(
        ticker=req.ticker.upper(),
        capital=capital,
        entry_price=req.entry_price,
        bullish_exit_price=req.bullish_exit,
        bearish_target_price=req.bearish_target,
        peak_price=req.peak_price,
        call_strike_override=req.call_strike_override,
        put_strike_override=req.put_strike_override,
        call_entry_override=req.call_entry_override,
        call_exit_override=req.call_exit_override,
        put_entry_override=req.put_entry_override,
        put_exit_override=req.put_exit_override
    )
    return result


@app.get("/api/scenarios/{ticker}")
def get_scenarios(ticker: str):
    from agents.squeeze.scenario_calculator import get_saved_scenarios
    result = get_saved_scenarios(ticker.upper())
    if not result:
        raise HTTPException(status_code=404, detail="No scenarios calculated yet")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE OPTIONS RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/options/recommend/{ticker}")
async def recommend_options(ticker: str, req: dict = None):
    """
    Ask Claude to recommend specific call and put options for a squeeze play.
    Returns strike prices, expiry, estimated premiums, and rationale.
    """
    import anthropic as _anthropic
    from datetime import date, timedelta

    ticker = ticker.upper().strip()

    # Get ticker data
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1", (ticker,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"{ticker} not found")

    r = dict(row)
    price        = r.get("price", 0)
    short_float  = r.get("short_float", 0)
    dtc          = r.get("days_to_cover", 0)
    si_trend     = r.get("si_trend", "FLAT")
    score        = r.get("score", 0)

    # Also get any user-supplied price overrides from request body
    entry_price  = (req or {}).get("entry_price", price) if req else price
    peak_price   = (req or {}).get("peak_price", price * 1.70) if req else price * 1.70
    bearish_tgt  = (req or {}).get("bearish_target", price * 0.60) if req else price * 0.60
    capital      = (req or {}).get("capital", 15000) if req else 15000

    today        = date.today()
    exp_near     = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    exp_mid      = (today + timedelta(days=60)).strftime("%Y-%m-%d")
    exp_far      = (today + timedelta(days=90)).strftime("%Y-%m-%d")

    prompt = f"""You are an options trading strategist specializing in short squeeze plays.

TICKER: {ticker}
Current Price: ${price:.2f}
Short Float: {short_float:.1f}%
Days to Cover: {dtc:.1f}
SI Trend: {si_trend}
Squeeze Score: {score:.0f}/100
Capital to deploy: ${capital:,.0f}
Entry price assumption: ${entry_price:.2f}
Bullish squeeze target: ${peak_price:.2f}
Post-squeeze bearish target: ${bearish_tgt:.2f}
Today's date: {today.isoformat()}
Near expiry: {exp_near}
Mid expiry: {exp_mid}
Far expiry: {exp_far}

Recommend SPECIFIC options contracts for this short squeeze play. Consider:
1. BULLISH CALL (to profit from squeeze): 
   - Strike should be slightly OTM (5-10% above current price) for leverage, or ATM for higher probability
   - Expiry should give enough time for the squeeze to play out (30-60 days typically)
   - Consider the days-to-cover ratio when choosing expiry
2. BEARISH PUT (to profit from post-squeeze collapse):
   - Strike near the squeeze peak target price
   - Shorter expiry (14-30 days) as post-squeeze collapses are faster
   - Or put spread if premium is expensive
3. Estimate realistic option premiums based on typical IV for this type of stock

Respond ONLY with valid JSON, no markdown, no explanation:
{{
  "call": {{
    "strike": 33.00,
    "expiry": "{exp_mid}",
    "expiry_label": "60 days",
    "est_premium": 1.40,
    "contracts_for_capital": 107,
    "rationale": "One sentence why this strike/expiry",
    "strategy": "Long Call",
    "max_loss": 14980,
    "breakeven": 34.40
  }},
  "put": {{
    "strike": 50.00,
    "expiry": "{exp_near}",
    "expiry_label": "30 days",
    "est_premium": 2.26,
    "contracts_for_capital": 66,
    "rationale": "One sentence why this strike/expiry",
    "strategy": "Long Put",
    "max_loss": 14916,
    "breakeven": 47.74
  }},
  "alternative_call": {{
    "strike": 35.00,
    "expiry": "{exp_far}",
    "expiry_label": "90 days",
    "est_premium": 0.95,
    "contracts_for_capital": 157,
    "rationale": "More aggressive OTM play with higher leverage",
    "strategy": "OTM Call Lottery",
    "max_loss": 14915,
    "breakeven": 35.95
  }},
  "summary": "Brief 2-sentence overview of the options strategy for this ticker"
}}"""

    try:
        client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        # Clean JSON
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    raw = part
                    break
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        rec = json.loads(raw)
        rec["ticker"]      = ticker
        rec["price"]       = price
        rec["entry_price"] = entry_price
        rec["capital"]     = capital
        logger.info(f"[Options] Claude recommended for {ticker}: "
                    f"Call ${rec.get('call',{}).get('strike')} | "
                    f"Put ${rec.get('put',{}).get('strike')}")
        return rec

    except Exception as e:
        logger.error(f"[Options] Claude recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/options/{ticker}")
async def get_options(ticker: str):
    from agents.squeeze.options_analyzer import analyze_options, get_latest_options
    
    settings = get_settings_dict()
    put_oi_threshold = int(settings.get("put_oi_threshold", 500))
    
    # Use cached if today's snapshot exists
    cached = get_latest_options(ticker.upper())
    if cached and cached.get("snapshot_date") == date.today().isoformat():
        return cached
    
    snap = analyze_options(ticker.upper(), put_oi_threshold)
    return snap.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE THESIS ON DEMAND
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/generate-thesis/{ticker}")
async def generate_thesis_on_demand(ticker: str):
    """Generate or regenerate thesis for a ticker already in the DB."""
    ticker = ticker.upper().strip()

    # Get ticker data from most recent result
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_results WHERE ticker=?
        ORDER BY screened_at DESC LIMIT 1
    """, (ticker,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"{ticker} not in results")

    r = dict(row)

    candidate = {
        "ticker":        ticker,
        "score":         r.get("score", 0),
        "short_float":   r.get("short_float", 0),
        "days_to_cover": r.get("days_to_cover", 0),
        "float_shares":  r.get("float_shares", 0),
        "price":         r.get("price", 0),
        "market_cap":    r.get("market_cap", 0),
        "volume_ratio":  r.get("volume_ratio", 1),
        "si_trend":      r.get("si_trend", "FLAT"),
        "has_options":   bool(r.get("has_options", 0)),
        "phase":         r.get("phase", "DETECTION"),
    }

    from agents.squeeze.chains import generate_thesis_direct
    from agents.squeeze.memory import format_history_for_context
    from agents.squeeze.learning_engine import format_calibration_for_prompt, query_episode_memory

    quant_data = {
        "short_float":   f"{candidate['short_float']:.1f}",
        "days_to_cover": f"{candidate['days_to_cover']:.1f}",
        "float_shares":  f"{candidate['float_shares']:.1f}",
        "price":         f"{candidate['price']:.2f}",
        "market_cap":    f"{candidate['market_cap']:.0f}",
        "volume_ratio":  f"{candidate['volume_ratio']:.1f}",
        "si_trend":      candidate["si_trend"]
    }

    prior_episodes  = query_episode_memory(ticker)
    calibration     = format_calibration_for_prompt()
    lifecycle_ctx   = format_history_for_context(ticker)
    sep = "\n\n"
    prior_block = f"PRIOR EPISODES:\n{prior_episodes}" if prior_episodes != "No prior prediction history." else ""
    adaptive_ctx = sep.join(filter(None, [calibration, lifecycle_ctx, prior_block]))

    thesis = await generate_thesis_direct(
        ticker, quant_data, "Current market data.", "No recent SEC filings.", adaptive_ctx
    )

    if not thesis:
        raise HTTPException(status_code=500, detail="Thesis generation failed")

    # Save to DB
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO squeeze_thesis_details
            (ticker, run_id, phase, setup, trigger, mechanics, risk,
             catalyst_types, confidence, time_horizon)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        ticker, r.get("run_id", "manual"), "BULLISH",
        thesis.setup, thesis.trigger, thesis.mechanics, thesis.risk,
        json.dumps(thesis.catalyst_types), thesis.confidence, thesis.time_horizon
    ))
    conn.commit()
    conn.close()

    return {
        "ticker": ticker,
        "thesis": thesis.model_dump()
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/explain-score")
async def explain_score_endpoint(req: ExplainRequest):
    from agents.squeeze.chains import explain_score
    
    result = await explain_score(req.ticker, req.model_dump())
    if not result:
        raise HTTPException(status_code=500, detail="Score explanation failed")
    return result.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY (closed/completed squeezes)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT sl.ticker, sl.status, sl.entry_price, sl.peak_price, sl.current_price,
               sl.short_interest, sl.si_change_pct, sl.price_chg_peak, sl.snapshot_date,
               sl.eval_triggered, sl.notes,
               std.confidence, std.setup, std.time_horizon
        FROM squeeze_lifecycle sl
        LEFT JOIN squeeze_thesis_details std ON std.ticker = sl.ticker AND std.phase='BULLISH'
        WHERE sl.status IN ('SQUEEZE_COMPLETE', 'REVERSAL', 'FAILED', 'STALE')
        AND sl.snapshot_date = (
            SELECT MAX(sl2.snapshot_date) FROM squeeze_lifecycle sl2 WHERE sl2.ticker = sl.ticker
        )
        ORDER BY sl.snapshot_date DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LOGS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(filter: str = "", limit: int = 200):
    """Read from rotating log file and return lines."""
    log_path = "logs/trankSP.log"
    lines = []
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        # Most recent first
        all_lines = list(reversed(all_lines))
        
        for line in all_lines:
            if not line.strip():
                continue
            if filter and filter.lower() not in line.lower():
                continue
            lines.append(line.strip())
            if len(lines) >= limit:
                break
    except FileNotFoundError:
        lines = ["Log file not found. Run a screen to generate logs."]
    except Exception as e:
        lines = [f"Log read error: {str(e)}"]
    
    return {"lines": lines}


# ─────────────────────────────────────────────────────────────────────────────
# THESIS HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/thesis-history/{ticker}")
def get_thesis_history(ticker: str):
    from agents.squeeze.memory import get_thesis_history
    return get_thesis_history(ticker.upper())


# ─────────────────────────────────────────────────────────────────────────────
# TICKER REFRESH — re-fetch SI data from Yahoo for a specific ticker
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ticker/refresh/{ticker}")
async def refresh_ticker_si(ticker: str):
    """
    Re-fetch short interest data from Yahoo for a specific ticker.
    Updates the most recent squeeze_results row and lifecycle snapshot.
    Called by the Re-eval / Refresh button on each card.
    """
    from agents.squeeze.yahoo_quote import YahooQuoteSession
    from agents.squeeze.massive_client import get_price_and_volume
    import os

    ticker = ticker.upper().strip()

    # Fetch fresh data
    session = YahooQuoteSession(delay=1.0)
    quote = await asyncio.to_thread(session.get_quote, ticker)

    if not quote or quote.get("price", 0) == 0:
        # Fallback to Massive for price
        snap = await asyncio.to_thread(get_price_and_volume, ticker)
        if snap:
            if not quote:
                quote = {"short_float": 0, "days_to_cover": 0, "si_trend": "FLAT",
                         "market_cap": 0, "float_shares": 0}
            quote["price"]        = snap["price"]
            quote["volume_ratio"] = snap["volume_ratio"]
        else:
            raise HTTPException(status_code=503, detail=f"Could not fetch data for {ticker}")

    # Update squeeze_results — most recent row for this ticker
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        UPDATE squeeze_results SET
            short_float   = ?,
            days_to_cover = ?,
            si_trend      = ?,
            price         = ?,
            volume_ratio  = ?,
            float_shares  = ?,
            market_cap    = ?
        WHERE ticker = ?
        AND run_id = (SELECT run_id FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1)
    """, (
        quote.get("short_float", 0),
        quote.get("days_to_cover", 0),
        quote.get("si_trend", "FLAT"),
        quote.get("price", 0),
        quote.get("volume_ratio", 1.0),
        quote.get("float_shares", 0),
        quote.get("market_cap", 0),
        ticker, ticker
    ))

    # Update lifecycle snapshot
    c.execute("""
        UPDATE squeeze_lifecycle SET
            current_price = ?,
            short_interest = ?
        WHERE ticker = ?
        AND snapshot_date = (SELECT MAX(snapshot_date) FROM squeeze_lifecycle WHERE ticker=?)
    """, (
        quote.get("price", 0),
        quote.get("short_float", 0),
        ticker, ticker
    ))

    conn.commit()
    conn.close()

    logger.info(f"[Refresh] {ticker}: SI={quote.get('short_float',0):.1f}% "
                f"DTC={quote.get('days_to_cover',0):.1f} ${quote.get('price',0):.2f}")

    return {
        "ticker": ticker,
        "short_float":   quote.get("short_float", 0),
        "days_to_cover": quote.get("days_to_cover", 0),
        "si_trend":      quote.get("si_trend", "FLAT"),
        "price":         quote.get("price", 0),
        "volume_ratio":  quote.get("volume_ratio", 1.0),
        "float_shares":  quote.get("float_shares", 0),
        "market_cap":    quote.get("market_cap", 0),
        "source":        quote.get("source", "yahoo"),
        "refreshed_at":  datetime.utcnow().isoformat()
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/predictions/stats")
def get_prediction_stats_endpoint():
    """Summary stats for the Predictions tab header."""
    from agents.squeeze.prediction_tracker import get_prediction_stats
    return get_prediction_stats()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "tavily_key": bool(os.environ.get("TAVILY_API_KEY")),
        "massive_key": bool(os.environ.get("MASSIVE_API_KEY")),
        "finnhub_key": bool(os.environ.get("FINNHUB_API_KEY"))
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS / LEARNING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/predictions")
def get_predictions(status: str = "ALL", limit: int = 100):
    conn = get_db()
    c = conn.cursor()
    if status == "ALL":
        c.execute("SELECT * FROM squeeze_predictions ORDER BY prediction_date DESC LIMIT ?", (limit,))
    else:
        c.execute("SELECT * FROM squeeze_predictions WHERE status=? ORDER BY prediction_date DESC LIMIT ?",
                  (status, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.get("/api/predictions/calibration")
def get_calibration():
    from agents.squeeze.learning_engine import get_calibration_stats
    return get_calibration_stats()


@app.get("/api/predictions/lessons")
def get_lessons():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM lessons_learned ORDER BY cycle_number DESC LIMIT 10")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


@app.post("/api/predictions/evaluate-now")
async def trigger_evaluation():
    """Manually trigger outcome evaluation for all open predictions."""
    from agents.squeeze.learning_engine import run_daily_outcome_check
    await run_daily_outcome_check()
    return {"status": "ok", "message": "Evaluation triggered"}


# ─────────────────────────────────────────────────────────────────────────────
# RUN DETAILS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/runs/detail/{run_id}")
def get_run_detail(run_id: str):
    from agents.squeeze.run_tracker import get_run_details
    return get_run_details(run_id)
