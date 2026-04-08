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


def get_consensus_badge(level: int) -> str:
    """Return visual consensus indicator for frontend."""
    if level == 4:   return "🔥🔥🔥🔥 4-LLM"
    elif level == 3: return "⭐⭐⭐ 3-LLM"
    elif level == 2: return "💎💎 2-LLM"
    else:            return "⚪ 1-LLM"
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

    scheduler = AsyncIOScheduler(timezone="America/Chicago")

    # AsyncIOScheduler runs in the same event loop as FastAPI.
    # Pass async functions DIRECTLY — never wrap in lambda/create_task.
    # APScheduler calls them with await internally.

    from agents.squeeze.lifecycle_tracker import run_daily_lifecycle_check
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
        daily_outcome_update,           # ← direct reference, NOT lambda/create_task
        CronTrigger(hour=9, minute=0),
        id="daily_outcome_update",
        replace_existing=True
    )

    scheduler.start()
    logger.info("[Scheduler] APScheduler started. Jobs: 8:30 lifecycle | 8:45 outcomes | 9:00 predictions")


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


@app.get("/api/universe/multi_llm")
def get_multi_llm_universe_endpoint():
    """Return ranked universe with consensus badges for Intake tab."""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("""
            SELECT ticker, llm_consensus, source_llm, short_float_est,
                   days_to_cover_est, catalyst, catalyst_type, confidence, added_at
            FROM squeeze_universe
            WHERE active=1
            ORDER BY llm_consensus DESC, confidence DESC, added_at DESC
        """)
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
    except Exception as e:
        # Columns may not exist yet — fall back to basic query
        logger.warning(f"[Universe] multi_llm query failed ({e}), falling back to basic")
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT ticker, source, added_at FROM squeeze_universe WHERE active=1")
            rows = [{"ticker": r["ticker"], "source": r["source"],
                     "llm_consensus": 1, "badge": "⚪ 1-LLM",
                     "added_at": r["added_at"]} for r in c.fetchall()]
            conn.close()
        except Exception as e2:
            logger.error(f"[Universe] fallback query also failed: {e2}")
            return {"tickers": [], "summary": {"total": 0, "4_llm": 0, "3_llm": 0, "2_llm": 0, "1_llm": 0}}

    for row in rows:
        row["badge"] = get_consensus_badge(row.get("llm_consensus") or 1)

    return {
        "tickers": rows,
        "summary": {
            "total": len(rows),
            "4_llm": sum(1 for r in rows if (r.get("llm_consensus") or 0) == 4),
            "3_llm": sum(1 for r in rows if (r.get("llm_consensus") or 0) == 3),
            "2_llm": sum(1 for r in rows if (r.get("llm_consensus") or 0) == 2),
            "1_llm": sum(1 for r in rows if (r.get("llm_consensus") or 0) == 1),
        }
    }


class UniverseBuildRequest(BaseModel):
    enabled_providers: Optional[List[str]] = None  # None = all 4


@app.post("/api/universe/build")
async def build_universe(req: UniverseBuildRequest = None):
    """Trigger Multi-LLM universe build — Claude + Grok + OpenAI + Gemini in parallel."""
    from agents.squeeze.multi_llm_client import build_multi_llm_universe

    providers = (req.enabled_providers if req and req.enabled_providers else None)
    result = await build_multi_llm_universe(count_per_llm=25, enabled_providers=providers)

    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail="All LLM providers failed")

    ranked   = result.get("ranked_tickers", [])
    consensus = result.get("consensus", {})
    logger.info(
        f"[Universe] Multi-LLM build complete — "
        f"{result.get('total_unique_tickers',0)} unique | "
        f"4-LLM: {consensus.get('4_llm',0)} | "
        f"3-LLM: {consensus.get('3_llm',0)}"
    )
    return {
        "added":           result.get("total_unique_tickers", 0),
        "total_found":     result.get("total_unique_tickers", 0),
        "consensus":       consensus,
        "message":         result.get("message", ""),
        "top_4_llm":       result.get("top_4_llm", []),
        "top_3_llm":       result.get("top_3_llm", []),
        "providers_used":  result.get("providers_used", []),
        "provider_counts": result.get("provider_counts", {}),
    }


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

            # ── Step 0: Ask Multi-LLM for today's squeeze candidates ─────────
            universe_count     = int(settings.get("universe_size", 30))
            enabled_providers  = settings.get("enabled_providers", None)  # None = all 4

            from agents.squeeze.multi_llm_client import (
                build_multi_llm_universe, get_tickers_only as _mlm_tickers_only
            )

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'universe', 'pct': 5, 'message': f'Asking LLMs to identify top {universe_count} short squeeze candidates...'})}\n\n"

            screen_start_time = datetime.utcnow()
            universe_result = await build_multi_llm_universe(
                count_per_llm=min(universe_count, 25),
                enabled_providers=enabled_providers
            )

            ranked_candidates  = universe_result.get("ranked_tickers", [])
            all_tickers        = _mlm_tickers_only(ranked_candidates)
            provider_counts    = universe_result.get("provider_counts", {})
            providers_used     = universe_result.get("providers_used", [])
            consensus_counts   = universe_result.get("consensus", {})

            # Store full candidate context for thesis generation
            settings["_claude_candidates"] = {c["ticker"]: c for c in ranked_candidates}

            # Build per-LLM display string: "Claude-25, Grok-24, OpenAI-22, Gemini-25"
            llm_counts_str = ", ".join(
                f"{p.capitalize()}-{provider_counts.get(p,0)}"
                for p in providers_used
            )
            total_identified = len(all_tickers)

            yield f"data: {json.dumps({'type': 'llm_counts', 'provider_counts': provider_counts, 'providers_used': providers_used, 'consensus': consensus_counts, 'total': total_identified, 'llm_counts_str': llm_counts_str})}\n\n"

            # Merge with any manually pinned tickers from the universe DB
            conn = get_db()
            c_db = conn.cursor()
            c_db.execute("SELECT ticker FROM squeeze_universe WHERE active=1")
            pinned = [r[0] for r in c_db.fetchall()]
            conn.close()

            tickers = all_tickers.copy()
            for t in pinned:
                if t not in tickers:
                    tickers.append(t)

            if not tickers:
                from agents.squeeze.universe_builder import SEED_TICKERS
                tickers = SEED_TICKERS

            total_tickers = len(tickers)
            logger.info(f"[Screen] Universe: {len(all_tickers)} from LLMs + {len(pinned)} pinned = {total_tickers} total")

            # Estimate: ~45s/ticker average (Yahoo 2s + Massive fallback + enrichment)
            avg_sec_per_ticker = 45
            est_total_sec      = total_tickers * avg_sec_per_ticker
            est_finish_utc     = screen_start_time + __import__('datetime').timedelta(seconds=est_total_sec)

            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id, 'ticker_count': total_tickers, 'llm_counts_str': llm_counts_str, 'start_time': screen_start_time.strftime('%H:%M:%S UTC'), 'est_finish': est_finish_utc.strftime('%H:%M:%S UTC'), 'est_total_min': round(est_total_sec/60,1)})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'screen', 'pct': 15, 'message': f'LLMs identified {total_tickers} tickers. Validating live data...', 'processing': f'Processing 0/{total_tickers}', 'avg_sec': avg_sec_per_ticker, 'start_time': screen_start_time.strftime('%H:%M:%S UTC'), 'est_finish': est_finish_utc.strftime('%H:%M:%S UTC')})}\n\n"

            result = await run_screener_pipeline(run_id, tickers, settings)

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'enrich', 'pct': 75, 'message': 'Enriching candidates with news + SEC data...'})}\n\n"
            
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



# ─────────────────────────────────────────────────────────────────────────────
# LLM CONSENSUS PAGE
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/llm_consensus")
def get_llm_consensus(run_date: str = None):
    """
    Return 4-LLM consensus table for the Consensus tab.
    One row per (date × ticker) showing which LLMs picked it.
    Color-coded by llm_consensus count.
    """
    conn = get_db()
    c    = conn.cursor()

    if run_date:
        c.execute("""
            SELECT run_date, llm_name, ticker,
                   est_short_float, est_dtc, catalyst, catalyst_type, confidence
            FROM llm_daily_suggestions
            WHERE run_date = ?
            ORDER BY ticker
        """, (run_date,))
    else:
        # Default: today's most recent date that has data
        c.execute("""
            SELECT run_date, llm_name, ticker,
                   est_short_float, est_dtc, catalyst, catalyst_type, confidence
            FROM llm_daily_suggestions
            WHERE run_date = (SELECT MAX(run_date) FROM llm_daily_suggestions)
            ORDER BY ticker
        """)

    rows = [dict(r) for r in c.fetchall()]

    # All available dates for the dropdown
    c.execute("SELECT DISTINCT run_date FROM llm_daily_suggestions ORDER BY run_date DESC LIMIT 30")
    dates = [r[0] for r in c.fetchall()]
    conn.close()

    if not rows:
        return {"data": [], "dates": dates}

    # Pivot: one row per ticker showing ✓/blank per LLM
    from collections import defaultdict
    pivot = defaultdict(lambda: {
        "ticker": "", "run_date": "",
        "claude": "", "grok": "", "openai": "", "gemini": "",
        "llm_consensus": 0, "est_short_float": 0, "est_dtc": 0,
        "catalyst": "", "catalyst_type": "", "confidence": 0,
    })

    for r in rows:
        key    = f"{r['run_date']}_{r['ticker']}"
        llm    = r["llm_name"].lower()
        ticker = r["ticker"]
        d      = pivot[key]
        d["ticker"]   = ticker
        d["run_date"] = r["run_date"]
        # Mark which LLM picked this ticker
        if   llm == "claude":  d["claude"]  = ticker
        elif llm == "grok":    d["grok"]    = ticker
        elif llm == "openai":  d["openai"]  = ticker
        elif llm == "gemini":  d["gemini"]  = ticker
        # Use highest-confidence values
        if r.get("confidence", 0) > d["confidence"]:
            d["est_short_float"] = r.get("est_short_float", 0) or 0
            d["est_dtc"]         = r.get("est_dtc", 0) or 0
            d["catalyst"]        = r.get("catalyst", "") or ""
            d["catalyst_type"]   = r.get("catalyst_type", "") or ""
            d["confidence"]      = r.get("confidence", 0) or 0

    # Count consensus and check DB membership
    conn2 = get_db()
    c2    = conn2.cursor()

    result = []
    for key, row in pivot.items():
        count = sum(1 for col in ["claude","grok","openai","gemini"] if row[col])
        row["llm_consensus"] = count
        row["badge"]         = get_consensus_badge(count)

        # Is ticker already in squeeze_results (active card)?
        c2.execute("SELECT COUNT(*) FROM squeeze_results WHERE ticker=?", (row["ticker"],))
        in_squeeze = c2.fetchone()[0] > 0
        row["in_squeeze"] = in_squeeze
        row["action"]     = "Already in Short Squeeze" if in_squeeze else "Add to Short Squeeze"
        result.append(row)

    conn2.close()

    # Sort: 4-LLM first, then 3, 2, 1, then alphabetical within group
    result.sort(key=lambda x: (-x["llm_consensus"], x["ticker"]))

    return {"data": result, "dates": dates}


@app.post("/api/llm_consensus/move/{ticker}")
async def move_ticker_to_squeeze(ticker: str):
    """
    Add ticker to universe and immediately re-evaluate it
    (fetch SI data, score, and generate thesis) — same as Re-Eval card button.
    """
    ticker = ticker.upper().strip()

    # Add to universe
    conn = get_db()
    c    = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO squeeze_universe (ticker, source) VALUES (?, ?)",
        (ticker, "consensus_page")
    )
    conn.commit()
    conn.close()

    # Re-evaluate: fetch live data + lifecycle eval
    try:
        from agents.squeeze.lifecycle_tracker import evaluate_ticker_lifecycle
        conn = get_db()
        c    = conn.cursor()
        c.execute("SELECT * FROM squeeze_lifecycle WHERE ticker=? LIMIT 1", (ticker,))
        row  = c.fetchone()
        conn.close()

        if row:
            await evaluate_ticker_lifecycle(ticker, dict(row))
        else:
            # New ticker — seed it with Yahoo data
            from agents.squeeze.yahoo_quote import get_quote_data
            import asyncio as _asyncio
            quote = await _asyncio.to_thread(get_quote_data, ticker)
            if quote and quote.get("price", 0) > 0:
                conn = get_db()
                c    = conn.cursor()
                c.execute("""
                    INSERT OR IGNORE INTO squeeze_lifecycle
                        (ticker, snapshot_date, status, entry_price, current_price, short_interest)
                    VALUES (?,?,?,?,?,?)
                """, (
                    ticker, date.today().isoformat(), "ACTIVE",
                    quote["price"], quote["price"], quote.get("short_float", 0)
                ))
                conn.commit()
                conn.close()

        logger.info(f"[Consensus] Moved {ticker} to Short Squeeze queue")
        return {"status": "ok", "ticker": ticker, "message": f"{ticker} added to Short Squeeze"}

    except Exception as e:
        logger.error(f"[Consensus] Move failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ─────────────────────────────────────────────────────────────────────────────
# CHECK PRICES — bulk price refresh for all tickers in DB
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/prices/check/stream")
async def check_prices_stream(request: Request):
    """
    SSE endpoint: fetches latest prices from Yahoo then Massive for all
    tickers in squeeze_results. Overwrites price in DB. Streams progress.
    """
    from agents.squeeze.yahoo_quote import YahooQuoteSession
    from agents.squeeze.massive_client import get_price_and_volume

    async def price_generator():
        try:
            conn = get_db()
            c    = conn.cursor()
            c.execute("SELECT DISTINCT ticker FROM squeeze_results ORDER BY ticker")
            tickers = [r[0] for r in c.fetchall()]
            conn.close()

            if not tickers:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No tickers in DB'})}\n\n"
                return

            total   = len(tickers)
            session = YahooQuoteSession(delay=1.5)
            updated = 0
            failed  = 0

            yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

            for idx, ticker in enumerate(tickers, 1):
                if await request.is_disconnected():
                    break

                price  = 0
                source = "none"

                # Try Yahoo first
                try:
                    quote = await asyncio.to_thread(session.get_quote, ticker)
                    if quote and quote.get("price", 0) > 0:
                        price  = quote["price"]
                        source = "yahoo"
                except Exception:
                    pass

                # Fallback to Massive
                if price == 0:
                    try:
                        snap = await asyncio.to_thread(get_price_and_volume, ticker)
                        if snap and snap.get("price", 0) > 0:
                            price  = snap["price"]
                            source = "massive"
                    except Exception:
                        pass

                if price > 0:
                    conn = get_db()
                    cc   = conn.cursor()
                    cc.execute("""
                        UPDATE squeeze_results SET price = ?
                        WHERE ticker = ?
                        AND run_id = (
                            SELECT run_id FROM squeeze_results
                            WHERE ticker=? ORDER BY screened_at DESC LIMIT 1
                        )
                    """, (price, ticker, ticker))
                    cc.execute("""
                        UPDATE squeeze_lifecycle SET current_price = ?
                        WHERE ticker = ?
                        AND snapshot_date = (
                            SELECT MAX(snapshot_date) FROM squeeze_lifecycle WHERE ticker=?
                        )
                    """, (price, ticker, ticker))
                    conn.commit()
                    conn.close()
                    updated += 1
                    logger.info(f"[CheckPrices] {ticker}: ${price:.2f} ({source})")
                else:
                    failed += 1
                    logger.warning(f"[CheckPrices] {ticker}: no price found")

                pct = round(idx / total * 100)
                yield f"data: {json.dumps({'type': 'tick', 'ticker': ticker, 'price': price, 'source': source, 'idx': idx, 'total': total, 'pct': pct})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'total': total, 'updated': updated, 'failed': failed})}\n\n"

        except Exception as e:
            logger.error(f"[CheckPrices] Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        price_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM PRICE / SI CONSENSUS — ask LLMs for prices and compare
# ─────────────────────────────────────────────────────────────────────────────

class LLMConsensusRequest(BaseModel):
    tickers: Optional[List[str]] = None
    enabled_providers: Optional[List[str]] = None


@app.get("/api/prices/llm_consensus/stream")
async def llm_price_consensus_stream(request: Request, providers: str = ""):
    """
    SSE streaming version of LLM price/SI consensus.
    Processes tickers in batches of 5, streams progress after each batch.
    Shows per-LLM values and consensus result for each ticker live.
    """
    from agents.squeeze.multi_llm_client import get_llm_price_si_consensus

    async def consensus_generator():
        try:
            enabled = [p.strip() for p in providers.split(",") if p.strip()] or None

            conn = get_db()
            c    = conn.cursor()
            c.execute("SELECT DISTINCT ticker FROM squeeze_results ORDER BY ticker")
            tickers = [r[0] for r in c.fetchall()]
            conn.close()

            if not tickers:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No tickers in DB'})}\n\n"
                return

            total         = len(tickers)
            batch_size    = 5
            start_time    = datetime.utcnow()
            run_id        = str(uuid.uuid4())[:8]
            updated_price = 0
            updated_si    = 0
            processed     = 0
            all_tick_rows = []   # accumulate for final summary table

            avg_sec_per_ticker = 8
            est_total_sec      = total * avg_sec_per_ticker
            est_finish         = start_time + __import__('datetime').timedelta(seconds=est_total_sec)

            yield f"data: {json.dumps({'type': 'start', 'total': total, 'run_id': run_id, 'batch_size': batch_size, 'start_time': start_time.strftime('%H:%M:%S UTC'), 'est_finish': est_finish.strftime('%H:%M:%S UTC'), 'est_total_min': round(est_total_sec/60,1)})}\n\n"

            for batch_start in range(0, total, batch_size):
                if await request.is_disconnected():
                    break

                batch = tickers[batch_start:batch_start + batch_size]

                # Log what we're about to ask
                logger.info(f"[LLMConsensus] Batch {batch_start//batch_size+1}: asking LLMs for {batch}")

                consensus_data = await get_llm_price_si_consensus(batch, enabled_providers=enabled)

                # Log raw returns for every ticker in this batch
                for t in batch:
                    d = consensus_data.get(t, {})
                    logger.info(
                        f"[LLMConsensus] RAW {t}: "
                        f"prices={d.get('all_prices',{})} "
                        f"si={d.get('all_si',{})} "
                        f"dtc={d.get('all_dtc',{})} "
                        f"price_consensus={d.get('price_consensus',False)} "
                        f"si_consensus={d.get('si_consensus',False)}"
                    )

                conn = get_db()
                cc   = conn.cursor()

                for ticker in batch:
                    data = consensus_data.get(ticker, {})
                    processed += 1

                    price_ok = data.get("price_consensus") and data.get("price", 0) > 0
                    si_ok    = data.get("si_consensus")    and data.get("short_float", 0) > 0
                    all_prices = data.get("all_prices", {})
                    all_si     = data.get("all_si", {})
                    all_dtc    = data.get("all_dtc", {})
                    note = data.get("price_consensus_note") or data.get("si_consensus_note") or ""

                    # ── Save raw LLM values to llm_price_runs table (always) ──
                    # One row per LLM that returned data for this ticker
                    all_llm_names = set(list(all_prices.keys()) + list(all_si.keys()))
                    if not all_llm_names:
                        # No LLM returned anything — save a "no data" row
                        try:
                            cc.execute("""
                                INSERT INTO llm_price_runs
                                (run_id, ticker, llm_name, price, short_float, days_to_cover,
                                 price_consensus, si_consensus, consensus_note)
                                VALUES (?,?,?,?,?,?,?,?,?)
                            """, (run_id, ticker, "none", 0, 0, 0, 0, 0, "No LLM returned data"))
                        except Exception:
                            pass
                    else:
                        for llm_name in all_llm_names:
                            try:
                                cc.execute("""
                                    INSERT INTO llm_price_runs
                                    (run_id, ticker, llm_name, price, short_float, days_to_cover,
                                     price_consensus, si_consensus, consensus_note)
                                    VALUES (?,?,?,?,?,?,?,?,?)
                                """, (
                                    run_id, ticker, llm_name,
                                    all_prices.get(llm_name, 0),
                                    all_si.get(llm_name, 0),
                                    all_dtc.get(llm_name, 0),
                                    int(price_ok), int(si_ok),
                                    note or ""
                                ))
                            except Exception:
                                pass

                    # ── Update squeeze_results if consensus reached ──
                    if price_ok:
                        cc.execute("""
                            UPDATE squeeze_results SET price = ?, price_consensus_note = ?
                            WHERE ticker = ?
                            AND run_id = (SELECT run_id FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1)
                        """, (data["price"], data["price_consensus_note"], ticker, ticker))
                        cc.execute("""
                            UPDATE squeeze_lifecycle SET current_price = ?
                            WHERE ticker = ?
                            AND snapshot_date = (SELECT MAX(snapshot_date) FROM squeeze_lifecycle WHERE ticker=?)
                        """, (data["price"], ticker, ticker))
                        updated_price += 1

                    if si_ok:
                        cc.execute("""
                            UPDATE squeeze_results SET short_float = ?, days_to_cover = ?, si_consensus_note = ?
                            WHERE ticker = ?
                            AND run_id = (SELECT run_id FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1)
                        """, (data["short_float"], data["days_to_cover"], note, ticker, ticker))
                        updated_si += 1

                    # Build tick row for live table
                    tick_row = {
                        "ticker":      ticker,
                        "claude":      all_prices.get("claude", "—"),
                        "grok":        all_prices.get("grok", "—"),
                        "openai":      all_prices.get("openai", "—"),
                        "gemini":      all_prices.get("gemini", "—"),
                        "si_claude":   all_si.get("claude", "—"),
                        "si_grok":     all_si.get("grok", "—"),
                        "si_openai":   all_si.get("openai", "—"),
                        "si_gemini":   all_si.get("gemini", "—"),
                        "price_ok":    price_ok,
                        "si_ok":       si_ok,
                        "note":        note or ("✓ Consensus" if (price_ok or si_ok) else "No consensus"),
                    }
                    all_tick_rows.append(tick_row)

                    elapsed_sec = (datetime.utcnow() - start_time).total_seconds()
                    avg_actual  = round(elapsed_sec / processed, 1) if processed > 0 else avg_sec_per_ticker
                    rem_sec     = avg_actual * (total - processed)
                    eta_str     = (datetime.utcnow() + __import__('datetime').timedelta(seconds=rem_sec)).strftime('%H:%M:%S UTC')
                    pct         = round(processed / total * 100)

                    yield f"data: {json.dumps({'type': 'tick', 'ticker': ticker, 'idx': processed, 'total': total, 'pct': pct, 'price': data.get('price', 0), 'short_float': data.get('short_float', 0), 'price_ok': price_ok, 'si_ok': si_ok, 'all_prices': all_prices, 'all_si': all_si, 'note': tick_row['note'], 'updated_price': updated_price, 'updated_si': updated_si, 'avg_sec': avg_actual, 'eta': eta_str, 'row': tick_row})}\n\n"

                conn.commit()
                conn.close()

            yield f"data: {json.dumps({'type': 'done', 'total': total, 'updated_price': updated_price, 'updated_si': updated_si, 'run_id': run_id, 'rows': all_tick_rows})}\n\n"
            logger.info(f"[LLMConsensus] Run {run_id} complete — {updated_price} prices, {updated_si} SI updated across {total} tickers")

        except Exception as e:
            logger.error(f"[LLMConsensus] Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        consensus_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/prices/llm_consensus")
async def llm_price_consensus(req: LLMConsensusRequest = None):
    """Non-streaming fallback — kept for API compatibility."""
    from agents.squeeze.multi_llm_client import get_llm_price_si_consensus

    tickers   = req.tickers if req and req.tickers else None
    providers = req.enabled_providers if req and req.enabled_providers else None

    if not tickers:
        conn = get_db()
        c    = conn.cursor()
        c.execute("SELECT DISTINCT ticker FROM squeeze_results ORDER BY ticker")
        tickers = [r[0] for r in c.fetchall()]
        conn.close()

    if not tickers:
        return {"status": "no_tickers", "results": []}

    consensus_data = await get_llm_price_si_consensus(tickers, enabled_providers=providers)

    conn = get_db()
    c    = conn.cursor()
    updated_price = 0
    updated_si    = 0

    for ticker, data in consensus_data.items():
        if data.get("price_consensus") and data.get("price", 0) > 0:
            c.execute("""
                UPDATE squeeze_results SET price = ?, price_consensus_note = ?
                WHERE ticker = ?
                AND run_id = (SELECT run_id FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1)
            """, (data["price"], data["price_consensus_note"], ticker, ticker))
            c.execute("""
                UPDATE squeeze_lifecycle SET current_price = ?
                WHERE ticker = ?
                AND snapshot_date = (SELECT MAX(snapshot_date) FROM squeeze_lifecycle WHERE ticker=?)
            """, (data["price"], ticker, ticker))
            updated_price += 1

        if data.get("si_consensus") and data.get("short_float", 0) > 0:
            c.execute("""
                UPDATE squeeze_results SET short_float = ?, days_to_cover = ?, si_consensus_note = ?
                WHERE ticker = ?
                AND run_id = (SELECT run_id FROM squeeze_results WHERE ticker=? ORDER BY screened_at DESC LIMIT 1)
            """, (data["short_float"], data["days_to_cover"], data["si_consensus_note"], ticker, ticker))
            updated_si += 1

    conn.commit()
    conn.close()

    return {"status": "ok", "tickers_checked": len(tickers), "updated_price": updated_price, "updated_si": updated_si}




@app.get("/api/llm_price_runs")
def get_llm_price_runs(run_id: str = None, limit: int = 200):
    """
    Return raw LLM price/SI values from llm_price_runs table.
    If run_id supplied, return that run. Otherwise return latest run.
    Returns pivoted rows: one row per ticker with columns per LLM.
    """
    conn = get_db()
    c    = conn.cursor()

    # Get the target run_id
    if not run_id:
        row = c.execute(
            "SELECT run_id, MAX(run_at) FROM llm_price_runs"
        ).fetchone()
        run_id = row[0] if row and row[0] else None

    if not run_id:
        conn.close()
        return {"run_id": None, "run_at": None, "rows": [], "available_runs": []}

    # All rows for this run
    c.execute("""
        SELECT ticker, llm_name, price, short_float, days_to_cover,
               price_consensus, si_consensus, consensus_note, run_at
        FROM llm_price_runs
        WHERE run_id = ?
        ORDER BY ticker, llm_name
    """, (run_id,))
    raw = c.fetchall()

    # Get run_at from first row
    run_at = raw[0][8] if raw else None

    # Available run IDs for dropdown
    c.execute("""
        SELECT DISTINCT run_id, MAX(run_at) as ra
        FROM llm_price_runs
        GROUP BY run_id
        ORDER BY ra DESC
        LIMIT 20
    """)
    available_runs = [{"run_id": r[0], "run_at": r[1]} for r in c.fetchall()]
    conn.close()

    # Pivot: one row per ticker
    from collections import defaultdict
    pivot = defaultdict(lambda: {
        "ticker": "", "price_consensus": False, "si_consensus": False, "consensus_note": "",
        "claude_price": None, "grok_price": None, "openai_price": None, "gemini_price": None,
        "claude_si": None, "grok_si": None, "openai_si": None, "gemini_si": None,
        "claude_dtc": None, "grok_dtc": None, "openai_dtc": None, "gemini_dtc": None,
    })

    for ticker, llm, price, si, dtc, p_ok, s_ok, note, _ in raw:
        d = pivot[ticker]
        d["ticker"] = ticker
        if p_ok: d["price_consensus"] = True
        if s_ok: d["si_consensus"]    = True
        if note: d["consensus_note"]  = note
        llm = (llm or "").lower()
        if llm in ("claude","grok","openai","gemini"):
            d[f"{llm}_price"] = round(price, 2) if price else None
            d[f"{llm}_si"]    = round(si, 1)    if si    else None
            d[f"{llm}_dtc"]   = round(dtc, 1)   if dtc   else None

    rows = sorted(pivot.values(), key=lambda x: x["ticker"])

    return {
        "run_id":         run_id,
        "run_at":         run_at,
        "rows":           rows,
        "total":          len(rows),
        "available_runs": available_runs,
    }

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "tavily_key":    bool(os.environ.get("TAVILY_API_KEY")),
        "massive_key":   bool(os.environ.get("MASSIVE_API_KEY")),
        "finnhub_key":   bool(os.environ.get("FINNHUB_API_KEY")),
        "openai_key":    bool(os.environ.get("OPENAI_API_KEY")),
        "grok_key":      bool(os.environ.get("XAI_API_KEY")),
        "gemini_key":    bool(os.environ.get("GOOGLE_API_KEY")),
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
