"""
TRaNKSP — Lifecycle Tracker
Tracks squeeze lifecycle daily. Triggers bearish evaluation based on 3 conditions.
Statuses: ACTIVE | SQUEEZE_FIRING | SQUEEZE_COMPLETE | REVERSAL | FAILED | STALE
"""

import os
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta

from .massive_client import get_price_and_volume
from .yahoo_quote import get_quote_data as _yq_get
# from .marketbeat_client import get_short_interest as _mb_get_si  # disabled (403)

from .rag import store_lifecycle_snapshot, query_lifecycle_memory
from .chains import generate_bearish_thesis

logger = logging.getLogger("tranksp.lifecycle")

DB_PATH = os.path.join("data", "tranksp.db")


def get_settings() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT key, value FROM settings")
    settings = dict(c.fetchall())
    conn.close()
    return settings


def get_active_tickers() -> List[Dict[str, Any]]:
    """Get all active lifecycle records (non-terminal statuses)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_lifecycle
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM squeeze_lifecycle sl2
            WHERE sl2.ticker = squeeze_lifecycle.ticker
        )
        AND status IN ('ACTIVE', 'SQUEEZE_FIRING')
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_ticker_lifecycle_history(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    c.execute("""
        SELECT * FROM squeeze_lifecycle
        WHERE ticker = ? AND snapshot_date >= ?
        ORDER BY snapshot_date ASC
    """, (ticker, cutoff))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def upsert_lifecycle_snapshot(data: Dict[str, Any]):
    """Insert or update a lifecycle snapshot for today."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO squeeze_lifecycle
            (ticker, snapshot_date, status, entry_price, peak_price, current_price,
             short_interest, si_change_pct, price_chg_peak, bearish_eval, eval_triggered, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, snapshot_date) DO UPDATE SET
            status=excluded.status,
            current_price=excluded.current_price,
            peak_price=excluded.peak_price,
            short_interest=excluded.short_interest,
            si_change_pct=excluded.si_change_pct,
            price_chg_peak=excluded.price_chg_peak,
            bearish_eval=excluded.bearish_eval,
            eval_triggered=excluded.eval_triggered,
            notes=excluded.notes
    """, (
        data["ticker"], today, data["status"],
        data.get("entry_price"), data.get("peak_price"), data.get("current_price"),
        data.get("short_interest"), data.get("si_change_pct"), data.get("price_chg_peak"),
        int(data.get("bearish_eval", False)), data.get("eval_triggered"), data.get("notes")
    ))
    conn.commit()
    conn.close()


def get_or_create_entry_price(ticker: str, current_price: float) -> float:
    """Get the original entry price when the ticker was first detected."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT entry_price FROM squeeze_lifecycle
        WHERE ticker = ? AND entry_price IS NOT NULL
        ORDER BY snapshot_date ASC LIMIT 1
    """, (ticker,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else current_price


def get_peak_price(ticker: str, current_price: float) -> float:
    """Get the highest price seen in lifecycle history."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT MAX(peak_price), MAX(current_price) FROM squeeze_lifecycle WHERE ticker=?", (ticker,))
    row = c.fetchone()
    conn.close()
    stored_peak = max(row[0] or 0, row[1] or 0)
    return max(stored_peak, current_price)


def days_since_detection(ticker: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT MIN(snapshot_date) FROM squeeze_lifecycle WHERE ticker=?", (ticker,))
    row = c.fetchone()
    conn.close()
    if not row or not row[0]:
        return 0
    first = datetime.strptime(row[0], "%Y-%m-%d").date()
    return (date.today() - first).days


def check_bearish_triggers(
    ticker: str,
    current_price: float,
    peak_price: float,
    entry_price: float,
    si_at_detection: float,
    si_current: float,
    days_active: int,
    auto_eval_hours: int
) -> Optional[str]:
    """
    Check 3 bearish trigger conditions. Returns trigger reason or None.
    
    1. squeeze_complete: SI dropped >50% from detection level
    2. post_spike_reversal: price down >30% from peak
    3. squeeze_failed: price down >20% from entry AND SI unchanged
    """
    # Must be past auto_eval threshold (default 72h)
    min_days = auto_eval_hours / 24
    if days_active < min_days:
        return None
    
    # Trigger 1: SI dropped >50%
    if si_at_detection and si_at_detection > 0:
        si_drop = (si_at_detection - si_current) / si_at_detection
        if si_drop >= 0.50:
            return "squeeze_complete"
    
    # Trigger 2: Price down >30% from peak
    if peak_price and peak_price > 0:
        price_drop_from_peak = (peak_price - current_price) / peak_price
        if price_drop_from_peak >= 0.30:
            return "post_spike_reversal"
    
    # Trigger 3: Price down >20% from entry, SI unchanged
    if entry_price and entry_price > 0:
        price_drop_from_entry = (entry_price - current_price) / entry_price
        si_unchanged = abs((si_current - si_at_detection) / (si_at_detection or 1)) < 0.10
        if price_drop_from_entry >= 0.20 and si_unchanged:
            return "squeeze_failed"
    
    return None


async def evaluate_ticker_lifecycle(ticker: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    """Run full lifecycle evaluation for one ticker."""
    settings = get_settings()
    auto_eval_hours = int(settings.get("auto_eval_hours", 72))
    
    # Waterfall: Yahoo first → Massive fallback → Finnhub fallback → last known
    quote_data = _yq_get(ticker)

    if quote_data and quote_data.get("price", 0) > 0:
        current_price = quote_data["price"]
        si_current    = quote_data.get("short_float", 0) or existing.get("short_interest", 0)
        logger.debug(f"[Lifecycle] {ticker}: Yahoo quote OK ${current_price:.2f} SI={si_current:.1f}%")
    else:
        # Fallback: Massive daily aggs for price
        snapshot = get_price_and_volume(ticker)
        current_price = (snapshot or {}).get("price", 0) or existing.get("current_price", 0)

        # Fallback: Finnhub for SI (if key configured)
        si_current = existing.get("short_interest", 0)
        if os.environ.get("FINNHUB_API_KEY"):
            from .finnhub_client import get_short_interest as _fh_si
            fh = _fh_si(ticker)
            if fh:
                si_current = fh.get("short_float", si_current)

        logger.debug(f"[Lifecycle] {ticker}: Yahoo failed, Massive=${current_price:.2f} SI={si_current:.1f}%")
    
    entry_price = get_or_create_entry_price(ticker, current_price)
    peak_price = get_peak_price(ticker, current_price)
    si_at_detection = existing.get("short_interest") or si_current
    days_active = days_since_detection(ticker)
    
    # Calculate derived metrics
    si_change_pct = ((si_current - si_at_detection) / si_at_detection * 100) if si_at_detection else 0
    price_chg_peak = ((current_price - peak_price) / peak_price * 100) if peak_price else 0
    
    # Check if price is significantly above entry (squeeze firing)
    price_chg_entry = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
    
    status = existing.get("status", "ACTIVE")
    if price_chg_entry >= 20 and status == "ACTIVE":
        status = "SQUEEZE_FIRING"
    
    # Check bearish triggers
    trigger_reason = check_bearish_triggers(
        ticker, current_price, peak_price, entry_price,
        si_at_detection, si_current, days_active, auto_eval_hours
    )
    
    bearish_eval = False
    bearish_thesis = None
    eval_triggered = None
    
    if trigger_reason:
        bearish_eval = True
        eval_triggered = trigger_reason
        
        # Map trigger to status
        status_map = {
            "squeeze_complete": "SQUEEZE_COMPLETE",
            "post_spike_reversal": "REVERSAL",
            "squeeze_failed": "FAILED"
        }
        status = status_map.get(trigger_reason, "REVERSAL")
        
        # Generate bearish thesis
        lifecycle_memory = query_lifecycle_memory(ticker)
        
        from .tools import search_news
        try:
            news_result = search_news.invoke(f"{ticker} stock news reversal")
            news_context = str(news_result)
        except Exception:
            news_context = "News unavailable."
        
        lifecycle_data = {
            "entry_price": entry_price,
            "peak_price": peak_price,
            "current_price": current_price,
            "si_entry": si_at_detection,
            "short_interest": si_current,
            "si_change_pct": si_change_pct,
            "price_chg_peak": price_chg_peak,
            "days_active": days_active
        }
        
        bearish_thesis = await generate_bearish_thesis(
            ticker, lifecycle_data, trigger_reason, lifecycle_memory, news_context
        )
        
        logger.info(f"[Lifecycle] {ticker}: bearish triggered — {trigger_reason}")
    
    snapshot = {
        "ticker": ticker,
        "status": status,
        "entry_price": entry_price,
        "peak_price": peak_price,
        "current_price": current_price,
        "short_interest": si_current,
        "si_change_pct": si_change_pct,
        "price_chg_peak": price_chg_peak,
        "bearish_eval": bearish_eval,
        "eval_triggered": eval_triggered,
        "notes": f"Auto-eval. Days active: {days_active}",
        "days_active": days_active
    }
    
    upsert_lifecycle_snapshot(snapshot)
    
    # Store in RAG lifecycle memory
    snapshot_text = (
        f"Ticker: {ticker} | Date: {date.today().isoformat()} | Status: {status}\n"
        f"Entry: ${entry_price:.2f} | Peak: ${peak_price:.2f} | Current: ${current_price:.2f}\n"
        f"SI: {si_current:.1f}% (was {si_at_detection:.1f}%, change: {si_change_pct:+.1f}%)\n"
        f"Price from peak: {price_chg_peak:+.1f}% | Days active: {days_active}\n"
        f"Bearish triggered: {eval_triggered or 'No'}"
    )
    store_lifecycle_snapshot(ticker, snapshot_text, date.today().isoformat(), status)
    
    return {
        "snapshot": snapshot,
        "bearish_thesis": bearish_thesis.model_dump() if bearish_thesis else None
    }


async def run_daily_lifecycle_check():
    """APScheduler job: evaluate all active tickers at 8:30 AM CT."""
    logger.info("[Lifecycle] Starting daily lifecycle check")
    active = get_active_tickers()
    logger.info(f"[Lifecycle] {len(active)} active tickers to evaluate")
    
    results = []
    for record in active:
        ticker = record["ticker"]
        try:
            result = await evaluate_ticker_lifecycle(ticker, record)
            results.append(result)
        except Exception as e:
            logger.error(f"[Lifecycle] Error evaluating {ticker}: {e}")
    
    logger.info(f"[Lifecycle] Daily check complete. {len(results)} tickers evaluated.")
    return results
