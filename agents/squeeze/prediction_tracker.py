"""
TRaNKSP — Layer 1: Outcome Tracking (Fully Complete)

Records every prediction and later fills outcomes for learning.
Exact implementation per the AI_Closed_Loop_with_code.docx specification.

squeeze_predictions table fields (all from document):
  ticker, prediction_date, predicted_direction (UP/DOWN)
  entry_price, target_price, confidence_score
  time_horizon ("3-7 days")
  catalyst_types that were cited
  thesis_id (links back to full thesis)
  outcome_price (filled in later)
  outcome_date
  outcome_result (HIT / MISS / PARTIAL / EXPIRED)
  actual_peak, actual_drawdown
  si_at_prediction, si_at_outcome
"""

import sqlite3
import os
import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger("tranksp.prediction_tracker")
DB_PATH = os.path.join("data", "tranksp.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Record new prediction ─────────────────────────────────────────────────────

def record_prediction(
    ticker:       str,
    run_id:       str,
    thesis:       Dict[str, Any],
    entry_price:  float,
    short_float:  Optional[float] = None,
    dtc:          Optional[float] = None,
    volume_ratio: Optional[float] = None,
    si_trend:     Optional[str]   = None,
    thesis_id:    Optional[int]   = None,
) -> Optional[int]:
    """
    Call right after generating a new thesis.
    Stores all Layer 1 required fields including si_at_prediction.
    Returns prediction id or None on failure.
    """
    conn = get_db()
    c    = conn.cursor()

    # Deduplicate — don't double-record same ticker+date
    c.execute("""
        SELECT id FROM squeeze_predictions
        WHERE ticker=? AND prediction_date=? AND status='OPEN' LIMIT 1
    """, (ticker, date.today().isoformat()))
    if c.fetchone():
        conn.close()
        logger.debug(f"[PredTracker] {ticker} already has OPEN prediction today — skipping")
        return None

    catalyst_str = (
        json.dumps(thesis.get("catalyst_types", []))
        if isinstance(thesis.get("catalyst_types"), list)
        else str(thesis.get("catalyst_types", "[]"))
    )

    confidence = float(thesis.get("confidence", 50.0))

    # Use thesis target_price if set, otherwise derive from entry + confidence
    target_price = thesis.get("target_price") or round(
        entry_price * (1 + confidence / 200), 2
    ) if entry_price and entry_price > 0 else None

    sql = """
        INSERT INTO squeeze_predictions (
            ticker, run_id, thesis_id,
            prediction_date, predicted_direction,
            entry_price, target_price, confidence_score,
            direction, confidence, time_horizon, catalyst_types,
            short_float_at_pred, si_at_prediction,
            dtc_at_pred, volume_ratio_at_pred,
            si_trend_at_pred, thesis_summary, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """

    values = (
        ticker, run_id, thesis_id,
        date.today().isoformat(), "UP",         # predicted_direction always UP for squeeze
        entry_price, target_price, confidence,   # confidence_score = confidence
        "UP", confidence,                        # direction + confidence (redundant but schema has both)
        thesis.get("time_horizon", "1-2 weeks"),
        catalyst_str,
        short_float, short_float,                # short_float_at_pred AND si_at_prediction
        dtc, volume_ratio,
        si_trend,
        (thesis.get("setup") or "")[:600],
    )

    try:
        c.execute(sql, values)
        pred_id = c.lastrowid
        conn.commit()
        logger.info(f"✅ [PredTracker] Recorded #{pred_id} for {ticker} "
                    f"entry=${entry_price:.2f} target=${target_price or 0:.2f} "
                    f"SI={short_float or 0:.1f}% conf={confidence:.0f}%")
        return pred_id
    except Exception as e:
        logger.error(f"❌ [PredTracker] Failed for {ticker}: {e}")
        return None
    finally:
        conn.close()


# ── Calculate outcome result ──────────────────────────────────────────────────

def calculate_outcome_result(
    entry_price:      float,
    target_price:     float,
    current_price:    float,
    pred_date_str:    str,
    time_horizon:     str = "1-2 weeks",
    lifecycle_status: str = "ACTIVE",
    si_at_pred:       float = 0,
    si_current:       float = 0,
) -> Optional[str]:
    """
    HIT      — price hit target OR lifecycle = SQUEEZE_COMPLETE
    MISS     — price down >20% from entry OR lifecycle FAILED/REVERSAL
    PARTIAL  — expired, made 40%+ progress toward target
    EXPIRED  — time window closed with little progress
    None     — still within time window, keep OPEN
    """
    if not entry_price or entry_price <= 0:
        return "EXPIRED"

    # Lifecycle shortcut
    if lifecycle_status == "SQUEEZE_COMPLETE":
        return "HIT"
    if lifecycle_status in ("FAILED", "REVERSAL"):
        return "MISS"

    # Days elapsed
    try:
        pred_date    = datetime.strptime(pred_date_str, "%Y-%m-%d").date()
        days_elapsed = (date.today() - pred_date).days
    except Exception:
        days_elapsed = 0

    # Max days from time_horizon string
    max_days = 14
    tl   = (time_horizon or "").lower()
    nums = [int(s) for s in tl.split() if s.isdigit()]
    if "month" in tl:
        max_days = (max(nums) if nums else 1) * 30
    elif "week" in tl:
        max_days = (max(nums) if nums else 2) * 7
    elif "day" in tl:
        max_days = max(nums) if nums else 7

    # Price-based HIT
    if target_price and target_price > 0 and current_price >= target_price:
        return "HIT"

    # Price-based MISS (down >20% from entry)
    if current_price <= entry_price * 0.80:
        return "MISS"

    # SI collapse check (SI dropped >50% = squeeze failed)
    if si_at_pred > 0 and si_current > 0:
        si_drop = (si_at_pred - si_current) / si_at_pred
        if si_drop > 0.50 and current_price < entry_price:
            return "MISS"

    # Time expired
    if days_elapsed >= max_days:
        if target_price and target_price > entry_price:
            pct_progress = (current_price - entry_price) / (target_price - entry_price)
            return "PARTIAL" if pct_progress >= 0.40 else "EXPIRED"
        return "EXPIRED"

    return None   # Still OPEN


# ── Update outcome ────────────────────────────────────────────────────────────

def update_prediction_outcome(
    ticker:          str,
    outcome_price:   float,
    actual_peak:     Optional[float] = None,
    outcome_result:  str             = "MISS",
    days_to_outcome: Optional[int]   = None,
    si_at_outcome:   Optional[float] = None,
) -> int:
    """Update outcome fields and close prediction. Called by daily job."""
    conn = get_db()
    c    = conn.cursor()

    # Calculate actual_drawdown (peak → outcome drop)
    actual_drawdown = None
    if actual_peak and outcome_price and actual_peak > 0:
        actual_drawdown = round((actual_peak - outcome_price) / actual_peak * 100, 2)

    sql = """
        UPDATE squeeze_predictions
        SET outcome_price    = ?,
            outcome_date     = ?,
            outcome          = ?,
            outcome_result   = ?,
            actual_peak      = ?,
            actual_drawdown  = ?,
            days_to_outcome  = ?,
            si_at_outcome    = ?,
            status           = 'CLOSED',
            outcome_pnl_pct  = ROUND((? - entry_price) / entry_price * 100, 2)
        WHERE ticker = ? AND status = 'OPEN'
    """
    c.execute(sql, (
        outcome_price, date.today().isoformat(),
        outcome_result, outcome_result,           # outcome + outcome_result both
        actual_peak, actual_drawdown,
        days_to_outcome, si_at_outcome,
        outcome_price, ticker
    ))
    updated = c.rowcount
    conn.commit()
    conn.close()

    if updated > 0:
        logger.info(f"✅ [PredTracker] Closed {updated} prediction(s) for {ticker} → "
                    f"{outcome_result} exit=${outcome_price:.2f} "
                    f"drawdown={actual_drawdown or 0:.1f}%")
    return updated


# ── Daily outcome check ───────────────────────────────────────────────────────

async def daily_outcome_update():
    """
    Called by APScheduler at 9:00 AM CT.
    Evaluates all open predictions against current lifecycle + price data.
    """
    from .yahoo_quote import get_quote_data

    open_preds = get_open_predictions()
    if not open_preds:
        logger.info("[PredTracker] No open predictions to evaluate")
        return

    logger.info(f"[PredTracker] Evaluating {len(open_preds)} open predictions...")

    # Get current lifecycle statuses
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT ticker, status, peak_price, short_interest
        FROM squeeze_lifecycle
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM squeeze_lifecycle sl2
            WHERE sl2.ticker = squeeze_lifecycle.ticker
        )
    """)
    lc_map: Dict[str, Dict] = {row["ticker"]: dict(row) for row in c.fetchall()}
    conn.close()

    evaluated = 0
    for pred in open_preds:
        ticker = pred["ticker"]
        try:
            quote = get_quote_data(ticker)
            if not quote or not quote.get("price"):
                continue

            current_price    = quote["price"]
            si_current       = quote.get("short_float", 0)
            lc               = lc_map.get(ticker, {})
            lifecycle_status = lc.get("status", "ACTIVE")
            actual_peak      = lc.get("peak_price") or current_price

            result = calculate_outcome_result(
                entry_price      = pred.get("entry_price", 0) or 0,
                target_price     = pred.get("target_price", 0) or 0,
                current_price    = current_price,
                pred_date_str    = pred.get("prediction_date", ""),
                time_horizon     = pred.get("time_horizon", "1-2 weeks"),
                lifecycle_status = lifecycle_status,
                si_at_pred       = pred.get("si_at_prediction", 0) or 0,
                si_current       = si_current,
            )

            if result:
                pred_date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d").date()
                days      = (date.today() - pred_date).days
                update_prediction_outcome(
                    ticker          = ticker,
                    outcome_price   = current_price,
                    actual_peak     = actual_peak,
                    outcome_result  = result,
                    days_to_outcome = days,
                    si_at_outcome   = si_current,
                )
                evaluated += 1

        except Exception as e:
            logger.warning(f"[PredTracker] Error evaluating {ticker}: {e}")

    logger.info(f"[PredTracker] Done — {evaluated} predictions closed")

    # Recompute calibration stats after closing predictions
    if evaluated > 0:
        try:
            from .learning_engine import compute_calibration_stats
            compute_calibration_stats()
        except Exception as e:
            logger.warning(f"[PredTracker] Calibration recompute error: {e}")


# ── Queries ───────────────────────────────────────────────────────────────────

def get_open_predictions(limit: int = 100) -> List[Dict]:
    """Daily job iterates these."""
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_predictions
        WHERE status = 'OPEN'
        ORDER BY prediction_date ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def get_prediction_stats() -> Dict:
    """Summary stats for Predictions tab dashboard."""
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*)                                                AS total,
            SUM(CASE WHEN status='OPEN'     THEN 1 ELSE 0 END)    AS open,
            SUM(CASE WHEN status='CLOSED'   THEN 1 ELSE 0 END)    AS closed,
            SUM(CASE WHEN outcome='HIT'     THEN 1 ELSE 0 END)    AS hits,
            SUM(CASE WHEN outcome='MISS'    THEN 1 ELSE 0 END)    AS misses,
            SUM(CASE WHEN outcome='PARTIAL' THEN 1 ELSE 0 END)    AS partials,
            ROUND(AVG(CASE WHEN outcome_pnl_pct IS NOT NULL
                           THEN outcome_pnl_pct END), 2)          AS avg_pnl_pct,
            ROUND(
              100.0 * SUM(CASE WHEN outcome IN ('HIT','PARTIAL') THEN 1 ELSE 0 END)
              / NULLIF(SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END), 0),
              1)                                                   AS accuracy_pct
        FROM squeeze_predictions
    """)
    row   = c.fetchone()
    stats = dict(row) if row else {}
    conn.close()
    return stats


# Backwards compat alias used by learning_engine.py
run_daily_outcome_check = daily_outcome_update
