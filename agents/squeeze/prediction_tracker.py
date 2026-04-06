"""
TRaNKSP — Layer 1: Outcome Tracking (Fully Complete)

Records every thesis as a structured prediction and later fills in
the outcome when the squeeze plays out (HIT / MISS / PARTIAL / EXPIRED).

This is the non-negotiable foundation of the self-learning loop:
  Predict → Track Outcome → Measure Accuracy → Update Model → Better Predict

Functions:
  record_prediction()        — called after every thesis generation
  update_prediction_outcome()— called by daily scheduler job
  calculate_outcome_result() — logic: HIT / MISS / PARTIAL / EXPIRED
  get_open_predictions()     — daily job iterates these
  get_prediction_stats()     — dashboard summary stats
"""

import sqlite3
import os
import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger("tranksp.prediction_tracker")
DB_PATH = os.path.join("data", "tranksp.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


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
) -> Optional[int]:
    """
    Call immediately after a new thesis is generated for a ticker.
    Stores the prediction so it can be evaluated later.

    Args:
        thesis: ThesisOutput.model_dump() dict — must contain
                confidence, time_horizon, catalyst_types, setup

    Returns:
        prediction id (int) or None on failure
    """
    conn = _conn()
    c    = conn.cursor()

    # Deduplicate: don't double-record same ticker on same date
    c.execute("""
        SELECT id FROM squeeze_predictions
        WHERE ticker=? AND prediction_date=? AND status='OPEN'
        LIMIT 1
    """, (ticker, date.today().isoformat()))
    if c.fetchone():
        conn.close()
        logger.debug(f"[PredTracker] {ticker} already has an open prediction today — skipping")
        return None

    catalyst_str = (
        json.dumps(thesis.get("catalyst_types", []))
        if isinstance(thesis.get("catalyst_types"), list)
        else str(thesis.get("catalyst_types", "[]"))
    )

    # Derive a target_price: entry × (1 + confidence/100 × 0.5)
    confidence   = float(thesis.get("confidence", 50.0))
    target_price = round(entry_price * (1 + confidence / 200), 2) if entry_price > 0 else None

    sql = """
        INSERT INTO squeeze_predictions (
            ticker, run_id, prediction_date, entry_price, target_price,
            direction, confidence, time_horizon, catalyst_types,
            short_float_at_pred, dtc_at_pred, volume_ratio_at_pred,
            si_trend_at_pred, thesis_summary, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """
    values = (
        ticker, run_id, date.today().isoformat(),
        entry_price, target_price,
        "UP",                                      # squeeze = always bullish at entry
        confidence,
        thesis.get("time_horizon", "1-2 weeks"),
        catalyst_str,
        short_float, dtc, volume_ratio, si_trend,
        (thesis.get("setup") or "")[:600],
    )

    try:
        c.execute(sql, values)
        pred_id = c.lastrowid
        conn.commit()
        logger.info(f"[PredTracker] ✓ Recorded prediction #{pred_id} for {ticker} "
                    f"entry=${entry_price:.2f} target=${target_price or 0:.2f} conf={confidence:.0f}%")
        return pred_id
    except Exception as e:
        logger.error(f"[PredTracker] Failed to record {ticker}: {e}")
        return None
    finally:
        conn.close()


# ── Calculate outcome ─────────────────────────────────────────────────────────

def calculate_outcome_result(
    entry_price:   float,
    target_price:  float,
    current_price: float,
    pred_date_str: str,
    time_horizon:  str = "1-2 weeks",
    lifecycle_status: str = "ACTIVE"
) -> str:
    """
    Determine HIT / MISS / PARTIAL / EXPIRED for a prediction.

    Rules:
      HIT      — current_price >= target_price  OR  lifecycle = SQUEEZE_COMPLETE
      MISS     — price dropped >20% below entry  OR  lifecycle = FAILED/REVERSAL
      PARTIAL  — time expired, made some progress (>50% to target)
      EXPIRED  — time expired, little/no progress
    """
    if not entry_price or entry_price <= 0:
        return "EXPIRED"

    # Days elapsed
    try:
        pred_date  = datetime.strptime(pred_date_str, "%Y-%m-%d").date()
        days_elapsed = (date.today() - pred_date).days
    except Exception:
        days_elapsed = 0

    # Max days from time_horizon string
    max_days = 14
    tl = (time_horizon or "").lower()
    nums = [int(s) for s in tl.split() if s.isdigit()]
    if "month" in tl:
        max_days = (max(nums) if nums else 1) * 30
    elif "week" in tl:
        max_days = (max(nums) if nums else 2) * 7
    elif "day" in tl:
        max_days = max(nums) if nums else 7

    # Lifecycle shortcut
    if lifecycle_status == "SQUEEZE_COMPLETE":
        return "HIT"
    if lifecycle_status in ("FAILED", "REVERSAL"):
        return "MISS"

    # Price-based
    if target_price and target_price > 0:
        if current_price >= target_price:
            return "HIT"

    if entry_price > 0 and current_price <= entry_price * 0.80:
        return "MISS"                              # down >20% = failed squeeze

    if days_elapsed >= max_days:
        # Expired — was there meaningful progress?
        if target_price and target_price > entry_price:
            pct_progress = (current_price - entry_price) / (target_price - entry_price)
            return "PARTIAL" if pct_progress >= 0.40 else "EXPIRED"
        return "EXPIRED"

    return None   # Still open — don't close yet


# ── Update outcome ────────────────────────────────────────────────────────────

def update_prediction_outcome(
    ticker:         str,
    outcome_price:  float,
    actual_peak:    Optional[float] = None,
    outcome_result: str             = "MISS",
    days_to_outcome: Optional[int] = None,
    si_at_outcome:  Optional[float] = None,
) -> int:
    """
    Close an open prediction with its final outcome.
    Called by the daily scheduler job.

    Returns: number of rows updated
    """
    conn = _conn()
    c    = conn.cursor()

    today = date.today().isoformat()
    sql   = """
        UPDATE squeeze_predictions
        SET outcome_price    = ?,
            outcome_date     = ?,
            outcome          = ?,
            actual_peak      = ?,
            days_to_outcome  = ?,
            status           = 'CLOSED',
            outcome_pnl_pct  = ROUND((?-entry_price)/entry_price*100, 2)
        WHERE ticker = ?
          AND status = 'OPEN'
    """
    c.execute(sql, (
        outcome_price, today, outcome_result, actual_peak,
        days_to_outcome, outcome_price, ticker
    ))
    updated = c.rowcount
    conn.commit()
    conn.close()

    if updated > 0:
        logger.info(f"[PredTracker] Closed {updated} prediction(s) for {ticker} → {outcome_result} "
                    f"exit=${outcome_price:.2f} peak=${actual_peak or 0:.2f}")
    return updated


# ── Daily outcome check (called by APScheduler) ───────────────────────────────

async def run_daily_outcome_check():
    """
    APScheduler job — runs daily at 8:45 AM CT.
    Fetches current price for each open prediction and evaluates outcome.
    """
    from .yahoo_quote import get_quote_data

    open_preds = get_open_predictions()
    if not open_preds:
        logger.info("[PredTracker] No open predictions to evaluate")
        return

    logger.info(f"[PredTracker] Evaluating {len(open_preds)} open predictions...")
    evaluated = 0

    # Get lifecycle statuses in bulk
    conn = _conn()
    c    = conn.cursor()
    lifecycle_map: Dict[str, str] = {}
    c.execute("SELECT ticker, status FROM squeeze_lifecycle WHERE snapshot_date=(SELECT MAX(snapshot_date) FROM squeeze_lifecycle sl2 WHERE sl2.ticker=squeeze_lifecycle.ticker)")
    for row in c.fetchall():
        lifecycle_map[row["ticker"]] = row["status"]
    conn.close()

    for pred in open_preds:
        ticker = pred["ticker"]
        try:
            quote = get_quote_data(ticker)
            if not quote or not quote.get("price"):
                continue

            current_price    = quote["price"]
            lifecycle_status = lifecycle_map.get(ticker, "ACTIVE")

            result = calculate_outcome_result(
                entry_price      = pred.get("entry_price", 0),
                target_price     = pred.get("target_price", 0),
                current_price    = current_price,
                pred_date_str    = pred.get("prediction_date", ""),
                time_horizon     = pred.get("time_horizon", "1-2 weeks"),
                lifecycle_status = lifecycle_status,
            )

            if result:
                # Get lifecycle peak for actual_peak
                lc_conn = _conn()
                lc_c    = lc_conn.cursor()
                lc_c.execute("SELECT peak_price FROM squeeze_lifecycle WHERE ticker=? ORDER BY snapshot_date DESC LIMIT 1", (ticker,))
                lc_row      = lc_c.fetchone()
                actual_peak = lc_row["peak_price"] if lc_row else current_price
                lc_conn.close()

                pred_date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d").date()
                days      = (date.today() - pred_date).days

                update_prediction_outcome(
                    ticker          = ticker,
                    outcome_price   = current_price,
                    actual_peak     = actual_peak,
                    outcome_result  = result,
                    days_to_outcome = days,
                    si_at_outcome   = quote.get("short_float"),
                )
                evaluated += 1

        except Exception as e:
            logger.warning(f"[PredTracker] Error evaluating {ticker}: {e}")

    logger.info(f"[PredTracker] Daily check complete — {evaluated} predictions closed")

    # Trigger calibration recompute
    if evaluated > 0:
        try:
            from .learning_engine import compute_calibration_stats
            compute_calibration_stats()
        except Exception:
            pass


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_open_predictions(limit: int = 200) -> List[Dict]:
    conn = _conn()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_predictions
        WHERE status = 'OPEN'
        ORDER BY prediction_date ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_prediction_stats() -> Dict:
    """Summary stats for dashboard Predictions tab."""
    conn = _conn()
    c    = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*)                                                  AS total,
            SUM(CASE WHEN status='OPEN'   THEN 1 ELSE 0 END)         AS open,
            SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END)         AS closed,
            SUM(CASE WHEN outcome='HIT'   THEN 1 ELSE 0 END)         AS hits,
            SUM(CASE WHEN outcome='MISS'  THEN 1 ELSE 0 END)         AS misses,
            SUM(CASE WHEN outcome='PARTIAL' THEN 1 ELSE 0 END)       AS partials,
            ROUND(AVG(CASE WHEN outcome_pnl_pct IS NOT NULL
                           THEN outcome_pnl_pct END), 2)             AS avg_pnl_pct,
            ROUND(
              100.0 * SUM(CASE WHEN outcome IN ('HIT','PARTIAL') THEN 1 ELSE 0 END)
              / NULLIF(SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END), 0),
              1)                                                      AS accuracy_pct
        FROM squeeze_predictions
    """)
    row   = c.fetchone()
    stats = dict(row) if row else {}
    conn.close()
    return stats
