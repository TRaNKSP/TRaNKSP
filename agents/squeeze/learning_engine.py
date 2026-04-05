"""
TRaNKSP — Self-Learning Engine (Phases 1-4)

Phase 1: Outcome tracking — every prediction stored + evaluated daily
Phase 2: Episodic memory — ChromaDB stores completed prediction stories
Phase 3: Calibration stats — accuracy by signal type, confidence band, sector
Phase 4: Meta-learning — LLM writes lessons learned every 10 cycles

Tables used:
  squeeze_predictions  — all predictions with outcomes
  calibration_stats    — computed accuracy per signal dimension
  lessons_learned      — LLM-generated insight documents
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta

logger = logging.getLogger("tranksp.learning")

DB_PATH = os.path.join("data", "tranksp.db")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Phase 1: Prediction Storage ───────────────────────────────────────────────

def save_prediction(
    ticker: str,
    run_id: str,
    entry_price: float,
    target_price: float,
    direction: str,
    confidence: float,
    time_horizon: str,
    catalyst_types: List[str],
    short_float: float,
    days_to_cover: float,
    volume_ratio: float,
    si_trend: str,
    thesis_summary: str = ""
) -> int:
    """Store a new prediction. Returns prediction_id."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO squeeze_predictions
            (ticker, run_id, prediction_date, entry_price, target_price,
             direction, confidence, time_horizon, catalyst_types,
             short_float_at_pred, dtc_at_pred, volume_ratio_at_pred,
             si_trend_at_pred, thesis_summary, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ticker, run_id, date.today().isoformat(),
        entry_price, target_price, direction, confidence,
        time_horizon, json.dumps(catalyst_types),
        short_float, days_to_cover, volume_ratio, si_trend,
        thesis_summary, "OPEN"
    ))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"[Learning] Saved prediction #{pred_id}: {ticker} {direction} target=${target_price:.2f} conf={confidence:.0f}%")
    return pred_id


def get_open_predictions() -> List[Dict]:
    """Get all predictions not yet evaluated."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_predictions
        WHERE status = 'OPEN'
        ORDER BY prediction_date ASC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def evaluate_prediction(pred_id: int, current_price: float, current_si: float = 0) -> Optional[str]:
    """
    Evaluate an open prediction against current market data.
    Returns outcome string: HIT / MISS / PARTIAL / REVERSAL / EXPIRED
    """
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM squeeze_predictions WHERE id=?", (pred_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None

    pred = dict(row)
    entry = pred["entry_price"] or 0
    target = pred["target_price"] or 0
    direction = pred["direction"]
    pred_date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d").date()
    days_elapsed = (date.today() - pred_date).days

    # Parse time horizon into max days
    horizon_str = pred["time_horizon"] or "2 weeks"
    max_days = 14
    if "day" in horizon_str.lower():
        nums = [int(s) for s in horizon_str.split() if s.isdigit()]
        max_days = max(nums) if nums else 7
    elif "week" in horizon_str.lower():
        nums = [int(s) for s in horizon_str.split() if s.isdigit()]
        max_days = max(nums) * 7 if nums else 14
    elif "month" in horizon_str.lower():
        max_days = 30

    outcome = None

    if direction == "UP":
        if target > 0 and current_price >= target:
            outcome = "HIT"
        elif entry > 0 and current_price <= entry * 0.85:
            outcome = "MISS"  # down >15% from entry = failed
        elif days_elapsed > max_days:
            # Partial if made some progress
            pct_progress = (current_price - entry) / (target - entry) if (target - entry) > 0 else 0
            outcome = "PARTIAL" if pct_progress >= 0.5 else "EXPIRED"
    elif direction == "DOWN":
        if target > 0 and current_price <= target:
            outcome = "HIT"
        elif entry > 0 and current_price >= entry * 1.15:
            outcome = "MISS"
        elif days_elapsed > max_days:
            pct_progress = (entry - current_price) / (entry - target) if (entry - target) > 0 else 0
            outcome = "PARTIAL" if pct_progress >= 0.5 else "EXPIRED"

    if outcome:
        pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
        c.execute("""
            UPDATE squeeze_predictions SET
                outcome = ?, outcome_price = ?, outcome_date = ?,
                outcome_pnl_pct = ?, days_to_outcome = ?, status = 'CLOSED'
            WHERE id = ?
        """, (outcome, current_price, date.today().isoformat(),
              round(pnl_pct, 2), days_elapsed, pred_id))
        conn.commit()

        # Store in ChromaDB episodic memory
        _store_episode(pred, outcome, current_price, pnl_pct, days_elapsed)

        logger.info(f"[Learning] Prediction #{pred_id} {pred['ticker']}: {outcome} "
                    f"${entry:.2f}→${current_price:.2f} ({pnl_pct:+.1f}%)")

    conn.close()
    return outcome


# ── Phase 2: Episodic Memory ──────────────────────────────────────────────────

def _store_episode(pred: Dict, outcome: str, exit_price: float,
                   pnl_pct: float, days_elapsed: int):
    """Store completed prediction as episodic memory in ChromaDB."""
    try:
        from .rag import _get_collection
        col = _get_collection("squeeze_episodes")

        catalyst_types = json.loads(pred.get("catalyst_types") or "[]")
        episode_text = (
            f"PREDICTION OUTCOME — {pred['ticker']} | {pred['prediction_date']}\n"
            f"Direction: {pred['direction']} | Outcome: {outcome}\n"
            f"Entry: ${pred['entry_price']:.2f} | Target: ${pred['target_price']:.2f} | "
            f"Exit: ${exit_price:.2f}\n"
            f"P&L: {pnl_pct:+.1f}% in {days_elapsed} days\n"
            f"Confidence at prediction: {pred['confidence']:.0f}%\n"
            f"Short Interest: {pred['short_float_at_pred']:.1f}% | "
            f"DTC: {pred['dtc_at_pred']:.1f} | SI Trend: {pred['si_trend_at_pred']}\n"
            f"Catalysts cited: {', '.join(catalyst_types)}\n"
            f"Thesis: {(pred.get('thesis_summary') or '')[:300]}\n"
            f"LESSON: {'Target was reached — setup was valid.' if outcome == 'HIT' else 'Setup failed — review catalyst assumptions.'}"
        )

        doc_id = f"episode_{pred['ticker']}_{pred['prediction_date']}_{pred['id']}"
        try:
            col.delete(ids=[doc_id])
        except Exception:
            pass
        col.add(
            documents=[episode_text],
            metadatas=[{
                "ticker": pred["ticker"],
                "outcome": outcome,
                "direction": pred["direction"],
                "catalyst_types": json.dumps(catalyst_types),
                "confidence": pred["confidence"],
                "pnl_pct": pnl_pct,
                "date": pred["prediction_date"]
            }],
            ids=[doc_id]
        )
        logger.debug(f"[Learning] Episode stored in ChromaDB: {pred['ticker']} {outcome}")
    except Exception as e:
        logger.error(f"[Learning] Episode storage error: {e}")


def query_episode_memory(ticker: str, context: str = "short squeeze prediction outcome") -> str:
    """Retrieve prior prediction episodes for a ticker."""
    try:
        from .rag import _get_collection
        col = _get_collection("squeeze_episodes")
        results = col.query(
            query_texts=[f"{ticker} {context}"],
            n_results=5,
            where={"ticker": ticker} if ticker else None
        )
        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs) if docs else "No prior prediction history."
    except Exception as e:
        logger.warning(f"[Learning] Episode query error: {e}")
        return "No prior prediction history."


def query_similar_episodes(catalyst_types: List[str], min_si: float = 20) -> str:
    """Query episodes with similar catalyst types for calibration context."""
    try:
        from .rag import _get_collection
        col = _get_collection("squeeze_episodes")
        query = f"short squeeze {' '.join(catalyst_types)} high short interest outcome"
        results = col.query(query_texts=[query], n_results=5)
        docs = results.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs) if docs else ""
    except Exception as e:
        logger.warning(f"[Learning] Similar episode query error: {e}")
        return ""


# ── Phase 3: Calibration Stats ────────────────────────────────────────────────

def compute_calibration_stats() -> Dict[str, Any]:
    """
    Compute accuracy statistics across all closed predictions.
    Returns dict of accuracy breakdowns by catalyst, SI band, confidence.
    """
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_predictions
        WHERE status = 'CLOSED'
        ORDER BY outcome_date DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    if not rows:
        return {"total": 0, "message": "No completed predictions yet"}

    total = len(rows)
    hits = sum(1 for r in rows if r["outcome"] in ("HIT", "PARTIAL"))
    overall_accuracy = round(hits / total * 100, 1) if total > 0 else 0

    # By catalyst type
    catalyst_stats: Dict[str, Dict] = {}
    for row in rows:
        cats = json.loads(row.get("catalyst_types") or "[]")
        hit = row["outcome"] in ("HIT", "PARTIAL")
        for cat in cats:
            if cat not in catalyst_stats:
                catalyst_stats[cat] = {"hits": 0, "total": 0}
            catalyst_stats[cat]["total"] += 1
            if hit:
                catalyst_stats[cat]["hits"] += 1

    catalyst_accuracy = {
        cat: {
            "accuracy": round(v["hits"] / v["total"] * 100, 1),
            "total": v["total"]
        }
        for cat, v in catalyst_stats.items() if v["total"] >= 2
    }

    # By confidence band
    conf_bands = {"40-60": {"hits":0,"total":0}, "60-80": {"hits":0,"total":0}, "80-100": {"hits":0,"total":0}}
    for row in rows:
        conf = row.get("confidence") or 0
        hit = row["outcome"] in ("HIT", "PARTIAL")
        band = "40-60" if conf < 60 else "60-80" if conf < 80 else "80-100"
        conf_bands[band]["total"] += 1
        if hit:
            conf_bands[band]["hits"] += 1

    confidence_calibration = {
        band: {
            "accuracy": round(v["hits"]/v["total"]*100,1) if v["total"] > 0 else 0,
            "total": v["total"]
        }
        for band, v in conf_bands.items()
    }

    # By SI band
    si_bands = {"<20": {"hits":0,"total":0}, "20-35": {"hits":0,"total":0}, ">35": {"hits":0,"total":0}}
    for row in rows:
        si = row.get("short_float_at_pred") or 0
        hit = row["outcome"] in ("HIT", "PARTIAL")
        band = "<20" if si < 20 else "20-35" if si < 35 else ">35"
        si_bands[band]["total"] += 1
        if hit:
            si_bands[band]["hits"] += 1

    si_accuracy = {
        band: {
            "accuracy": round(v["hits"]/v["total"]*100,1) if v["total"] > 0 else 0,
            "total": v["total"]
        }
        for band, v in si_bands.items()
    }

    # Avg P&L
    pnl_values = [r["outcome_pnl_pct"] for r in rows if r.get("outcome_pnl_pct") is not None]
    avg_pnl = round(sum(pnl_values) / len(pnl_values), 1) if pnl_values else 0

    stats = {
        "total": total,
        "hits": hits,
        "overall_accuracy": overall_accuracy,
        "avg_pnl_pct": avg_pnl,
        "by_catalyst": catalyst_accuracy,
        "by_confidence": confidence_calibration,
        "by_si_band": si_accuracy,
        "computed_at": datetime.utcnow().isoformat()
    }

    # Save to DB
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO calibration_stats (key, value, computed_at)
        VALUES ('latest', ?, ?)
    """, (json.dumps(stats), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    logger.info(f"[Learning] Calibration: {total} predictions, {overall_accuracy}% accuracy, avg P&L {avg_pnl:+.1f}%")
    return stats


def get_calibration_stats() -> Dict[str, Any]:
    """Get latest calibration stats from DB."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT value FROM calibration_stats WHERE key='latest' ORDER BY computed_at DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            pass
    return {"total": 0, "message": "No calibration data yet"}


def format_calibration_for_prompt() -> str:
    """Format calibration stats as LLM context for adaptive prompting."""
    stats = get_calibration_stats()
    if stats.get("total", 0) < 5:
        return "PERFORMANCE CONTEXT: Insufficient history for calibration (need 5+ completed predictions)."

    lines = [
        f"PERFORMANCE CONTEXT (last {stats['total']} predictions):",
        f"Overall accuracy: {stats['overall_accuracy']}% | Avg P&L: {stats['avg_pnl_pct']:+.1f}%",
        ""
    ]

    # Catalyst accuracy
    by_cat = stats.get("by_catalyst", {})
    if by_cat:
        lines.append("Accuracy by catalyst type:")
        sorted_cats = sorted(by_cat.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for cat, data in sorted_cats:
            if data["total"] >= 2:
                lines.append(f"  {cat}: {data['accuracy']}% ({data['total']} predictions)")

    # Confidence calibration
    by_conf = stats.get("by_confidence", {})
    if by_conf:
        lines.append("\nConfidence calibration:")
        for band, data in by_conf.items():
            if data["total"] > 0:
                overunder = "OVERCONFIDENT" if (int(band.split("-")[0]) - data["accuracy"]) > 15 else "CALIBRATED"
                lines.append(f"  {band}% confidence → {data['accuracy']}% actual ({overunder})")

    # SI band accuracy
    by_si = stats.get("by_si_band", {})
    if by_si:
        lines.append("\nAccuracy by short interest level:")
        for band, data in by_si.items():
            if data["total"] > 0:
                lines.append(f"  SI {band}%: {data['accuracy']}% hit rate ({data['total']} cases)")

    return "\n".join(lines)


# ── Phase 4: Meta-Learning ────────────────────────────────────────────────────

async def generate_lessons_learned() -> Optional[str]:
    """
    Every 10 completed predictions, LLM generates a 'lessons learned' document.
    Stored in ChromaDB and DB for future context injection.
    """
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM squeeze_predictions WHERE status='CLOSED'")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM lessons_learned")
    lesson_count = c.fetchone()[0]
    conn.close()

    # Generate every 10 new completions
    if total == 0 or total < (lesson_count + 1) * 10:
        return None

    logger.info(f"[Learning] Generating lessons learned (cycle {lesson_count + 1}, {total} predictions)")

    stats = compute_calibration_stats()
    recent_episodes = query_similar_episodes([], 0)

    # Use Haiku to generate concise lessons
    try:
        import os
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
            max_tokens=800,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        prompt = f"""You are a trading systems analyst reviewing prediction outcomes for a short squeeze detection AI.

PERFORMANCE STATS:
{json.dumps(stats, indent=2)}

RECENT PREDICTION EPISODES:
{recent_episodes[:2000]}

Generate a concise "Lessons Learned" document (200-300 words) that:
1. Identifies the 2-3 strongest predictive signals (highest accuracy)
2. Identifies the 2-3 weakest signals to be skeptical of
3. Notes any confidence calibration issues (overconfidence in specific setups)
4. Gives 3 specific actionable rules for future predictions
5. Notes any patterns in failed predictions

Be specific and data-driven. This document will be injected into future prediction prompts."""

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        lessons = response.content

        # Store in DB
        conn = _conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO lessons_learned (cycle_number, prediction_count, lessons_text, stats_snapshot)
            VALUES (?, ?, ?, ?)
        """, (lesson_count + 1, total, lessons, json.dumps(stats)))
        conn.commit()
        conn.close()

        # Store in ChromaDB
        try:
            from .rag import _get_collection
            col = _get_collection("squeeze_episodes")
            col.add(
                documents=[f"LESSONS LEARNED (Cycle {lesson_count+1}):\n{lessons}"],
                metadatas=[{"type": "lessons_learned", "cycle": lesson_count+1, "date": date.today().isoformat()}],
                ids=[f"lessons_cycle_{lesson_count+1}"]
            )
        except Exception:
            pass

        logger.info(f"[Learning] Lessons learned generated for cycle {lesson_count+1}")
        return lessons

    except Exception as e:
        logger.error(f"[Learning] Meta-learning error: {e}")
        return None


def get_latest_lessons() -> str:
    """Get most recent lessons learned document."""
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT lessons_text FROM lessons_learned ORDER BY cycle_number DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""


# ── Prediction outcome daily check (APScheduler job) ─────────────────────────

async def run_daily_outcome_check():
    """
    Run daily by APScheduler. Checks all open predictions against current prices.
    Uses Massive API for current prices.
    """
    from .massive_client import get_price_and_volume

    open_preds = get_open_predictions()
    if not open_preds:
        logger.info("[Learning] No open predictions to evaluate")
        return

    logger.info(f"[Learning] Evaluating {len(open_preds)} open predictions")
    evaluated = 0

    for pred in open_preds:
        ticker = pred["ticker"]
        quote = _yq_get(ticker)
        snapshot = quote or get_price_and_volume(ticker)
        if not snapshot:
            continue

        current_price = snapshot.get("price", 0)
        outcome = evaluate_prediction(pred["id"], current_price)
        if outcome:
            evaluated += 1

    # Recompute calibration after evaluations
    if evaluated > 0:
        compute_calibration_stats()
        await generate_lessons_learned()

    logger.info(f"[Learning] Daily outcome check complete: {evaluated} predictions evaluated")
