"""
TRaNKSP — Run Tracker
Tracks every screen run and universe build with per-ticker service timestamps.
Feeds the Runs tab in the UI.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("tranksp.run_tracker")
DB_PATH = os.path.join("data", "tranksp.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def start_run(run_id: str, trigger: str, ticker_count: int) -> None:
    """Record the start of a run."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO screen_runs
            (run_id, run_at, tickers_screened, candidates_found,
             new_tickers, updated_tickers, total_in_db, prev_total)
        VALUES (?,?,?,0,0,0,0,0)
    """, (run_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), ticker_count))
    conn.commit()
    conn.close()


def save_run_detail(
    run_id: str,
    ticker: str,
    db_operation: str,          # INSERT or UPDATE
    used_massive: bool = False,
    used_yahoo: bool = False,
    used_finviz: bool = False,
    db_write_time: str = "",
    llm_time: str = "",
    rag_time: str = "",
    chromadb_time: str = "",
    agent_time: str = "",
    mcp_time: str = "",
    error: str = ""
) -> None:
    """Save per-ticker run detail row."""
    conn = _conn()
    c = conn.cursor()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    c.execute("""
        INSERT INTO run_details
            (run_id, ticker, db_operation,
             used_massive, used_yahoo, used_finviz,
             db_write_time, llm_time, rag_time, chromadb_time,
             agent_time, mcp_time, error, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, ticker, db_operation,
        int(used_massive), int(used_yahoo), int(used_finviz),
        db_write_time or now, llm_time, rag_time, chromadb_time,
        agent_time, mcp_time, error, now
    ))
    conn.commit()
    conn.close()


def get_runs(limit: int = 30) -> List[Dict]:
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT run_id, run_at, tickers_screened, candidates_found,
               new_tickers, updated_tickers, total_in_db, prev_total,
               (total_in_db - prev_total) as delta
        FROM screen_runs
        ORDER BY run_at DESC LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_run_details(run_id: str) -> List[Dict]:
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM run_details WHERE run_id=? ORDER BY created_at ASC
    """, (run_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
