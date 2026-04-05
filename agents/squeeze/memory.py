"""
TRaNKSP — Per-Ticker Thesis Memory
Uses SQLChatMessageHistory (langchain_community) with manual history management.
Keeps last 5 thesis generations per ticker.
"""

import os
import sqlite3
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("tranksp.memory")

DB_PATH = os.path.join("data", "tranksp.db")
MAX_HISTORY = 5


def save_thesis_to_history(ticker: str, thesis: Dict[str, Any], phase: str = "BULLISH"):
    """Save a thesis generation to per-ticker history (max 5 kept)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        session_id = f"{ticker}_{phase}"
        content = json.dumps({"ticker": ticker, "phase": phase, "thesis": thesis, "ts": datetime.utcnow().isoformat()})
        
        c.execute(
            "INSERT INTO langchain_chat_history (session_id, message_type, content) VALUES (?, ?, ?)",
            (session_id, "ai", content)
        )
        conn.commit()
        
        # Prune to max 5
        c.execute(
            "SELECT id FROM langchain_chat_history WHERE session_id=? ORDER BY created_at DESC",
            (session_id,)
        )
        rows = c.fetchall()
        if len(rows) > MAX_HISTORY:
            ids_to_delete = [r[0] for r in rows[MAX_HISTORY:]]
            c.execute(f"DELETE FROM langchain_chat_history WHERE id IN ({','.join('?'*len(ids_to_delete))})", ids_to_delete)
            conn.commit()
        
        conn.close()
    except Exception as e:
        logger.error(f"save_thesis_to_history error for {ticker}: {e}")


def get_thesis_history(ticker: str, phase: str = "BULLISH") -> List[Dict[str, Any]]:
    """Retrieve last N thesis generations for a ticker."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        session_id = f"{ticker}_{phase}"
        c.execute(
            "SELECT content, created_at FROM langchain_chat_history WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, MAX_HISTORY)
        )
        rows = c.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            try:
                data = json.loads(row[0])
                data["saved_at"] = row[1]
                results.append(data)
            except Exception:
                pass
        return results
    except Exception as e:
        logger.error(f"get_thesis_history error for {ticker}: {e}")
        return []


def format_history_for_context(ticker: str) -> str:
    """Format thesis history as LLM context string."""
    history = get_thesis_history(ticker)
    if not history:
        return "No previous thesis history."
    
    parts = []
    for i, item in enumerate(history):
        thesis = item.get("thesis", {})
        ts = item.get("saved_at", "unknown")
        confidence = thesis.get("confidence", "?")
        setup = thesis.get("setup", "N/A")[:200]
        parts.append(f"[Generation {i+1} - {ts}]\nConfidence: {confidence}\nSetup: {setup}")
    
    return "\n\n".join(parts)
