"""
TRaNKSP — Database Migration (idempotent)
Run with: python migrate_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join("data", "tranksp.db")

MIGRATIONS = [
    # ── Settings ──────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # Default settings
    """
    INSERT OR IGNORE INTO settings (key, value) VALUES
        ('capital_per_scenario', '15000'),
        ('portfolio_size', '10'),
        ('min_score_threshold', '40'),
        ('mc_floor', '250000000'),
        ('put_oi_threshold', '500'),
        ('auto_eval_hours', '72'),
        ('universe_size', '30'),
        ('yahoo_screener_pages', '5'),
        ('yahoo_page_delay', '5')
    """,

    # ── Universe (live candidates pulled from internet) ────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_universe (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker        TEXT NOT NULL UNIQUE,
        source        TEXT,
        added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_checked  TIMESTAMP,
        active        INTEGER DEFAULT 1
    )
    """,

    # ── Screener results ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL,
        ticker        TEXT NOT NULL,
        score         REAL,
        short_float   REAL,
        days_to_cover REAL,
        float_shares  REAL,
        price         REAL,
        market_cap    REAL,
        volume_ratio  REAL,
        si_trend      TEXT,
        has_options   INTEGER DEFAULT 0,
        thesis_id     INTEGER,
        phase         TEXT DEFAULT 'DETECTION',
        screened_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Lifecycle tracking ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_lifecycle (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker         TEXT NOT NULL,
        snapshot_date  DATE NOT NULL,
        status         TEXT DEFAULT 'ACTIVE',
        entry_price    REAL,
        peak_price     REAL,
        current_price  REAL,
        short_interest REAL,
        si_change_pct  REAL,
        price_chg_peak REAL,
        bearish_eval   INTEGER DEFAULT 0,
        eval_triggered TEXT,
        notes          TEXT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, snapshot_date)
    )
    """,

    # ── Options chain snapshots ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_options (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker           TEXT NOT NULL,
        snapshot_date    DATE NOT NULL,
        has_options      INTEGER DEFAULT 0,
        iv_rank          REAL,
        atm_iv           REAL,
        best_call_strike REAL,
        best_call_oi     INTEGER,
        best_call_expiry TEXT,
        best_put_strike  REAL,
        best_put_oi      INTEGER,
        best_put_expiry  TEXT,
        nearest_expiry   TEXT,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, snapshot_date)
    )
    """,

    # ── Scenario P&L ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_scenarios (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                TEXT NOT NULL,
        capital               REAL,
        entry_price           REAL,
        exit_price_bullish    REAL,
        exit_price_bearish    REAL,
        scenario_a_shares     REAL,
        scenario_a_pnl        REAL,
        scenario_a_pct        REAL,
        scenario_b_contracts  REAL,
        scenario_b_entry_est  REAL,
        scenario_b_exit_actual REAL,
        scenario_b_pnl        REAL,
        scenario_b_pct        REAL,
        scenario_c_contracts  REAL,
        scenario_c_entry_est  REAL,
        scenario_c_exit_actual REAL,
        scenario_c_pnl        REAL,
        scenario_c_pct        REAL,
        combined_ac_pnl       REAL,
        combined_ac_pct       REAL,
        calculated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Thesis details (structured ThesisOutput) ───────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_thesis_details (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        run_id          TEXT,
        phase           TEXT DEFAULT 'BULLISH',
        setup           TEXT,
        trigger         TEXT,
        mechanics       TEXT,
        risk            TEXT,
        catalyst_types  TEXT,
        confidence      REAL,
        time_horizon    TEXT,
        raw_thesis      TEXT,
        score_explanation TEXT,
        generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── LangChain per-ticker chat history ─────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS langchain_chat_history (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL,
        message_type  TEXT NOT NULL,
        content       TEXT NOT NULL,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Agent logs ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id     TEXT,
        ticker     TEXT,
        component  TEXT,
        level      TEXT DEFAULT 'INFO',
        message    TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Screen run stats (freshness + delta tracking) ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS screen_runs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           TEXT NOT NULL UNIQUE,
        run_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tickers_screened INTEGER DEFAULT 0,
        candidates_found INTEGER DEFAULT 0,
        new_tickers      INTEGER DEFAULT 0,
        updated_tickers  INTEGER DEFAULT 0,
        total_in_db      INTEGER DEFAULT 0,
        prev_run_id      TEXT,
        prev_total       INTEGER DEFAULT 0
    )
    """,
    # ── Prediction tracking (Phase 1) ────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS squeeze_predictions (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker               TEXT NOT NULL,
        run_id               TEXT,
        thesis_id            INTEGER,
        prediction_date      DATE NOT NULL,
        predicted_direction  TEXT DEFAULT 'UP',
        entry_price          REAL,
        target_price         REAL,
        confidence_score     REAL,
        direction            TEXT DEFAULT 'UP',
        confidence           REAL,
        time_horizon         TEXT,
        catalyst_types       TEXT,
        short_float_at_pred  REAL,
        si_at_prediction     REAL,
        dtc_at_pred          REAL,
        volume_ratio_at_pred REAL,
        si_trend_at_pred     TEXT,
        thesis_summary       TEXT,
        status               TEXT DEFAULT 'OPEN',
        outcome              TEXT,
        outcome_result       TEXT,
        outcome_price        REAL,
        outcome_date         DATE,
        outcome_pnl_pct      REAL,
        days_to_outcome      INTEGER,
        actual_peak          REAL,
        actual_drawdown      REAL,
        si_at_outcome        REAL,
        created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (thesis_id) REFERENCES squeeze_thesis_details(id)
    )
    """,

    # ── Calibration stats (Phase 3) ───────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS calibration_stats (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(key)
    )
    """,

    # ── Lessons learned (Phase 4) ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS lessons_learned (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_number     INTEGER NOT NULL,
        prediction_count INTEGER,
        lessons_text     TEXT,
        stats_snapshot   TEXT,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Run detail per ticker ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS run_details (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id         TEXT NOT NULL,
        ticker         TEXT NOT NULL,
        db_operation   TEXT,
        used_massive   INTEGER DEFAULT 0,
        used_yahoo     INTEGER DEFAULT 0,
        used_finviz    INTEGER DEFAULT 0,
        db_write_time  TEXT,
        llm_time       TEXT,
        rag_time       TEXT,
        chromadb_time  TEXT,
        agent_time     TEXT,
        mcp_time       TEXT,
        error          TEXT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def run_migrations():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"[migrate_db] Opening {DB_PATH}")
    for i, sql in enumerate(MIGRATIONS):
        sql = sql.strip()
        if not sql:
            continue
        try:
            cursor.executescript(sql) if ";" in sql and sql.count(";") > 1 else cursor.execute(sql)
            conn.commit()
            print(f"[migrate_db] Step {i+1:02d} OK")
        except sqlite3.Error as e:
            print(f"[migrate_db] Step {i+1:02d} WARNING: {e}")

    conn.close()
    print("[migrate_db] Migration complete.")


if __name__ == "__main__":
    run_migrations()
