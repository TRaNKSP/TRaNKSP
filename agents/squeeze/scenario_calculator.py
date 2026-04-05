"""
TRaNKSP — Scenario P&L Calculator
Scenario A: Long stock
Scenario B: Call options (Black-Scholes estimated at detection, actual on exit)
Scenario C: Put options (triggered at squeeze peak)
Combined A+C
"""

import os
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .options_analyzer import (
    estimate_call_price, estimate_put_price,
    get_latest_options
)
from .output_schema import ScenarioPnL

logger = logging.getLogger("tranksp.scenarios")

DB_PATH = os.path.join("data", "tranksp.db")


def get_capital(settings: Dict[str, Any]) -> float:
    return float(settings.get("capital_per_scenario", 15000))


# ── Scenario A: Long Stock ────────────────────────────────────────────────────

def calc_scenario_a(
    capital: float,
    entry_price: float,
    exit_price: float
) -> ScenarioPnL:
    """Long stock: buy shares at entry, sell at exit."""
    if entry_price <= 0:
        return ScenarioPnL(
            label="A: Long Stock", capital=capital,
            entry_price=entry_price, exit_price=exit_price,
            quantity=0, entry_cost=0, exit_value=0, pnl=0, pct=0,
            notes="Invalid entry price"
        )
    
    shares = int(capital / entry_price)
    entry_cost = shares * entry_price
    exit_value = shares * exit_price
    pnl = exit_value - entry_cost
    pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0
    
    return ScenarioPnL(
        label="A: Long Stock",
        capital=capital,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=float(shares),
        entry_cost=entry_cost,
        exit_value=exit_value,
        pnl=pnl,
        pct=round(pct, 2),
        notes=f"{shares} shares @ ${entry_price:.2f}"
    )


# ── Scenario B: Call Options ──────────────────────────────────────────────────

def calc_scenario_b(
    capital: float,
    entry_price: float,
    exit_price: float,
    call_strike: Optional[float],
    call_expiry_days: int = 30,
    iv: float = 0.60,
    actual_exit_premium: Optional[float] = None
) -> ScenarioPnL:
    """
    Call options: buy ATM calls at detection (Black-Scholes estimate),
    exit at squeeze peak (actual or estimated).
    """
    if not call_strike:
        call_strike = round(entry_price * 1.05, 0)  # 5% OTM default
    
    # Entry premium (Black-Scholes estimate)
    entry_premium = estimate_call_price(entry_price, call_strike, max(call_expiry_days, 5), iv)
    if entry_premium <= 0:
        entry_premium = max(entry_price * 0.05, 0.50)
    
    # Contracts purchasable (1 contract = 100 shares)
    contract_cost = entry_premium * 100
    contracts = int(capital / contract_cost) if contract_cost > 0 else 0
    
    if contracts == 0:
        return ScenarioPnL(
            label="B: Call Options", capital=capital,
            entry_price=entry_price, exit_price=exit_price,
            quantity=0, entry_cost=0, exit_value=0, pnl=0, pct=0,
            notes="Insufficient capital for even 1 contract"
        )
    
    entry_cost = contracts * contract_cost
    
    # Exit premium
    if actual_exit_premium is not None:
        exit_premium = actual_exit_premium
        notes_suffix = "(actual exit)"
    else:
        # Estimate exit at squeeze peak price
        days_remaining = max(call_expiry_days - 14, 1)  # assume 2 weeks into trade
        exit_premium = estimate_call_price(exit_price, call_strike, days_remaining, iv * 1.3)  # IV expansion
        notes_suffix = "(BS estimated exit)"
    
    exit_value = contracts * exit_premium * 100
    pnl = exit_value - entry_cost
    pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0
    
    return ScenarioPnL(
        label="B: Call Options",
        capital=capital,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=float(contracts),
        entry_cost=entry_cost,
        exit_value=exit_value,
        pnl=pnl,
        pct=round(pct, 2),
        notes=f"{contracts} contracts | Strike ${call_strike:.0f} | Entry: ${entry_premium:.2f} {notes_suffix}"
    )


# ── Scenario C: Put Options ───────────────────────────────────────────────────

def calc_scenario_c(
    capital: float,
    peak_price: float,
    bearish_target: float,
    put_strike: Optional[float],
    put_expiry_days: int = 21,
    iv: float = 0.70,
    actual_exit_premium: Optional[float] = None
) -> ScenarioPnL:
    """
    Put options: triggered at squeeze peak (bearish phase).
    Buy ATM-ish puts at peak, sell when price hits target.
    """
    if not put_strike:
        put_strike = round(peak_price * 0.95, 0)  # 5% ITM put
    
    # Entry premium (at squeeze peak)
    entry_premium = estimate_put_price(peak_price, put_strike, max(put_expiry_days, 5), iv)
    if entry_premium <= 0:
        entry_premium = max(peak_price * 0.04, 0.50)
    
    contract_cost = entry_premium * 100
    contracts = int(capital / contract_cost) if contract_cost > 0 else 0
    
    if contracts == 0:
        return ScenarioPnL(
            label="C: Put Options", capital=capital,
            entry_price=peak_price, exit_price=bearish_target,
            quantity=0, entry_cost=0, exit_value=0, pnl=0, pct=0,
            notes="Insufficient capital for even 1 put contract"
        )
    
    entry_cost = contracts * contract_cost
    
    # Exit premium
    if actual_exit_premium is not None:
        exit_premium = actual_exit_premium
        notes_suffix = "(actual exit)"
    else:
        days_remaining = max(put_expiry_days - 10, 1)
        # At bearish target, put should be deep ITM
        exit_premium = estimate_put_price(bearish_target, put_strike, days_remaining, iv * 0.8)
        notes_suffix = "(BS estimated exit)"
    
    exit_value = contracts * exit_premium * 100
    pnl = exit_value - entry_cost
    pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0
    
    return ScenarioPnL(
        label="C: Put Options",
        capital=capital,
        entry_price=peak_price,
        exit_price=bearish_target,
        quantity=float(contracts),
        entry_cost=entry_cost,
        exit_value=exit_value,
        pnl=pnl,
        pct=round(pct, 2),
        notes=f"{contracts} contracts | Strike ${put_strike:.0f} | Entry: ${entry_premium:.2f} {notes_suffix}"
    )


# ── Combined A+C ──────────────────────────────────────────────────────────────

def calc_combined_ac(scenario_a: ScenarioPnL, scenario_c: ScenarioPnL) -> Dict[str, float]:
    combined_pnl = scenario_a.pnl + scenario_c.pnl
    combined_cost = scenario_a.entry_cost + scenario_c.entry_cost
    combined_pct = (combined_pnl / combined_cost * 100) if combined_cost > 0 else 0
    return {"pnl": round(combined_pnl, 2), "pct": round(combined_pct, 2)}


# ── Full Scenario Calculator ──────────────────────────────────────────────────

def calculate_all_scenarios(
    ticker: str,
    capital: float,
    entry_price: float,
    bullish_exit_price: float,
    bearish_target_price: float,
    peak_price: Optional[float] = None,
    # Optional overrides from user input
    call_strike_override: Optional[float] = None,
    put_strike_override:  Optional[float] = None,
    call_entry_override: Optional[float] = None,
    call_exit_override: Optional[float] = None,
    put_entry_override: Optional[float] = None,
    put_exit_override: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate all 3 scenarios + combined for a ticker.
    Returns dict with all scenario results and saves to DB.
    """
    if not peak_price:
        peak_price = bullish_exit_price
    
    # Get options data
    options = get_latest_options(ticker)
    call_strike = call_strike_override or (options.get("best_call_strike") if options else None)
    put_strike  = put_strike_override  or (options.get("best_put_strike")  if options else None)
    atm_iv_pct = options.get("atm_iv", 60) if options else 60
    iv = (atm_iv_pct or 60) / 100.0  # convert % to decimal
    iv = max(0.20, min(iv, 3.0))  # clamp
    
    # Scenario A
    s_a = calc_scenario_a(capital, entry_price, bullish_exit_price)
    
    # Scenario B (calls)
    s_b = calc_scenario_b(
        capital, entry_price, bullish_exit_price,
        call_strike=call_strike,
        call_expiry_days=30,
        iv=iv,
        actual_exit_premium=call_exit_override
    )
    if call_entry_override:
        # Recalculate with actual entry premium
        contracts = int(capital / (call_entry_override * 100)) if call_entry_override > 0 else 0
        entry_cost = contracts * call_entry_override * 100
        exit_prem = call_exit_override or estimate_call_price(bullish_exit_price, call_strike or entry_price * 1.05, 15, iv * 1.3)
        exit_value = contracts * exit_prem * 100
        s_b = ScenarioPnL(
            label="B: Call Options",
            capital=capital, entry_price=entry_price, exit_price=bullish_exit_price,
            quantity=float(contracts), entry_cost=entry_cost, exit_value=exit_value,
            pnl=exit_value - entry_cost,
            pct=round((exit_value - entry_cost) / entry_cost * 100, 2) if entry_cost > 0 else 0,
            notes=f"{contracts} contracts | Actual entry: ${call_entry_override:.2f}"
        )
    
    # Scenario C (puts)
    s_c = calc_scenario_c(
        capital, peak_price, bearish_target_price,
        put_strike=put_strike,
        put_expiry_days=21,
        iv=iv * 1.2,  # IV typically higher for puts
        actual_exit_premium=put_exit_override
    )
    if put_entry_override:
        contracts = int(capital / (put_entry_override * 100)) if put_entry_override > 0 else 0
        entry_cost = contracts * put_entry_override * 100
        exit_prem = put_exit_override or estimate_put_price(bearish_target_price, put_strike or peak_price * 0.95, 10, iv)
        exit_value = contracts * exit_prem * 100
        s_c = ScenarioPnL(
            label="C: Put Options",
            capital=capital, entry_price=peak_price, exit_price=bearish_target_price,
            quantity=float(contracts), entry_cost=entry_cost, exit_value=exit_value,
            pnl=exit_value - entry_cost,
            pct=round((exit_value - entry_cost) / entry_cost * 100, 2) if entry_cost > 0 else 0,
            notes=f"{contracts} contracts | Actual entry: ${put_entry_override:.2f}"
        )
    
    # Combined A+C
    combined = calc_combined_ac(s_a, s_c)
    
    result = {
        "ticker": ticker,
        "capital": capital,
        "entry_price": entry_price,
        "bullish_exit": bullish_exit_price,
        "bearish_target": bearish_target_price,
        "peak_price": peak_price,
        "scenario_a": s_a.model_dump(),
        "scenario_b": s_b.model_dump(),
        "scenario_c": s_c.model_dump(),
        "combined_ac": combined,
        "options_used": {
            "call_strike": call_strike,
            "put_strike": put_strike,
            "iv_pct": atm_iv_pct
        }
    }
    
    _save_scenarios(ticker, capital, entry_price, bullish_exit_price, bearish_target_price, s_a, s_b, s_c, combined)
    
    return result


def _save_scenarios(ticker, capital, entry_price, exit_bullish, exit_bearish, s_a, s_b, s_c, combined):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO squeeze_scenarios
            (ticker, capital, entry_price, exit_price_bullish, exit_price_bearish,
             scenario_a_shares, scenario_a_pnl, scenario_a_pct,
             scenario_b_contracts, scenario_b_entry_est, scenario_b_exit_actual, scenario_b_pnl, scenario_b_pct,
             scenario_c_contracts, scenario_c_entry_est, scenario_c_exit_actual, scenario_c_pnl, scenario_c_pct,
             combined_ac_pnl, combined_ac_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ticker, capital, entry_price, exit_bullish, exit_bearish,
        s_a.quantity, s_a.pnl, s_a.pct,
        s_b.quantity, s_b.entry_cost / max(s_b.quantity * 100, 1), None, s_b.pnl, s_b.pct,
        s_c.quantity, s_c.entry_cost / max(s_c.quantity * 100, 1), None, s_c.pnl, s_c.pct,
        combined["pnl"], combined["pct"]
    ))
    conn.commit()
    conn.close()


def get_saved_scenarios(ticker: str) -> Optional[Dict[str, Any]]:
    """Get most recent saved scenarios for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM squeeze_scenarios WHERE ticker=? ORDER BY calculated_at DESC LIMIT 1", (ticker,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None
