"""
TRaNKSP — Options Analyzer
Pulls yfinance options chain, calculates IV rank, finds best strikes.
Black-Scholes estimation for scenario planning.
"""

import os
import sqlite3
import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime, date

# yfinance removed — options analysis uses Massive API
import numpy as np

from .output_schema import OptionsSnapshot

logger = logging.getLogger("tranksp.options")

DB_PATH = os.path.join("data", "tranksp.db")


# ── Black-Scholes Helpers ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years, r risk-free rate, sigma IV."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def estimate_call_price(current_price: float, strike: float, days_to_expiry: int, iv: float) -> float:
    """Estimate call option price via Black-Scholes."""
    T = days_to_expiry / 365.0
    return black_scholes_call(current_price, strike, T, 0.045, iv)


def estimate_put_price(current_price: float, strike: float, days_to_expiry: int, iv: float) -> float:
    """Estimate put option price via Black-Scholes."""
    T = days_to_expiry / 365.0
    return black_scholes_put(current_price, strike, T, 0.045, iv)


# ── IV Rank Calculation ───────────────────────────────────────────────────────

def calculate_iv_rank(ticker_obj) -> Optional[float]:
    """
    Calculate IV Rank = (current IV - 52w low IV) / (52w high IV - 52w low IV) * 100
    Uses historical option chain data if available, else approximates from price history.
    """
    try:
        hist = ticker_obj.history(period="1y")
        if hist.empty:
            return None
        
        # Approximate HV as proxy for IV percentile
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(20).std() * math.sqrt(252) * 100  # annualized %
        
        if rolling_vol.empty:
            return None
        
        current = rolling_vol.iloc[-1]
        low_52 = rolling_vol.min()
        high_52 = rolling_vol.max()
        
        if high_52 == low_52:
            return 50.0
        
        iv_rank = (current - low_52) / (high_52 - low_52) * 100
        return round(float(iv_rank), 1)
    except Exception as e:
        logger.warning(f"IV rank calculation error: {e}")
        return None


# ── Main Options Analyzer ─────────────────────────────────────────────────────

def analyze_options(ticker: str, put_oi_threshold: int = 500) -> OptionsSnapshot:
    """
    Full options chain analysis for a ticker.
    Returns OptionsSnapshot with best strikes and liquidity assessment.
    """
    try:
        # Massive options chain not yet implemented — returning empty snapshot
        expirations = stock.options
        
        if not expirations:
            return OptionsSnapshot(has_options=False)
        
        nearest_expiry = expirations[0]
        current_price = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice") or 0
        
        if current_price <= 0:
            hist = stock.history(period="1d")
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
        
        # Get ATM IV from nearest expiry chain
        try:
            chain = stock.option_chain(nearest_expiry)
            calls = chain.calls
            puts = chain.puts
            
            # Find ATM call/put (strike closest to current price)
            if not calls.empty:
                calls = calls.copy()
                calls["dist"] = abs(calls["strike"] - current_price)
                atm_call = calls.nsmallest(1, "dist")
                atm_iv = float(atm_call["impliedVolatility"].iloc[0]) if not atm_call.empty else None
                
                # Best call: highest OI near ATM (within 10% of price)
                near_calls = calls[
                    (calls["strike"] >= current_price * 0.95) &
                    (calls["strike"] <= current_price * 1.15)
                ]
                if not near_calls.empty:
                    best_call = near_calls.nlargest(1, "openInterest")
                    best_call_strike = float(best_call["strike"].iloc[0])
                    best_call_oi = int(best_call["openInterest"].iloc[0])
                else:
                    best_call_strike = None
                    best_call_oi = None
            else:
                atm_iv = None
                best_call_strike = None
                best_call_oi = None
            
            # Best put: highest OI below current price (for bearish scenario)
            liquid_puts = False
            best_put_strike = None
            best_put_oi = None
            
            if not puts.empty:
                puts = puts.copy()
                below_puts = puts[
                    (puts["strike"] >= current_price * 0.70) &
                    (puts["strike"] <= current_price * 0.98)
                ]
                if not below_puts.empty:
                    best_put = below_puts.nlargest(1, "openInterest")
                    best_put_strike = float(best_put["strike"].iloc[0])
                    best_put_oi = int(best_put["openInterest"].iloc[0])
                    liquid_puts = best_put_oi >= put_oi_threshold
            
        except Exception as e:
            logger.warning(f"[Options] Chain error for {ticker}: {e}")
            atm_iv = None
            best_call_strike = None
            best_call_oi = None
            best_put_strike = None
            best_put_oi = None
            liquid_puts = False
        
        # IV Rank
        iv_rank = calculate_iv_rank(stock)
        
        snapshot = OptionsSnapshot(
            has_options=True,
            iv_rank=iv_rank,
            atm_iv=round(float(atm_iv) * 100, 1) if atm_iv else None,
            best_call_strike=best_call_strike,
            best_call_oi=best_call_oi,
            best_call_expiry=nearest_expiry,
            best_put_strike=best_put_strike,
            best_put_oi=best_put_oi,
            best_put_expiry=nearest_expiry,
            nearest_expiry=nearest_expiry,
            liquid_puts_available=liquid_puts
        )
        
        # Persist to DB
        _save_options_snapshot(ticker, snapshot)
        
        return snapshot
        
    except Exception as e:
        logger.error(f"[Options] Error for {ticker}: {e}")
        return OptionsSnapshot(has_options=False)


def _save_options_snapshot(ticker: str, snap: OptionsSnapshot):
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO squeeze_options
            (ticker, snapshot_date, has_options, iv_rank, atm_iv,
             best_call_strike, best_call_oi, best_call_expiry,
             best_put_strike, best_put_oi, best_put_expiry, nearest_expiry)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker, snapshot_date) DO UPDATE SET
            has_options=excluded.has_options,
            iv_rank=excluded.iv_rank,
            atm_iv=excluded.atm_iv,
            best_call_strike=excluded.best_call_strike,
            best_call_oi=excluded.best_call_oi,
            best_put_strike=excluded.best_put_strike,
            best_put_oi=excluded.best_put_oi
    """, (
        ticker, today, int(snap.has_options), snap.iv_rank, snap.atm_iv,
        snap.best_call_strike, snap.best_call_oi, snap.best_call_expiry,
        snap.best_put_strike, snap.best_put_oi, snap.best_put_expiry,
        snap.nearest_expiry
    ))
    conn.commit()
    conn.close()


def get_latest_options(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieve most recent options snapshot from DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM squeeze_options WHERE ticker=?
        ORDER BY snapshot_date DESC LIMIT 1
    """, (ticker,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None
