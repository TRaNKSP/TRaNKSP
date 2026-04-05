"""
TRaNKSP — Finnhub API Client (Short Interest + Fundamentals)

FREE TIER: 60 calls/minute — no credit card required
Get key at: https://finnhub.io/dashboard

Used for:
  - Short interest (shares short, short float %, days to cover)
  - Basic company profile (market cap, float, sector)
  - Earnings calendar (upcoming binary events)

Finnhub short interest endpoint:
  GET https://finnhub.io/api/v1/stock/short-interest?symbol=GME&token=...
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import date, timedelta

logger = logging.getLogger("tranksp.finnhub")

# 60 req/min = 1 req/sec. Use 1.2s gap to stay safely under.
_FINNHUB_DELAY = 1.2
_last_call_time: float = 0.0


def _get(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Rate-limited GET to Finnhub API."""
    global _last_call_time
    import requests

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.debug("[Finnhub] No FINNHUB_API_KEY in .env — skipping")
        return None

    elapsed = time.time() - _last_call_time
    if elapsed < _FINNHUB_DELAY:
        time.sleep(_FINNHUB_DELAY - elapsed)

    url = f"https://finnhub.io/api/v1{endpoint}"
    p = {"token": api_key}
    if params:
        p.update(params)

    try:
        _last_call_time = time.time()
        resp = requests.get(url, params=p, timeout=10)

        if resp.status_code == 429:
            logger.warning("[Finnhub] 429 rate limit — waiting 30s")
            time.sleep(30)
            _last_call_time = time.time()
            resp = requests.get(url, params=p, timeout=10)

        if resp.status_code != 200:
            logger.warning(f"[Finnhub] HTTP {resp.status_code} for {endpoint}")
            return None

        return resp.json()
    except Exception as e:
        logger.error(f"[Finnhub] Error for {endpoint}: {e}")
        return None


def get_short_interest(ticker: str) -> Optional[Dict[str, Any]]:
    """
    GET /stock/short-interest?symbol={ticker}
    Returns short interest data from FINRA (updated bi-monthly).
    Fields: shortInterest, shortPercent, daysTocover, date
    """
    # Get last 2 periods for trend calculation
    today = date.today().isoformat()
    three_months_ago = (date.today() - timedelta(days=90)).isoformat()

    data = _get("/stock/short-interest", {
        "symbol": ticker,
        "from": three_months_ago,
        "to": today
    })

    if not data or "data" not in data:
        logger.debug(f"[Finnhub] No short interest for {ticker}")
        return None

    records = data.get("data", [])
    if not records:
        return None

    # Sort by date desc, most recent first
    records = sorted(records, key=lambda x: x.get("date", ""), reverse=True)
    latest = records[0]

    shares_short = latest.get("shortInterest", 0) or 0
    short_pct = latest.get("shortPercent", 0) or 0
    # shortPercent from Finnhub is already a percentage (e.g. 0.25 = 25%)
    if short_pct and short_pct < 1:
        short_pct = short_pct * 100

    days_to_cover = latest.get("daysToCover", 0) or 0

    # SI trend
    si_trend = "FLAT"
    if len(records) >= 2:
        prev = records[1].get("shortInterest", 0) or 0
        if prev > 0 and shares_short > 0:
            delta = (shares_short - prev) / prev
            si_trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else "FLAT"

    logger.info(f"[Finnhub] SI {ticker}: {short_pct:.1f}% DTC={days_to_cover:.1f} {si_trend}")

    return {
        "short_float":   round(float(short_pct), 2),
        "shares_short":  shares_short,
        "days_to_cover": round(float(days_to_cover), 2),
        "si_trend":      si_trend,
        "settlement_date": latest.get("date", ""),
        "source": "finnhub"
    }


def get_company_profile(ticker: str) -> Optional[Dict[str, Any]]:
    """
    GET /stock/profile2?symbol={ticker}
    Returns market cap, float, sector, industry.
    """
    data = _get("/stock/profile2", {"symbol": ticker})
    if not data or not data.get("marketCapitalization"):
        return None

    market_cap = (data.get("marketCapitalization", 0) or 0) * 1_000_000  # FH returns in millions
    float_shares = data.get("shareOutstanding", 0) or 0  # in millions

    return {
        "market_cap":   market_cap,
        "float_shares": float(float_shares),  # already in millions
        "name":         data.get("name", ticker),
        "sector":       data.get("finnhubIndustry", ""),
        "exchange":     data.get("exchange", ""),
    }


def get_earnings_date(ticker: str) -> Optional[str]:
    """
    GET /calendar/earnings?symbol={ticker}
    Returns next upcoming earnings date string or None.
    """
    today = date.today().isoformat()
    three_months = (date.today() + timedelta(days=90)).isoformat()

    data = _get("/calendar/earnings", {
        "symbol": ticker,
        "from": today,
        "to": three_months
    })

    if not data:
        return None

    earnings = data.get("earningsCalendar", [])
    if not earnings:
        return None

    # Sort by date, return nearest
    earnings = sorted(earnings, key=lambda x: x.get("date", ""))
    return earnings[0].get("date") if earnings else None
