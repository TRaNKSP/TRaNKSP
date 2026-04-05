"""
TRaNKSP — Massive.com (formerly Polygon.io) API Client

FREE TIER ENDPOINTS ONLY (5 req/min):
  ✓ /v2/aggs/ticker/{ticker}/range/1/day/...  — daily OHLCV bars (price + volume)
  ✓ /v3/reference/tickers/{ticker}            — market cap, float shares
  ✓ /v3/reference/short-interest              — short float %, DTC, SI trend
  ✗ /v2/snapshot/...                          — PAID only (403 on free tier)
  ✗ /v2/last/trade/...                        — PAID only

Rate limit: 5 requests/minute → enforce 13s minimum gap between ALL calls.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta

logger = logging.getLogger("tranksp.massive")

_FREE_TIER_DELAY = 13.0   # 60s / 5 req = 12s + 1s safety buffer
_last_call_time: float = 0.0


def _rate_limited_get(endpoint: str, params: dict = None) -> Optional[Dict]:
    """
    Rate-limited GET to Massive API. Enforces 13s gap between all calls.
    Returns parsed JSON or None on any error.
    """
    global _last_call_time
    import requests

    api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        logger.error("[Massive] No API key. Set MASSIVE_API_KEY in .env")
        return None

    elapsed = time.time() - _last_call_time
    if elapsed < _FREE_TIER_DELAY:
        wait = _FREE_TIER_DELAY - elapsed
        logger.debug(f"[Massive] Rate limit wait: {wait:.1f}s")
        time.sleep(wait)

    url = f"https://api.polygon.io{endpoint}"   # polygon.io and massive.com both work
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        _last_call_time = time.time()
        resp = requests.get(url, headers=headers, params=params or {}, timeout=15)

        if resp.status_code == 429:
            logger.warning("[Massive] 429 rate limit — waiting 60s then retrying")
            time.sleep(60)
            _last_call_time = time.time()
            resp = requests.get(url, headers=headers, params=params or {}, timeout=15)

        if resp.status_code == 403:
            logger.error(f"[Massive] 403 Forbidden — endpoint requires paid plan: {endpoint}")
            return None

        if resp.status_code != 200:
            logger.warning(f"[Massive] HTTP {resp.status_code} for {endpoint}")
            return None

        return resp.json()

    except Exception as e:
        logger.error(f"[Massive] Request error for {endpoint}: {e}")
        return None


# ── Daily Aggregates (FREE) — price + volume ──────────────────────────────────

def get_daily_aggregates(ticker: str, days: int = 32) -> Optional[List[Dict]]:
    """
    GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
    FREE tier endpoint. Returns daily OHLCV bars sorted ascending.
    """
    today = date.today()
    from_date = (today - timedelta(days=days)).isoformat()
    to_date = today.isoformat()

    data = _rate_limited_get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        params={"adjusted": "true", "sort": "asc", "limit": 50}
    )

    if not data:
        return None

    status = data.get("status", "")
    if status not in ("OK", "ok", "DELAYED"):
        logger.debug(f"[Massive] Aggs status={status} for {ticker}")
        return None

    results = data.get("results", [])
    if not results:
        logger.debug(f"[Massive] No agg bars for {ticker}")
        return None

    return results


def get_price_and_volume(ticker: str) -> Optional[Dict[str, float]]:
    """
    Derive current price and volume ratio from daily aggregate bars (FREE).
    Uses most recent bar's close as price, compares to 30-day avg volume.
    Returns {price, volume_ratio} or None.
    """
    bars = get_daily_aggregates(ticker, days=32)
    if not bars or len(bars) < 2:
        return None

    latest = bars[-1]
    price = latest.get("c", 0)  # close price
    if not price or price <= 0:
        return None

    cur_vol = latest.get("v", 0)
    prior_vols = [b.get("v", 0) for b in bars[:-1] if b.get("v", 0) > 0]
    avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else cur_vol
    volume_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    logger.info(f"[Massive] Aggs {ticker}: ${price:.2f} vol×{volume_ratio:.1f}")

    return {
        "price": float(price),
        "volume_ratio": volume_ratio,
        "open": latest.get("o", price),
        "high": latest.get("h", price),
        "low":  latest.get("l", price),
        "volume": cur_vol,
    }


# ── Short Interest (FREE) ─────────────────────────────────────────────────────

def get_short_interest(ticker: str) -> Optional[Dict[str, Any]]:
    """
    GET /v3/reference/short-interest?ticker={ticker}
    FREE tier. Updated bi-monthly by FINRA. Returns short float %, DTC, SI trend.
    """
    data = _rate_limited_get(
        "/v3/reference/short-interest",
        params={"ticker": ticker, "limit": 4, "order": "desc"}
    )

    if not data or data.get("status") not in ("OK", "ok"):
        logger.debug(f"[Massive] No short interest data for {ticker}")
        return None

    results = data.get("results", [])
    if not results:
        return None

    latest = results[0]
    shares_short  = latest.get("short_interest", 0) or 0
    shares_float  = latest.get("shares_float",   0) or 0
    avg_daily_vol = latest.get("avg_daily_volume", 0) or 0

    short_float   = (shares_short / shares_float * 100) if shares_float > 0 else 0
    days_to_cover = (shares_short / avg_daily_vol) if avg_daily_vol > 0 else 0

    si_trend = "FLAT"
    if len(results) >= 2:
        prior = results[1].get("short_interest", 0) or 0
        if prior > 0:
            delta = (shares_short - prior) / prior
            si_trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else "FLAT"

    logger.info(f"[Massive] SI {ticker}: {short_float:.1f}% DTC={days_to_cover:.1f} {si_trend}")

    return {
        "short_float":   round(short_float, 2),
        "shares_short":  shares_short,
        "shares_float":  shares_float,
        "days_to_cover": round(days_to_cover, 2),
        "si_trend":      si_trend,
        "settlement_date": latest.get("settlement_date", ""),
    }


# ── Ticker Reference Details (FREE) ──────────────────────────────────────────

def get_ticker_details(ticker: str) -> Optional[Dict[str, Any]]:
    """
    GET /v3/reference/tickers/{ticker}
    FREE tier. Returns market cap, float shares, sector.
    """
    data = _rate_limited_get(f"/v3/reference/tickers/{ticker}")
    if not data or data.get("status") not in ("OK", "ok"):
        return None

    r = data.get("results", {})
    market_cap   = r.get("market_cap", 0) or 0
    float_shares = r.get("share_class_shares_outstanding", 0) or \
                   r.get("weighted_shares_outstanding", 0) or 0

    return {
        "market_cap":   market_cap,
        "float_shares": float_shares / 1_000_000 if float_shares else 0,
        "name":         r.get("name", ticker),
        "sector":       r.get("sic_description", ""),
    }


# ── Backwards-compat alias used by lifecycle_tracker & tools ─────────────────

def get_ticker_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Alias for get_price_and_volume() using FREE aggregate bars endpoint.
    Named 'snapshot' for API compatibility with existing callers.
    """
    return get_price_and_volume(ticker)
