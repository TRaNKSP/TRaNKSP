"""
TRaNKSP — Financial Datasets API Client
https://financialdatasets.ai

Supplements Yahoo Finance / Massive with:
  ✓ Income statements        — revenue, earnings, profitability trends
  ✓ Balance sheets           — cash, debt, equity
  ✓ Cash flow statements     — operating/free cash flow
  ✓ Current stock price      — real-time price fallback
  ✓ Historical stock prices  — OHLCV history (up to 2 years)
  ✓ Company news             — catalyst feed for squeeze thesis

Authentication:
  - Set FINANCIAL_DATASETS_API_KEY in .env
  - REST calls use X-API-KEY header (no OAuth needed for programmatic use)

Rate limits:
  - Paid plan: generous — no enforced delay needed
  - Free plan:  ~5 req/min — set FD_CALL_DELAY=13 in .env if on free tier

Usage in TRaNKSP:
  - Called by nodes.py as supplemental fundamentals layer
  - Does NOT replace Yahoo short interest data (Yahoo is best for SI/DTC)
  - Used to enrich thesis with revenue trend, debt load, cash runway
  - Current price used as fallback if Yahoo and Massive both fail
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import date, timedelta

import httpx

logger = logging.getLogger("tranksp.financial_datasets")

BASE_URL = "https://api.financialdatasets.ai"

# Default 1s gap between calls — adequate for paid plans.
# Override with FD_CALL_DELAY env var if on free tier (set to 13).
_CALL_DELAY = float(os.environ.get("FD_CALL_DELAY", "1.0"))
_last_call_time: float = 0.0


def _get_api_key() -> Optional[str]:
    return os.environ.get("FINANCIAL_DATASETS_API_KEY", "").strip() or None


def _rate_limited_get(endpoint: str, params: dict = None) -> Optional[Dict]:
    """
    Rate-limited GET to Financial Datasets API.
    Returns parsed JSON dict or None on any error.
    """
    global _last_call_time

    api_key = _get_api_key()
    if not api_key:
        logger.debug("[FD] FINANCIAL_DATASETS_API_KEY not set — skipping")
        return None

    elapsed = time.time() - _last_call_time
    if elapsed < _CALL_DELAY:
        time.sleep(_CALL_DELAY - elapsed)

    url = f"{BASE_URL}{endpoint}"
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
    }

    try:
        _last_call_time = time.time()
        resp = httpx.get(url, headers=headers, params=params or {}, timeout=15)

        if resp.status_code == 429:
            logger.warning("[FD] 429 rate limit — waiting 60s then retrying")
            time.sleep(60)
            _last_call_time = time.time()
            resp = httpx.get(url, headers=headers, params=params or {}, timeout=15)

        if resp.status_code == 401:
            logger.error("[FD] 401 Unauthorized — check FINANCIAL_DATASETS_API_KEY")
            return None

        if resp.status_code == 402:
            logger.warning(f"[FD] 402 Payment Required — endpoint requires paid plan: {endpoint}")
            return None

        if resp.status_code != 200:
            logger.debug(f"[FD] HTTP {resp.status_code} for {endpoint}")
            return None

        return resp.json()

    except Exception as e:
        logger.debug(f"[FD] Request error for {endpoint}: {e}")
        return None


# ── Current Stock Price ───────────────────────────────────────────────────────

def get_current_price(ticker: str) -> Optional[Dict[str, Any]]:
    """
    GET /prices/snapshot?ticker={ticker}
    Returns current price, volume, open, high, low.
    Used as fallback price source if Yahoo and Massive both fail.
    """
    data = _rate_limited_get("/prices/snapshot", params={"ticker": ticker.upper()})
    if not data:
        return None

    snapshot = data.get("snapshot") or {}
    price = snapshot.get("price", 0) or 0

    if not price or price <= 0:
        return None

    logger.info(f"[FD] Price {ticker}: ${price:.2f}")
    return {
        "price": float(price),
        "open":  float(snapshot.get("open", price)),
        "high":  float(snapshot.get("high", price)),
        "low":   float(snapshot.get("low", price)),
        "volume": float(snapshot.get("volume", 0)),
        "source": "financial_datasets",
    }


# ── Historical Prices ─────────────────────────────────────────────────────────

def get_historical_prices(ticker: str, days: int = 30) -> Optional[List[Dict]]:
    """
    GET /prices?ticker={ticker}&start={start}&end={end}&interval=day
    Returns list of daily OHLCV bars sorted ascending.
    Used for volume ratio calculation if Yahoo/Massive unavailable.
    """
    today = date.today()
    start = (today - timedelta(days=days)).isoformat()
    end   = today.isoformat()

    data = _rate_limited_get("/prices", params={
        "ticker":   ticker.upper(),
        "start":    start,
        "end":      end,
        "interval": "day",
    })

    if not data:
        return None

    prices = data.get("prices", [])
    if not prices:
        return None

    logger.debug(f"[FD] Historical prices {ticker}: {len(prices)} bars")
    return prices


def get_volume_ratio_from_history(ticker: str) -> float:
    """
    Derive volume ratio from historical prices.
    Returns latest_volume / 30-day_avg_volume, or 1.0 if unavailable.
    """
    bars = get_historical_prices(ticker, days=32)
    if not bars or len(bars) < 2:
        return 1.0

    # Sort by date ascending — latest is last
    try:
        bars_sorted = sorted(bars, key=lambda b: b.get("date", ""))
    except Exception:
        bars_sorted = bars

    latest_vol = float(bars_sorted[-1].get("volume", 0) or 0)
    prior_vols = [float(b.get("volume", 0) or 0) for b in bars_sorted[:-1] if b.get("volume", 0)]
    avg_vol    = sum(prior_vols) / len(prior_vols) if prior_vols else latest_vol

    return round(latest_vol / avg_vol, 2) if avg_vol > 0 else 1.0


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_income_statements(ticker: str, period: str = "annual", limit: int = 3) -> Optional[List[Dict]]:
    """
    GET /financials/income-statements?ticker={ticker}&period={period}&limit={limit}
    period: 'annual' or 'quarterly'
    Returns revenue trend, earnings, profit margins for thesis enrichment.
    """
    data = _rate_limited_get("/financials/income-statements", params={
        "ticker": ticker.upper(),
        "period": period,
        "limit":  limit,
    })

    if not data:
        return None

    statements = data.get("income_statements", [])
    if not statements:
        return None

    logger.debug(f"[FD] Income statements {ticker}: {len(statements)} periods")
    return statements


def get_balance_sheets(ticker: str, period: str = "annual", limit: int = 2) -> Optional[List[Dict]]:
    """
    GET /financials/balance-sheets?ticker={ticker}&period={period}&limit={limit}
    Returns cash, debt, equity for squeeze context (debt load = squeeze risk amplifier).
    """
    data = _rate_limited_get("/financials/balance-sheets", params={
        "ticker": ticker.upper(),
        "period": period,
        "limit":  limit,
    })

    if not data:
        return None

    sheets = data.get("balance_sheets", [])
    if not sheets:
        return None

    logger.debug(f"[FD] Balance sheets {ticker}: {len(sheets)} periods")
    return sheets


def get_cash_flow_statements(ticker: str, period: str = "annual", limit: int = 2) -> Optional[List[Dict]]:
    """
    GET /financials/cash-flow-statements?ticker={ticker}&period={period}&limit={limit}
    Returns operating cash flow, free cash flow.
    Negative FCF + high SI = classic squeeze setup (company burning cash).
    """
    data = _rate_limited_get("/financials/cash-flow-statements", params={
        "ticker": ticker.upper(),
        "period": period,
        "limit":  limit,
    })

    if not data:
        return None

    statements = data.get("cash_flow_statements", [])
    if not statements:
        return None

    logger.debug(f"[FD] Cash flow statements {ticker}: {len(statements)} periods")
    return statements


# ── Company News ──────────────────────────────────────────────────────────────

def get_company_news(ticker: str, limit: int = 10) -> Optional[List[Dict]]:
    """
    GET /news?ticker={ticker}&limit={limit}
    Returns recent news headlines and summaries.
    Supplemental to Tavily — provides structured catalyst feed.
    """
    data = _rate_limited_get("/news", params={
        "ticker": ticker.upper(),
        "limit":  limit,
    })

    if not data:
        return None

    news = data.get("news", [])
    if not news:
        return None

    logger.debug(f"[FD] Company news {ticker}: {len(news)} articles")
    return news


# ── Composite Fundamentals Summary ───────────────────────────────────────────

def get_fundamentals_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Pull income statement + balance sheet + cash flow in sequence.
    Returns a flat dict suitable for injecting into the thesis prompt.
    This is the main entry point called by nodes.py enrich stage.

    Returns None if FINANCIAL_DATASETS_API_KEY is not set.
    Returns partial data if some endpoints fail (graceful degradation).
    """
    if not _get_api_key():
        return None

    ticker = ticker.upper()
    result: Dict[str, Any] = {"ticker": ticker, "source": "financial_datasets"}

    # Income statement — revenue trend, net income
    income = get_income_statements(ticker, period="annual", limit=3)
    if income:
        latest = income[0]
        result["revenue"]        = latest.get("revenue", 0) or 0
        result["net_income"]     = latest.get("net_income", 0) or 0
        result["gross_profit"]   = latest.get("gross_profit", 0) or 0
        result["operating_income"] = latest.get("operating_income", 0) or 0
        result["period_end"]     = latest.get("period_of_report_date", "")

        # Revenue trend: compare latest vs prior year
        if len(income) >= 2:
            prior_rev = income[1].get("revenue", 0) or 0
            cur_rev   = result["revenue"]
            if prior_rev > 0:
                rev_change = (cur_rev - prior_rev) / prior_rev * 100
                result["revenue_trend_pct"] = round(rev_change, 1)
                result["revenue_trend"]     = (
                    "GROWING" if rev_change > 5
                    else "DECLINING" if rev_change < -5
                    else "FLAT"
                )
            else:
                result["revenue_trend_pct"] = 0
                result["revenue_trend"]     = "FLAT"

    # Balance sheet — cash vs debt
    balance = get_balance_sheets(ticker, period="annual", limit=1)
    if balance:
        sheet = balance[0]
        cash      = sheet.get("cash_and_equivalents", 0) or 0
        total_debt = (sheet.get("long_term_debt", 0) or 0) + (sheet.get("short_term_debt", 0) or 0)
        total_equity = sheet.get("total_equity", 0) or 0
        result["cash"]         = cash
        result["total_debt"]   = total_debt
        result["total_equity"] = total_equity
        result["debt_to_equity"] = round(total_debt / total_equity, 2) if total_equity > 0 else None
        # High debt + squeeze setup = potential forced-cover catalyst
        result["debt_load"] = (
            "HIGH"   if total_debt > 0 and (total_equity <= 0 or total_debt / max(total_equity, 1) > 2)
            else "MODERATE" if total_debt > 0
            else "LOW"
        )

    # Cash flow — free cash flow
    cashflow = get_cash_flow_statements(ticker, period="annual", limit=1)
    if cashflow:
        cf = cashflow[0]
        operating_cf = cf.get("operating_cash_flow", 0) or 0
        capex        = cf.get("capital_expenditure", 0) or 0
        free_cf      = operating_cf - abs(capex)
        result["operating_cash_flow"] = operating_cf
        result["free_cash_flow"]      = free_cf
        result["cash_burn"] = "BURNING" if free_cf < 0 else "GENERATING"

    if len(result) <= 2:  # only ticker + source
        logger.debug(f"[FD] No fundamentals data for {ticker}")
        return None

    logger.info(
        f"[FD] Fundamentals {ticker}: "
        f"Rev={result.get('revenue', 0)/1e6:.0f}M "
        f"trend={result.get('revenue_trend', 'N/A')} "
        f"debt={result.get('debt_load', 'N/A')} "
        f"cash={result.get('cash_burn', 'N/A')}"
    )
    return result


def format_fundamentals_for_thesis(fundamentals: Dict[str, Any]) -> str:
    """
    Format the fundamentals summary dict into a human-readable
    string block for injection into the thesis prompt.
    """
    if not fundamentals:
        return ""

    ticker = fundamentals.get("ticker", "")
    lines  = [f"FINANCIAL DATASETS FUNDAMENTALS — {ticker}:"]

    rev = fundamentals.get("revenue", 0)
    if rev:
        lines.append(f"  Revenue: ${rev/1e6:.1f}M  ({fundamentals.get('revenue_trend', 'N/A')}, {fundamentals.get('revenue_trend_pct', 0):+.1f}%)")

    ni = fundamentals.get("net_income", 0)
    if ni is not None:
        lines.append(f"  Net Income: ${ni/1e6:.1f}M")

    fcf = fundamentals.get("free_cash_flow")
    if fcf is not None:
        lines.append(f"  Free Cash Flow: ${fcf/1e6:.1f}M  ({fundamentals.get('cash_burn', 'N/A')})")

    cash = fundamentals.get("cash", 0)
    debt = fundamentals.get("total_debt", 0)
    if cash or debt:
        lines.append(f"  Cash: ${cash/1e6:.1f}M  |  Total Debt: ${debt/1e6:.1f}M  ({fundamentals.get('debt_load', 'N/A')})")

    dte = fundamentals.get("debt_to_equity")
    if dte is not None:
        lines.append(f"  Debt/Equity: {dte:.2f}x")

    period = fundamentals.get("period_end", "")
    if period:
        lines.append(f"  Period: {period}")

    return "\n".join(lines)
