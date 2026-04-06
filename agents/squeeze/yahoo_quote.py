"""
TRaNKSP — Yahoo Finance Quote Scraper

Uses Yahoo's internal quoteSummary API:
  GET https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}
      ?modules=defaultKeyStatistics,summaryDetail,price&crumb={crumb}

Key design:
  - Module-level singleton session — warms up ONCE, reused for all tickers
  - Crumb acquired once per session lifetime
  - 2s delay between calls to avoid 429s
  - If 429 hit: wait 30s then retry once, then give up gracefully
  - No re-warming per ticker — that was causing the 429 storm
"""

import re
import json
import time
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("tranksp.yahoo_quote")

SUMMARY_API  = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
SUMMARY_API2 = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
YAHOO_HOME   = "https://finance.yahoo.com"
CRUMB_URL1   = "https://query1.finance.yahoo.com/v1/test/getcrumb"
CRUMB_URL2   = "https://query2.finance.yahoo.com/v1/test/getcrumb"

CALL_DELAY   = 2.0   # seconds between calls

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def _safe_raw(val, mult: float = 1.0) -> float:
    if isinstance(val, dict):
        val = val.get("raw", val.get("fmt", 0))
    try:
        return float(val) * mult
    except (TypeError, ValueError):
        return 0.0


def _build_result(stats: dict, detail: dict, price_data: dict) -> Dict[str, Any]:
    short_pct   = _safe_raw(stats.get("shortPercentOfFloat"))
    if 0 < short_pct < 1:
        short_pct *= 100

    short_ratio = _safe_raw(stats.get("shortRatio"))
    float_shares= _safe_raw(stats.get("floatShares"), 1/1_000_000)
    market_cap  = _safe_raw(stats.get("marketCap") or detail.get("marketCap"), 1/1_000_000)
    shares_short      = _safe_raw(stats.get("sharesShort"))
    shares_short_prior= _safe_raw(stats.get("sharesShortPriorMonth"))

    cur_price  = _safe_raw(price_data.get("regularMarketPrice") or detail.get("regularMarketPrice"))
    cur_vol    = _safe_raw(price_data.get("regularMarketVolume") or detail.get("volume"))
    avg_vol    = _safe_raw(price_data.get("averageDailyVolume3Month") or detail.get("averageVolume")) or 1
    vol_ratio  = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    si_trend = "FLAT"
    if shares_short_prior > 0 and shares_short > 0:
        delta = (shares_short - shares_short_prior) / shares_short_prior
        si_trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else "FLAT"

    return {
        "short_float":   round(short_pct, 2),
        "days_to_cover": round(short_ratio, 2),
        "float_shares":  round(float_shares, 2),
        "market_cap":    round(market_cap, 2),
        "si_trend":      si_trend,
        "shares_short":  shares_short,
        "price":         round(cur_price, 2),
        "volume_ratio":  vol_ratio,
        "source":        "yahoo_quote",
    }


# ── Singleton session ─────────────────────────────────────────────────────────

class _YahooSession:
    """Module-level singleton. Warms once, stays warm."""

    def __init__(self):
        import requests
        self._session  = requests.Session()
        self._crumb    = None
        self._warmed   = False
        self._last_call= 0.0

    def _ua(self) -> str:
        return random.choice(_USER_AGENTS)

    def _base_headers(self) -> dict:
        return {
            "User-Agent":      self._ua(),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         YAHOO_HOME,
        }

    def warm_up(self):
        """Warm session + acquire crumb. Called once at startup or after session expiry."""
        if self._warmed:
            return
        logger.info("[YahooQuote] Warming session and acquiring crumb...")
        try:
            # Step 1: Homepage cookies
            self._session.get(
                YAHOO_HOME,
                headers={**self._base_headers(), "Accept": "text/html"},
                timeout=10
            )
            time.sleep(random.uniform(1.0, 2.0))

            # Step 2: SPY quote page for full cookie set
            spy_resp = self._session.get(
                "https://finance.yahoo.com/quote/SPY/",
                headers={**self._base_headers(), "Accept": "text/html"},
                timeout=10
            )
            time.sleep(random.uniform(0.8, 1.5))

            # Strategy A: Dedicated crumb endpoints (q2 then q1)
            for crumb_url in [CRUMB_URL2, CRUMB_URL1]:
                try:
                    r = self._session.get(
                        crumb_url,
                        headers={**self._base_headers(), "Accept": "*/*"},
                        timeout=8
                    )
                    if r.status_code == 200 and r.text and len(r.text.strip()) < 50:
                        self._crumb = r.text.strip()
                        logger.info(f"[YahooQuote] Crumb acquired via endpoint "
                                    f"({'q2' if 'query2' in crumb_url else 'q1'})")
                        break
                except Exception:
                    pass

            # Strategy B: Extract crumb from SPY page HTML (if endpoint failed)
            if not self._crumb and spy_resp and spy_resp.status_code == 200:
                import re as _re
                html = spy_resp.text
                # Yahoo embeds crumb in page as "CrumbStore":{"crumb":"XXXXXX"}
                for pattern in [
                    r'"CrumbStore":\{"crumb":"([^"]{8,20})"',
                    r'"crumb":"([^"]{8,20})"',
                    r'crumb=([A-Za-z0-9%._-]{8,20})',
                ]:
                    m = _re.search(pattern, html)
                    if m:
                        crumb_raw = m.group(1).replace(r'\u002F', '/')
                        if crumb_raw and len(crumb_raw) >= 8:
                            self._crumb = crumb_raw
                            logger.info(f"[YahooQuote] Crumb extracted from page HTML")
                            break

            # Strategy C: Try the v2 consent / finance.yahoo.com API directly
            if not self._crumb:
                try:
                    r3 = self._session.get(
                        "https://finance.yahoo.com/quote/SPY/?p=SPY",
                        headers={**self._base_headers(),
                                 "Accept": "text/html,application/xhtml+xml"},
                        timeout=10
                    )
                    import re as _re
                    for pattern in [r'"crumb":"([^"]{8,20})"', r'crumb=([A-Za-z0-9._-]{8,20})']:
                        m = _re.search(pattern, r3.text)
                        if m:
                            self._crumb = m.group(1).replace(r'\u002F', '/')
                            logger.info("[YahooQuote] Crumb extracted via Strategy C")
                            break
                except Exception:
                    pass

            if self._crumb:
                logger.info(f"[YahooQuote] Session ready — crumb: {self._crumb[:6]}***")
            else:
                logger.warning("[YahooQuote] No crumb acquired — will try crumb-free requests "
                               "(price data OK, SI fields may be limited)")

            self._warmed = True

        except Exception as e:
            logger.warning(f"[YahooQuote] Warm-up error: {e}")
            self._warmed = True  # mark done so we don't loop

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < CALL_DELAY:
            time.sleep(CALL_DELAY - elapsed)
        self._last_call = time.time()

    def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch quoteSummary for one ticker. Uses shared session + crumb."""
        self.warm_up()
        self._wait()

        ticker  = ticker.upper().strip()
        modules = "defaultKeyStatistics,summaryDetail,price"

        headers = {**self._base_headers(), "Accept": "application/json, */*"}

        # Try with crumb first, then without (Yahoo sometimes works crumb-free)
        param_sets = []
        if self._crumb:
            param_sets.append({"modules": modules, "formatted": "false", "crumb": self._crumb})
        param_sets.append({"modules": modules, "formatted": "false"})  # crumb-free fallback

        for base_url in [SUMMARY_API, SUMMARY_API2]:
            url = base_url.format(ticker=ticker)
            for params in param_sets:
                try:
                    resp = self._session.get(url, params=params, headers=headers, timeout=12)

                    if resp.status_code == 429:
                        logger.warning(f"[YahooQuote] 429 on {ticker} — waiting 30s")
                        time.sleep(30)
                        self._last_call = time.time()
                        resp = self._session.get(url, params=params, headers=headers, timeout=12)
                        if resp.status_code == 429:
                            logger.warning(f"[YahooQuote] {ticker}: 429 again — skipping")
                            return None

                    if resp.status_code == 401:
                        # Crumb expired — force re-warm on next call
                        logger.warning(f"[YahooQuote] 401 on {ticker} — crumb expired, will re-warm")
                        self._warmed = False
                        self._crumb  = None
                        break  # break param_sets loop, try next base_url

                    if resp.status_code != 200:
                        logger.debug(f"[YahooQuote] HTTP {resp.status_code} for {ticker}")
                        continue

                    data   = resp.json()
                    result = (data.get("quoteSummary", {}).get("result") or [{}])[0]
                    stats  = result.get("defaultKeyStatistics", {})
                    detail = result.get("summaryDetail", {})
                    price  = result.get("price", {})

                    if not (stats or detail):
                        continue

                    parsed = _build_result(stats, detail, price)
                    if parsed["price"] > 0 or parsed["short_float"] > 0:
                        logger.info(
                            f"[YahooQuote] {ticker}: "
                            f"SI={parsed['short_float']:.1f}% "
                            f"DTC={parsed['days_to_cover']:.1f} "
                            f"${parsed['price']:.2f} {parsed['si_trend']}"
                        )
                        return parsed

                except Exception as e:
                    logger.debug(f"[YahooQuote] Error for {ticker}: {e}")

        logger.debug(f"[YahooQuote] No data for {ticker}")
        return None

    def reset(self):
        """Force re-warm (e.g. after long pause or repeated 401s)."""
        self._warmed = False
        self._crumb  = None
        import requests
        self._session = requests.Session()


# ── Module-level singleton ────────────────────────────────────────────────────

_SESSION = _YahooSession()


def get_quote_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get quote data using the shared module-level session."""
    return _SESSION.get_quote(ticker)


def reset_session():
    """Force session re-warm."""
    _SESSION.reset()


# ── YahooQuoteSession class kept for backwards compatibility ──────────────────

class YahooQuoteSession:
    """
    Thin wrapper kept for API compatibility with nodes.py.
    Delegates to the module-level singleton so warm-up only happens once.
    """
    def __init__(self, delay: float = 2.0):
        pass  # singleton handles delay

    def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        return _SESSION.get_quote(ticker)
