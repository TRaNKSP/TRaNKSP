"""
TRaNKSP — Yahoo Finance Most Shorted Stocks Scraper

The new /research-hub/screener/ URL is a React SPA — it returns 404/empty
on direct HTTP fetch because the data is loaded client-side via XHR.

The correct approach is to call Yahoo's internal screener query API directly,
which is what the browser's XHR calls when you load that page.

Primary API endpoint (JSON, paginated):
  GET https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved
    ?formatted=false
    &scrIds=most_shorted_stocks
    &count=100
    &start=0

This returns clean JSON with full quote data including:
  shortPercentOfFloat, shortRatio (DTC), regularMarketPrice,
  regularMarketVolume, averageDailyVolume3Month, marketCap, floatShares,
  sharesShort, sharesShortPriorMonth

Fallback: older SSR screener page
  https://finance.yahoo.com/screener/predefined/most_shorted_stocks?offset=0&count=100

Rate limits: 5s between pages, cookie warm-up, random UA rotation.
"""

import re
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("tranksp.yahoo_screener")

# Primary: Yahoo internal screener API (JSON response, no JS needed)
QUERY_API = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
QUERY_API_V2 = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"

# Fallback: older SSR screener page
SSR_URL = "https://finance.yahoo.com/screener/predefined/most_shorted_stocks"

# Cookie warm-up URL
YAHOO_HOME = "https://finance.yahoo.com"

PAGE_SIZE = 100
PAGE_DELAY = 5.0
MAX_PAGES_DEFAULT = 5

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

def _browser_headers(referer: str = "") -> dict:
    h = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Origin": "https://finance.yahoo.com",
        "Referer": referer or "https://finance.yahoo.com/",
    }
    return h


def _safe_float(val, mult: float = 1.0) -> float:
    if isinstance(val, dict):
        val = val.get("raw", val.get("fmt", 0))
    try:
        return float(val) * mult
    except (TypeError, ValueError):
        return 0.0


def _parse_quote(q: dict) -> Optional[Dict[str, Any]]:
    """Parse a single quote from Yahoo screener JSON into TRaNKSP format."""
    ticker = q.get("symbol", "")
    if not ticker or not re.match(r'^[A-Z]{1,6}$', ticker):
        return None

    price = _safe_float(q.get("regularMarketPrice"))
    if price <= 0:
        return None

    volume     = _safe_float(q.get("regularMarketVolume"))
    avg_vol    = _safe_float(q.get("averageDailyVolume3Month")) or \
                 _safe_float(q.get("averageDailyVolume10Day")) or 1
    vol_ratio  = round(volume / avg_vol, 2) if avg_vol > 0 else 1.0

    short_pct  = _safe_float(q.get("shortPercentOfFloat"))
    if 0 < short_pct < 1:
        short_pct *= 100   # decimal → percent

    short_ratio    = _safe_float(q.get("shortRatio"))      # days to cover
    float_shares   = _safe_float(q.get("floatShares"),    1/1_000_000)
    market_cap     = _safe_float(q.get("marketCap"),      1/1_000_000)
    shares_short   = _safe_float(q.get("sharesShort"))
    shares_prior   = _safe_float(q.get("sharesShortPriorMonth"))

    si_trend = "FLAT"
    if shares_prior > 0 and shares_short > 0:
        delta = (shares_short - shares_prior) / shares_prior
        si_trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else "FLAT"

    return {
        "ticker":        ticker,
        "price":         round(price, 2),
        "volume_ratio":  vol_ratio,
        "short_float":   round(short_pct, 2),
        "days_to_cover": round(short_ratio, 2),
        "float_shares":  round(float_shares, 2),
        "market_cap":    round(market_cap, 2),
        "si_trend":      si_trend,
        "shares_short":  shares_short,
        "source":        "yahoo_most_shorted",
    }


def _fetch_query_api(session, start: int, count: int) -> List[Dict]:
    """
    Fetch one page from Yahoo's internal screener query API.
    Returns list of parsed ticker dicts.
    Tries query1 then query2 as fallback.
    """
    params = {
        "formatted":  "false",
        "scrIds":     "most_shorted_stocks",
        "count":      count,
        "start":      start,
    }
    referer = f"https://finance.yahoo.com/screener/predefined/most_shorted_stocks?offset={start}&count={count}"

    for base_url in [QUERY_API, QUERY_API_V2]:
        try:
            resp = session.get(
                base_url,
                params=params,
                headers=_browser_headers(referer),
                timeout=20,
            )
            if resp.status_code == 429:
                return []  # caller handles wait
            if resp.status_code != 200:
                logger.debug(f"[YahooScreener] Query API {base_url[-7:]}: {resp.status_code}")
                continue

            data = resp.json()
            # Navigate: finance.screener.result[0].quotes
            quotes = (data.get("finance", {})
                          .get("result", [{}])[0]
                          .get("quotes", []))

            if not quotes:
                # Try alternate path
                quotes = (data.get("finance", {})
                              .get("result", [{}])[0]
                              .get("rows", []))

            results = [_parse_quote(q) for q in quotes]
            results = [r for r in results if r is not None]
            logger.info(f"[YahooScreener] Query API page start={start}: {len(results)} tickers")
            return results

        except Exception as e:
            logger.warning(f"[YahooScreener] Query API error (start={start}): {e}")

    return []


def _fetch_ssr_page(session, offset: int, count: int) -> List[Dict]:
    """
    Fallback: fetch the older SSR screener page and parse HTML/JSON.
    """
    params = {"offset": offset, "count": count}
    referer = "https://finance.yahoo.com/"

    try:
        resp = session.get(
            SSR_URL,
            params=params,
            headers=_browser_headers(referer),
            timeout=20,
        )
        if resp.status_code != 200 or len(resp.text) < 5000:
            return []

        html = resp.text

        # Try embedded JSON first
        match = re.search(
            r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
            html, re.DOTALL
        )
        if match:
            try:
                data = json.loads(match.group(1))
                results = []

                def _walk(obj, depth=0):
                    if depth > 12 or not obj:
                        return
                    if isinstance(obj, dict):
                        if "symbol" in obj and "regularMarketPrice" in obj:
                            parsed = _parse_quote(obj)
                            if parsed:
                                results.append(parsed)
                            return
                        for v in obj.values():
                            _walk(v, depth + 1)
                    elif isinstance(obj, list):
                        for item in obj:
                            _walk(item, depth + 1)

                _walk(data)
                if results:
                    logger.info(f"[YahooScreener] SSR JSON page offset={offset}: {len(results)} tickers")
                    return results
            except Exception:
                pass

        # HTML table fallback
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            link = cells[0].find("a")
            if not link:
                continue
            ticker = link.get_text(strip=True)
            if not re.match(r'^[A-Z]{1,6}$', ticker):
                continue

            def cv(cell):
                s = cell.find("fin-streamer")
                raw = (s.get("data-value") if s else None) or cell.get_text(strip=True)
                raw = re.sub(r'[,%\s]', '', str(raw))
                try:
                    if raw.upper().endswith('B'): return float(raw[:-1]) * 1000
                    if raw.upper().endswith('M'): return float(raw[:-1])
                    return float(raw)
                except: return 0.0

            vals = [cv(c) for c in cells]
            price = vals[2] if len(vals) > 2 else 0
            if price <= 0:
                continue
            short_pct = next((v for v in reversed(vals[4:]) if 1 <= v <= 100), 0)
            results.append({
                "ticker":       ticker,
                "price":        round(price, 2),
                "volume_ratio": 1.0,
                "short_float":  round(short_pct, 2),
                "days_to_cover": 0,
                "float_shares": 0,
                "market_cap":   vals[7] if len(vals) > 7 else 0,
                "si_trend":     "FLAT",
                "source":       "yahoo_most_shorted_html",
            })
        return results

    except Exception as e:
        logger.warning(f"[YahooScreener] SSR page error (offset={offset}): {e}")
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

def scrape_most_shorted(
    max_pages: int = MAX_PAGES_DEFAULT,
    page_delay: float = PAGE_DELAY,
    min_short_pct: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Scrape Yahoo Finance most-shorted stocks across multiple pages.

    Args:
        max_pages:     Pages to fetch (100 tickers each). Default 5 = 500 tickers.
                       Set 40 to get all ~3,895 records (~4 min at 5s/page).
        page_delay:    Seconds between page requests. Default 5s (polite crawl).
        min_short_pct: Filter to only return tickers with SI >= this %.

    Returns:
        List of dicts with ticker, price, short_float, days_to_cover,
        volume_ratio, float_shares, market_cap, si_trend, source.
    """
    import requests

    session = requests.Session()
    all_results: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # ── Warm up session with cookies ─────────────────────────────────────────
    logger.info("[YahooScreener] Warming up session...")
    try:
        session.get(YAHOO_HOME, headers=_browser_headers(), timeout=10)
        time.sleep(random.uniform(1.5, 2.5))
        # Also visit the screener page to get the right referer cookies
        session.get(
            "https://finance.yahoo.com/screener/predefined/most_shorted_stocks",
            headers=_browser_headers(),
            timeout=10
        )
        time.sleep(random.uniform(1.0, 2.0))
    except Exception as e:
        logger.warning(f"[YahooScreener] Warm-up failed: {e}")

    # ── Paginate ─────────────────────────────────────────────────────────────
    for page_num in range(max_pages):
        start = page_num * PAGE_SIZE

        logger.info(
            f"[YahooScreener] Page {page_num+1}/{max_pages} "
            f"(records {start+1}–{start+PAGE_SIZE})..."
        )

        if page_num > 0:
            delay = page_delay + random.uniform(-1.0, 1.0)
            delay = max(3.0, delay)
            logger.debug(f"[YahooScreener] Waiting {delay:.1f}s")
            time.sleep(delay)

        # Try primary query API first, then SSR fallback
        page_rows = []
        for attempt in range(3):
            if attempt > 0:
                time.sleep(5 * attempt)

            # Primary: query API
            page_rows = _fetch_query_api(session, start, PAGE_SIZE)

            if not page_rows:
                # Fallback: SSR page
                logger.debug(f"[YahooScreener] Query API empty, trying SSR fallback...")
                page_rows = _fetch_ssr_page(session, start, PAGE_SIZE)

            if page_rows:
                break

            logger.warning(f"[YahooScreener] Page {page_num+1}: no data (attempt {attempt+1})")

        # Deduplicate and filter
        new_count = 0
        for row in page_rows:
            t = row.get("ticker", "")
            if not t or t in seen:
                continue
            if min_short_pct > 0 and row.get("short_float", 0) < min_short_pct:
                continue
            seen.add(t)
            all_results.append(row)
            new_count += 1

        logger.info(
            f"[YahooScreener] Page {page_num+1}: {new_count} new tickers "
            f"(total: {len(all_results)})"
        )

        if not page_rows:
            logger.info(f"[YahooScreener] Stopping — empty page at {page_num+1}")
            break

    logger.info(f"[YahooScreener] ✓ Done: {len(all_results)} tickers from {page_num+1} pages")
    return all_results


def get_most_shorted_tickers(max_pages: int = MAX_PAGES_DEFAULT) -> List[str]:
    return [d["ticker"] for d in scrape_most_shorted(max_pages=max_pages)]
