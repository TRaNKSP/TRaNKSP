"""
TRaNKSP — MarketBeat Short Interest Scraper

Scrapes: https://www.marketbeat.com/stocks/{EXCHANGE}/{TICKER}/short-interest/

Fields extracted:
  - Short Percent of Float   → short_float
  - Short Interest Ratio     → days_to_cover  (labeled "X Days to Cover")
  - Current Short Interest   → shares_short
  - Previous Short Interest  → for si_trend
  - Outstanding Shares       → float_shares (used as proxy)
  - Today's Volume Vs. Average → volume_ratio

URL pattern:
  NASDAQ stocks: /stocks/NASDAQ/{TICKER}/short-interest/
  NYSE stocks:   /stocks/NYSE/{TICKER}/short-interest/
  We try NASDAQ first, then NYSE, then AMEX/OTC as fallbacks.

Rate limiting:
  2-3 second delay per request (polite, not per-ticker in bulk).
  MarketBeat is a public data site — single page load per ticker.

No API key needed. Pure HTML scraping.
"""

import re
import time
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("tranksp.marketbeat")

BASE_URL = "https://www.marketbeat.com/stocks/{exchange}/{ticker}/short-interest/"
EXCHANGES = ["NASDAQ", "NYSE", "AMEX", "OTC", "NYSEARCA"]

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

# Cache exchange per ticker to avoid repeated 404s
_exchange_cache: Dict[str, str] = {}

_last_call: float = 0.0
_DELAY = 2.5  # seconds between requests


def _headers(referer: str = "https://www.google.com/") -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": referer,
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
    }


def _wait():
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _DELAY:
        time.sleep(_DELAY - elapsed + random.uniform(0, 0.5))
    _last_call = time.time()


def _parse_number(text: str) -> float:
    """Parse numbers like '7,380,768', '14.02%', '$4.98 million', '8.6 Days to Cover'."""
    if not text:
        return 0.0
    text = text.strip()
    # Extract the numeric portion
    match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
    if not match:
        return 0.0
    try:
        val = float(match.group().replace(',', ''))
        # Handle millions/billions
        tl = text.lower()
        if 'billion' in tl or ' b' in tl:
            val *= 1_000_000_000
        elif 'million' in tl or ' m' in tl:
            val *= 1_000_000
        return val
    except (ValueError, TypeError):
        return 0.0


def _parse_page(html: str, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Parse MarketBeat short interest page HTML.
    
    Page structure (from screenshot):
      Table with rows like:
        | Current Short Interest    | 7,380,768 shares      |
        | Previous Short Interest   | 6,790,396 shares      |
        | Change Vs. Previous Month | 8.69%                 |
        | Short Interest Ratio      | 8.6 Days to Cover     |
        | Outstanding Shares        | 55,070,000 shares     |
        | Short Percent of Float    | 14.02%                |
        | Today's Trading Volume    | 2,137,597 shares      |
        | Average Trading Volume    | 1,595,547 shares      |
        | Today's Volume Vs. Average| 134%                  |
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Check we got valid content (not a block page)
        if len(html) < 3000 or ticker.upper() not in html.upper():
            return None

        # Build a label → value map from ALL table rows on the page
        data = {}
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                data[label] = value

        # Also scan definition-list style elements (MarketBeat sometimes uses these)
        for dt in soup.find_all(["dt", "th"]):
            label = dt.get_text(strip=True).lower()
            dd = dt.find_next_sibling(["dd", "td"])
            if dd:
                data[label] = dd.get_text(strip=True)

        # Scan for labeled spans / divs in stat blocks
        for el in soup.find_all(["div", "span"], class_=re.compile(r'label|title|key', re.I)):
            label = el.get_text(strip=True).lower()
            val_el = el.find_next_sibling() or el.parent.find_next_sibling()
            if val_el:
                data[label] = val_el.get_text(strip=True)

        if not data:
            logger.debug(f"[MarketBeat] No table data found for {ticker}")
            return None

        # Extract fields using fuzzy key matching
        def get(keys):
            for k in keys:
                for dk, dv in data.items():
                    if k in dk:
                        return dv
            return ""

        current_si_text  = get(["current short interest"])
        previous_si_text = get(["previous short interest"])
        short_pct_text   = get(["short percent of float", "percent of float"])
        si_ratio_text    = get(["short interest ratio", "days to cover"])
        outstanding_text = get(["outstanding shares", "shares outstanding"])
        avg_vol_text     = get(["average trading volume", "average volume"])
        cur_vol_text     = get(["today's trading volume", "current volume"])
        vol_vs_avg_text  = get(["today's volume vs. average", "volume vs"])

        # Parse values
        shares_short  = _parse_number(current_si_text)
        shares_prior  = _parse_number(previous_si_text)
        short_pct     = _parse_number(short_pct_text)   # already in %
        days_to_cover = _parse_number(si_ratio_text)
        outstanding   = _parse_number(outstanding_text)
        avg_vol       = _parse_number(avg_vol_text)
        cur_vol       = _parse_number(cur_vol_text)

        # Volume ratio
        vol_vs_avg = _parse_number(vol_vs_avg_text)
        if vol_vs_avg > 0:
            volume_ratio = round(vol_vs_avg / 100, 2)
        elif avg_vol > 0 and cur_vol > 0:
            volume_ratio = round(cur_vol / avg_vol, 2)
        else:
            volume_ratio = 1.0

        # Float shares in millions
        float_m = round(outstanding / 1_000_000, 2) if outstanding > 0 else 0

        # SI trend
        si_trend = "FLAT"
        if shares_prior > 0 and shares_short > 0:
            delta = (shares_short - shares_prior) / shares_prior
            si_trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else "FLAT"

        if short_pct == 0 and shares_short == 0:
            logger.debug(f"[MarketBeat] Parsed but all zeros for {ticker} — likely wrong page")
            return None

        logger.info(
            f"[MarketBeat] {ticker}: SI={short_pct:.1f}% "
            f"DTC={days_to_cover:.1f} shares={shares_short:,.0f} {si_trend}"
        )

        return {
            "short_float":    round(short_pct, 2),
            "days_to_cover":  round(days_to_cover, 2),
            "shares_short":   shares_short,
            "float_shares":   float_m,
            "volume_ratio":   volume_ratio,
            "si_trend":       si_trend,
            "source":         "marketbeat",
        }

    except Exception as e:
        logger.warning(f"[MarketBeat] Parse error for {ticker}: {e}")
        return None


def get_short_interest(ticker: str, exchange: str = None) -> Optional[Dict[str, Any]]:
    """
    Scrape MarketBeat short interest data for a ticker.
    
    Args:
        ticker:   Stock symbol e.g. 'CLDX'
        exchange: Optional exchange hint e.g. 'NASDAQ'. If None, tries all.
    
    Returns:
        Dict with short_float, days_to_cover, shares_short, float_shares,
        volume_ratio, si_trend, source — or None if not found.
    """
    import requests

    ticker = ticker.upper().strip()
    _wait()

    # Build exchange list: cached first, then hinted, then all
    exchanges_to_try = []
    if ticker in _exchange_cache:
        exchanges_to_try = [_exchange_cache[ticker]]
    elif exchange:
        exchanges_to_try = [exchange.upper()] + [e for e in EXCHANGES if e != exchange.upper()]
    else:
        exchanges_to_try = EXCHANGES

    session = requests.Session()

    for exch in exchanges_to_try:
        url = BASE_URL.format(exchange=exch, ticker=ticker)
        referer = f"https://www.marketbeat.com/stocks/{exch}/{ticker}/"

        try:
            resp = session.get(url, headers=_headers(referer), timeout=15)

            if resp.status_code == 404:
                logger.debug(f"[MarketBeat] 404 for {ticker} on {exch}")
                continue

            if resp.status_code == 429:
                logger.warning(f"[MarketBeat] 429 rate limit — waiting 30s")
                time.sleep(30)
                _last_call = time.time()
                resp = session.get(url, headers=_headers(referer), timeout=15)

            if resp.status_code == 200:
                result = _parse_page(resp.text, ticker)
                if result:
                    _exchange_cache[ticker] = exch  # cache for next time
                    return result
                else:
                    logger.debug(f"[MarketBeat] Page loaded for {ticker}/{exch} but no data parsed")

            else:
                logger.debug(f"[MarketBeat] HTTP {resp.status_code} for {ticker}/{exch}")

        except Exception as e:
            logger.debug(f"[MarketBeat] Error fetching {ticker}/{exch}: {e}")

        # Small gap between exchange attempts
        time.sleep(random.uniform(0.5, 1.0))

    logger.debug(f"[MarketBeat] No data found for {ticker} across all exchanges")
    return None
