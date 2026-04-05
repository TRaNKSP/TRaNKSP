"""
TRaNKSP — Universe Builder
Pulls short squeeze candidates from live internet sources:
- Finviz high short interest screener
- Yahoo Finance most-shorted
- StockAnalysis short interest rankings
- Reddit r/WallStreetBets mentions
- FINRA short volume (via Tavily)
"""

import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup
from .yahoo_screener import scrape_most_shorted

logger = logging.getLogger("tranksp.universe")

# Known high-short-interest tickers used as seed + fallback
SEED_TICKERS = [
    # Meme/squeeze classics (still trading)
    "GME", "AMC", "MSTR", "COIN",
    # High SI fintech/EV
    "RIVN", "LCID", "NKLA", "WKHS", "SOFI", "HOOD", "OPEN", "UPST",
    # High SI growth/speculative
    "PLTR", "DKNG", "AFRM", "BYND", "CVNA", "SPCE",
    # Small/mid high SI
    "PRTY", "MVIS", "ATER", "PROG", "GNUS", "XELA", "IMPP",
    # Crypto-adjacent high SI
    "MARA", "RIOT", "HUT", "CLSK",
]


async def fetch_finviz_high_short() -> List[Dict[str, Any]]:
    """Scrape Finviz screener for high short interest stocks."""
    candidates = []
    url = "https://finviz.com/screener.ashx?v=111&f=sh_short_o30&ft=4&o=-short"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(f"Finviz returned {resp.status}")
                    return []
                html = await resp.text()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Finviz ticker links
        ticker_cells = soup.find_all("a", class_="screener-link-primary")
        for cell in ticker_cells[:30]:
            ticker = cell.text.strip()
            if ticker and re.match(r'^[A-Z]{1,5}$', ticker):
                candidates.append({"ticker": ticker, "source": "finviz", "reason": "High short interest screener"})
        
        logger.info(f"[Universe] Finviz: {len(candidates)} tickers")
    except Exception as e:
        logger.warning(f"[Universe] Finviz error: {e}")
    
    return candidates


async def fetch_wsb_mentions() -> List[Dict[str, Any]]:
    """Fetch trending tickers from Reddit WSB (public JSON endpoint)."""
    candidates = []
    url = "https://www.reddit.com/r/wallstreetbets/hot.json?limit=25"
    
    try:
        headers = {"User-Agent": "TRaNKSP/1.0 (research tool)"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        
        posts = data.get("data", {}).get("children", [])
        ticker_pattern = re.compile(r'\b([A-Z]{2,5})\b')
        
        # Common false positives to exclude
        exclude = {"DD", "EV", "THE", "AI", "FDA", "SEC", "CEO", "OTC", "FOR", "ETF", 
                   "IPO", "IMO", "WSB", "YOLO", "GG", "RIP", "ATH", "WTF", "EOD"}
        
        ticker_counts: Dict[str, int] = {}
        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "")
            matches = ticker_pattern.findall(title)
            for m in matches:
                if m not in exclude and len(m) >= 2:
                    ticker_counts[m] = ticker_counts.get(m, 0) + 1
        
        # Top mentioned tickers
        top = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for ticker, count in top:
            candidates.append({
                "ticker": ticker,
                "source": "reddit_wsb",
                "reason": f"WSB mentions: {count} posts in hot"
            })
        
        logger.info(f"[Universe] Reddit WSB: {len(candidates)} tickers")
    except Exception as e:
        logger.warning(f"[Universe] Reddit WSB error: {e}")
    
    return candidates


async def fetch_yahoo_most_shorted() -> List[Dict[str, Any]]:
    """Scrape Yahoo Finance most-shorted stocks page."""
    candidates = []
    url = "https://finance.yahoo.com/screener/predefined/most_shorted_stocks"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Find ticker links in Yahoo's table
        links = soup.find_all("a", {"data-symbol": True})
        seen = set()
        for link in links[:25]:
            ticker = link.get("data-symbol", "").strip()
            if ticker and ticker not in seen and re.match(r'^[A-Z]{1,5}$', ticker):
                candidates.append({"ticker": ticker, "source": "yahoo_most_shorted", "reason": "Yahoo most-shorted list"})
                seen.add(ticker)
        
        logger.info(f"[Universe] Yahoo: {len(candidates)} tickers")
    except Exception as e:
        logger.warning(f"[Universe] Yahoo error: {e}")
    
    return candidates


async def fetch_stockanalysis_short() -> List[Dict[str, Any]]:
    """Fetch short interest rankings from StockAnalysis."""
    candidates = []
    url = "https://stockanalysis.com/stocks/screener/?f=marketCap,shortFloat,shortRatio&o=shortFloat&ob=d&p=1"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()
        
        soup = BeautifulSoup(html, "html.parser")
        ticker_links = soup.select("td a[href*='/stocks/']")
        
        seen = set()
        for link in ticker_links[:20]:
            href = link.get("href", "")
            match = re.search(r'/stocks/([A-Z]{1,5})/', href)
            if match:
                ticker = match.group(1)
                if ticker not in seen:
                    candidates.append({
                        "ticker": ticker,
                        "source": "stockanalysis",
                        "reason": "StockAnalysis high short float ranking"
                    })
                    seen.add(ticker)
        
        logger.info(f"[Universe] StockAnalysis: {len(candidates)} tickers")
    except Exception as e:
        logger.warning(f"[Universe] StockAnalysis error: {e}")
    
    return candidates


async def build_universe(existing_tickers: List[str] = None) -> List[Dict[str, Any]]:
    """
    Pull candidates from all sources, deduplicate, and return.
    Primary source: Yahoo most-shorted screener (structured SI data).
    Secondary sources: WSB mentions, StockAnalysis, seed tickers.
    existing_tickers: already in the universe (skip re-adding)
    """
    existing = set(existing_tickers or [])

    logger.info("[Universe] Starting live universe build from all sources...")

    # Primary: Yahoo most-shorted screener — single clean request with SI data
    yahoo_screener_data = await asyncio.to_thread(scrape_most_shorted)
    yahoo_candidates = []
    for row in yahoo_screener_data:
        ticker = row.get("ticker", "")
        if ticker and ticker not in existing:
            yahoo_candidates.append({
                "ticker": ticker,
                "source": "yahoo_most_shorted",
                "reason": f"Yahoo most-shorted screener: SI={row.get('short_float',0):.1f}%"
            })
    logger.info(f"[Universe] Yahoo screener: {len(yahoo_candidates)} tickers")

    # Secondary: other web sources in parallel
    results = await asyncio.gather(
        fetch_wsb_mentions(),
        fetch_stockanalysis_short(),
        return_exceptions=True
    )
    
    all_candidates: Dict[str, Dict[str, Any]] = {}

    # Add Yahoo screener results first (highest quality)
    for item in yahoo_candidates:
        ticker = item["ticker"]
        if ticker not in all_candidates:
            all_candidates[ticker] = item

    for result in results:
        if isinstance(result, list):
            for item in result:
                ticker = item.get("ticker", "")
                if ticker and ticker not in existing:
                    if ticker not in all_candidates:
                        all_candidates[ticker] = item
                    else:
                        all_candidates[ticker]["source"] += f"+{item['source']}"
    
    # Add seed tickers as fallback if universe is thin
    if len(all_candidates) < 10:
        logger.warning("[Universe] Thin results from scrapers, adding seed tickers")
        for ticker in SEED_TICKERS:
            if ticker not in existing and ticker not in all_candidates:
                all_candidates[ticker] = {"ticker": ticker, "source": "seed", "reason": "Known squeeze candidate"}
    
    final = list(all_candidates.values())
    logger.info(f"[Universe] Built universe: {len(final)} new candidates")
    return final
