"""
TRaNKSP — ReAct Agent Tools (5 tools)
"""

import os
import json
import logging
from typing import Optional
from langchain_core.tools import tool
from .massive_client import get_price_and_volume
from .yahoo_quote import get_quote_data as _yq_get

logger = logging.getLogger("tranksp.tools")

# ── Tool: search_news ─────────────────────────────────────────────────────────

@tool
def search_news(query: str) -> str:
    """Search for recent news about a stock ticker or topic. 
    Use for: recent catalysts, earnings, analyst upgrades, regulatory news.
    Input: natural language query like 'GME short squeeze news 2024'
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        results = client.search(query, max_results=5, search_depth="basic")
        
        if not results or "results" not in results:
            return "No news results found."
        
        summaries = []
        for r in results["results"][:5]:
            title = r.get("title", "No title")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            summaries.append(f"• {title}\n  {content}\n  Source: {url}")
        
        return "\n\n".join(summaries)
    except Exception as e:
        logger.error(f"search_news error: {e}")
        return f"News search failed: {str(e)}"


# ── Tool: get_sec_filings ─────────────────────────────────────────────────────

@tool
def get_sec_filings(ticker: str) -> str:
    """Retrieve recent SEC 8-K filings for a ticker. 
    Use to find material events: earnings, M&A, executive changes, regulatory actions.
    Input: ticker symbol like 'GME'
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        query = f"SEC 8-K filing {ticker} site:sec.gov OR site:secedgar.gov"
        results = client.search(query, max_results=3, search_depth="basic")
        
        if not results or "results" not in results:
            return f"No recent SEC filings found for {ticker}."
        
        summaries = []
        for r in results["results"][:3]:
            title = r.get("title", "")
            content = r.get("content", "")[:400]
            summaries.append(f"• {title}\n  {content}")
        
        return "\n\n".join(summaries) if summaries else f"No SEC filings found for {ticker}."
    except Exception as e:
        logger.error(f"get_sec_filings error for {ticker}: {e}")
        return f"SEC filing lookup failed: {str(e)}"


# ── Tool: get_short_data ──────────────────────────────────────────────────────

@tool
def get_short_data(ticker: str) -> str:
    """Get current short interest data for a ticker from yfinance.
    Returns: short float %, shares short, days to cover, institutional ownership.
    Input: ticker symbol like 'GME'
    """
    try:
        si = _yq_get(ticker)
        snap = get_price_and_volume(ticker)
        info = si or {}
        
        short_float = info.get("shortPercentOfFloat", 0)
        if short_float and short_float < 1:
            short_float = short_float * 100  # convert decimal to %
        
        shares_short = info.get("sharesShort", 0)
        days_to_cover = info.get("shortRatio", 0)
        float_shares = info.get("floatShares", 0)
        inst_ownership = info.get("heldPercentInstitutions", 0)
        if inst_ownership and inst_ownership < 1:
            inst_ownership = inst_ownership * 100

        return (
            f"Short Data for {ticker}:\n"
            f"  Short Float: {short_float:.1f}%\n"
            f"  Shares Short: {shares_short:,}\n"
            f"  Days to Cover: {days_to_cover:.1f}\n"
            f"  Float Shares: {float_shares:,}\n"
            f"  Institutional Ownership: {inst_ownership:.1f}%"
        )
    except Exception as e:
        logger.error(f"get_short_data error for {ticker}: {e}")
        return f"Short data unavailable for {ticker}: {str(e)}"


# ── Tool: get_earnings_date ───────────────────────────────────────────────────

@tool
def get_earnings_date(ticker: str) -> str:
    """Get the next earnings date for a ticker.
    Earnings are major binary events that can trigger short squeezes.
    Input: ticker symbol like 'GME'
    """
    try:
        # Earnings date lookup via Massive is not on free tier
        # Return informational message
        return f"Earnings date lookup: use a financial calendar for {ticker} earnings schedule."
        if False:  # kept for structure
            stock = None
            cal = None
        
        if cal is not None and not cal.empty:
            # calendar can be a dict or DataFrame depending on yfinance version
            if hasattr(cal, 'to_dict'):
                cal_dict = cal.to_dict()
            else:
                cal_dict = cal
            
            earnings_date = cal_dict.get("Earnings Date", ["Unknown"])[0]
            return f"Next earnings date for {ticker}: {earnings_date}"
        
        # Fallback to earnings_dates
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            next_date = earnings.index[0]
            return f"Next earnings date for {ticker}: {next_date.strftime('%Y-%m-%d')}"
        
        return f"No upcoming earnings date found for {ticker}."
    except Exception as e:
        logger.error(f"get_earnings_date error for {ticker}: {e}")
        return f"Earnings date lookup failed for {ticker}: {str(e)}"


# ── Tool: search_competitors ──────────────────────────────────────────────────

@tool
def search_competitors(ticker: str) -> str:
    """Find competitors and sector peers for a ticker that might be co-moving.
    Useful for identifying sector-wide squeeze momentum or relative value.
    Input: ticker symbol like 'GME'
    """
    try:
        si = _yq_get(ticker)
        snap = get_price_and_volume(ticker)
        info = si or {}
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        query = f"{ticker} {industry} competitors short squeeze peers 2024"
        results = client.search(query, max_results=3, search_depth="basic")
        
        context = f"Sector: {sector}\nIndustry: {industry}\n\n"
        if results and "results" in results:
            for r in results["results"][:3]:
                context += f"• {r.get('title', '')}\n  {r.get('content', '')[:200]}\n"
        
        return context
    except Exception as e:
        logger.error(f"search_competitors error for {ticker}: {e}")
        return f"Competitor search failed for {ticker}: {str(e)}"


# All tools list for agent
ALL_TOOLS = [search_news, get_sec_filings, get_short_data, get_earnings_date, search_competitors]
