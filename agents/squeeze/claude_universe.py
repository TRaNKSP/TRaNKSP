"""
TRaNKSP — Claude-Powered Universe Builder

Asks Claude (via Anthropic API) to identify the most heavily shorted
stocks right now, based on its knowledge of current market conditions,
FINRA short interest data, and known squeeze candidates.

Claude returns a structured JSON list of tickers with:
  - ticker symbol
  - estimated short float %
  - reason why it's a squeeze candidate
  - catalyst type
  - confidence

This replaces the fragile Yahoo/MarketBeat scraping for universe building.
We still validate/enrich each ticker with Yahoo quoteSummary for live prices.

Model: claude-haiku-4-5 (fast + cheap — this is just a list generation call)
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import date

logger = logging.getLogger("tranksp.claude_universe")


UNIVERSE_PROMPT = """You are a short squeeze intelligence analyst with deep knowledge of current US equity markets.

Today's date: {today}

Your task: Identify the {count} most compelling short squeeze candidates in the US market RIGHT NOW.

Focus on stocks with:
- High short interest (>15% of float ideally >25%)
- High days-to-cover (>3, ideally >5)  
- Small to mid float (under 50M shares ideally)
- Recent or upcoming catalysts (earnings, FDA decisions, contracts, news)
- Active retail/WSB attention OR institutional momentum

Include a mix of:
- Classic meme squeeze candidates (GME, AMC type setups if relevant)
- Biotech/small-cap with high SI + upcoming binary events
- Any sector where shorts are currently very exposed

Respond ONLY with a JSON array. No preamble, no explanation, no markdown fences.
Each item must have exactly these fields:
{{
  "ticker": "SYMBOL",
  "est_short_float": 35.2,
  "est_days_to_cover": 4.5,
  "float_size_m": 12.5,
  "catalyst": "brief catalyst description",
  "catalyst_type": "earnings|fda|momentum|technical|news|squeeze_history",
  "squeeze_reason": "one sentence why this is a squeeze candidate",
  "confidence": 72
}}

Return exactly {count} tickers as a JSON array."""


async def get_claude_universe(count: int = 30) -> List[Dict[str, Any]]:
    """
    Ask Claude to identify today's top short squeeze candidates.
    Returns list of dicts with ticker + context data.
    
    Uses claude-haiku for speed and cost efficiency.
    Falls back to empty list if API call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("[ClaudeUniverse] No ANTHROPIC_API_KEY — cannot build Claude universe")
        return []

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        today = date.today().strftime("%B %d, %Y")

        logger.info(f"[ClaudeUniverse] Asking Claude for top {count} short squeeze candidates...")

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": UNIVERSE_PROMPT.format(count=count, today=today)
            }]
        )

        raw = message.content[0].text.strip()

        # Strip any accidental markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for p in parts:
                p = p.strip().lstrip("json").strip()
                if p.startswith("["):
                    raw = p
                    break

        # Find JSON array
        start = raw.find("[")
        end   = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        candidates = json.loads(raw)

        # Validate and normalize
        result = []
        for item in candidates:
            ticker = item.get("ticker", "").upper().strip()
            if not ticker or len(ticker) > 6:
                continue
            result.append({
                "ticker":          ticker,
                "est_short_float": float(item.get("est_short_float", 0)),
                "est_dtc":         float(item.get("est_days_to_cover", 0)),
                "float_size_m":    float(item.get("float_size_m", 0)),
                "catalyst":        str(item.get("catalyst", "")),
                "catalyst_type":   str(item.get("catalyst_type", "momentum")),
                "squeeze_reason":  str(item.get("squeeze_reason", "")),
                "confidence":      float(item.get("confidence", 50)),
                "source":          "claude_universe",
            })

        logger.info(f"[ClaudeUniverse] Claude returned {len(result)} candidates: "
                    f"{', '.join(r['ticker'] for r in result[:10])}{'...' if len(result)>10 else ''}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"[ClaudeUniverse] JSON parse error: {e}\nRaw: {raw[:500]}")
        return []
    except Exception as e:
        logger.error(f"[ClaudeUniverse] API error: {e}")
        return []


def get_tickers_only(candidates: List[Dict]) -> List[str]:
    """Extract just ticker symbols from Claude universe result."""
    return [c["ticker"] for c in candidates]
