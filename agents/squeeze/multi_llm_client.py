"""
TRaNKSP — Multi-LLM Client

Queries multiple LLM providers in parallel for universe building and thesis
generation. Aggregating across models makes the output more robust — a ticker
that multiple independent LLMs agree on is a stronger signal than one only
one model picks.

Supported providers:
  - Anthropic Claude  (claude-haiku-4-5-20251001)   → ANTHROPIC_API_KEY
  - OpenAI GPT-4o-mini                              → OPENAI_API_KEY
  - xAI Grok-2                                      → GROK_API_KEY
  - Google Gemini 1.5 Flash                         → GEMINI_API_KEY

If a key is missing the provider is skipped gracefully.
Results are deduplicated + ranked by mention count across models.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import date

logger = logging.getLogger("tranksp.multi_llm")

# ── Shared prompt template ─────────────────────────────────────────────────────

UNIVERSE_PROMPT = """You are a short squeeze intelligence analyst with deep knowledge of current US equity markets.

Today's date: {today}

Identify the top {count} most compelling short squeeze candidates in the US market RIGHT NOW.

Focus on:
- Short interest > 15% of float (ideally > 25%)
- Days to cover > 3 (ideally > 5)
- Float under 50M shares
- Upcoming catalysts (earnings, FDA, contracts, news)
- Active retail / WSB attention OR institutional momentum

Include a mix of sectors and market caps.

Respond ONLY with a valid JSON array. No markdown fences, no preamble.
Each item MUST have exactly these fields:
[
  {{
    "ticker": "SYMBOL",
    "est_short_float": 35.2,
    "est_days_to_cover": 4.5,
    "float_size_m": 12.5,
    "catalyst": "brief catalyst",
    "catalyst_type": "earnings|fda|momentum|technical|news|squeeze_history",
    "squeeze_reason": "one sentence why this is a squeeze candidate",
    "confidence": 72
  }}
]

Return exactly {count} items."""


def _clean_json(raw: str) -> List[Dict]:
    """Strip markdown fences and extract JSON array from LLM response."""
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                raw = part
                break
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        return json.loads(raw)
    except Exception:
        return []


def _normalize(items: List[Dict], source: str) -> List[Dict]:
    """Normalize and validate items from any provider."""
    result = []
    for item in items:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker or len(ticker) > 6 or not ticker.isalpha():
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
            "source":          source,
        })
    return result


# ── Per-provider fetchers ──────────────────────────────────────────────────────

async def _fetch_anthropic(count: int, today: str) -> List[Dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        msg    = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        items = _clean_json(msg.content[0].text)
        result = _normalize(items, "claude")
        logger.info(f"[MultiLLM] Claude: {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"[MultiLLM] Claude error: {e}")
        return []


async def _fetch_openai(count: int, today: str) -> List[Dict]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("[MultiLLM] OpenAI: no key — skipping")
        return []
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp   = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        items  = _clean_json(resp.choices[0].message.content or "")
        result = _normalize(items, "openai")
        logger.info(f"[MultiLLM] OpenAI: {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"[MultiLLM] OpenAI error: {e}")
        return []


async def _fetch_grok(count: int, today: str) -> List[Dict]:
    api_key = os.environ.get("GROK_API_KEY", "")
    if not api_key:
        logger.debug("[MultiLLM] Grok: no key — skipping")
        return []
    try:
        import openai  # Grok uses OpenAI-compatible API
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp   = await client.chat.completions.create(
            model="grok-2-latest",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        items  = _clean_json(resp.choices[0].message.content or "")
        result = _normalize(items, "grok")
        logger.info(f"[MultiLLM] Grok: {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"[MultiLLM] Grok error: {e}")
        return []


async def _fetch_gemini(count: int, today: str) -> List[Dict]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.debug("[MultiLLM] Gemini: no key — skipping")
        return []
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp   = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"max_output_tokens": 2000, "temperature": 0.3}
        )
        items  = _clean_json(resp.text or "")
        result = _normalize(items, "gemini")
        logger.info(f"[MultiLLM] Gemini: {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"[MultiLLM] Gemini error: {e}")
        return []


# ── Consensus builder ─────────────────────────────────────────────────────────

def _build_consensus(all_results: List[List[Dict]]) -> List[Dict]:
    """
    Merge results from all providers.

    Ranking:
      - Tickers mentioned by more providers rank higher (consensus = confidence)
      - Within same mention count, rank by average confidence score
      - Provider count stored in 'llm_consensus' field

    Returns deduplicated list sorted by (mention_count DESC, avg_confidence DESC).
    """
    from collections import defaultdict
    ticker_data: Dict[str, Dict] = {}
    mention_count: Dict[str, int] = defaultdict(int)
    confidence_sum: Dict[str, float] = defaultdict(float)
    sources: Dict[str, List[str]] = defaultdict(list)

    for provider_results in all_results:
        for item in provider_results:
            t = item["ticker"]
            mention_count[t] += 1
            confidence_sum[t] += item["confidence"]
            sources[t].append(item["source"])
            if t not in ticker_data:
                ticker_data[t] = item

    # Sort: consensus first, then confidence
    ranked = sorted(
        ticker_data.keys(),
        key=lambda t: (mention_count[t], confidence_sum[t] / mention_count[t]),
        reverse=True
    )

    result = []
    for t in ranked:
        item = dict(ticker_data[t])
        item["llm_consensus"]   = mention_count[t]
        item["avg_confidence"]  = round(confidence_sum[t] / mention_count[t], 1)
        item["sources"]         = sources[t]
        item["source"]          = f"multi_llm({','.join(sorted(set(sources[t])))})"
        result.append(item)

    # Log consensus summary
    consensus_tickers = [t for t in ranked if mention_count[t] > 1]
    logger.info(
        f"[MultiLLM] Consensus: {len(result)} unique tickers, "
        f"{len(consensus_tickers)} agreed on by 2+ models: "
        f"{', '.join(consensus_tickers[:10])}"
    )
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

async def get_multi_llm_universe(count: int = 30) -> List[Dict]:
    """
    Query all configured LLM providers in PARALLEL for squeeze candidates.
    Returns consensus-ranked list.

    Falls back to Claude-only if no other keys are configured.
    """
    today = date.today().strftime("%B %d, %Y")
    per_model_count = min(count, 25)   # ask each model for up to 25

    providers_configured = sum([
        bool(os.environ.get("ANTHROPIC_API_KEY")),
        bool(os.environ.get("OPENAI_API_KEY")),
        bool(os.environ.get("GROK_API_KEY")),
        bool(os.environ.get("GEMINI_API_KEY")),
    ])
    logger.info(f"[MultiLLM] Querying {providers_configured} LLM provider(s) in parallel...")

    # Run all providers simultaneously
    all_results = await asyncio.gather(
        _fetch_anthropic(per_model_count, today),
        _fetch_openai(per_model_count, today),
        _fetch_grok(per_model_count, today),
        _fetch_gemini(per_model_count, today),
        return_exceptions=False
    )

    # Filter empty / exception results
    valid = [r for r in all_results if isinstance(r, list) and r]

    if not valid:
        logger.error("[MultiLLM] All providers failed — no universe built")
        return []

    consensus = _build_consensus(valid)
    # Cap at requested count
    return consensus[:count]


def get_tickers_only(candidates: List[Dict]) -> List[str]:
    return [c["ticker"] for c in candidates]
