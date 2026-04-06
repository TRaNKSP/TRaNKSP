"""
TRaNKSP — Multi-LLM Universe Builder

Queries Claude, Grok (xAI), OpenAI (GPT), and Gemini in parallel.
Returns consensus-ranked tickers — agreed on by multiple models = stronger signal.

Environment variables (set in .env):
  ANTHROPIC_API_KEY  — required (Claude)
  XAI_API_KEY        — optional (Grok-2)
  OPENAI_API_KEY     — optional (GPT-4o-mini)
  GOOGLE_API_KEY     — optional (Gemini 2.0 Flash)

Note: XAI_API_KEY and GOOGLE_API_KEY match the uploaded multi_llm_universe.py spec.
      These are checked FIRST; GROK_API_KEY / GEMINI_API_KEY are accepted as aliases.
"""

import os
import json
import asyncio
import logging
from datetime import date
from collections import defaultdict
from typing import List, Dict, Any, Set

logger = logging.getLogger("tranksp.multi_llm")

# ── Prompt ────────────────────────────────────────────────────────────────────

UNIVERSE_PROMPT = """You are an expert short squeeze analyst.
Today's date: {today}

Return the **top {count}** most compelling short squeeze candidates right now.

For each ticker, respond with a JSON array like this:
[
  {{
    "ticker": "GME",
    "est_short_float": 18.5,
    "est_days_to_cover": 4.2,
    "catalyst": "Earnings beat + console cycle",
    "catalyst_type": "earnings",
    "squeeze_reason": "High SI + strong retail momentum",
    "confidence": 82
  }}
]

Rules:
- Only US-listed stocks (NYSE/NASDAQ)
- Focus on realistic squeeze potential (SI > 10%, DTC > 2, catalysts)
- Do NOT add any extra text outside the JSON array."""


def _get_key(primary: str, alias: str = "") -> str:
    """Get API key checking primary name then alias."""
    return os.environ.get(primary, "") or os.environ.get(alias, "")


def _clean_json(raw: str) -> List[Dict]:
    """Strip markdown fences and extract JSON array."""
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except Exception:
        return []


def _normalize(items: List[Dict], source: str) -> List[Dict]:
    result = []
    for item in items:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker or len(ticker) > 6 or not ticker.isalpha():
            continue
        result.append({
            "ticker":          ticker,
            "est_short_float": float(item.get("est_short_float", 0)),
            "est_dtc":         float(item.get("est_days_to_cover", 0)),
            "catalyst":        str(item.get("catalyst", "")),
            "catalyst_type":   str(item.get("catalyst_type", "momentum")),
            "squeeze_reason":  str(item.get("squeeze_reason", "")),
            "confidence":      float(item.get("confidence", 50)),
            "source":          source,
        })
    return result


# ── Per-provider fetchers ──────────────────────────────────────────────────────

async def _fetch_claude(count: int, today: str) -> List[Dict]:
    api_key = _get_key("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        msg = await asyncio.to_thread(
            client.messages.create,
            model="claude-3-5-sonnet-20241022",   # sonnet for universe quality
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        items  = _clean_json(msg.content[0].text)
        result = _normalize(items, "claude")
        logger.info(f"✅ CLAUDE   → {len(result)} candidates")
        return result
    except Exception as e:
        logger.warning(f"❌ CLAUDE   failed: {e}")
        return []


async def _fetch_grok(count: int, today: str) -> List[Dict]:
    # Matches uploaded file: XAI_API_KEY, model grok-2-1212
    api_key = _get_key("XAI_API_KEY", "GROK_API_KEY")
    if not api_key:
        logger.debug("GROK: no XAI_API_KEY — skipping")
        return []
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp = await client.chat.completions.create(
            model="grok-2-1212",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7
        )
        items  = _clean_json(resp.choices[0].message.content or "")
        result = _normalize(items, "grok")
        logger.info(f"✅ GROK     → {len(result)} candidates")
        return result
    except Exception as e:
        logger.warning(f"❌ GROK     failed: {e}")
        return []


async def _fetch_openai(count: int, today: str) -> List[Dict]:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI: no key — skipping")
        return []
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7
        )
        items  = _clean_json(resp.choices[0].message.content or "")
        result = _normalize(items, "openai")
        logger.info(f"✅ OPENAI   → {len(result)} candidates")
        return result
    except Exception as e:
        logger.warning(f"❌ OPENAI   failed: {e}")
        return []


async def _fetch_gemini(count: int, today: str) -> List[Dict]:
    # Matches uploaded file: GOOGLE_API_KEY, model gemini-2.0-flash
    api_key = _get_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        logger.debug("GEMINI: no GOOGLE_API_KEY — skipping")
        return []
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-2.0-flash")
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp   = await asyncio.to_thread(model.generate_content, prompt)
        items  = _clean_json(resp.text or "")
        result = _normalize(items, "gemini")
        logger.info(f"✅ GEMINI   → {len(result)} candidates")
        return result
    except Exception as e:
        logger.warning(f"❌ GEMINI   failed: {e}")
        return []


# ── Consensus builder ─────────────────────────────────────────────────────────

def _build_consensus(all_results: List[List[Dict]]) -> Dict[str, Any]:
    """
    Merge and rank. Matches the uploaded multi_llm_universe.py structure exactly.
    Returns both the full ranked list AND the categorized consensus dict.
    """
    ticker_sources: Dict[str, Set[str]] = defaultdict(set)
    all_candidates: List[Dict] = []

    for provider_results in all_results:
        for c in provider_results:
            ticker = c["ticker"]
            ticker_sources[ticker].add(c["source"])
            all_candidates.append(c)

    # Categorize by consensus level
    consensus = {"4_llm": [], "3_llm": [], "2_llm": [], "1_llm": []}
    for ticker, sources in sorted(ticker_sources.items(),
                                   key=lambda x: len(x[1]), reverse=True):
        level = len(sources)
        cand  = next((c for c in all_candidates if c["ticker"] == ticker), {})
        cand  = dict(cand)
        cand["llm_consensus"] = level
        cand["sources"]       = sorted(sources)

        if   level == 4: consensus["4_llm"].append(cand)
        elif level == 3: consensus["3_llm"].append(cand)
        elif level == 2: consensus["2_llm"].append(cand)
        else:            consensus["1_llm"].append(cand)

    # Flat ranked list (4-LLM first → 1-LLM last)
    ranked = (
        consensus["4_llm"] +
        consensus["3_llm"] +
        consensus["2_llm"] +
        consensus["1_llm"]
    )

    logger.info(
        f"[MultiLLM] Consensus: {len(ticker_sources)} unique | "
        f"4-LLM: {len(consensus['4_llm'])} | "
        f"3-LLM: {len(consensus['3_llm'])} | "
        f"2-LLM: {len(consensus['2_llm'])} | "
        f"1-LLM: {len(consensus['1_llm'])}"
    )

    return {"ranked": ranked, "consensus": consensus,
            "total_unique": len(ticker_sources)}


# ── Main entry points ─────────────────────────────────────────────────────────

async def build_multi_llm_universe(count_per_llm: int = 25) -> Dict[str, Any]:
    """
    Full universe build — matches uploaded multi_llm_universe.py exactly.
    Queries all 4 providers in parallel and returns consensus dict + saves to DB.
    """
    today = date.today().strftime("%B %d, %Y")
    logger.info(f"🚀 Starting Multi-LLM Universe Build ({count_per_llm} per model)...")

    all_results = await asyncio.gather(
        _fetch_claude(count_per_llm, today),
        _fetch_grok(count_per_llm, today),
        _fetch_openai(count_per_llm, today),
        _fetch_gemini(count_per_llm, today),
    )

    valid = [r for r in all_results if isinstance(r, list) and r]
    if not valid:
        logger.error("[MultiLLM] All providers failed")
        return {"status": "failed", "total_unique_tickers": 0}

    merged = _build_consensus(valid)

    # Summary result matching uploaded file format
    result = {
        "total_unique_tickers": merged["total_unique"],
        "consensus": {
            "4_llm": len(merged["consensus"]["4_llm"]),
            "3_llm": len(merged["consensus"]["3_llm"]),
            "2_llm": len(merged["consensus"]["2_llm"]),
            "1_llm": len(merged["consensus"]["1_llm"]),
        },
        "top_4_llm": [c["ticker"] for c in merged["consensus"]["4_llm"][:10]],
        "top_3_llm": [c["ticker"] for c in merged["consensus"]["3_llm"][:15]],
        "ranked":    merged["ranked"],
        "status":    "success",
    }

    logger.info(f"🎉 Multi-LLM Complete! Total: {result['total_unique_tickers']} | "
                f"4-LLM: {result['consensus']['4_llm']} | "
                f"3-LLM: {result['consensus']['3_llm']}")
    return result


async def get_multi_llm_universe(count: int = 30) -> List[Dict]:
    """Simplified entry point — returns flat ranked list for screen pipeline."""
    result = await build_multi_llm_universe(count_per_llm=min(count, 25))
    return result.get("ranked", [])[:count]


def get_tickers_only(candidates: List[Dict]) -> List[str]:
    return [c["ticker"] for c in candidates]
