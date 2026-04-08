"""
TRaNKSP — Multi-LLM Universe Builder

Queries Claude, Grok (xAI), OpenAI (GPT), and Gemini in parallel.
Returns consensus-ranked tickers — agreed on by multiple models = stronger signal.

NEW: enabled_providers param lets you run any subset of LLMs.
NEW: LLM price/SI consensus — ask LLMs for price + SI estimates, compare across
     providers. If ≤1% variance between ≥3 LLMs, accept Claude's value and tag
     the record with consensus note.

Environment variables (set in .env):
  ANTHROPIC_API_KEY  — required (Claude Haiku 4.5)
  XAI_API_KEY        — optional (Grok-3, alias → latest stable)
  OPENAI_API_KEY     — optional (GPT-4.1-mini)
  GOOGLE_API_KEY     — optional (Gemini 2.5 Flash, google-genai SDK)

Model versions (April 2026):
  Claude:  claude-haiku-4-5-20251001   — fast + cheap for universe building
  Grok:    grok-3                       — aliased to latest stable xAI model
  OpenAI:  gpt-4.1-mini                 — upgraded from gpt-4o-mini (better + cheaper)
  Gemini:  gemini-2.5-flash             — uses new google-genai SDK with 60s timeout
                                          (fixes known hang bug in google-generativeai)
"""

import os
import json
import asyncio
import logging
from datetime import date
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional

logger = logging.getLogger("tranksp.multi_llm")

# All supported provider names (lowercase)
ALL_PROVIDERS = ["claude", "grok", "openai", "gemini"]

def get_consensus_badge(level: int) -> str:
    """Return visual consensus indicator for frontend display."""
    if level == 4:   return "🔥🔥🔥🔥 4-LLM"
    elif level == 3: return "⭐⭐⭐ 3-LLM"
    elif level == 2: return "💎💎 2-LLM"
    else:            return "⚪ 1-LLM"


# ── Prompts ───────────────────────────────────────────────────────────────────

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


PRICE_SI_PROMPT = """You are a financial data analyst with access to current market knowledge.
Today's date: {today}

For each ticker in the list below, provide your best estimate of:
1. Current stock price (USD)
2. Short interest as % of float
3. Days to cover (short ratio)

Tickers: {tickers}

Respond ONLY with a JSON array — no markdown, no extra text:
[
  {{
    "ticker": "GME",
    "price": 15.42,
    "short_float": 18.5,
    "days_to_cover": 4.2
  }}
]

Base your estimates on the most recent data you have. If uncertain, give your best estimate."""


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


# ── Per-provider universe fetchers ────────────────────────────────────────────

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
            model="claude-haiku-4-5-20251001",
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
    api_key = _get_key("XAI_API_KEY", "GROK_API_KEY")
    if not api_key:
        logger.debug("GROK: no XAI_API_KEY — skipping")
        return []
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp = await client.chat.completions.create(
            model="grok-3",
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
            model="gpt-4.1-mini",
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
    # Uses new google-genai SDK (replaces deprecated google-generativeai).
    # Adds explicit 60s timeout — gemini-2.5-flash has a known SDK bug where
    # overloaded requests hang indefinitely instead of raising a timeout error.
    api_key = _get_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        logger.debug("GEMINI: no GOOGLE_API_KEY — skipping")
        return []
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)
        prompt = UNIVERSE_PROMPT.format(count=count, today=today)
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            ),
            timeout=60.0
        )
        items  = _clean_json(resp.text or "")
        result = _normalize(items, "gemini")
        logger.info(f"✅ GEMINI   → {len(result)} candidates")
        return result
    except asyncio.TimeoutError:
        logger.warning("❌ GEMINI   timed out after 60s — skipping")
        return []
    except Exception as e:
        logger.warning(f"❌ GEMINI   failed: {e}")
        return []


# ── Per-provider price/SI fetchers ────────────────────────────────────────────

async def _fetch_price_si_claude(tickers: List[str], today: str) -> Dict[str, Dict]:
    api_key = _get_key("ANTHROPIC_API_KEY")
    if not api_key:
        return {}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = PRICE_SI_PROMPT.format(today=today, tickers=", ".join(tickers))
        msg = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        items = _clean_json(msg.content[0].text)
        result = {i["ticker"].upper(): i for i in items if i.get("ticker")}
        logger.info(f"✅ CLAUDE price/SI → {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"❌ CLAUDE price/SI failed: {e}")
        return {}


async def _fetch_price_si_grok(tickers: List[str], today: str) -> Dict[str, Dict]:
    api_key = _get_key("XAI_API_KEY", "GROK_API_KEY")
    if not api_key:
        return {}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        prompt = PRICE_SI_PROMPT.format(today=today, tickers=", ".join(tickers))
        resp = await client.chat.completions.create(
            model="grok-3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        items = _clean_json(resp.choices[0].message.content or "")
        result = {i["ticker"].upper(): i for i in items if i.get("ticker")}
        logger.info(f"✅ GROK price/SI → {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"❌ GROK price/SI failed: {e}")
        return {}


async def _fetch_price_si_openai(tickers: List[str], today: str) -> Dict[str, Dict]:
    api_key = _get_key("OPENAI_API_KEY")
    if not api_key:
        return {}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        prompt = PRICE_SI_PROMPT.format(today=today, tickers=", ".join(tickers))
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        items = _clean_json(resp.choices[0].message.content or "")
        result = {i["ticker"].upper(): i for i in items if i.get("ticker")}
        logger.info(f"✅ OPENAI price/SI → {len(result)} tickers")
        return result
    except Exception as e:
        logger.warning(f"❌ OPENAI price/SI failed: {e}")
        return {}


async def _fetch_price_si_gemini(tickers: List[str], today: str) -> Dict[str, Dict]:
    api_key = _get_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        return {}
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)
        prompt = PRICE_SI_PROMPT.format(today=today, tickers=", ".join(tickers))
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            ),
            timeout=60.0
        )
        items  = _clean_json(resp.text or "")
        result = {i["ticker"].upper(): i for i in items if i.get("ticker")}
        logger.info(f"✅ GEMINI price/SI → {len(result)} tickers")
        return result
    except asyncio.TimeoutError:
        logger.warning("❌ GEMINI price/SI timed out after 60s — skipping")
        return {}
    except Exception as e:
        logger.warning(f"❌ GEMINI price/SI failed: {e}")
        return {}


# ── LLM Price/SI Consensus ────────────────────────────────────────────────────

def _check_consensus_pct(values: List[float], threshold_pct: float = 1.0) -> bool:
    """Return True if all values are within threshold_pct % of each other."""
    if len(values) < 2:
        return False
    min_v = min(values)
    max_v = max(values)
    if min_v <= 0:
        return False
    return ((max_v - min_v) / min_v * 100) <= threshold_pct


async def get_llm_price_si_consensus(
    tickers: List[str],
    enabled_providers: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Ask enabled LLMs for price + SI estimates for each ticker.
    For each field, if ≥3 LLMs agree within 1%, accept Claude's value
    and tag with consensus note listing agreeing LLMs.

    Returns: dict keyed by ticker with keys:
      price, short_float, days_to_cover,
      price_consensus_note, si_consensus_note,
      price_consensus (bool), si_consensus (bool)
    """
    if not tickers:
        return {}

    providers = enabled_providers or ALL_PROVIDERS
    today = date.today().strftime("%B %d, %Y")

    # Fetch in parallel from enabled providers
    fetch_map = {
        "claude": _fetch_price_si_claude,
        "grok":   _fetch_price_si_grok,
        "openai": _fetch_price_si_openai,
        "gemini": _fetch_price_si_gemini,
    }

    tasks = {
        name: fn(tickers, today)
        for name, fn in fetch_map.items()
        if name in providers
    }

    results_raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
    provider_data: Dict[str, Dict[str, Dict]] = {}
    for name, res in zip(tasks.keys(), results_raw):
        if isinstance(res, dict):
            provider_data[name] = res
        else:
            logger.warning(f"[PriceConsensus] {name} failed: {res}")
            provider_data[name] = {}

    output: Dict[str, Dict] = {}
    for ticker in tickers:
        ticker = ticker.upper()

        # Collect values per provider
        prices, si_floats, dtcs = {}, {}, {}
        for pname, pdata in provider_data.items():
            row = pdata.get(ticker, {})
            if row.get("price", 0) > 0:
                prices[pname]   = float(row["price"])
            if row.get("short_float", 0) > 0:
                si_floats[pname] = float(row["short_float"])
            if row.get("days_to_cover", 0) > 0:
                dtcs[pname]     = float(row["days_to_cover"])

        # Claude is the basis — skip if Claude has no data
        claude_price = prices.get("claude", 0)
        claude_si    = si_floats.get("claude", 0)
        claude_dtc   = dtcs.get("claude", 0)

        # Price consensus: ≥3 LLMs within 1%
        price_consensus     = False
        price_consensus_note = ""
        if claude_price > 0 and len(prices) >= 3:
            agreeing = [n for n, v in prices.items() if _check_consensus_pct([claude_price, v], 1.0)]
            if len(agreeing) >= 3:
                price_consensus = True
                agreeing_names  = ", ".join(n.capitalize() for n in sorted(agreeing))
                price_consensus_note = (
                    f"Price updated based on LLM consensus: {agreeing_names} "
                    f"(all within 1% of ${claude_price:.2f})"
                )
                logger.info(f"[PriceConsensus] {ticker} PRICE consensus: {agreeing_names} → ${claude_price:.2f}")

        # SI consensus: ≥3 LLMs within 1%
        si_consensus     = False
        si_consensus_note = ""
        if claude_si > 0 and len(si_floats) >= 3:
            agreeing_si = [n for n, v in si_floats.items() if _check_consensus_pct([claude_si, v], 1.0)]
            if len(agreeing_si) >= 3:
                si_consensus = True
                agreeing_si_names = ", ".join(n.capitalize() for n in sorted(agreeing_si))
                si_consensus_note = (
                    f"SI updated based on LLM consensus: {agreeing_si_names} "
                    f"(all within 1% of {claude_si:.1f}%)"
                )
                logger.info(f"[PriceConsensus] {ticker} SI consensus: {agreeing_si_names} → {claude_si:.1f}%")

        output[ticker] = {
            "price":               claude_price,
            "short_float":         claude_si,
            "days_to_cover":       claude_dtc,
            "price_consensus":     price_consensus,
            "si_consensus":        si_consensus,
            "price_consensus_note": price_consensus_note,
            "si_consensus_note":   si_consensus_note,
            "all_prices":          prices,
            "all_si":              si_floats,
            "all_dtc":             dtcs,
        }

    return output


# ── Consensus builder ─────────────────────────────────────────────────────────

def _build_consensus(all_results: List[List[Dict]]) -> Dict[str, Any]:
    """
    Merge and rank by consensus level.
    Returns both full ranked list and per-level categorization.
    """
    ticker_sources: Dict[str, Set[str]] = defaultdict(set)
    all_candidates: List[Dict] = []

    for provider_results in all_results:
        for c in provider_results:
            ticker = c["ticker"]
            ticker_sources[ticker].add(c["source"])
            all_candidates.append(c)

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

async def build_multi_llm_universe(
    count_per_llm: int = 25,
    enabled_providers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Full universe build — queries enabled providers in parallel,
    builds consensus ranking, saves to DB.

    enabled_providers: list of provider names to use, e.g. ["claude", "gemini"]
                       Defaults to all 4 providers.
    """
    providers = enabled_providers or ALL_PROVIDERS
    today_str = date.today().strftime("%B %d, %Y")
    logger.info(f"🚀 Multi-LLM Universe Build ({count_per_llm}/model) providers={providers}")

    # Map provider names to fetch functions
    fetch_map = {
        "claude": _fetch_claude,
        "grok":   _fetch_grok,
        "openai": _fetch_openai,
        "gemini": _fetch_gemini,
    }

    tasks = [
        fetch_map[p](count_per_llm, today_str)
        for p in providers
        if p in fetch_map
    ]

    all_results = await asyncio.gather(*tasks)

    valid = [r for r in all_results if isinstance(r, list) and r]
    if not valid:
        logger.error("[MultiLLM] All providers failed")
        return {"status": "failed", "total_unique_tickers": 0}

    merged = _build_consensus(valid)

    # Save to DB
    import sqlite3 as _sqlite3
    DB_PATH = os.path.join("data", "tranksp.db")
    conn = _sqlite3.connect(DB_PATH)
    conn.row_factory = _sqlite3.Row
    c = conn.cursor()
    added = 0
    today_iso = date.today().isoformat()

    for cand in merged["ranked"]:
        level = cand.get("llm_consensus", 1)
        c.execute("""
            INSERT OR REPLACE INTO squeeze_universe
            (ticker, source, source_llm, llm_consensus,
             short_float_est, days_to_cover_est,
             catalyst, catalyst_type, confidence, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            cand["ticker"],
            f"multi_llm_{level}",
            ",".join(cand.get("sources", [])),
            level,
            cand.get("est_short_float"),
            cand.get("est_dtc"),
            cand.get("catalyst"),
            cand.get("catalyst_type"),
            cand.get("confidence"),
        ))
        if c.rowcount > 0:
            added += 1

        for source in cand.get("sources", []):
            c.execute("""
                INSERT OR IGNORE INTO llm_daily_suggestions
                (run_date, llm_name, ticker, est_short_float,
                 est_dtc, catalyst, catalyst_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today_iso, source, cand["ticker"],
                cand.get("est_short_float"),
                cand.get("est_dtc"),
                cand.get("catalyst"),
                cand.get("catalyst_type"),
                cand.get("confidence"),
            ))

    conn.commit()
    conn.close()
    logger.info(f"[MultiLLM] Saved {added} tickers to squeeze_universe")

    for cand in merged["ranked"]:
        cand["badge"] = get_consensus_badge(cand.get("llm_consensus", 1))

    # Per-provider counts for UI display
    provider_counts = {}
    for p in providers:
        provider_counts[p] = sum(
            1 for cand in merged["ranked"]
            if p in cand.get("sources", [])
        )

    result = {
        "status":               "success",
        "total_unique_tickers": merged["total_unique"],
        "providers_used":       providers,
        "provider_counts":      provider_counts,
        "consensus": {
            "4_llm": len(merged["consensus"]["4_llm"]),
            "3_llm": len(merged["consensus"]["3_llm"]),
            "2_llm": len(merged["consensus"]["2_llm"]),
            "1_llm": len(merged["consensus"]["1_llm"]),
        },
        "top_4_llm":      [c["ticker"] for c in merged["consensus"]["4_llm"][:10]],
        "top_3_llm":      [c["ticker"] for c in merged["consensus"]["3_llm"][:15]],
        "ranked_tickers": merged["ranked"],
        "message":        f"Built universe with {merged['total_unique']} unique tickers",
    }

    logger.info(
        f"🎉 Multi-LLM Finished! Total: {result['total_unique_tickers']} | "
        f"Providers: {providers} | Counts: {provider_counts}"
    )
    return result


async def get_multi_llm_universe(
    count: int = 30,
    enabled_providers: Optional[List[str]] = None
) -> List[Dict]:
    """Simplified entry point — returns flat ranked list for screen pipeline."""
    result = await build_multi_llm_universe(
        count_per_llm=min(count, 25),
        enabled_providers=enabled_providers
    )
    return result.get("ranked_tickers", [])[:count]


def get_tickers_only(candidates: List[Dict]) -> List[str]:
    return [c["ticker"] for c in candidates]
