"""
TRaNKSP — LangGraph Screener Pipeline Nodes

Data source priority per ticker:
  1. Yahoo Finance quoteSummary API  — price + full SI data in one call
  2. Massive.com daily aggregates    — price/volume fallback (free tier)
  3. Finnhub short-interest API      — SI fallback (if FINNHUB_API_KEY set)
  4. Defaults (score=0, skipped)     — if all sources fail

Stage 1: screen  — Yahoo quote → Massive fallback → Finnhub fallback
Stage 2: enrich  — RAG news + SEC via ReAct agent
Stage 3: thesis  — MapReduce + adaptive prompting from calibration stats
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date

from .state import ScreenerState
from .rag import store_news, store_filing, query_news, query_filings
from .react_agent import run_react_research
from .chains import mapreduce_synthesize, generate_thesis_direct
from .memory import format_history_for_context, save_thesis_to_history
from .massive_client import get_price_and_volume, get_ticker_details
from .yahoo_quote import YahooQuoteSession
from .finnhub_client import get_short_interest as _fh_get_si  # used only if FINNHUB_API_KEY set
# from .marketbeat_client import get_short_interest as _mb_get_si  # MarketBeat (403ing — disabled)
from .learning_engine import (
    format_calibration_for_prompt, query_episode_memory
)
from .run_tracker import save_run_detail
from .prediction_tracker import record_prediction as _record_pred
from .yahoo_screener import scrape_most_shorted

logger = logging.getLogger("tranksp.nodes")


# ── Scoring ───────────────────────────────────────────────────────────────────

def calculate_squeeze_score(
    short_float: float, days_to_cover: float,
    float_shares: float, volume_ratio: float, si_trend: str
) -> float:
    if short_float >= 50:    sf = 35
    elif short_float >= 30:  sf = 28
    elif short_float >= 20:  sf = 21
    elif short_float >= 10:  sf = 14
    else:                    sf = 7

    if days_to_cover >= 10:  dtc = 25
    elif days_to_cover >= 7: dtc = 20
    elif days_to_cover >= 5: dtc = 15
    elif days_to_cover >= 3: dtc = 10
    else:                    dtc = 5

    if volume_ratio >= 5:    vol = 20
    elif volume_ratio >= 3:  vol = 16
    elif volume_ratio >= 2:  vol = 12
    elif volume_ratio >= 1:  vol = 8
    else:                    vol = 4

    if float_shares <= 5:    flt = 10
    elif float_shares <= 10: flt = 8
    elif float_shares <= 20: flt = 6
    elif float_shares <= 50: flt = 4
    else:                    flt = 2

    trend = 10 if si_trend == "RISING" else 5 if si_trend == "FLAT" else 0

    return min(100.0, float(sf + dtc + vol + flt + trend))


# ── Stage 1: Screen Node ──────────────────────────────────────────────────────

async def _screen_one_ticker(ticker: str, settings: Dict, run_id: str) -> Optional[Dict]:
    """
    Screen one ticker with waterfall data source priority:

      Priority 1 — Yahoo quoteSummary API (finance.yahoo.com/quote/{ticker}/)
        Single call returns price, SI %, DTC, market cap, float, volume.
        Best data quality, no API key needed.

      Priority 2 — MarketBeat short interest scraper (marketbeat.com)
        Used when Yahoo returns 0 SI data. Has Current SI, SI%, DTC, trend.
        Tries NASDAQ then NYSE then other exchanges automatically.
        No API key needed.

      Priority 3 — Massive daily aggregates (fallback for price/volume)
        Used when Yahoo fails for price. Free tier, 13s gap.

      Priority 4 — Finnhub short interest (fallback for SI data)
        Used when Yahoo + MarketBeat both fail AND FINNHUB_API_KEY is set.

      Priority 5 — Massive reference data (fallback for market cap/float)
        Used when all other sources fail for market cap/float.
    """
    mc_floor   = float(settings.get("mc_floor",            250_000_000))
    min_score  = float(settings.get("min_score_threshold", 40))
    ts_snapshot = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    price = volume_ratio = short_float = days_to_cover = 0
    float_shares = market_cap = 0
    si_trend = "FLAT"
    data_source = "none"

    # ── Priority 1: Yahoo quoteSummary (price + SI in one call) ──────────────
    quote = await asyncio.to_thread(_yahoo_session.get_quote, ticker)

    if quote and quote.get("price", 0) > 0:
        price         = quote["price"]
        volume_ratio  = quote.get("volume_ratio", 1.0)
        short_float   = quote.get("short_float", 0)
        days_to_cover = quote.get("days_to_cover", 0)
        si_trend      = quote.get("si_trend", "FLAT")
        market_cap    = quote.get("market_cap", 0) * 1_000_000
        float_shares  = quote.get("float_shares", 0)
        data_source   = "yahoo"
        logger.debug(f"[Screen] {ticker}: Yahoo OK — ${price:.2f} SI={short_float:.1f}%")

    if price == 0:
        # ── Priority 3: Massive daily aggregates (price/volume) ───────────────
        logger.debug(f"[Screen] {ticker}: no price yet, trying Massive aggs...")
        snapshot = await asyncio.to_thread(get_price_and_volume, ticker)
        if snapshot and snapshot.get("price", 0) > 0:
            price        = snapshot["price"]
            volume_ratio = volume_ratio if volume_ratio > 1.0 else snapshot["volume_ratio"]
            data_source  = data_source + "+massive" if data_source != "none" else "massive"
            logger.debug(f"[Screen] {ticker}: Massive aggs OK — ${price:.2f}")

    if short_float == 0:
        # ── Priority 4: Finnhub SI (if key is configured) ─────────────────────
        import os
        if os.environ.get("FINNHUB_API_KEY"):
            logger.debug(f"[Screen] {ticker}: trying Finnhub for SI data...")
            si_data = await asyncio.to_thread(_fh_get_si, ticker)
            if si_data and si_data.get("short_float", 0) > 0:
                short_float   = si_data["short_float"]
                days_to_cover = si_data["days_to_cover"]
                si_trend      = si_data["si_trend"]
                data_source  += "+finnhub"
                logger.debug(f"[Screen] {ticker}: Finnhub SI={short_float:.1f}%")

    # ── Priority 5: Massive reference (market cap / float fallback) ───────────
    if market_cap == 0 or float_shares == 0:
        details = await asyncio.to_thread(get_ticker_details, ticker)
        if details:
            market_cap   = market_cap   or details.get("market_cap", 0)
            float_shares = float_shares or details.get("float_shares", 0)

    # ── Filters ───────────────────────────────────────────────────────────────
    if price < 1.0:
        logger.debug(f"[Screen] {ticker}: price ${price:.2f} < $1 — skip")
        save_run_detail(run_id, ticker, "SKIP", error=f"price<$1 (source:{data_source})")
        return None

    if market_cap > 0 and market_cap < mc_floor:
        logger.debug(f"[Screen] {ticker}: mktcap ${market_cap/1e6:.0f}M < floor — skip")
        return None

    score = calculate_squeeze_score(short_float, days_to_cover, float_shares, volume_ratio, si_trend)

    if score < min_score:
        logger.debug(f"[Screen] {ticker}: score {score:.0f} < {min_score} — skip")
        return None

    logger.info(
        f"[Screen] ✓ {ticker} [{data_source}]: score={score:.0f} "
        f"SI={short_float:.1f}% DTC={days_to_cover:.1f} "
        f"vol×{volume_ratio:.1f} ${price:.2f}"
    )

    return {
        "ticker":        ticker,
        "short_float":   short_float,
        "days_to_cover": days_to_cover,
        "float_shares":  float_shares,
        "price":         price,
        "market_cap":    market_cap / 1_000_000,
        "volume_ratio":  volume_ratio,
        "si_trend":      si_trend,
        "score":         score,
        "has_options":   True,
        "phase":         "DETECTION",
        "_ts_snapshot":  ts_snapshot,
        "_data_source":  data_source,
    }


# Module-level Yahoo quote session (shared across tickers in a run)
_yahoo_session: YahooQuoteSession = None  # type: ignore

def _get_yahoo_session(delay: float = 2.0) -> YahooQuoteSession:
    global _yahoo_session
    if _yahoo_session is None:
        _yahoo_session = YahooQuoteSession(delay=delay)
    return _yahoo_session


async def node_screen(state: ScreenerState) -> Dict[str, Any]:
    """
    Stage 1: Two-source screening strategy.

    Source A — Yahoo Finance most-shorted screener (single HTTP request):
      Scrapes finance.yahoo.com/research-hub/screener/most_shorted_stocks/
      Returns tickers WITH short interest data pre-populated.
      No per-ticker API calls needed for SI data on these tickers.

    Source B — Universe tickers not covered by Yahoo screener:
      Uses Massive API for price/volume + Finnhub for SI data.
      Rate-limited: Massive 13s/call, Finnhub 1.2s/call.

    Combined result is deduplicated, scored, and sorted by squeeze score.
    """
    tickers  = state["tickers"]
    settings = state.get("settings", {})
    run_id   = state["run_id"]
    mc_floor   = float(settings.get("mc_floor",            250_000_000))
    min_score  = float(settings.get("min_score_threshold", 40))

    candidates  = []
    errors      = []
    yahoo_tickers = set()

    # ── Source A: Yahoo most-shorted screener ─────────────────────────────────
    # Initialize Yahoo quote session for this run (shared, warm cookies)
    global _yahoo_session
    quote_delay = float(settings.get("yahoo_page_delay", 2.0))
    _yahoo_session = YahooQuoteSession(delay=quote_delay)

    max_pages  = int(settings.get("yahoo_screener_pages", 5))
    page_delay = float(settings.get("yahoo_page_delay", 5.0))
    logger.info(f"[Stage1:Screen] Source A: Yahoo screener ({max_pages} pages × 100 tickers, {page_delay:.0f}s gap)...")
    yahoo_data = await asyncio.to_thread(
        scrape_most_shorted,
        max_pages=max_pages,
        page_delay=page_delay
    )

    for row in yahoo_data:
        ticker = row["ticker"]
        yahoo_tickers.add(ticker)

        price       = row.get("price", 0)
        short_float = row.get("short_float", 0)
        dtc         = row.get("days_to_cover", 0)
        vol_ratio   = row.get("volume_ratio", 1.0)
        float_m     = row.get("float_shares", 0)
        mktcap      = (row.get("market_cap", 0) or 0) * 1_000_000

        if price < 1.0:
            continue
        if mktcap > 0 and mktcap < mc_floor:
            continue

        score = calculate_squeeze_score(short_float, dtc, float_m, vol_ratio, "FLAT")
        if score < min_score:
            continue

        candidates.append({
            "ticker":       ticker,
            "short_float":  short_float,
            "days_to_cover": dtc,
            "float_shares": float_m,
            "price":        price,
            "market_cap":   row.get("market_cap", 0),
            "volume_ratio": vol_ratio,
            "si_trend":     "FLAT",
            "score":        score,
            "has_options":  True,
            "phase":        "DETECTION",
            "source":       "yahoo_screener",
        })
        logger.info(f"[Screen] ✓ {ticker} (Yahoo): score={score:.0f} SI={short_float:.1f}%")

    logger.info(f"[Screen] Source A complete: {len(candidates)} candidates from {len(yahoo_data)} Yahoo tickers")

    # ── Source B: Universe tickers not in Yahoo screener ─────────────────────
    extra_tickers = [t for t in tickers if t not in yahoo_tickers]
    if extra_tickers:
        logger.info(f"[Stage1:Screen] Source B: {len(extra_tickers)} extra tickers via Massive+Finnhub")
        for i, ticker in enumerate(extra_tickers):
            logger.info(f"[Screen] {i+1}/{len(extra_tickers)}: {ticker}")
            try:
                result = await _screen_one_ticker(ticker, settings, run_id)
                if result:
                    candidates.append(result)
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                logger.warning(f"[Screen] {ticker} error: {e}")

    # Deduplicate by ticker (Yahoo data wins — more complete)
    seen = {}
    for c in candidates:
        t = c["ticker"]
        if t not in seen or c.get("source") == "yahoo_screener":
            seen[t] = c
    candidates = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    logger.info(f"[Stage1:Screen] Complete: {len(candidates)} candidates total")

    return {
        "raw_candidates":  candidates,
        "screen_errors":   errors,
        "status":          "enriching",
        "log_messages":    [f"Screen: {len(candidates)} candidates ({len(yahoo_data)} from Yahoo screener, {len(extra_tickers)} from Massive/Finnhub)"]
    }


# ── Stage 2: Enrich Node ──────────────────────────────────────────────────────

async def enrich_ticker(ticker: str, run_id: str) -> Dict[str, Any]:
    ts_agent = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    research  = await run_react_research(ticker)
    ts_rag    = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if research.get("news_context"):
        store_news(ticker, research["news_context"], source="react_agent", run_id=run_id)
    if research.get("sec_context"):
        store_filing(ticker, research["sec_context"], filing_type="8-K", run_id=run_id)

    ts_chroma  = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    news_chunks   = query_news(ticker,    f"{ticker} short squeeze catalyst", n_results=5)
    filing_chunks = query_filings(ticker, f"{ticker} material events",        n_results=3)

    return {
        ticker: {
            "news_context":  research.get("news_context", ""),
            "sec_context":   research.get("sec_context", ""),
            "agent_summary": research.get("agent_summary", ""),
            "news_chunks":   news_chunks,
            "filing_chunks": filing_chunks,
            "_ts_agent":     ts_agent,
            "_ts_rag":       ts_rag,
            "_ts_chroma":    ts_chroma,
        }
    }


async def node_enrich(state: ScreenerState) -> Dict[str, Any]:
    candidates     = state.get("raw_candidates", [])
    portfolio_size = int(state.get("settings", {}).get("portfolio_size", 10))
    top_candidates = candidates[:portfolio_size]

    logger.info(f"[Stage2:Enrich] Enriching {len(top_candidates)} candidates")
    enriched = {}
    for candidate in top_candidates:
        result = await enrich_ticker(candidate["ticker"], state["run_id"])
        enriched.update(result)

    return {
        "enriched":     enriched,
        "status":       "generating_thesis",
        "log_messages": [f"Enrichment complete for {len(top_candidates)} tickers"]
    }


# ── Stage 3: Thesis Node (adaptive prompting) ─────────────────────────────────

async def generate_ticker_thesis(
    candidate: Dict[str, Any], enriched_data: Dict[str, Any], run_id: str
) -> Dict[str, Any]:
    ticker  = candidate["ticker"]
    enrich  = enriched_data.get(ticker, {})
    ts_llm  = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    # Phase 2: episodic memory for this ticker
    prior_episodes = query_episode_memory(ticker)

    # Phase 3: calibration context
    calibration_context = format_calibration_for_prompt()

    # Claude universe context — reasoning Claude already generated for this ticker
    state_settings    = candidate.get("_settings", {})
    claude_candidates = state_settings.get("_claude_candidates", {})
    claude_ctx        = claude_candidates.get(ticker, {})
    claude_universe_context = ""
    if claude_ctx:
        claude_universe_context = (
            f"CLAUDE INITIAL ASSESSMENT FOR {ticker}:\n"
            f"  Estimated SI: {claude_ctx.get('est_short_float',0):.1f}% | "
            f"  DTC: {claude_ctx.get('est_dtc',0):.1f} | "
            f"  Float: {claude_ctx.get('float_size_m',0):.1f}M\n"
            f"  Catalyst: {claude_ctx.get('catalyst','')}\n"
            f"  Squeeze reason: {claude_ctx.get('squeeze_reason','')}\n"
            f"  Catalyst type: {claude_ctx.get('catalyst_type','')} | "
            f"  Initial confidence: {claude_ctx.get('confidence',0):.0f}%"
        )

    all_chunks = enrich.get("news_chunks", []) + enrich.get("filing_chunks", [])
    quant_data = {
        "short_float":   f"{candidate['short_float']:.1f}",
        "days_to_cover": f"{candidate['days_to_cover']:.1f}",
        "float_shares":  f"{candidate['float_shares']:.1f}",
        "price":         f"{candidate['price']:.2f}",
        "market_cap":    f"{candidate['market_cap']:.0f}",
        "volume_ratio":  f"{candidate['volume_ratio']:.1f}",
        "si_trend":      candidate["si_trend"]
    }

    # Build adaptive context from all learning layers
    from .learning_engine import get_latest_lessons
    lessons          = get_latest_lessons()
    lifecycle_context = format_history_for_context(ticker)
    adaptive_context = "\n\n".join(filter(None, [
        claude_universe_context,
        calibration_context,
        f"PRIOR EPISODES FOR {ticker}:\n{prior_episodes}" if prior_episodes != "No prior prediction history." else "",
        f"LESSONS LEARNED:\n{lessons}" if lessons else "",
        f"LIFECYCLE HISTORY:\n{lifecycle_context}" if lifecycle_context else ""
    ]))

    # ── Async thesis generation (await required — chains use ainvoke) ──────────
    thesis = None
    try:
        if len(all_chunks) >= 3:
            thesis = await mapreduce_synthesize(ticker, all_chunks, quant_data)
        else:
            thesis = await generate_thesis_direct(
                ticker, quant_data,
                enrich.get("news_context",  "No news."),
                enrich.get("sec_context",   "No filings."),
                adaptive_context
            )
    except Exception as e:
        logger.error(f"❌ Thesis generation failed for {ticker}: {e}")

    # ── Post-thesis actions ────────────────────────────────────────────────────
    if thesis:
        # Safe conversion: Pydantic model → dict
        thesis_dict = thesis.model_dump() if hasattr(thesis, "model_dump") else dict(thesis)

        save_thesis_to_history(ticker, thesis_dict, phase="BULLISH")

        # ── Layer 1: Record prediction for outcome tracking ────────────────────
        # Uses prediction_tracker (the authoritative Layer 1 module).
        # Note: learning_engine.save_prediction() is intentionally NOT called
        # here — _record_pred is the single source of truth for predictions.
        try:
            current_price = candidate.get("price", 0)
            pred_id = _record_pred(
                ticker       = ticker,
                run_id       = run_id,
                thesis       = thesis_dict,
                entry_price  = current_price,
                short_float  = candidate.get("short_float"),
                dtc          = candidate.get("days_to_cover"),
                volume_ratio = candidate.get("volume_ratio"),
                si_trend     = candidate.get("si_trend"),
            )
            if pred_id:
                logger.info(f"✅ Thesis generated and prediction #{pred_id} recorded for {ticker}")
            else:
                logger.debug(f"[{ticker}] Prediction not recorded (duplicate or error)")
        except Exception as e:
            logger.error(f"❌ record_prediction failed for {ticker}: {e}")

    return {
        ticker: {
            "candidate":    candidate,
            "thesis":       thesis.model_dump() if thesis else None,
            "run_id":       run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "_ts_llm":      ts_llm,
        }
    }


async def node_thesis(state: ScreenerState) -> Dict[str, Any]:
    candidates     = state.get("raw_candidates", [])
    enriched       = state.get("enriched", {})
    portfolio_size = int(state.get("settings", {}).get("portfolio_size", 10))
    top_candidates = candidates[:portfolio_size]

    logger.info(f"[Stage3:Thesis] Generating theses for {len(top_candidates)} candidates")

    final_results = []
    for candidate in top_candidates:
        ticker = candidate["ticker"]
        candidate["_settings"] = state.get("settings", {})
        result = await generate_ticker_thesis(candidate, enriched, state["run_id"])
        thesis_data = result[ticker]
        final_results.append({
            "ticker":       ticker,
            "score":        candidate["score"],
            "short_float":  candidate["short_float"],
            "days_to_cover": candidate["days_to_cover"],
            "float_shares": candidate["float_shares"],
            "price":        candidate["price"],
            "market_cap":   candidate["market_cap"],
            "volume_ratio": candidate["volume_ratio"],
            "si_trend":     candidate["si_trend"],
            "has_options":  candidate["has_options"],
            "phase":        "DETECTION",
            "thesis":       thesis_data.get("thesis"),
            "generated_at": thesis_data.get("generated_at")
        })

    return {
        "final_results": final_results,
        "status":        "complete",
        "log_messages":  [f"Thesis complete for {len(top_candidates)} tickers"]
    }
