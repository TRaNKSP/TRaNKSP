"""
TRaNKSP — LangChain Chains
- MapReduce thesis synthesis (Haiku map → Sonnet reduce)
- Score explanation chain
- Bearish thesis chain
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .prompts import (
    MAP_STEP_PROMPT, REDUCE_STEP_PROMPT,
    SQUEEZE_THESIS_SYSTEM, SQUEEZE_THESIS_PROMPT,
    BEARISH_THESIS_PROMPT, SCORE_EXPLANATION_PROMPT
)
from .output_schema import ThesisOutput, BearishThesisOutput, ScoreExplanationOutput


def _extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles: markdown fences, preamble text, trailing text, partial responses.
    """
    text = text.strip()
    
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    
    # Find first { and last } to extract JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common LLM JSON errors: trailing commas, single quotes
        import re
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Replace single quotes with double (careful with apostrophes)
        # Only replace standalone single-quoted keys/values
        try:
            return json.loads(text)
        except Exception:
            raise



logger = logging.getLogger("tranksp.chains")


def _haiku_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.1,
        max_tokens=800,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
    )


def _sonnet_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.2,
        max_tokens=2000,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
    )


# ── MapReduce Synthesis ───────────────────────────────────────────────────────

async def mapreduce_synthesize(
    ticker: str,
    chunks: List[str],
    quant_data: Dict[str, Any]
) -> Optional[ThesisOutput]:
    """
    Map: Haiku summarizes each chunk independently (prevents lost-in-the-middle)
    Reduce: Sonnet synthesizes all summaries into final ThesisOutput
    """
    if not chunks:
        chunks = ["No research data available."]
    
    logger.info(f"[MapReduce] {ticker}: mapping {len(chunks)} chunks")
    
    # MAP step — Haiku processes each chunk
    map_llm = _haiku_llm()
    summaries = []
    
    for i, chunk in enumerate(chunks[:10]):  # cap at 10 chunks
        try:
            prompt = MAP_STEP_PROMPT.format(ticker=ticker, chunk=chunk[:1500])
            response = await map_llm.ainvoke([{"role": "user", "content": prompt}])
            summaries.append(f"[Chunk {i+1}]: {response.content}")
        except Exception as e:
            logger.warning(f"[MapReduce] Map error chunk {i+1} for {ticker}: {e}")
            summaries.append(f"[Chunk {i+1}]: Error - {str(e)}")
    
    logger.info(f"[MapReduce] {ticker}: reducing {len(summaries)} summaries with Sonnet")
    
    # REDUCE step — Sonnet synthesizes
    reduce_llm = _sonnet_llm()
    
    quant_str = "\n".join([f"  {k}: {v}" for k, v in quant_data.items()])
    reduce_prompt = REDUCE_STEP_PROMPT.format(
        ticker=ticker,
        summaries="\n\n".join(summaries),
        quant_data=quant_str
    )
    
    system = SQUEEZE_THESIS_SYSTEM + "\n\nRespond ONLY with valid JSON matching the ThesisOutput schema. No preamble."
    schema_hint = """
{
  "setup": "...",
  "trigger": "...",
  "mechanics": "...",
  "risk": "...",
  "catalyst_types": ["earnings", "momentum"],
  "confidence": 72.5,
  "time_horizon": "1-2 weeks",
  "bullish_score": 68.0
}"""
    
    try:
        response = await reduce_llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": reduce_prompt + f"\n\nRespond with JSON only:\n{schema_hint}"}
        ])
        
        data = _extract_json(response.content)
        thesis = ThesisOutput(**data)
        logger.info(f"[MapReduce] {ticker}: thesis complete. Score={thesis.bullish_score}, Confidence={thesis.confidence}")
        return thesis
        
    except Exception as e:
        logger.error(f"[MapReduce] Reduce error for {ticker}: {e}")
        return None


# ── Direct Thesis Chain (fallback / fast path) ────────────────────────────────

async def generate_thesis_direct(
    ticker: str,
    quant_data: Dict[str, Any],
    news_context: str,
    sec_context: str,
    lifecycle_context: str = ""
) -> Optional[ThesisOutput]:
    """Direct Sonnet thesis generation (no MapReduce, for single-ticker fast path)."""
    llm = _sonnet_llm()
    
    prompt = SQUEEZE_THESIS_PROMPT.format(
        ticker=ticker,
        short_float=quant_data.get("short_float", "N/A"),
        days_to_cover=quant_data.get("days_to_cover", "N/A"),
        float_shares=quant_data.get("float_shares", "N/A"),
        price=quant_data.get("price", "N/A"),
        market_cap=quant_data.get("market_cap", "N/A"),
        volume_ratio=quant_data.get("volume_ratio", "N/A"),
        si_trend=quant_data.get("si_trend", "N/A"),
        news_context=news_context[:2000],
        sec_context=sec_context[:1000],
        lifecycle_context=lifecycle_context[:500]
    )
    
    system = SQUEEZE_THESIS_SYSTEM + "\n\nRespond ONLY with valid JSON. No preamble or markdown."
    
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```")
        
        data = json.loads(raw)
        return ThesisOutput(**data)
    except Exception as e:
        logger.error(f"[ThesisChain] Error for {ticker}: {e}")
        return None


# ── Bearish Thesis Chain ──────────────────────────────────────────────────────

async def generate_bearish_thesis(
    ticker: str,
    lifecycle: Dict[str, Any],
    trigger_reason: str,
    lifecycle_memory: str,
    news_context: str
) -> Optional[BearishThesisOutput]:
    """Generate bearish reversal thesis triggered by lifecycle conditions."""
    llm = _sonnet_llm()
    
    prompt = BEARISH_THESIS_PROMPT.format(
        ticker=ticker,
        trigger_reason=trigger_reason,
        entry_price=lifecycle.get("entry_price", 0),
        peak_price=lifecycle.get("peak_price", 0),
        current_price=lifecycle.get("current_price", 0),
        si_entry=lifecycle.get("si_entry", 0),
        si_current=lifecycle.get("short_interest", 0),
        si_change_pct=lifecycle.get("si_change_pct", 0),
        price_chg_peak=lifecycle.get("price_chg_peak", 0),
        days_active=lifecycle.get("days_active", 0),
        lifecycle_memory=lifecycle_memory[:1500],
        news_context=news_context[:1000]
    )
    
    system = "You are a bearish thesis generator. Respond ONLY with valid JSON matching BearishThesisOutput schema. No preamble."
    
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        
        data = _extract_json(response.content)
        data["trigger_reason"] = trigger_reason
        return BearishThesisOutput(**data)
    except Exception as e:
        logger.error(f"[BearishChain] Error for {ticker}: {e}")
        return None


# ── Score Explanation Chain ───────────────────────────────────────────────────

async def explain_score(ticker: str, score_data: Dict[str, Any]) -> Optional[ScoreExplanationOutput]:
    """On-demand LLMChain to explain a squeeze score to the user."""
    llm = _haiku_llm()
    
    prompt = SCORE_EXPLANATION_PROMPT.format(
        ticker=ticker,
        score=score_data.get("score", 0),
        short_float=score_data.get("short_float", "N/A"),
        days_to_cover=score_data.get("days_to_cover", "N/A"),
        volume_ratio=score_data.get("volume_ratio", "N/A"),
        float_shares=score_data.get("float_shares", "N/A"),
        si_trend=score_data.get("si_trend", "N/A")
    )
    
    system = "You are a score explainer. Respond ONLY with valid JSON matching ScoreExplanationOutput schema."
    
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        
        data = _extract_json(response.content)
        return ScoreExplanationOutput(**data)
    except Exception as e:
        logger.error(f"[ScoreExplain] Error for {ticker}: {e}")
        return None
