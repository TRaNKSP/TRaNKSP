"""
TRaNKSP — Claude Universe Builder

Thin wrapper that now delegates to multi_llm_client.py for consensus-ranked
universe building across all configured LLM providers (Claude + OpenAI + Grok + Gemini).

Kept for backwards compatibility — callers import from here unchanged.
"""

from .multi_llm_client import get_multi_llm_universe, get_tickers_only
import logging

logger = logging.getLogger("tranksp.claude_universe")


async def get_claude_universe(count: int = 30):
    """
    Build squeeze candidate universe using all configured LLM providers.
    Claude is always used. OpenAI, Grok, Gemini are used if keys are configured.
    Returns consensus-ranked list — tickers agreed on by multiple models rank first.
    """
    return await get_multi_llm_universe(count=count)
