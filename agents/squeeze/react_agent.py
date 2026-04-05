"""
TRaNKSP — ReAct Research Agent (LangGraph native)
Uses langgraph.prebuilt.create_react_agent (confirmed working import path)
"""

import os
import logging
from typing import Dict, Any

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from .tools import ALL_TOOLS
from .prompts import REACT_SYSTEM_PROMPT

logger = logging.getLogger("tranksp.react_agent")


def build_react_agent(ticker: str):
    """Build a ReAct agent for a specific ticker."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # Haiku for speed in ReAct loop
        temperature=0,
        max_tokens=2000,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    system_prompt = REACT_SYSTEM_PROMPT.format(ticker=ticker)
    
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        state_modifier=system_prompt
    )
    return agent


async def run_react_research(ticker: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Run the ReAct agent to gather research for a ticker.
    Returns: {news_context, sec_context, agent_summary}
    """
    logger.info(f"[ReAct] Starting research for {ticker}")
    
    try:
        agent = build_react_agent(ticker)
        
        initial_message = {
            "messages": [{
                "role": "user",
                "content": (
                    f"Research {ticker} for short squeeze potential. "
                    f"Use your tools to: (1) search recent news, (2) check SEC filings, "
                    f"(3) get current short data, (4) find earnings date, (5) check competitors. "
                    f"After gathering data, provide a comprehensive summary."
                )
            }]
        }
        
        result = await agent.ainvoke(initial_message)
        
        messages = result.get("messages", [])
        
        # Extract the final AI message
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        final_summary = ai_messages[-1].content if ai_messages else "No research summary available."
        
        # Extract tool results
        tool_messages = [m for m in messages if hasattr(m, "type") and m.type == "tool"]
        news_chunks = []
        sec_chunks = []
        
        for msg in tool_messages:
            content = str(msg.content) if msg.content else ""
            if "SEC" in content or "8-K" in content or "filing" in content.lower():
                sec_chunks.append(content)
            else:
                news_chunks.append(content)
        
        logger.info(f"[ReAct] Completed research for {ticker}: {len(tool_messages)} tool calls")
        
        return {
            "news_context": "\n\n".join(news_chunks[:3]) or "No news data gathered.",
            "sec_context": "\n\n".join(sec_chunks[:2]) or "No SEC filings found.",
            "agent_summary": final_summary
        }
        
    except Exception as e:
        logger.error(f"[ReAct] Error for {ticker}: {e}")
        return {
            "news_context": f"ReAct agent error: {str(e)}",
            "sec_context": "Unavailable.",
            "agent_summary": f"Research failed: {str(e)}"
        }
