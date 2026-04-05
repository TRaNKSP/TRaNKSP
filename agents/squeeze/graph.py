"""
TRaNKSP — LangGraph Screener Pipeline Graph
3-stage: screen → enrich → thesis
"""

import logging
from langgraph.graph import StateGraph, END

from .state import ScreenerState
from .nodes import node_screen, node_enrich, node_thesis

logger = logging.getLogger("tranksp.graph")


def build_screener_graph():
    """Build the 3-stage screener pipeline."""
    graph = StateGraph(ScreenerState)
    
    graph.add_node("screen", node_screen)
    graph.add_node("enrich", node_enrich)
    graph.add_node("thesis", node_thesis)
    
    graph.set_entry_point("screen")
    graph.add_edge("screen", "enrich")
    graph.add_edge("enrich", "thesis")
    graph.add_edge("thesis", END)
    
    return graph.compile()


# Singleton compiled graph
_screener_graph = None


def get_screener_graph():
    global _screener_graph
    if _screener_graph is None:
        _screener_graph = build_screener_graph()
    return _screener_graph


async def run_screener_pipeline(
    run_id: str,
    tickers: list,
    settings: dict,
    log_callback=None
) -> dict:
    """
    Run the full 3-stage screener pipeline.
    Returns final_results list with all ticker data + theses.
    """
    graph = get_screener_graph()
    
    initial_state = {
        "run_id": run_id,
        "tickers": tickers,
        "settings": settings,
        "raw_candidates": [],
        "screen_errors": [],
        "enriched": {},
        "theses": {},
        "final_results": [],
        "status": "screening",
        "log_messages": []
    }
    
    logger.info(f"[Pipeline] Starting run {run_id} with {len(tickers)} tickers")
    
    final_state = await graph.ainvoke(initial_state)
    
    logger.info(f"[Pipeline] Run {run_id} complete. {len(final_state.get('final_results', []))} results.")
    
    return {
        "run_id": run_id,
        "results": final_state.get("final_results", []),
        "errors": final_state.get("screen_errors", []),
        "status": final_state.get("status", "unknown"),
        "log_messages": final_state.get("log_messages", [])
    }
