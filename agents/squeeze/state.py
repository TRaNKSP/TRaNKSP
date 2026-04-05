"""
TRaNKSP — LangGraph State Definitions
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator


class SqueezeCandidate(TypedDict):
    ticker: str
    short_float: float
    days_to_cover: float
    float_shares: float
    price: float
    market_cap: float
    volume_ratio: float
    si_trend: str
    score: float
    has_options: bool
    phase: str


class ScreenerState(TypedDict):
    """State for the 3-stage LangGraph screener pipeline."""
    run_id: str
    tickers: List[str]
    
    # Stage 1: raw screen results
    raw_candidates: Annotated[List[SqueezeCandidate], operator.add]
    screen_errors: Annotated[List[str], operator.add]
    
    # Stage 2: enrichment (RAG news + SEC)
    enriched: Annotated[Dict[str, Dict[str, Any]], lambda a, b: {**a, **b}]
    
    # Stage 3: thesis generation
    theses: Annotated[Dict[str, Any], lambda a, b: {**a, **b}]
    final_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Metadata
    status: str
    log_messages: Annotated[List[str], operator.add]


class ReactAgentState(TypedDict):
    """State for the per-ticker ReAct research agent."""
    ticker: str
    messages: List[Any]
    news_context: str
    sec_context: str
    short_data: Dict[str, Any]
    earnings_date: Optional[str]
    competitors: List[str]
    iteration: int


class LifecycleState(TypedDict):
    """State for lifecycle evaluation."""
    ticker: str
    lifecycle_record: Dict[str, Any]
    should_trigger_bearish: bool
    trigger_reason: Optional[str]
    bearish_thesis: Optional[Dict[str, Any]]
    updated_status: str
