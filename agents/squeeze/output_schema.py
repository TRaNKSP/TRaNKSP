"""
TRaNKSP — Pydantic Output Schemas
Fields are Optional with defaults so partial LLM responses degrade gracefully.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ThesisOutput(BaseModel):
    """Structured short squeeze thesis. All fields optional — LLM may omit some."""
    setup: str = Field(default="Analysis in progress.", description="Current short squeeze setup description")
    trigger: str = Field(default="Catalyst not identified.", description="Most likely catalyst that could trigger the squeeze")
    mechanics: str = Field(default="Standard short squeeze mechanics apply.", description="How the squeeze would mechanically unfold")
    risk: str = Field(default="Market risk and SI reversal.", description="Key risks that could prevent the squeeze")
    catalyst_types: List[str] = Field(default_factory=list, description="List of catalyst types: earnings, news, momentum, technical, etc.")
    confidence: float = Field(default=50.0, ge=0.0, le=100.0, description="Confidence score 0-100")
    time_horizon: str = Field(default="1-2 weeks", description="Expected time horizon: days, weeks, 1-2 months, etc.")
    bullish_score: float = Field(default=50.0, ge=0.0, le=100.0, description="Overall bullish squeeze score 0-100")


class BearishThesisOutput(BaseModel):
    """Structured bearish / reversal thesis."""
    reversal_setup: str = Field(default="Squeeze appears exhausted.", description="Why the squeeze has completed or is reversing")
    put_thesis: str = Field(default="Put options on reversal.", description="Rationale for put option position")
    downside_target: str = Field(default="10-20% below current price", description="Price target or % decline expected")
    risk: str = Field(default="Squeeze could reignite.", description="Risks to the bearish thesis")
    timing: str = Field(default="1-2 weeks", description="Expected timing for bearish move")
    confidence: float = Field(default=50.0, ge=0.0, le=100.0, description="Confidence in bearish thesis 0-100")
    trigger_reason: str = Field(default="manual", description="What triggered bearish evaluation")


class ScoreExplanationOutput(BaseModel):
    """On-demand score explanation."""
    score_breakdown: str = Field(default="Score based on SI, DTC, volume, float, and trend.", description="Breakdown of how the score was calculated")
    strongest_factor: str = Field(default="Short interest level", description="The single strongest bullish factor")
    weakest_factor: str = Field(default="Volume confirmation", description="The single weakest or most concerning factor")
    comparison: str = Field(default="Comparable to typical squeeze candidates.", description="How this ticker compares to typical squeeze candidates")
    recommendation: str = Field(default="Monitor for catalyst confirmation.", description="Actionable recommendation given the score")


class OptionsSnapshot(BaseModel):
    """Options chain analysis result."""
    has_options: bool
    iv_rank: Optional[float] = None
    atm_iv: Optional[float] = None
    best_call_strike: Optional[float] = None
    best_call_oi: Optional[int] = None
    best_call_expiry: Optional[str] = None
    best_put_strike: Optional[float] = None
    best_put_oi: Optional[int] = None
    best_put_expiry: Optional[str] = None
    nearest_expiry: Optional[str] = None
    liquid_puts_available: bool = False


class ScenarioPnL(BaseModel):
    """P&L calculation for one scenario."""
    label: str
    capital: float
    entry_price: float
    exit_price: float
    quantity: float
    entry_cost: float
    exit_value: float
    pnl: float
    pct: float
    notes: str = ""


class LifecycleSnapshot(BaseModel):
    """Daily lifecycle snapshot for a ticker."""
    ticker: str
    status: str
    entry_price: Optional[float]
    peak_price: Optional[float]
    current_price: Optional[float]
    short_interest: Optional[float]
    si_change_pct: Optional[float]
    price_chg_peak: Optional[float]
    bearish_eval: bool
    eval_triggered: Optional[str]
