from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MarketType(str, Enum):
    CFD_INDEX = "cfd_index"
    COMMODITY = "commodity"
    FOREX = "forex"


class SignalSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


class RiskDecisionType(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"


class PipelineStage(str, Enum):
    META = "meta"
    DATA = "data"
    RESEARCH = "research"
    REGIME = "regime"
    SIGNAL = "signal"
    DECISION = "decision"
    RISK = "risk"
    EXECUTION = "execution"
    MONITOR = "monitor"
    POSITION = "position"
    HEDGING = "hedging"


class ArbitrationMode(str, Enum):
    HIERARCHY = "hierarchy"
    VOTING = "voting"
    VETO = "veto"


class AgentRole(str, Enum):
    META = "meta"
    DATA = "data"
    RESEARCH = "research"
    REGIME = "regime"
    SIGNAL = "signal"
    DECISION = "decision"
    RISK = "risk"
    EXECUTION = "execution"
    MONITOR = "monitor"
    POSITION = "position"
    HEDGING = "hedging"


class MessagePattern(str, Enum):
    REQUEST_RESPONSE = "request_response"
    PUB_SUB = "pub_sub"
    QUEUE = "queue"
    SHARED_STATE = "shared_state"


class MarketRegime(str, Enum):
    BULL = "bull"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class SignalMode(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    NONE = "none"


class StrategyKind(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    PAIRS = "pairs"
    GRID_DRY_RUN = "grid_dry_run"
    UNCERTAIN = "uncertain"
    CRISIS_HALT = "crisis_halt"


@dataclass
class TradeSignal:
    symbol: str
    side: SignalSide
    timeframe: str
    strength: float
    reason: str
    mode: SignalMode = SignalMode.NONE
    strategy: StrategyKind = StrategyKind.UNCERTAIN
    predicted_return: float | None = None
    confidence: float | None = None
    requested_lots: float | None = None
    portfolio_weight: float | None = None
    group_id: str | None = None
    pair_id: str | None = None
    trade_mode: str = "single"  # single | pair


@dataclass
class AgentState:
    """Agent context: market + account + positions (Lesson 10.2)."""
    equity: float
    free_margin: float
    current_exposure_pct: float
    open_position_lots: dict[str, float]


@dataclass
class AgentSpec:
    role: AgentRole
    name: str
    responsibilities: list[str]
    not_responsible: list[str]
    metric: str


@dataclass
class AgentVote:
    agent: str
    vote: int
    reason: str


@dataclass
class ArbitrationResult:
    symbol: str
    approved: bool
    net_score: int
    votes: list[AgentVote]
    selected_signal: TradeSignal | None = None
    reason: str = ""


@dataclass
class AgentHealth:
    agent: str
    healthy: bool
    failures: int
    circuit_open: bool
    elapsed_ms: float = 0.0
    last_error: str | None = None


@dataclass
class MultiAgentReport:
    evolution_stage: int
    arbitration_mode: str
    parallel_analysis: bool
    agent_health: list[AgentHealth]
    parallel_elapsed_ms: float
    serial_estimate_ms: float
    bus_events: int
    arbitration_results: list[ArbitrationResult]


@dataclass
class DecisionReport:
    symbol: str
    side: SignalSide
    predicted_return: float
    confidence: float
    annualized_volatility: float
    half_kelly_cap_pct: float
    van_tharp_cap_pct: float
    hard_cap_pct: float
    portfolio_weight_pct: float
    final_position_pct: float
    requested_lots: float
    reason: str


@dataclass
class RiskDecision:
    decision: RiskDecisionType
    approved_lots: float
    reason: str
    drawdown_level: str = "normal"
    stop_distance: float | None = None
    audit_id: int | None = None


@dataclass
class ExecutionPlan:
    symbol: str
    side: SignalSide
    lots: float
    order_type: str
    estimated_cost_jpy: float
    dry_run: bool
    reason: str
    expected_price: float | None = None
    average_fill_price: float | None = None
    filled_lots: float = 0.0
    latency_ms: float = 0.0
    slippage_pct: float = 0.0
    fill_ratio: float = 0.0
    child_orders: int = 1
    execution_record_id: str | None = None
    status: str = "planned"
    algo: str = "single"
    canonical_symbol: str | None = None
    ticket: int | None = None
    group_id: str | None = None
    trade_mode: str = "single"
    pair_id: str | None = None


@dataclass
class MonitorReport:
    symbol: str
    status: str
    message: str


@dataclass
class PositionAction:
    symbol: str
    action: str
    reason: str


@dataclass
class SymbolStatsReport:
    symbol: str
    timeframe: str
    bars: int
    cumulative_log_return: float
    annualized_return: float
    daily_volatility: float
    annualized_volatility: float
    autocorr_lag1: float
    autocorr_lag5: float
    skewness: float
    excess_kurtosis: float
    price_stationary: bool
    return_stationary: bool
    vol_autocorr: float
    regime: MarketRegime
    tail_warning: str
    rsi: float | None = None
    macd_diff: float | None = None
    macd_histogram: float | None = None
    bb_position: float | None = None
    atr: float | None = None
    atr_pct: float | None = None
    macd_divergence: str | None = None
    rsi_divergence: str | None = None


@dataclass
class RegimeAssessment:
    symbol: str
    regime: MarketRegime
    annualized_volatility: float
    recent_return: float
    recommended_mode: SignalMode
    reason: str
    adx: float = 0.0
    position_scale: float = 1.0
    selected_strategy: StrategyKind = StrategyKind.UNCERTAIN
    trend_confirmed: bool = False
    sideways_confirmed: bool = False
    regime_label: str = "unknown"
    probabilities: dict[str, float] = field(default_factory=dict)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    is_transition: bool = False
    asset_correlation: float = 0.0
    detection_method: str = "rule_based"
    vol_cluster: str = ""
    uncertain: bool = False
    max_probability: float = 0.0
    degradation_level: int = 0
    health_warnings: list[str] = field(default_factory=list)
    misjudgment_pattern: str | None = None
    switches_per_week: float = 0.0
    avg_regime_duration_days: float = 0.0
    adx_oscillation: bool = False
    uncertain_reasons: list[str] = field(default_factory=list)


@dataclass
class ResilienceReport:
    degradation_level: int
    level_name: str
    position_scale_multiplier: float
    data_quality_ok: bool
    warnings: list[str] = field(default_factory=list)
    market_health: Any | None = None
    symbol_health: list[Any] = field(default_factory=list)
