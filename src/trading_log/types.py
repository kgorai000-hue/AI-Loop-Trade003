from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderLogRecord:
    """Appendix A1.1 order-level fields."""

    order_id: str
    trace_id: str
    symbol: str
    side: str
    order_type: str
    order_price: float
    order_qty: float
    submit_ts_ms: int
    cancel_ts_ms: int | None = None
    status: str = "submitted"
    dry_run: bool = True


@dataclass
class FillLogRecord:
    """Appendix A1.1.2 fill-level fields."""

    fill_id: str
    order_id: str
    fill_price: float
    fill_qty: float
    fill_ts_ms: int
    slippage_pct: float = 0.0


@dataclass
class TradeMetrics:
    """Appendix A1.1.3 derived execution metrics."""

    order_id: str
    expected_price: float
    average_fill_price: float
    slippage_pct: float
    latency_ms: float
    fill_ratio: float
    commission: float = 0.0
    tax: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class MarketSnapshot:
    """Appendix A1.2.2 market state at decision time."""

    order_id: str
    bar_ts: int
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    bar_vwap: float
    atr_5min: float
    bid1: float
    ask1: float


@dataclass
class AgentDecisionMeta:
    """Appendix A1.3 RL / agent decision context."""

    order_id: str
    trace_id: str
    agent_id: str
    agent_version: str
    action: str
    target_position: float
    confidence: float
    signal_strength: float = 0.0
    predicted_return: float | None = None
    strategy: str = ""


@dataclass
class TradeLogBundle:
    order: OrderLogRecord
    fills: list[FillLogRecord] = field(default_factory=list)
    metrics: TradeMetrics | None = None
    market: MarketSnapshot | None = None
    agent: AgentDecisionMeta | None = None


@dataclass
class TradeLogSummary:
    order_count: int = 0
    fill_count: int = 0
    avg_slippage_pct: float = 0.0
    avg_fill_ratio: float = 0.0
    avg_latency_ms: float = 0.0
    total_commission: float = 0.0
