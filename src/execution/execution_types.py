from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionFill:
    fill_id: str
    lots: float
    price: float
    timestamp_ms: int
    slippage_pct: float


@dataclass
class OrderRoutingPlan:
    total_lots: float
    child_lots: list[float]
    algo: str
    order_type: str
    reason: str


@dataclass
class SimulatedExecution:
    expected_price: float
    average_fill_price: float
    filled_lots: float
    requested_lots: float
    latency_ms: float
    slippage_pct: float
    fill_ratio: float
    fills: list[ExecutionFill]
    status: str
    reason: str


@dataclass
class ExecutionLogRecord:
    record_id: str
    timestamp: int
    symbol: str
    canonical_symbol: str
    side: str
    order_type: str
    expected_price: float
    average_fill_price: float
    requested_lots: float
    filled_lots: float
    slippage_pct: float
    latency_ms: float
    fill_ratio: float
    commission_jpy: float
    dry_run: bool
    status: str
    child_orders: int
    reason: str


@dataclass
class ExecutionPipelineReport:
    records: list[ExecutionLogRecord] = field(default_factory=list)
    avg_slippage_pct: float = 0.0
    avg_fill_ratio: float = 0.0
    avg_latency_ms: float = 0.0
    partial_fill_count: int = 0
    warnings: list[str] = field(default_factory=list)
