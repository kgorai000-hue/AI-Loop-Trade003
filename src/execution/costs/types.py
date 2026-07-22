from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class CostComponents:
    explicit_pct: float
    slippage_pct: float
    market_impact_pct: float
    opportunity_cost_pct: float
    spread_pct: float = 0.0

    @property
    def total_pct(self) -> float:
        return self.explicit_pct + self.slippage_pct + self.market_impact_pct + self.opportunity_cost_pct


@dataclass
class TradabilityAssessment:
    symbol: str
    side: str
    lots: float
    notional: float
    gross_alpha_pct: float
    costs: CostComponents
    net_alpha_pct: float
    tradable: bool
    fill_probability: float
    order_adv_ratio: float
    slippage_model: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class GrossNetStrategyReport:
    strategy_name: str
    gross_alpha_pct: float
    turnover_pct: float
    cost_per_trade_pct: float
    annual_cost_pct: float
    net_alpha_pct: float
    tradable: bool


@dataclass
class CostPipelineReport:
    assessments: list[TradabilityAssessment] = field(default_factory=list)
    blocked_count: int = 0
    total_estimated_cost_jpy: float = 0.0
    warnings: list[str] = field(default_factory=list)
