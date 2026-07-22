from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SymbolAllocation:
    symbol: str
    raw_weight: float
    adjusted_weight: float
    signal_strength: float
    annualized_volatility: float
    correlation_penalty: float = 0.0


@dataclass
class FactorExposure:
    factor: str
    exposure: float
    limit: float | None = None
    breached: bool = False


@dataclass
class PortfolioReport:
    weight_method: str
    allocations: list[SymbolAllocation] = field(default_factory=list)
    factor_exposures: list[FactorExposure] = field(default_factory=list)
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    notional_leverage: float = 0.0
    risk_leverage: float = 0.0
    portfolio_volatility: float = 0.0
    shrinkage: float | None = None
    rebalance_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
