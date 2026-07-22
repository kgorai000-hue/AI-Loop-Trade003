from __future__ import annotations

from dataclasses import dataclass, field

from src.stats.performance import PerformanceReport


@dataclass
class TradeRecord:
    signal_bar: int
    execution_bar: int
    side: float
    cost_pct: float


@dataclass
class BacktestResult:
    strategy_name: str
    returns: list[float]
    positions: list[float]
    trades: list[TradeRecord]
    performance: PerformanceReport
    cost_per_trade_pct: float
    total_cost_pct: float


@dataclass
class WalkForwardRound:
    round_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float


@dataclass
class MonteCarloResult:
    simulations: int
    mean_return: float
    std_return: float
    percentile_5: float
    percentile_50: float
    percentile_95: float
    prob_positive: float


@dataclass
class OOSSplitResult:
    train_return: float
    val_return: float
    test_return: float
    train_sharpe: float
    val_sharpe: float
    test_sharpe: float
    oos_ratio: float


@dataclass
class ParameterSensitivityResult:
    parameter: str
    base_return: float
    low_return: float
    high_return: float
    max_change_pct: float
    stable: bool


@dataclass
class QualityGateCheck:
    layer: str
    check_id: str
    name: str
    passed: bool
    detail: str


@dataclass
class QualityGateReport:
    strategy_name: str
    checks: list[QualityGateCheck] = field(default_factory=list)
    live_expected_return: float = 0.0
    bonferroni_threshold: float = 0.05

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)
