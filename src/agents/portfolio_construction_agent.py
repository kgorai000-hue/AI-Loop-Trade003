from __future__ import annotations

from src.agents.portfolio_agent import PortfolioAgent
from src.core.config import AppConfig
from src.core.types import SymbolStatsReport, TradeSignal
from src.data.store import OHLCVStore
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.types import PortfolioReport


class PortfolioConstructionAgent:
    """
    Portfolio optimization layer between signals and risk (Lesson 16).
    Signal scanning stays in PortfolioAgent; this agent allocates weights.
    """

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.optimizer = PortfolioOptimizer(config, store)

    def allocate(
        self,
        signals: list[TradeSignal],
        research_reports: list[SymbolStatsReport] | None = None,
        *,
        equity: float = 0.0,
        benchmark_vol: float = 0.15,
    ) -> tuple[list[TradeSignal], PortfolioReport]:
        return self.optimizer.allocate(
            signals,
            research_reports,
            equity=equity,
            benchmark_vol=benchmark_vol,
        )
