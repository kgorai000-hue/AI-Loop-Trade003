from __future__ import annotations

from src.agents.portfolio_agent import PortfolioAgent
from src.core.config import AppConfig
from src.core.types import SymbolStatsReport, TradeSignal
from src.data.store import OHLCVStore
from src.portfolio.asset_rotation import apply_asset_rotation
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.types import PortfolioReport


class PortfolioConstructionAgent:
    """
    Portfolio optimization layer between signals and risk (Lesson 16).
    Signal scanning stays in PortfolioAgent; this agent allocates weights.
    Also applies AssetRotation for multi-group concurrency and migration.
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
        sized, report = self.optimizer.allocate(
            signals,
            research_reports,
            equity=equity,
            benchmark_vol=benchmark_vol,
        )
        rotated, plan = apply_asset_rotation(self.config, sized)
        if plan.notes:
            report.warnings.extend(plan.notes)
        if plan.active_groups:
            report.warnings.append(
                f"asset_rotation active_groups={plan.active_groups} "
                f"selected={plan.selected_symbols}"
            )
        return rotated, report
