from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.data_agent import DataAgent
from src.agents.meta_agent import PipelineResult
from src.agents.pipeline import TradingPipeline
from src.core.config import AppConfig
from src.core.data_manager import DataManager
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.system.integration import SystemIntegrator
from src.system.types import IntegrationReport

logger = logging.getLogger(__name__)


@dataclass
class SystemRunResult:
    pipeline: PipelineResult
    integration: IntegrationReport
    synced_bars: int = 0


class TradingSystem:
    """
    End-to-end Multi-Agent Trading System (Lesson 21).
    Modular monolith: single process, clean agent boundaries.
    """

    def __init__(self, config: AppConfig, connector: MT5Connector, store: OHLCVStore) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.pipeline = TradingPipeline(config, connector, store)
        self.data_manager = DataManager(config, store)
        self.data_agent = DataAgent(config, connector, store)
        self.integrator = SystemIntegrator(config, connector, store)

    def sync_data(self, symbols: list[str] | None = None) -> int:
        summary = self.data_agent.sync_all(symbols=symbols)
        return summary.total_stored

    def run(
        self,
        symbols: list[str] | None = None,
        *,
        sync_first: bool = False,
        backtest_passed: bool | None = None,
    ) -> SystemRunResult:
        synced = 0
        if sync_first:
            synced = self.sync_data(symbols)

        pipeline_ok = False
        pipeline_result = PipelineResult()
        try:
            pipeline_result = self.pipeline.run(symbols)
            pipeline_ok = True
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)

        ops_healthy = True
        if pipeline_result.ops_report:
            ops_healthy = pipeline_result.ops_report.healthy

        integration = self.integrator.assess(
            pipeline_ok=pipeline_ok,
            backtest_passed=backtest_passed,
            ops_healthy=ops_healthy,
        )

        return SystemRunResult(
            pipeline=pipeline_result,
            integration=integration,
            synced_bars=synced,
        )
