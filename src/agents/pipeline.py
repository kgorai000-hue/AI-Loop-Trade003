from __future__ import annotations

import logging

from src.agents.meta_agent import MetaAgent, PipelineResult
from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore

logger = logging.getLogger(__name__)

__all__ = ["PipelineResult", "TradingPipeline"]


class TradingPipeline:
    """Lesson 02-11 lifecycle; delegates orchestration to MetaAgent."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.meta_agent = MetaAgent(config, connector, store)

    def run(self, symbols: list[str] | None = None) -> PipelineResult:
        return self.meta_agent.run(symbols)
