from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, SignalMode, StrategyKind, TradeSignal
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.strategies.mean_reversion import evaluate_mean_reversion

logger = logging.getLogger(__name__)


class MeanReversionStrategyAgent:
    """RSI + Bollinger mean reversion (Lesson 5.2)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.strategies = config.strategies
        self.engine = FeatureEngine(config)

    def generate(
        self,
        symbol: str,
        regime: RegimeAssessment | None = None,
        timeframe: str | None = None,
    ) -> TradeSignal | None:
        if regime is not None and regime.selected_strategy != StrategyKind.MEAN_REVERSION:
            return None

        timeframe = timeframe or self.config.stats.signal_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.config.stats.min_bars:
            return None

        features = self.engine.build_from_bars(symbol, bars, timeframe)
        if features is None:
            return None

        cfg = self.strategies
        result = evaluate_mean_reversion(
            features.snapshot,
            cfg.mr_rsi_oversold,
            cfg.mr_rsi_overbought,
            cfg.mr_bb_entry_low,
            cfg.mr_bb_entry_high,
        )
        if result is None:
            return None

        return TradeSignal(
            symbol=symbol,
            side=result.side,
            timeframe=timeframe,
            strength=result.strength,
            mode=SignalMode.MEAN_REVERSION,
            strategy=StrategyKind.MEAN_REVERSION,
            predicted_return=result.strength * 0.01,
            confidence=result.strength,
            reason=result.reason,
        )
