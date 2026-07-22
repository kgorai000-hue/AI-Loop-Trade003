from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, SignalMode, StrategyKind, TradeSignal
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays
from src.strategies.trend_following import evaluate_trend_following

logger = logging.getLogger(__name__)


class TrendStrategyAgent:
    """Dual MA + ADX trend following (Lesson 5.1)."""

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
        if regime is not None and regime.selected_strategy != StrategyKind.TREND_FOLLOWING:
            return None

        timeframe = timeframe or self.config.stats.signal_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.config.stats.min_bars:
            return None

        series = self.engine.compute_series(bars)
        if series is None:
            return None

        _, _, _, closes, _ = bars_to_arrays(bars)
        result = evaluate_trend_following(
            closes,
            series["adx"],
            self.strategies.trend_ma_short,
            self.strategies.trend_ma_long,
            self.strategies.adx_trend_threshold,
            adx_sideways_threshold=self.strategies.adx_sideways_threshold,
        )
        if result is None:
            return None

        return TradeSignal(
            symbol=symbol,
            side=result.side,
            timeframe=timeframe,
            strength=result.strength,
            mode=SignalMode.MOMENTUM,
            strategy=StrategyKind.TREND_FOLLOWING,
            predicted_return=result.strength * 0.01,
            confidence=result.strength,
            reason=result.reason,
        )
