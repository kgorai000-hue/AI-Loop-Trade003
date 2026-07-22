from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import SignalMode, SignalSide, TradeSignal, RegimeAssessment
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine

logger = logging.getLogger(__name__)


class SignalAgent:
    """Feature-vector based signals with regime weighting (Lesson 04)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.stats = config.stats
        self.indicators = config.indicators
        self.engine = FeatureEngine(config)

    def generate(
        self,
        symbol: str,
        regime: RegimeAssessment | None = None,
        timeframe: str | None = None,
    ) -> TradeSignal | None:
        timeframe = timeframe or self.stats.signal_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.stats.min_bars:
            return None

        if regime is not None and regime.recommended_mode == SignalMode.NONE:
            logger.debug("RegimeAgent blocked %s: %s", symbol, regime.reason)
            return None

        market_regime = regime.regime if regime else None
        features = self.engine.build_from_bars(symbol, bars, timeframe, market_regime)
        if features is None or features.mode == SignalMode.NONE:
            return None

        score = features.score
        threshold = self.indicators.signal_score_threshold
        snap = features.snapshot

        if score > threshold:
            return TradeSignal(
                symbol=symbol,
                side=SignalSide.BUY,
                timeframe=timeframe,
                strength=min(abs(score), 1.0),
                mode=features.mode,
                reason=(
                    f"feature score={score:.3f} mode={features.mode.value} "
                    f"RSI={snap.rsi:.1f} MACD={snap.macd_diff:.5f} BBpos={snap.bb_position:.2f}"
                ),
            )

        if score < -threshold:
            return TradeSignal(
                symbol=symbol,
                side=SignalSide.SELL,
                timeframe=timeframe,
                strength=min(abs(score), 1.0),
                mode=features.mode,
                reason=(
                    f"feature score={score:.3f} mode={features.mode.value} "
                    f"RSI={snap.rsi:.1f} MACD={snap.macd_diff:.5f} BBpos={snap.bb_position:.2f}"
                ),
            )

        return None

    def scan(
        self,
        symbols: list[str] | None = None,
        regime_map: dict[str, RegimeAssessment] | None = None,
    ) -> list[TradeSignal]:
        symbols = symbols or self.config.symbols
        regime_map = regime_map or {}
        signals: list[TradeSignal] = []
        for symbol in symbols:
            signal = self.generate(symbol, regime_map.get(symbol))
            if signal is not None:
                signals.append(signal)
        return signals
