from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import SignalMode, SignalSide, TradeSignal, StrategyKind
from src.data.store import OHLCVStore
from src.ml.ic import detect_ic_decay, rolling_ic
from src.ml.trainer import MLTrainer

logger = logging.getLogger(__name__)


class MLSignalAgent:
    """Supervised learning signals with IC self-doubt (Lesson 9.6)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.ml = config.ml
        self.trainer = MLTrainer(config)
        self._last_report = None

    def generate(
        self,
        symbol: str,
        timeframe: str | None = None,
    ) -> TradeSignal | None:
        if not self.ml.enabled:
            return None

        timeframe = timeframe or self.config.stats.signal_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.config.stats.min_bars:
            return None

        try:
            report = self.trainer.train_and_evaluate(bars, symbol, timeframe)
            self._last_report = report
        except ValueError as exc:
            logger.debug("MLSignalAgent skip %s: %s", symbol, exc)
            return None

        if report.ic_decay_warning:
            logger.warning("MLSignalAgent IC decay detected for %s; halving signal strength", symbol)

        pred = self.trainer.predict_latest(bars, report)
        if pred is None:
            return None

        score, confidence = pred
        strength_scale = 0.5 if report.ic_decay_warning else 1.0
        if not report.viable:
            strength_scale *= 0.5

        threshold = self.ml.signal_probability_threshold
        if score > threshold:
            side = SignalSide.BUY
        elif score < (1.0 - threshold):
            side = SignalSide.SELL
        else:
            return None

        strength = min(confidence * strength_scale, 1.0)
        if strength < 0.1:
            return None

        predicted_return = (score - 0.5) * 0.02
        return TradeSignal(
            symbol=symbol,
            side=side,
            timeframe=timeframe,
            strength=strength,
            mode=SignalMode.MOMENTUM,
            strategy=StrategyKind.TREND_FOLLOWING,
            predicted_return=predicted_return,
            confidence=confidence,
            reason=(
                f"ML {report.model_type} prob={score:.3f} IC={report.mean_ic:.3f} "
                f"IR={report.mean_ir:.2f} viable={report.viable}"
            ),
        )
