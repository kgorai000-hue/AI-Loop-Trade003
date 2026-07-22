from __future__ import annotations

from src.core.types import SignalSide
from src.features.indicators import IndicatorSnapshot
from src.strategies.trend_following import StrategySignal


def evaluate_mean_reversion(
    snapshot: IndicatorSnapshot,
    rsi_oversold: float,
    rsi_overbought: float,
    bb_entry_low: float,
    bb_entry_high: float,
) -> StrategySignal | None:
    if snapshot.rsi <= rsi_oversold and snapshot.bb_position <= bb_entry_low:
        strength = min((rsi_oversold - snapshot.rsi) / rsi_oversold, 1.0)
        return StrategySignal(
            side=SignalSide.BUY,
            strength=max(strength, 0.3),
            reason=(
                f"mean reversion buy RSI={snapshot.rsi:.1f} "
                f"BBpos={snapshot.bb_position:.2f}"
            ),
        )

    if snapshot.rsi >= rsi_overbought and snapshot.bb_position >= bb_entry_high:
        strength = min((snapshot.rsi - rsi_overbought) / (100 - rsi_overbought), 1.0)
        return StrategySignal(
            side=SignalSide.SELL,
            strength=max(strength, 0.3),
            reason=(
                f"mean reversion sell RSI={snapshot.rsi:.1f} "
                f"BBpos={snapshot.bb_position:.2f}"
            ),
        )

    return None
