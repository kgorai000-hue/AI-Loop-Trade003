from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import SignalSide


@dataclass
class StrategySignal:
    side: SignalSide
    strength: float
    reason: str


def evaluate_trend_following(
    closes: np.ndarray,
    adx: np.ndarray,
    ma_short: int,
    ma_long: int,
    adx_threshold: float,
    adx_sideways_threshold: float | None = None,
) -> StrategySignal | None:
    if len(closes) < ma_long + 2:
        return None

    short_val = float(np.mean(closes[-ma_short:]))
    long_val = float(np.mean(closes[-ma_long:]))
    adx_val = float(adx[-1]) if len(adx) and not np.isnan(adx[-1]) else 0.0

    # Explicit flat when ADX is in sideways regime (below sideways threshold).
    if adx_sideways_threshold is not None and adx_val < adx_sideways_threshold:
        return None

    if adx_val < adx_threshold:
        return None

    spread_pct = (short_val - long_val) / long_val if long_val else 0.0
    strength = min(abs(spread_pct) * 50, 1.0)

    if short_val > long_val:
        return StrategySignal(
            side=SignalSide.BUY,
            strength=max(strength, 0.3),
            reason=f"dual MA bullish (SMA{ma_short}>{ma_long}) ADX={adx_val:.1f}",
        )

    if short_val < long_val:
        return StrategySignal(
            side=SignalSide.SELL,
            strength=max(strength, 0.3),
            reason=f"dual MA bearish (SMA{ma_short}<{ma_long}) ADX={adx_val:.1f}",
        )

    return None
