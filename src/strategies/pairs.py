from __future__ import annotations

import numpy as np

from src.core.types import SignalSide
from src.stats.returns import log_returns
from src.strategies.trend_following import StrategySignal


def compute_spread_zscore(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
) -> float | None:
    min_len = min(len(closes_a), len(closes_b))
    if min_len < lookback + 5:
        return None

    a = closes_a[-min_len:]
    b = closes_b[-min_len:]
    spread = np.log(a / b)
    window = spread[-lookback:]
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    if std == 0:
        return None
    return float((spread[-1] - mean) / std)


def evaluate_pair(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
    zscore_entry: float,
    zscore_exit: float,
    symbol_a: str,
    symbol_b: str,
) -> tuple[StrategySignal | None, StrategySignal | None]:
    z = compute_spread_zscore(closes_a, closes_b, lookback)
    if z is None:
        return None, None

    if z <= -zscore_entry:
        strength = min(abs(z) / (zscore_entry * 2), 1.0)
        leg_a = StrategySignal(
            side=SignalSide.BUY,
            strength=max(strength, 0.3),
            reason=f"pair long spread z={z:.2f} ({symbol_a}/{symbol_b})",
        )
        leg_b = StrategySignal(
            side=SignalSide.SELL,
            strength=max(strength, 0.3),
            reason=f"pair short spread z={z:.2f} ({symbol_b}/{symbol_a})",
        )
        return leg_a, leg_b

    if z >= zscore_entry:
        strength = min(abs(z) / (zscore_entry * 2), 1.0)
        leg_a = StrategySignal(
            side=SignalSide.SELL,
            strength=max(strength, 0.3),
            reason=f"pair short spread z={z:.2f} ({symbol_a}/{symbol_b})",
        )
        leg_b = StrategySignal(
            side=SignalSide.BUY,
            strength=max(strength, 0.3),
            reason=f"pair long spread z={z:.2f} ({symbol_b}/{symbol_a})",
        )
        return leg_a, leg_b

    return None, None


def backtest_pair_returns(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
    zscore_entry: float,
) -> np.ndarray:
    """Simplified pairs backtest: long spread when z < -entry, short when z > entry."""
    min_len = min(len(closes_a), len(closes_b))
    if min_len < lookback + 10:
        return np.array([])

    a = closes_a[-min_len:]
    b = closes_b[-min_len:]
    ret_a = log_returns(a)
    ret_b = log_returns(b)
    spread = np.log(a / b)

    strategy = np.zeros(len(ret_a))
    position = 0.0
    for idx in range(lookback, len(ret_a)):
        window = spread[idx - lookback : idx]
        std = np.std(window, ddof=1)
        if std == 0:
            strategy[idx - 1] = position * (ret_a[idx - 1] - ret_b[idx - 1])
            continue
        z = (spread[idx - 1] - np.mean(window)) / std
        if z <= -zscore_entry:
            position = 1.0
        elif z >= zscore_entry:
            position = -1.0
        strategy[idx - 1] = position * (ret_a[idx - 1] - ret_b[idx - 1])

    return strategy
