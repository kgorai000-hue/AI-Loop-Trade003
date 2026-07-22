from __future__ import annotations

import numpy as np

from src.core.types import SignalSide
from src.pairs.spread import build_spread_snapshot
from src.stats.returns import log_returns
from src.strategies.trend_following import StrategySignal


def compute_spread_zscore(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
    *,
    beta_window: int | None = None,
) -> float | None:
    """Z-score of hedge spread S = log A - β log B (β from rolling OLS)."""
    snap = build_spread_snapshot(
        closes_a,
        closes_b,
        z_lookback=lookback,
        beta_window=beta_window or max(lookback * 2, 40),
    )
    if snap is None:
        return None
    return snap.zscore


def evaluate_pair(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
    zscore_entry: float,
    zscore_exit: float,
    symbol_a: str,
    symbol_b: str,
    *,
    beta_window: int | None = None,
    z_entry_mult: float = 1.0,
    size_scale: float = 1.0,
    beta: float | None = None,
) -> tuple[StrategySignal | None, StrategySignal | None]:
    snap = build_spread_snapshot(
        closes_a,
        closes_b,
        z_lookback=lookback,
        beta_window=beta_window or max(lookback * 2, 40),
    )
    if snap is None or snap.zscore is None:
        return None, None

    z = snap.zscore
    used_beta = beta if beta is not None else snap.beta
    entry = zscore_entry * z_entry_mult
    strength_base = min(abs(z) / (entry * 2), 1.0) * size_scale

    if z <= -entry:
        strength = max(strength_base, 0.3 * size_scale)
        leg_a = StrategySignal(
            side=SignalSide.BUY,
            strength=strength,
            reason=(
                f"pair long spread z={z:.2f} β={used_beta:.3f} "
                f"({symbol_a}/{symbol_b})"
            ),
        )
        leg_b = StrategySignal(
            side=SignalSide.SELL,
            strength=strength,
            reason=(
                f"pair short hedge z={z:.2f} β={used_beta:.3f} "
                f"({symbol_b}/{symbol_a})"
            ),
        )
        return leg_a, leg_b

    if z >= entry:
        strength = max(strength_base, 0.3 * size_scale)
        leg_a = StrategySignal(
            side=SignalSide.SELL,
            strength=strength,
            reason=(
                f"pair short spread z={z:.2f} β={used_beta:.3f} "
                f"({symbol_a}/{symbol_b})"
            ),
        )
        leg_b = StrategySignal(
            side=SignalSide.BUY,
            strength=strength,
            reason=(
                f"pair long hedge z={z:.2f} β={used_beta:.3f} "
                f"({symbol_b}/{symbol_a})"
            ),
        )
        return leg_a, leg_b

    _ = zscore_exit  # reserved for exit management
    return None, None


def backtest_pair_returns(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    lookback: int,
    zscore_entry: float,
    *,
    beta_window: int | None = None,
) -> np.ndarray:
    """Pairs backtest using hedge spread residual returns approx ret_a - β ret_b."""
    min_len = min(len(closes_a), len(closes_b))
    bw = beta_window or max(lookback * 2, 40)
    if min_len < max(lookback, bw) + 10:
        return np.array([])

    a = closes_a[-min_len:]
    b = closes_b[-min_len:]
    ret_a = log_returns(a)
    ret_b = log_returns(b)

    strategy = np.zeros(len(ret_a))
    position = 0.0
    for idx in range(max(lookback, bw), len(ret_a)):
        snap = build_spread_snapshot(
            a[: idx + 1],
            b[: idx + 1],
            z_lookback=lookback,
            beta_window=bw,
        )
        if snap is None or snap.zscore is None:
            if snap is not None:
                strategy[idx - 1] = position * (ret_a[idx - 1] - snap.beta * ret_b[idx - 1])
            continue
        z = snap.zscore
        if z <= -zscore_entry:
            position = 1.0
        elif z >= zscore_entry:
            position = -1.0
        strategy[idx - 1] = position * (ret_a[idx - 1] - snap.beta * ret_b[idx - 1])

    return strategy
