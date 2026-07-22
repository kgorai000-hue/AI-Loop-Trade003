from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backtest.engine import run_backtest
from src.backtest.types import BacktestResult, OOSSplitResult
from src.stats.performance import evaluate_returns


@dataclass
class TemporalSplit:
    train_end: int
    val_end: int
    test_end: int


def temporal_split_indices(
    n_bars: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TemporalSplit:
    """Chronological train/val/test split (Lesson 7.2)."""
    train_end = int(n_bars * train_ratio)
    val_end = int(n_bars * (train_ratio + val_ratio))
    return TemporalSplit(train_end=train_end, val_end=val_end, test_end=n_bars)


def evaluate_oos_split(
    closes: np.ndarray,
    signals: np.ndarray,
    strategy_name: str,
    cost_pct: float,
    train_ratio: float,
    val_ratio: float,
    risk_free_rate: float,
    periods_per_year: int,
    min_oos_ratio: float,
    *,
    zero_means_flat: bool = False,
) -> tuple[OOSSplitResult, BacktestResult]:
    split = temporal_split_indices(len(closes), train_ratio, val_ratio)

    full = run_backtest(
        closes,
        signals,
        strategy_name,
        cost_pct,
        risk_free_rate,
        periods_per_year,
        zero_means_flat=zero_means_flat,
    )

    def _segment(start: int, end: int) -> tuple[float, float]:
        seg_returns = np.array(full.returns[start:end])
        if len(seg_returns) == 0:
            return 0.0, 0.0
        perf = evaluate_returns(seg_returns, risk_free_rate, periods_per_year)
        return perf.total_return, perf.sharpe_ratio

    train_ret, train_sh = _segment(0, split.train_end)
    val_ret, val_sh = _segment(split.train_end, split.val_end)
    test_ret, test_sh = _segment(split.val_end, split.test_end)

    oos_ratio = test_ret / train_ret if abs(train_ret) > 1e-8 else 0.0

    return (
        OOSSplitResult(
            train_return=train_ret,
            val_return=val_ret,
            test_return=test_ret,
            train_sharpe=train_sh,
            val_sharpe=val_sh,
            test_sharpe=test_sh,
            oos_ratio=oos_ratio,
        ),
        full,
    )
