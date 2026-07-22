from __future__ import annotations

import numpy as np

from src.backtest.engine import run_backtest
from src.backtest.types import WalkForwardRound


def walk_forward_validation(
    closes: np.ndarray,
    signals: np.ndarray,
    strategy_name: str,
    train_window: int,
    test_window: int,
    step: int,
    cost_pct: float,
    risk_free_rate: float,
    periods_per_year: int,
    *,
    zero_means_flat: bool = False,
) -> list[WalkForwardRound]:
    """Rolling walk-forward validation (Lesson 7.5)."""
    results: list[WalkForwardRound] = []
    n = len(closes)
    round_idx = 0

    for start in range(0, n - train_window - test_window + 1, step):
        train_end = start + train_window
        test_end = train_end + test_window
        if test_end > n:
            break

        train_sig = signals[start:train_end]
        test_sig = signals[train_end:test_end]
        train_closes = closes[start : train_end + 1]
        test_closes = closes[train_end : test_end + 1]

        train_bt = run_backtest(
            train_closes,
            train_sig,
            strategy_name,
            cost_pct,
            risk_free_rate,
            periods_per_year,
            zero_means_flat=zero_means_flat,
        )
        test_bt = run_backtest(
            test_closes,
            test_sig,
            strategy_name,
            cost_pct,
            risk_free_rate,
            periods_per_year,
            zero_means_flat=zero_means_flat,
        )

        results.append(
            WalkForwardRound(
                round_index=round_idx,
                train_start=start,
                train_end=train_end - 1,
                test_start=train_end,
                test_end=test_end - 1,
                train_return=train_bt.performance.total_return,
                test_return=test_bt.performance.total_return,
                train_sharpe=train_bt.performance.sharpe_ratio,
                test_sharpe=test_bt.performance.sharpe_ratio,
            )
        )
        round_idx += 1

    return results


def summarize_walk_forward(rounds: list[WalkForwardRound]) -> dict[str, float]:
    if not rounds:
        return {"rounds": 0, "avg_test_return": 0.0, "avg_test_sharpe": 0.0, "positive_rounds_pct": 0.0}

    test_returns = [r.test_return for r in rounds]
    test_sharpes = [r.test_sharpe for r in rounds]
    return {
        "rounds": len(rounds),
        "avg_test_return": float(np.mean(test_returns)),
        "avg_test_sharpe": float(np.mean(test_sharpes)),
        "positive_rounds_pct": float(np.mean([1.0 if r > 0 else 0.0 for r in test_returns])),
    }
