from __future__ import annotations

import numpy as np

from src.backtest.types import BacktestResult, TradeRecord
from src.stats.performance import evaluate_returns, round_trip_pnls
from src.stats.returns import log_returns


def signals_to_positions(signals: np.ndarray, *, zero_means_flat: bool = False) -> np.ndarray:
    """Map {-1,0,1} signals to positions with T+1 execution (Lesson 7.1)."""
    positions = np.zeros(len(signals))
    current = 0.0
    for idx in range(len(signals)):
        if zero_means_flat:
            current = float(signals[idx])
        elif signals[idx] != 0:
            current = float(signals[idx])
        positions[idx] = current
    return positions


def run_backtest(
    closes: np.ndarray,
    signals: np.ndarray,
    strategy_name: str,
    cost_per_trade_pct: float = 0.0,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 252,
    *,
    zero_means_flat: bool = False,
) -> BacktestResult:
    """
    Backtest with signal at bar T, execution at T+1 (no look-ahead).

    signals[i] uses data through bar i; PnL applies to return from bar i to i+1.

    By default signal 0 means "hold previous position".
    With zero_means_flat=True, signal 0 means flatten (used by trend+ADX sideways).

    Performance.trades / win_rate / profit_factor are round-trip based
    (continuous non-flat position stretches), not non-zero bar counts.
    """
    if len(closes) < 2 or len(signals) != len(closes):
        return _empty_result(strategy_name, cost_per_trade_pct, risk_free_rate, periods_per_year)

    bar_returns = log_returns(closes)
    n = len(bar_returns)
    strategy_returns = np.zeros(n)
    held_positions = np.zeros(n)
    trades: list[TradeRecord] = []
    total_cost = 0.0

    prev_position = 0.0
    for idx in range(n):
        if zero_means_flat:
            position = float(signals[idx])
        else:
            position = float(signals[idx]) if signals[idx] != 0 else prev_position
        held_positions[idx] = position
        strategy_returns[idx] = position * bar_returns[idx]

        if position != prev_position and (position != 0.0 or prev_position != 0.0):
            cost = cost_per_trade_pct / 100.0
            strategy_returns[idx] -= cost
            total_cost += cost
            trades.append(
                TradeRecord(
                    signal_bar=idx,
                    execution_bar=idx + 1,
                    side=position,
                    cost_pct=cost_per_trade_pct,
                )
            )
        prev_position = position

    if trades:
        for trade in trades:
            if trade.signal_bar >= trade.execution_bar:
                raise ValueError(
                    f"Look-ahead detected: signal_bar {trade.signal_bar} >= execution_bar {trade.execution_bar}"
                )

    trade_pnls = round_trip_pnls(strategy_returns, held_positions)
    perf = evaluate_returns(
        strategy_returns,
        risk_free_rate,
        periods_per_year,
        trade_pnls=trade_pnls,
    )
    positions = signals_to_positions(signals, zero_means_flat=zero_means_flat)

    return BacktestResult(
        strategy_name=strategy_name,
        returns=strategy_returns.tolist(),
        positions=positions.tolist(),
        trades=trades,
        performance=perf,
        cost_per_trade_pct=cost_per_trade_pct,
        total_cost_pct=total_cost * 100,
    )


def run_backtest_from_scores(
    closes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    strategy_name: str,
    cost_per_trade_pct: float = 0.0,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 252,
) -> BacktestResult:
    signals = np.zeros(len(closes))
    for idx in range(len(scores)):
        if scores[idx] > threshold:
            signals[idx] = 1.0
        elif scores[idx] < -threshold:
            signals[idx] = -1.0
    return run_backtest(
        closes,
        signals,
        strategy_name,
        cost_per_trade_pct,
        risk_free_rate,
        periods_per_year,
    )


def _empty_result(
    strategy_name: str,
    cost_per_trade_pct: float,
    risk_free_rate: float,
    periods_per_year: float,
) -> BacktestResult:
    perf = evaluate_returns(
        np.array([]),
        risk_free_rate,
        periods_per_year,
        trade_pnls=np.array([]),
    )
    return BacktestResult(
        strategy_name=strategy_name,
        returns=[],
        positions=[],
        trades=[],
        performance=perf,
        cost_per_trade_pct=cost_per_trade_pct,
        total_cost_pct=0.0,
    )
