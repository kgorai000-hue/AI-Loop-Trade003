from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stats.returns import annualize_return, log_returns


@dataclass
class PerformanceReport:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    trades: int
    win_rate: float
    profit_factor: float = 0.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    return float(abs(np.min(drawdowns)))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 252,
) -> float:
    if len(returns) < 2 or np.std(returns, ddof=1) == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * np.mean(excess) / np.std(excess, ddof=1))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 252,
) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / downside_std)


def calmar_ratio(annualized_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return annualized_return / max_dd


def profit_factor(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def round_trip_pnls(strategy_returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sum strategy returns for each continuous non-flat position stretch.

    `positions` must align with `strategy_returns` (one position per return bar).
    An open position at the end is counted as one round-trip.
    """
    if len(strategy_returns) == 0 or len(positions) != len(strategy_returns):
        return np.array([], dtype=float)

    pnls: list[float] = []
    start: int | None = None
    side = 0.0
    for idx, pos in enumerate(positions):
        pos_f = float(pos)
        if start is None:
            if pos_f != 0.0:
                start = idx
                side = pos_f
            continue
        if pos_f != side:
            pnls.append(float(np.sum(strategy_returns[start:idx])))
            if pos_f != 0.0:
                start = idx
                side = pos_f
            else:
                start = None
                side = 0.0
    if start is not None:
        pnls.append(float(np.sum(strategy_returns[start:])))
    return np.asarray(pnls, dtype=float)


def evaluate_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 252,
    *,
    trade_pnls: np.ndarray | None = None,
) -> PerformanceReport:
    if len(returns) == 0:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    ann_return = annualize_return(total_return, len(returns), periods_per_year)
    mdd = max_drawdown(equity)

    if trade_pnls is None:
        # Fallback for raw return series without position path: do not pretend
        # non-zero bars are trades — leave trade metrics empty.
        trades = 0
        win_rate = 0.0
        pf = 0.0
    else:
        trades = int(len(trade_pnls))
        if trades:
            wins = int(np.sum(trade_pnls > 0))
            win_rate = wins / trades
            pf = profit_factor(trade_pnls)
        else:
            win_rate = 0.0
            pf = 0.0

    return PerformanceReport(
        total_return=total_return,
        annualized_return=ann_return,
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=mdd,
        calmar_ratio=calmar_ratio(ann_return, mdd),
        trades=trades,
        win_rate=win_rate,
        profit_factor=pf,
    )


def backtest_scores(
    closes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.15,
) -> np.ndarray:
    """Simple backtest: long when score > threshold, short when score < -threshold."""
    returns = log_returns(closes)
    if len(returns) == 0 or len(scores) != len(closes):
        return np.array([])

    strategy_returns = np.zeros(len(returns))
    positions = np.zeros(len(closes))

    for idx in range(1, len(closes)):
        if scores[idx - 1] > threshold:
            positions[idx] = 1.0
        elif scores[idx - 1] < -threshold:
            positions[idx] = -1.0
        else:
            positions[idx] = positions[idx - 1]
        strategy_returns[idx - 1] = positions[idx] * returns[idx - 1]

    return strategy_returns
