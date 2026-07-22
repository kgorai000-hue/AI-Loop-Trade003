from __future__ import annotations

import numpy as np

from src.backtest.engine import run_backtest
from src.core.history import periods_per_year_for_timeframe
from src.stats.performance import evaluate_returns, round_trip_pnls, sharpe_ratio
from src.stats.returns import annualize_return


def test_periods_per_year_m30() -> None:
    assert periods_per_year_for_timeframe("M30", 252) == 252 * 48
    assert periods_per_year_for_timeframe("H1", 252) == 252 * 24
    assert periods_per_year_for_timeframe("D1", 252) == 252


def test_annualize_return_uses_bar_periods() -> None:
    half_year_bars = int(0.5 * 252 * 48)
    ann = annualize_return(0.10, half_year_bars, 252 * 48)
    assert 0.20 < ann < 0.22


def test_sharpe_scales_with_intraday_periods() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0001, 0.001, size=2000)
    daily = sharpe_ratio(rets, 0.0, 252)
    m30 = sharpe_ratio(rets, 0.0, 252 * 48)
    assert m30 > daily


def test_round_trip_pnls_counts_stretches_not_bars() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02], dtype=float)
    positions = np.array([1.0, 1.0, 0.0, -1.0, -1.0], dtype=float)
    pnls = round_trip_pnls(returns, positions)
    assert len(pnls) == 2
    assert abs(pnls[0] - 0.03) < 1e-12
    assert abs(pnls[1] - 0.01) < 1e-12


def test_evaluate_returns_without_trade_pnls_leaves_trades_zero() -> None:
    rets = np.array([0.01, -0.005, 0.002], dtype=float)
    perf = evaluate_returns(rets, 0.0, 252)
    assert perf.trades == 0
    assert perf.win_rate == 0.0


def test_run_backtest_round_trip_metrics() -> None:
    closes = np.array([100.0, 100.0, 101.0, 102.0, 102.0], dtype=float)
    signals = np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=float)
    bt = run_backtest(
        closes,
        signals,
        "unit",
        cost_per_trade_pct=0.0,
        risk_free_rate=0.0,
        periods_per_year=252,
        zero_means_flat=True,
    )
    assert bt.performance.trades == 1
    assert bt.performance.win_rate == 1.0
    assert bt.performance.total_return > 0
