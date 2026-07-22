from __future__ import annotations

import numpy as np

from src.backtest.engine import run_backtest
from src.backtest.strategies import build_trend_signals
from src.backtest.types import ParameterSensitivityResult
from src.core.config import AppConfig


def bonferroni_threshold(n_strategies: int, alpha: float = 0.05) -> float:
    """Multiple testing correction (Lesson 7.3)."""
    if n_strategies <= 0:
        return alpha
    return alpha / n_strategies


def trend_parameter_sensitivity(
    closes: np.ndarray,
    adx: np.ndarray,
    config: AppConfig,
    cost_pct: float,
    perturbation: float = 0.20,
    max_return_change: float = 0.30,
    timeframe: str | None = None,
) -> list[ParameterSensitivityResult]:
    """Test +/- perturbation on trend MA parameters."""
    from src.core.history import periods_per_year_for_timeframe

    tf = timeframe or config.stats.signal_timeframe
    periods = periods_per_year_for_timeframe(tf, config.trading.trading_days_per_year)
    cfg = config.strategies
    base_signals = build_trend_signals(closes, adx, config)
    base = run_backtest(
        closes,
        base_signals,
        "trend",
        cost_pct,
        config.indicators.risk_free_rate,
        periods,
        zero_means_flat=True,
    )
    base_ret = base.performance.total_return

    results: list[ParameterSensitivityResult] = []
    for param_name, base_val, low_val, high_val in [
        ("ma_short", cfg.trend_ma_short, max(2, int(cfg.trend_ma_short * (1 - perturbation))), int(cfg.trend_ma_short * (1 + perturbation))),
        ("ma_long", cfg.trend_ma_long, max(5, int(cfg.trend_ma_long * (1 - perturbation))), int(cfg.trend_ma_long * (1 + perturbation))),
    ]:
        low_ret = _trend_return_with_params(
            closes, adx, config, cost_pct, param_name, low_val, periods
        )
        high_ret = _trend_return_with_params(
            closes, adx, config, cost_pct, param_name, high_val, periods
        )
        max_change = max(
            abs(low_ret - base_ret) / max(abs(base_ret), 1e-8),
            abs(high_ret - base_ret) / max(abs(base_ret), 1e-8),
        )
        results.append(
            ParameterSensitivityResult(
                parameter=param_name,
                base_return=base_ret,
                low_return=low_ret,
                high_return=high_ret,
                max_change_pct=max_change,
                stable=max_change <= max_return_change,
            )
        )

    return results


def _trend_return_with_params(
    closes: np.ndarray,
    adx: np.ndarray,
    config: AppConfig,
    cost_pct: float,
    param_name: str,
    value: int,
    periods_per_year: float,
) -> float:
    cfg = config.strategies
    ma_short = value if param_name == "ma_short" else cfg.trend_ma_short
    ma_long = value if param_name == "ma_long" else cfg.trend_ma_long

    from src.strategies.trend_following import evaluate_trend_following

    signals = np.zeros(len(closes))
    min_bars = ma_long + 5
    current = 0.0
    sideways = float(cfg.adx_sideways_threshold)
    for idx in range(min_bars, len(closes)):
        adx_slice = adx[:idx]
        adx_val = float(adx_slice[-1]) if len(adx_slice) and not np.isnan(adx_slice[-1]) else 0.0
        if adx_val < sideways:
            current = 0.0
        else:
            sig = evaluate_trend_following(
                closes[:idx],
                adx_slice,
                ma_short,
                ma_long,
                cfg.adx_trend_threshold,
                adx_sideways_threshold=sideways,
            )
            if sig is not None:
                current = 1.0 if sig.side.value == "buy" else -1.0
        signals[idx - 1] = current

    bt = run_backtest(
        closes,
        signals,
        "trend_sensitivity",
        cost_pct,
        config.indicators.risk_free_rate,
        periods_per_year,
        zero_means_flat=True,
    )
    return bt.performance.total_return


def expected_live_return(
    backtest_annual_return: float,
    decay_factor: float,
    hidden_cost_pct: float,
) -> float:
    """Lesson 7.7: conservative live expectation."""
    return backtest_annual_return * decay_factor - hidden_cost_pct
