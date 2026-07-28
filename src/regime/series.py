"""Build bar-by-bar five-state label series for regime-conditioned backtests."""

from __future__ import annotations

import numpy as np

from src.regime.five_state import FiveStateThresholds, classify_five_state
from src.regime.scores import RegimeScores, efficiency_ratio, regression_slope_t
from src.regime.states import MR_ALLOWED, TREND_ALLOWED
from src.stats.returns import log_returns
from src.stats.risk import volatility


def build_five_state_labels(
    closes: np.ndarray,
    *,
    asset_correlation: float = 0.0,
    trading_days: int = 252,
    vol_window: int = 20,
    trend_lookback: int = 40,
    er_lookback: int = 20,
    vol_hist_window: int = 120,
    thresholds: FiveStateThresholds | None = None,
    min_bars: int = 60,
) -> np.ndarray:
    """Return object array of five-state labels aligned to closes.

    Precomputes rolling volatility once (O(n)) instead of recomputing the
    full history at every bar (previous O(n²) path).
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    labels = np.array(["uncertain"] * n, dtype=object)
    previous: str | None = None
    th = thresholds or FiveStateThresholds()
    start = max(min_bars, trend_lookback, er_lookback + 1, vol_window + 5)
    if n <= start:
        return labels

    returns = log_returns(closes)
    vol_series = _precompute_rolling_vols(returns, vol_window, trading_days)

    for idx in range(start, n):
        vol_pct = _vol_percentile_at(vol_series, close_idx=idx, vol_window=vol_window, hist_window=vol_hist_window)
        slope, slope_t = regression_slope_t(closes[: idx + 1], trend_lookback)
        er = efficiency_ratio(closes[: idx + 1], er_lookback)
        trend_score = float(np.tanh(slope_t / 2.5))
        if slope < 0:
            trend_score = -abs(trend_score) if er > 0.3 else trend_score * 0.5
        scores = RegimeScores(
            trend_score=trend_score,
            vol_percentile=vol_pct,
            efficiency_ratio=er,
            slope=slope,
            slope_t=slope_t,
            asset_correlation=asset_correlation,
            spread_stress=0.0,
            reason_bits=[
                f"slope_t={slope_t:.2f}",
                f"ER={er:.2f}",
                f"vol_pctl={vol_pct:.0%}",
                f"corr={asset_correlation:.2f}",
            ],
        )
        result = classify_five_state(scores, previous=previous, thresholds=th)
        labels[idx] = result.label
        previous = result.label
    return labels


def _precompute_rolling_vols(
    returns: np.ndarray,
    vol_window: int,
    trading_days: int,
) -> np.ndarray:
    """vol_series[j] = annualized vol of returns[j-vol_window+1 : j+1]."""
    m = len(returns)
    out = np.full(m, np.nan, dtype=float)
    if m < vol_window:
        return out
    for j in range(vol_window - 1, m):
        out[j] = volatility(
            returns[j - vol_window + 1 : j + 1],
            annualize=True,
            trading_days=trading_days,
        )
    return out


def _vol_percentile_at(
    vol_series: np.ndarray,
    *,
    close_idx: int,
    vol_window: int,
    hist_window: int,
) -> float:
    """Percentile of current vol vs trailing hist at close index."""
    # closes[:close_idx+1] ↔ returns[:close_idx] → last return index = close_idx - 1
    j = close_idx - 1
    if j < vol_window - 1 or j >= len(vol_series) or np.isnan(vol_series[j]):
        return 0.5
    hist_start = max(vol_window - 1, j - hist_window + 1)
    window = vol_series[hist_start : j + 1]
    window = window[~np.isnan(window)]
    if len(window) < 5:
        return 0.5
    return float(np.mean(window <= window[-1]))


def mask_signals_for_strategy(
    signals: np.ndarray,
    labels: np.ndarray,
    strategy_name: str,
) -> np.ndarray:
    """Zero out signals outside the strategy's allowed regimes."""
    out = np.array(signals, dtype=float, copy=True)
    if strategy_name == "trend_following":
        allowed = TREND_ALLOWED
    elif strategy_name == "mean_reversion":
        allowed = MR_ALLOWED
    else:
        return out
    label_str = np.asarray([str(x) for x in labels], dtype=object)
    mask = np.array([lab not in allowed for lab in label_str], dtype=bool)
    out[mask] = 0.0
    return out
