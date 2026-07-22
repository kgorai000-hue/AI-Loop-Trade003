"""Multi-axis regime scores from OHLCV (direction, vol, efficiency, liquidity proxy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stats.returns import log_returns
from src.stats.risk import volatility


@dataclass
class RegimeScores:
    trend_score: float  # [-1, 1] approx
    vol_percentile: float  # [0, 1]
    efficiency_ratio: float  # [0, 1]
    slope: float
    slope_t: float
    asset_correlation: float
    spread_stress: float  # 0 normal, 1 stressed (proxy)
    reason_bits: list[str]


def efficiency_ratio(closes: np.ndarray, lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    window = closes[-(lookback + 1) :]
    net = abs(float(window[-1] - window[0]))
    path = float(np.sum(np.abs(np.diff(window))))
    if path <= 1e-12:
        return 0.0
    return float(np.clip(net / path, 0.0, 1.0))


def regression_slope_t(closes: np.ndarray, lookback: int) -> tuple[float, float]:
    if len(closes) < lookback:
        return 0.0, 0.0
    y = np.log(np.asarray(closes[-lookback:], dtype=float))
    if np.any(~np.isfinite(y)):
        return 0.0, 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0, 0.0
    slope = float(np.dot(x, y_c) / denom)
    resid = y_c - slope * x
    dof = max(len(y) - 2, 1)
    se = float(np.sqrt(np.sum(resid**2) / dof / denom))
    t_stat = slope / se if se > 1e-12 else 0.0
    return slope, float(t_stat)


def rolling_vol_percentile(
    returns: np.ndarray,
    vol_window: int,
    hist_window: int,
    trading_days: int,
) -> float:
    if len(returns) < vol_window + 5:
        return 0.5
    vols: list[float] = []
    start = max(vol_window, len(returns) - hist_window)
    for idx in range(start, len(returns) + 1):
        chunk = returns[idx - vol_window : idx]
        if len(chunk) < vol_window:
            continue
        vols.append(float(volatility(chunk, annualize=True, trading_days=trading_days)))
    if len(vols) < 5:
        return 0.5
    current = vols[-1]
    return float(np.mean(np.asarray(vols) <= current))


def compute_regime_scores(
    closes: np.ndarray,
    *,
    trend_lookback: int = 40,
    er_lookback: int = 20,
    vol_window: int = 20,
    vol_hist_window: int = 120,
    trading_days: int = 252,
    asset_correlation: float = 0.0,
    atr_pct: float | None = None,
    atr_pct_median: float | None = None,
) -> RegimeScores:
    closes = np.asarray(closes, dtype=float)
    returns = log_returns(closes.tolist())
    slope, slope_t = regression_slope_t(closes, trend_lookback)
    er = efficiency_ratio(closes, er_lookback)
    vol_pct = rolling_vol_percentile(returns, vol_window, vol_hist_window, trading_days)

    # Map slope_t into [-1, 1] soft score
    trend_score = float(np.tanh(slope_t / 2.5))
    # Blend with ER sign of slope
    if slope < 0:
        trend_score = -abs(trend_score) if er > 0.3 else trend_score * 0.5

    spread_stress = 0.0
    if atr_pct is not None and atr_pct_median is not None and atr_pct_median > 0:
        ratio = atr_pct / atr_pct_median
        spread_stress = float(np.clip((ratio - 1.5) / 1.5, 0.0, 1.0))

    bits: list[str] = [
        f"slope_t={slope_t:.2f}",
        f"ER={er:.2f}",
        f"vol_pctl={vol_pct:.0%}",
        f"corr={asset_correlation:.2f}",
    ]
    return RegimeScores(
        trend_score=trend_score,
        vol_percentile=vol_pct,
        efficiency_ratio=er,
        slope=slope,
        slope_t=slope_t,
        asset_correlation=asset_correlation,
        spread_stress=spread_stress,
        reason_bits=bits,
    )
