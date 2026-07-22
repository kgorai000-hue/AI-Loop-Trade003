"""Build bar-by-bar five-state label series for regime-conditioned backtests."""

from __future__ import annotations

import numpy as np

from src.regime.five_state import FiveStateThresholds, classify_five_state
from src.regime.scores import compute_regime_scores
from src.regime.states import MR_ALLOWED, TREND_ALLOWED


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
    """Return object array of five-state labels aligned to closes."""
    n = len(closes)
    labels = np.array(["uncertain"] * n, dtype=object)
    previous: str | None = None
    th = thresholds or FiveStateThresholds()
    start = max(min_bars, trend_lookback, er_lookback + 1, vol_window + 5)
    for idx in range(start, n):
        scores = compute_regime_scores(
            closes[: idx + 1],
            trend_lookback=trend_lookback,
            er_lookback=er_lookback,
            vol_window=vol_window,
            vol_hist_window=min(vol_hist_window, idx),
            trading_days=trading_days,
            asset_correlation=asset_correlation,
        )
        result = classify_five_state(scores, previous=previous, thresholds=th)
        labels[idx] = result.label
        previous = result.label
    return labels


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
    for idx, label in enumerate(labels):
        if str(label) not in allowed:
            out[idx] = 0.0
    return out
