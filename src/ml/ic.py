from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def calculate_ic(
    signals: np.ndarray,
    returns: np.ndarray,
    method: str = "spearman",
) -> float:
    """Information Coefficient between signals and forward returns (Lesson 9.5)."""
    if len(signals) != len(returns):
        raise ValueError("signals and returns must have same length")
    if len(signals) < 2:
        return 0.0

    mask = ~(np.isnan(signals) | np.isnan(returns))
    sig = signals[mask]
    ret = returns[mask]
    if len(sig) < 2:
        return 0.0

    if method == "spearman":
        ic, _ = spearmanr(sig, ret)
    else:
        ic = float(np.corrcoef(sig, ret)[0, 1])

    return float(ic) if ic is not None and not np.isnan(ic) else 0.0


def rolling_ic(
    signals: np.ndarray,
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    n = min(len(signals), len(returns))
    out = np.full(n, np.nan)
    for idx in range(window, n):
        out[idx] = calculate_ic(signals[idx - window : idx], returns[idx - window : idx])
    return out


def calculate_ir(ic_series: np.ndarray) -> float:
    """Information Ratio = mean(IC) / std(IC)."""
    clean = ic_series[~np.isnan(ic_series)]
    if len(clean) < 2 or np.std(clean, ddof=1) == 0:
        return 0.0
    return float(np.mean(clean) / np.std(clean, ddof=1))


def long_short_spread(
    signals: np.ndarray,
    returns: np.ndarray,
    quantile: float = 0.2,
) -> float:
    """Top minus bottom quantile return spread."""
    if len(signals) < 10:
        return 0.0
    q_high = np.quantile(signals, 1 - quantile)
    q_low = np.quantile(signals, quantile)
    long_mask = signals >= q_high
    short_mask = signals <= q_low
    if not long_mask.any() or not short_mask.any():
        return 0.0
    return float(np.mean(returns[long_mask]) - np.mean(returns[short_mask]))


def detect_ic_decay(
    rolling_values: np.ndarray,
    recent_window: int = 20,
    historical_window: int = 40,
    decay_ratio: float = 0.5,
) -> tuple[bool, float, float]:
    clean = rolling_values[~np.isnan(rolling_values)]
    if len(clean) < recent_window + 5:
        return False, 0.0, 0.0
    recent = float(np.mean(clean[-recent_window:]))
    hist_end = max(len(clean) - recent_window, historical_window)
    hist_start = max(0, hist_end - historical_window)
    historical = float(np.mean(clean[hist_start:hist_end])) if hist_end > hist_start else recent
    decayed = historical > 0 and recent < historical * decay_ratio
    return decayed, recent, historical
