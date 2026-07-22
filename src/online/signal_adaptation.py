from __future__ import annotations

import numpy as np


def dynamic_signal_threshold(
    signal_history: list[float],
    *,
    k: float = 1.5,
    default: float = 0.5,
) -> float:
    """Level 1 signal adaptation: mean + k * std (Lesson 17.3)."""
    if len(signal_history) < 5:
        return default
    arr = np.array(signal_history, dtype=float)
    return float(np.mean(arr) + k * np.std(arr))


def scale_signals_by_threshold(signals: list[float], threshold: float) -> list[float]:
    if threshold <= 0:
        return signals
    return [min(1.0, max(0.0, s / threshold)) for s in signals]
