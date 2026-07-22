from __future__ import annotations

import numpy as np


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index (Lesson 17.5)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
    actual_pct = (actual_counts + 1) / (len(actual) + n_bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_severity(psi: float) -> str:
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "medium"
    return "low"


class DDMDetector:
    """Drift Detection Method on error rate (Lesson 17.4)."""

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._min_error = float("inf")
        self._mean = 0.0
        self._variance = 0.0
        self._count = 0

    def update(self, error: float) -> bool:
        self._count += 1
        delta = error - self._mean
        self._mean += delta / self._count
        self._variance += delta * (error - self._mean)

        if self._count < 2:
            return False

        std = max(np.sqrt(self._variance / (self._count - 1)), 1e-9)
        if error < self._min_error:
            self._min_error = error

        p = (error - self._min_error) / std
        return p >= self.drift_level

    def reset(self) -> None:
        self._min_error = float("inf")
        self._mean = 0.0
        self._variance = 0.0
        self._count = 0


def sliding_accuracy_drift(
    accuracies: list[float],
    *,
    window: int = 5,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Simple sliding accuracy drift check (Lesson 17.4 paper exercise)."""
    if len(accuracies) < window:
        return False, float(np.mean(accuracies)) if accuracies else 0.0
    avg = float(np.mean(accuracies[-window:]))
    return avg < threshold, avg
