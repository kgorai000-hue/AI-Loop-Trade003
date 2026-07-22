from __future__ import annotations

import math

import numpy as np


def effective_lookback_days(decay_factor: float, threshold: float = 0.1) -> int:
    """Effective lookback ~ ln(threshold) / ln(lambda) (Lesson 17.2)."""
    if decay_factor <= 0 or decay_factor >= 1:
        return 0
    return max(1, int(math.log(threshold) / math.log(decay_factor)))


class ExponentialMovingModel:
    """Online learning with exponential forgetting (Lesson 17.2)."""

    def __init__(self, decay_factor: float = 0.95) -> None:
        self.lambda_ = decay_factor
        self.weights: np.ndarray | None = None
        self.cumulative_weight = 0.0

    def update(self, features: np.ndarray, target: float, learning_rate: float = 0.01) -> tuple[float, float]:
        if self.weights is None:
            self.weights = np.zeros(features.shape[0], dtype=float)

        pred = float(np.dot(self.weights, features))
        error = target - pred
        self.weights = self.lambda_ * self.weights + learning_rate * error * features
        self.cumulative_weight = self.lambda_ * self.cumulative_weight + 1.0
        return pred, error

    def predict(self, features: np.ndarray) -> float:
        if self.weights is None:
            return 0.0
        return float(np.dot(self.weights, features))

    def get_effective_lookback(self, threshold: float = 0.1) -> int:
        return effective_lookback_days(self.lambda_, threshold)
