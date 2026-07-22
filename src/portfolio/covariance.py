from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    if returns.ndim != 2 or returns.shape[0] < 2:
        n = returns.shape[1] if returns.ndim == 2 else 1
        return np.eye(n) * 0.04
    return np.cov(returns, rowvar=False)


def shrunk_covariance(returns: np.ndarray, method: str = "ledoit_wolf") -> dict[str, np.ndarray | float]:
    """Ledoit-Wolf shrinkage covariance (Lesson 16.3)."""
    if returns.ndim != 2 or returns.shape[0] < 3 or returns.shape[1] < 1:
        n = returns.shape[1] if returns.ndim == 2 and returns.shape[1] else 1
        eye = np.eye(n) * 0.04
        return {"covariance": eye, "shrinkage": 1.0, "sample_cov": eye}

    sample = sample_covariance(returns)
    if method == "sample":
        return {"covariance": sample, "shrinkage": 0.0, "sample_cov": sample}

    estimator = LedoitWolf()
    estimator.fit(returns)
    return {
        "covariance": estimator.covariance_,
        "shrinkage": float(estimator.shrinkage_),
        "sample_cov": sample,
    }


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    outer = np.outer(std, std)
    return cov / np.maximum(outer, 1e-12)
