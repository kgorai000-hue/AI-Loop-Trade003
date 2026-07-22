from __future__ import annotations


def notional_leverage(total_notional: float, equity: float) -> float:
    """True leverage = sum(|notional|) / capital (Lesson 16.5)."""
    if equity <= 0:
        return 0.0
    return total_notional / equity


def risk_leverage(portfolio_volatility: float, benchmark_volatility: float) -> float:
    if benchmark_volatility <= 0:
        return 0.0
    return portfolio_volatility / benchmark_volatility


def portfolio_volatility(weights: dict[str, float], volatilities: dict[str, float], correlation) -> float:
    import numpy as np

    symbols = list(weights.keys())
    if not symbols:
        return 0.0
    w = np.array([weights[s] for s in symbols])
    vols = np.array([volatilities.get(s, 0.2) for s in symbols])
    cov = np.outer(vols, vols) * correlation
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)))
