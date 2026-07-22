from __future__ import annotations

import numpy as np


def equal_weights(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    share = 1.0 / len(symbols)
    return {symbol: share for symbol in symbols}


def inverse_volatility_weights(volatilities: dict[str, float]) -> dict[str, float]:
    if not volatilities:
        return {}
    inv = {symbol: 1.0 / max(vol, 1e-6) for symbol, vol in volatilities.items()}
    total = sum(inv.values())
    if total <= 0:
        return equal_weights(list(volatilities))
    return {symbol: weight / total for symbol, weight in inv.items()}


def equal_risk_contribution_weights(cov: np.ndarray, symbols: list[str], max_iter: int = 500) -> dict[str, float]:
    """ERC weights via iterative risk parity (Lesson 16.2)."""
    n = len(symbols)
    if n == 0:
        return {}
    if n == 1:
        return {symbols[0]: 1.0}

    cov = np.asarray(cov, dtype=float)
    off_diag = cov - np.diag(np.diag(cov))
    if np.allclose(off_diag, 0.0):
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        inv = 1.0 / vols
        w = inv / inv.sum()
        return {symbols[i]: float(w[i]) for i in range(n)}

    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        if port_var <= 1e-12:
            break
        port_vol = np.sqrt(port_var)
        marginal = cov @ w
        rc = w * marginal / port_vol
        target = port_vol / n
        rc = np.maximum(rc, 1e-12)
        w_new = w * (target / rc)
        w_new = np.maximum(w_new, 0.0)
        total = w_new.sum()
        if total <= 0:
            break
        w_new = w_new / total
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new

    return {symbols[i]: float(w[i]) for i in range(n)}


def cap_weights(weights: dict[str, float], max_weight: float) -> dict[str, float]:
    if not weights or max_weight <= 0:
        return weights
    capped = {symbol: min(weight, max_weight) for symbol, weight in weights.items()}
    total = sum(capped.values())
    if total <= 0:
        return equal_weights(list(weights))
    return {symbol: weight / total for symbol, weight in capped.items()}


def apply_correlation_penalty(
    weights: dict[str, float],
    correlation: np.ndarray,
    symbols: list[str],
    *,
    penalty_strength: float = 0.5,
    threshold: float = 0.75,
) -> dict[str, float]:
    """Down-weight symbols highly correlated with the rest (Lesson 16.6)."""
    if not weights or penalty_strength <= 0 or len(symbols) < 2:
        return weights

    adjusted: dict[str, float] = {}
    for idx, symbol in enumerate(symbols):
        base = weights.get(symbol, 0.0)
        others = [correlation[idx, j] for j in range(len(symbols)) if j != idx]
        avg_corr = float(np.mean(others)) if others else 0.0
        if avg_corr >= threshold:
            penalty = 1.0 - penalty_strength * (avg_corr - threshold) / max(1.0 - threshold, 1e-6)
            adjusted[symbol] = base * max(0.1, penalty)
        else:
            adjusted[symbol] = base

    total = sum(adjusted.values())
    if total <= 0:
        return weights
    return {symbol: weight / total for symbol, weight in adjusted.items()}
