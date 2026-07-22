from __future__ import annotations

import math

import numpy as np


def simple_returns(prices: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        return np.array([])
    return np.diff(arr) / arr[:-1]


def log_returns(prices: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        return np.array([])
    arr = arr[arr > 0]
    if len(arr) < 2:
        return np.array([])
    return np.diff(np.log(arr))


def cumulative_return(returns: np.ndarray, method: str = "log") -> float:
    if len(returns) == 0:
        return 0.0
    if method == "simple":
        return float(np.prod(1.0 + returns) - 1.0)
    return float(np.sum(returns))


def annualize_return(
    total_return: float,
    n_periods: int,
    periods_per_year: float = 252,
) -> float:
    """Annualize a total return earned over ``n_periods`` bars.

    ``periods_per_year`` must match the bar frequency
    (e.g. M30 -> 252 * 48).
    """
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualize_volatility(
    period_volatility: float,
    periods_per_year: float = 252,
) -> float:
    return period_volatility * math.sqrt(periods_per_year)
