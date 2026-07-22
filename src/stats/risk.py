from __future__ import annotations

import numpy as np

from src.stats.returns import annualize_volatility, log_returns


def volatility(returns: np.ndarray, annualize: bool = True, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    daily_vol = float(np.std(returns, ddof=1))
    if annualize:
        return annualize_volatility(daily_vol, trading_days)
    return daily_vol


def skewness(returns: np.ndarray) -> float:
    if len(returns) < 3:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((returns - mean) / std) ** 3))


def excess_kurtosis(returns: np.ndarray) -> float:
    if len(returns) < 4:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((returns - mean) / std) ** 4) - 3.0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    length = min(len(a), len(b))
    a = a[-length:]
    b = b[-length:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlation_matrix(
    series_map: dict[str, list[float]],
    method: str = "log",
) -> tuple[list[str], np.ndarray]:
    symbols = sorted(series_map.keys())
    returns_list: list[np.ndarray] = []

    for symbol in symbols:
        prices = series_map[symbol]
        rets = log_returns(prices) if method == "log" else np.asarray(prices)
        returns_list.append(rets)

    min_len = min(len(r) for r in returns_list)
    if min_len < 2:
        return symbols, np.eye(len(symbols))

    aligned = np.column_stack([r[-min_len:] for r in returns_list])
    return symbols, np.corrcoef(aligned, rowvar=True)


def tail_warning(skew: float, kurt: float) -> str:
    notes: list[str] = []
    if skew < -0.5:
        notes.append("negative skew (crash risk)")
    if kurt > 0:
        notes.append("fat tails (normal assumption unsafe)")
    return "; ".join(notes) if notes else "within typical equity-like ranges"
