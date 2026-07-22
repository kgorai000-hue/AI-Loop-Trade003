"""Hedge-ratio spread: S = log A - β log B with rolling OLS β."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpreadSnapshot:
    beta: float
    beta_se: float
    spread: np.ndarray  # aligned to closes length (nan-padded head)
    zscore: float | None
    spread_vol: float
    half_life: float | None
    phi: float | None
    trend_slope: float
    zero_cross_rate: float


def rolling_ols_beta(
    log_a: np.ndarray,
    log_b: np.ndarray,
    window: int,
) -> tuple[float, float]:
    """OLS of log_a ~ alpha + beta * log_b on the trailing window. Returns (beta, se)."""
    if len(log_a) < window or len(log_b) < window:
        return 1.0, 0.0
    y = np.asarray(log_a[-window:], dtype=float)
    x = np.asarray(log_b[-window:], dtype=float)
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(x)):
        return 1.0, 0.0
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x_c, x_c))
    if denom <= 1e-18:
        return 1.0, 0.0
    beta = float(np.dot(x_c, y_c) / denom)
    resid = y_c - beta * x_c
    dof = max(len(y) - 2, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    se = float(np.sqrt(sigma2 / denom)) if denom > 0 else 0.0
    return beta, se


def hedge_spread_series(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    beta: float,
) -> np.ndarray:
    a = np.asarray(closes_a, dtype=float)
    b = np.asarray(closes_b, dtype=float)
    n = min(len(a), len(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.log(a[-n:])
        lb = np.log(b[-n:])
    return la - beta * lb


def estimate_half_life(spread: np.ndarray, lookback: int) -> tuple[float | None, float | None]:
    """AR(1) half-life on trailing spread. Returns (half_life_bars, phi)."""
    if len(spread) < lookback + 2:
        return None, None
    s = np.asarray(spread[-lookback:], dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return None, None
    y = s[1:]
    x = s[:-1]
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x_c, x_c))
    if denom <= 1e-18:
        return None, None
    phi = float(np.dot(x_c, y_c) / denom)
    if abs(phi) >= 0.999 or phi <= 0:
        return None, phi
    half = float(np.log(2.0) / -np.log(phi))
    if not np.isfinite(half) or half <= 0:
        return None, phi
    return half, phi


def spread_trend_slope(spread: np.ndarray, lookback: int) -> float:
    if len(spread) < lookback:
        return 0.0
    y = np.asarray(spread[-lookback:], dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return float(np.dot(x, y_c) / denom)


def zero_cross_rate(spread: np.ndarray, lookback: int) -> float:
    if len(spread) < lookback:
        return 0.0
    s = np.asarray(spread[-lookback:], dtype=float)
    s = s - np.nanmean(s)
    signs = np.sign(s)
    signs = signs[np.isfinite(signs)]
    if len(signs) < 3:
        return 0.0
    crosses = int(np.sum(signs[1:] * signs[:-1] < 0))
    return crosses / max(len(signs) - 1, 1)


def build_spread_snapshot(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    *,
    z_lookback: int,
    beta_window: int,
    half_life_window: int | None = None,
) -> SpreadSnapshot | None:
    min_len = min(len(closes_a), len(closes_b))
    need = max(z_lookback, beta_window) + 5
    if min_len < need:
        return None

    a = np.asarray(closes_a[-min_len:], dtype=float)
    b = np.asarray(closes_b[-min_len:], dtype=float)
    if np.any(a <= 0) or np.any(b <= 0):
        return None

    log_a = np.log(a)
    log_b = np.log(b)
    beta, beta_se = rolling_ols_beta(log_a, log_b, beta_window)
    spread = hedge_spread_series(a, b, beta)

    window = spread[-z_lookback:]
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    if std <= 1e-12:
        z = None
    else:
        z = float((spread[-1] - mean) / std)

    hl_win = half_life_window or max(z_lookback * 2, 40)
    half_life, phi = estimate_half_life(spread, hl_win)
    trend = spread_trend_slope(spread, z_lookback)
    zcr = zero_cross_rate(spread, z_lookback)

    return SpreadSnapshot(
        beta=beta,
        beta_se=beta_se,
        spread=spread,
        zscore=z,
        spread_vol=std if std > 0 else 0.0,
        half_life=half_life,
        phi=phi,
        trend_slope=trend,
        zero_cross_rate=zcr,
    )


def beta_drift_ratio(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    *,
    short_window: int,
    long_window: int,
) -> float:
    """|beta_short - beta_long| / max(|beta_long|, eps)."""
    min_len = min(len(closes_a), len(closes_b))
    if min_len < long_window + 2:
        return 0.0
    a = np.asarray(closes_a[-min_len:], dtype=float)
    b = np.asarray(closes_b[-min_len:], dtype=float)
    if np.any(a <= 0) or np.any(b <= 0):
        return 0.0
    la, lb = np.log(a), np.log(b)
    beta_s, _ = rolling_ols_beta(la, lb, short_window)
    beta_l, _ = rolling_ols_beta(la, lb, long_window)
    return abs(beta_s - beta_l) / max(abs(beta_l), 1e-6)
