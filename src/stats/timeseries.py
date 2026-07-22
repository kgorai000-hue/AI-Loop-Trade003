from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller

from src.core.types import MarketRegime, SignalMode
from src.stats.risk import volatility


def autocorrelation(series: np.ndarray, lag: int = 1) -> float:
    if len(series) <= lag:
        return 0.0
    x = series[lag:]
    y = series[:-lag]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def adf_stationarity(series: np.ndarray | list[float]) -> dict[str, float | bool]:
    clean = np.asarray(series, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) < 10:
        return {"adf_stat": 0.0, "pvalue": 1.0, "stationary": False}

    result = adfuller(clean, autolag="AIC")
    pvalue = float(result[1])
    return {
        "adf_stat": float(result[0]),
        "pvalue": pvalue,
        "stationary": pvalue < 0.05,
    }


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    if len(returns) < window:
        return np.array([])
    vols = []
    for idx in range(window, len(returns) + 1):
        vols.append(float(np.std(returns[idx - window : idx], ddof=1)))
    return np.asarray(vols)


def volatility_autocorrelation(returns: np.ndarray, window: int = 20) -> float:
    vols = rolling_volatility(returns, window)
    if len(vols) < 2:
        return 0.0
    return autocorrelation(vols, lag=1)


def detect_regime(
    returns: np.ndarray,
    vol_crisis: float = 0.40,
    vol_bull_max: float = 0.20,
    lookback: int = 20,
    trading_days: int = 252,
) -> tuple[MarketRegime, float, float]:
    if len(returns) < lookback:
        return MarketRegime.SIDEWAYS, 0.0, 0.0

    recent = returns[-lookback:]
    ann_vol = volatility(recent, annualize=True, trading_days=trading_days)
    recent_return = float(np.sum(recent))

    if ann_vol >= vol_crisis:
        return MarketRegime.CRISIS, ann_vol, recent_return

    if ann_vol <= vol_bull_max and recent_return > 0:
        return MarketRegime.BULL, ann_vol, recent_return

    return MarketRegime.SIDEWAYS, ann_vol, recent_return


def regime_to_signal_mode(regime: MarketRegime) -> SignalMode:
    if regime == MarketRegime.CRISIS:
        return SignalMode.NONE
    if regime == MarketRegime.BULL:
        return SignalMode.MOMENTUM
    return SignalMode.MEAN_REVERSION


def classify_signal_mode(
    autocorr_lag1: float,
    momentum_threshold: float,
    mean_reversion_threshold: float,
) -> SignalMode:
    if autocorr_lag1 >= momentum_threshold:
        return SignalMode.MOMENTUM
    if autocorr_lag1 <= mean_reversion_threshold:
        return SignalMode.MEAN_REVERSION
    return SignalMode.NONE
