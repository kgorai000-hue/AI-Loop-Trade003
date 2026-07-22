from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import AppConfig
from src.features.indicators import bars_to_arrays, compute_indicators


def build_feature_frame(bars: list[dict], config: AppConfig) -> pd.DataFrame:
    """Build quant ML features with shift(1) to avoid look-ahead (Lesson 9.2)."""
    opens, highs, lows, closes, volumes = bars_to_arrays(bars)
    close = pd.Series(closes, dtype=float)
    high = pd.Series(highs, dtype=float)
    low = pd.Series(lows, dtype=float)
    volume = pd.Series(volumes, dtype=float)

    features = pd.DataFrame(index=range(len(close)))
    ret = close.pct_change()
    lookback = config.ml.lookback

    features["ret_1"] = ret
    features["ret_5"] = close.pct_change(5)
    features["ret_20"] = close.pct_change(lookback)
    features["vol_20"] = ret.rolling(lookback).std()
    vol_mean = volume.rolling(lookback).mean()
    features["vol_ratio"] = volume / vol_mean.replace(0, np.nan)
    hl_range = high.rolling(lookback).max() - low.rolling(lookback).min()
    features["price_pos"] = (close - low.rolling(lookback).min()) / hl_range.replace(0, np.nan)

    ind = compute_indicators(
        opens,
        highs,
        lows,
        closes,
        volumes,
        macd_fast=config.indicators.macd_fast,
        macd_slow=config.indicators.macd_slow,
        macd_signal=config.indicators.macd_signal,
        macd_histogram_double=config.indicators.macd_histogram_double,
        rsi_period=config.indicators.rsi_period,
        bb_period=config.indicators.bb_period,
        bb_std=config.indicators.bb_std,
        atr_period=config.indicators.atr_period,
        adx_period=config.strategies.adx_period,
    )
    features["rsi"] = ind["rsi"]
    features["macd_diff"] = ind["macd_diff"]
    features["bb_position"] = ind["bb_position"]
    atr = ind["atr"]
    features["atr_pct"] = np.where(close > 0, atr / close, 0.0)
    features["adx"] = ind["adx"]

    # Use prior bar only (T signal -> T+1 label alignment)
    return features.shift(1)


def create_forward_return_labels(
    bars: list[dict],
    horizon: int,
) -> pd.Series:
    _, _, _, closes, _ = bars_to_arrays(bars)
    close = pd.Series(closes, dtype=float)
    return close.pct_change(horizon).shift(-horizon)


def create_classification_labels(
    bars: list[dict],
    horizon: int,
    threshold: float,
) -> pd.Series:
    future_ret = create_forward_return_labels(bars, horizon)
    return (future_ret > threshold).astype(float)


def align_xy(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    valid = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid].astype(float)
    y = labels.loc[valid].astype(float)
    return X, y


def univariate_ic_screen(
    X: pd.DataFrame,
    forward_returns: pd.Series,
    min_ic: float = 0.03,
) -> list[tuple[str, float]]:
    """Rank features by |Spearman IC| (Lesson 9.2)."""
    from src.ml.ic import calculate_ic

    scores: list[tuple[str, float]] = []
    for col in X.columns:
        ic = calculate_ic(X[col].to_numpy(), forward_returns.loc[X.index].to_numpy())
        if abs(ic) >= min_ic:
            scores.append((col, ic))
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores
