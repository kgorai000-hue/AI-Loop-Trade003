from __future__ import annotations

import numpy as np

from src.core.config import AppConfig
from src.core.types import MarketRegime
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays, latest_snapshot
from src.strategies.mean_reversion import evaluate_mean_reversion
from src.strategies.trend_following import evaluate_trend_following


def build_trend_signals(closes: np.ndarray, adx: np.ndarray, config: AppConfig) -> np.ndarray:
    """
    Build trend positions using ADX dual thresholds.

    - ADX >= trend_threshold: take MA direction (±1)
    - ADX < sideways_threshold: flatten (0)
    - between: hold previous position
    """
    cfg = config.strategies
    signals = np.zeros(len(closes))
    min_bars = cfg.trend_ma_long + 5
    current = 0.0
    sideways = float(cfg.adx_sideways_threshold)

    for idx in range(min_bars, len(closes)):
        adx_slice = adx[:idx]
        adx_val = float(adx_slice[-1]) if len(adx_slice) and not np.isnan(adx_slice[-1]) else 0.0

        if adx_val < sideways:
            current = 0.0
        else:
            sig = evaluate_trend_following(
                closes[:idx],
                adx_slice,
                cfg.trend_ma_short,
                cfg.trend_ma_long,
                cfg.adx_trend_threshold,
                adx_sideways_threshold=sideways,
            )
            if sig is not None:
                current = 1.0 if sig.side.value == "buy" else -1.0
            # else: between sideways and trend -> hold current
        signals[idx - 1] = current

    return signals


def build_mean_reversion_signals(bars: list[dict], config: AppConfig) -> np.ndarray:
    engine = FeatureEngine(config)
    _, _, _, closes, _ = bars_to_arrays(bars)
    signals = np.zeros(len(closes))
    cfg = config.strategies
    min_bars = config.stats.min_bars

    for idx in range(min_bars, len(bars)):
        subset = bars[:idx]
        opens, highs, lows, c, volumes = bars_to_arrays(subset)
        snap = latest_snapshot(
            opens,
            highs,
            lows,
            c,
            volumes,
            **engine._indicator_kwargs(),
        )
        if snap is None:
            continue
        sig = evaluate_mean_reversion(
            snap,
            cfg.mr_rsi_oversold,
            cfg.mr_rsi_overbought,
            cfg.mr_bb_entry_low,
            cfg.mr_bb_entry_high,
        )
        if sig is not None:
            signals[idx - 1] = 1.0 if sig.side.value == "buy" else -1.0

    return signals


def build_feature_score_signals(
    bars: list[dict],
    config: AppConfig,
    regime: MarketRegime | None = None,
) -> np.ndarray:
    engine = FeatureEngine(config)
    _, _, _, closes, _ = bars_to_arrays(bars)
    signals = np.zeros(len(closes))
    min_bars = config.stats.min_bars
    threshold = config.indicators.signal_score_threshold

    for idx in range(min_bars, len(bars)):
        features = engine.build_from_bars("SYM", bars[:idx], "TF", regime)
        if features is None:
            continue
        score = features.score
        if score > threshold:
            signals[idx - 1] = 1.0
        elif score < -threshold:
            signals[idx - 1] = -1.0

    return signals
