from __future__ import annotations

import numpy as np

from src.core.config import StrategiesConfig
from src.core.types import MarketRegime, SignalMode, StrategyKind
from src.stats.returns import log_returns
from src.stats.risk import volatility


def sustained(values: np.ndarray, threshold: float, days: int, above: bool = True) -> bool:
    tail = values[-days:]
    if len(tail) < days:
        return False
    valid = tail[~np.isnan(tail)]
    if len(valid) < days:
        return False
    if above:
        return bool(np.all(valid > threshold))
    return bool(np.all(valid < threshold))


def vol_crisis_percentile(
    returns: np.ndarray,
    lookback: int,
    percentile: float = 90.0,
    trading_days: int = 252,
) -> tuple[bool, float]:
    if len(returns) < lookback + 10:
        return False, 0.0

    rolling_vols: list[float] = []
    window = min(20, lookback)
    for idx in range(window, len(returns)):
        chunk = returns[idx - window : idx]
        rolling_vols.append(
            volatility(chunk, annualize=True, trading_days=trading_days)
        )

    if not rolling_vols:
        return False, 0.0

    current = rolling_vols[-1]
    threshold = float(np.percentile(rolling_vols, percentile))
    return current >= threshold, current


def select_strategy_kind(
    regime: MarketRegime,
    adx_series: np.ndarray,
    closes: list[float],
    strategies: StrategiesConfig,
    vol_window: int,
    trading_days: int,
    crisis_vol_threshold: float,
) -> tuple[StrategyKind, SignalMode, float, bool, bool, str]:
    """Lesson 5.6: ADX-integrated regime routing."""
    returns = log_returns(closes)
    crisis_pct, crisis_vol = vol_crisis_percentile(
        returns,
        lookback=vol_window,
        percentile=90.0,
        trading_days=trading_days,
    )

    if regime == MarketRegime.CRISIS or crisis_pct:
        return (
            StrategyKind.CRISIS_HALT,
            SignalMode.NONE,
            0.0,
            False,
            False,
            f"crisis vol {crisis_vol:.1%} >= 90th percentile or regime crisis",
        )

    trend_confirmed = sustained(
        adx_series,
        strategies.adx_trend_threshold,
        strategies.adx_confirm_days,
        above=True,
    )
    sideways_confirmed = sustained(
        adx_series,
        strategies.adx_sideways_threshold,
        strategies.adx_confirm_days,
        above=False,
    )

    if trend_confirmed:
        return (
            StrategyKind.TREND_FOLLOWING,
            SignalMode.MOMENTUM,
            1.0,
            True,
            False,
            f"ADX > {strategies.adx_trend_threshold} for {strategies.adx_confirm_days} bars",
        )

    if sideways_confirmed:
        return (
            StrategyKind.MEAN_REVERSION,
            SignalMode.MEAN_REVERSION,
            1.0,
            False,
            True,
            f"ADX < {strategies.adx_sideways_threshold} for {strategies.adx_confirm_days} bars",
        )

    return (
        StrategyKind.UNCERTAIN,
        SignalMode.MOMENTUM,
        strategies.uncertain_position_scale,
        False,
        False,
        "ADX regime unclear; reduced position scale",
    )
