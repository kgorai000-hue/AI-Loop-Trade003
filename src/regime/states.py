"""Five-state regime labels and strategy routing (hard cutover)."""

from __future__ import annotations

from enum import Enum

from src.core.types import MarketRegime, SignalMode, StrategyKind


class FiveState(str, Enum):
    """Professional multi-axis regimes for strategy control."""

    STABLE_TREND = "stable_trend"  # A
    HIGH_VOL_TREND = "high_vol_trend"  # B
    STABLE_RANGE = "stable_range"  # C
    HIGH_VOL_CHOP = "high_vol_chop"  # D
    STRESS = "stress"  # E
    UNCERTAIN = "uncertain"


# Backtest / optimize masks
TREND_ALLOWED = frozenset({FiveState.STABLE_TREND.value, FiveState.HIGH_VOL_TREND.value})
MR_ALLOWED = frozenset({FiveState.STABLE_RANGE.value})
PAIR_ALLOWED = frozenset(
    {
        FiveState.STABLE_TREND.value,
        FiveState.HIGH_VOL_TREND.value,
        FiveState.STABLE_RANGE.value,
    }
)
HALT_STATES = frozenset(
    {FiveState.HIGH_VOL_CHOP.value, FiveState.STRESS.value, FiveState.UNCERTAIN.value}
)


def five_state_to_market_regime(label: str) -> MarketRegime:
    if label in (FiveState.STABLE_TREND.value, FiveState.HIGH_VOL_TREND.value):
        return MarketRegime.BULL
    if label == FiveState.STRESS.value:
        return MarketRegime.CRISIS
    return MarketRegime.SIDEWAYS


def strategy_for_five_state(label: str) -> StrategyKind:
    if label in TREND_ALLOWED:
        return StrategyKind.TREND_FOLLOWING
    if label in MR_ALLOWED:
        return StrategyKind.MEAN_REVERSION
    if label == FiveState.STRESS.value:
        return StrategyKind.CRISIS_HALT
    return StrategyKind.CRISIS_HALT  # D + uncertain → no new entries


def mode_for_five_state(label: str) -> SignalMode:
    if label in TREND_ALLOWED:
        return SignalMode.MOMENTUM
    if label in MR_ALLOWED:
        return SignalMode.MEAN_REVERSION
    return SignalMode.NONE


def position_scale_for_five_state(
    label: str,
    *,
    high_vol_trend_scale: float = 0.5,
) -> float:
    if label == FiveState.STABLE_TREND.value:
        return 1.0
    if label == FiveState.HIGH_VOL_TREND.value:
        return float(high_vol_trend_scale)
    if label == FiveState.STABLE_RANGE.value:
        return 1.0
    return 0.0


def strategy_weights_for_five_state(label: str) -> dict[str, float]:
    if label in TREND_ALLOWED:
        return {"trend": 1.0, "mean_reversion": 0.0, "defensive": 0.0}
    if label in MR_ALLOWED:
        return {"trend": 0.0, "mean_reversion": 1.0, "defensive": 0.0}
    return {"trend": 0.0, "mean_reversion": 0.0, "defensive": 1.0}
