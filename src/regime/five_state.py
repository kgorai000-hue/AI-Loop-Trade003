"""Classify multi-axis scores into five states with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

from src.regime.scores import RegimeScores
from src.regime.states import FiveState


@dataclass
class FiveStateThresholds:
    # Enter A (stable trend)
    enter_trend_score: float = 0.55
    enter_er_trend: float = 0.45
    enter_vol_max_for_stable: float = 0.70
    # Exit A / stay in trend family
    exit_trend_score: float = 0.35
    exit_er_trend: float = 0.30
    # High vol cut
    high_vol_enter: float = 0.70
    stress_vol: float = 0.90
    # Range
    enter_range_abs_trend: float = 0.30
    enter_er_range_max: float = 0.35
    exit_range_abs_trend: float = 0.45
    # Correlation stress
    stress_corr: float = 0.85
    # Liquidity / ATR proxy
    stress_spread: float = 0.70


@dataclass
class FiveStateResult:
    label: str
    raw_label: str
    confidence: float
    is_transition: bool
    reason: str


def classify_five_state(
    scores: RegimeScores,
    *,
    previous: str | None = None,
    thresholds: FiveStateThresholds | None = None,
) -> FiveStateResult:
    th = thresholds or FiveStateThresholds()
    raw = _raw_classify(scores, th)
    confirmed = _apply_hysteresis(raw, previous, scores, th)
    conf = _confidence(scores, confirmed)
    is_trans = confirmed == FiveState.UNCERTAIN.value or (
        previous is not None and previous != confirmed and previous != FiveState.UNCERTAIN.value
    )
    reason = f"{confirmed} (raw={raw}); " + ", ".join(scores.reason_bits[:4])
    return FiveStateResult(
        label=confirmed,
        raw_label=raw,
        confidence=conf,
        is_transition=is_trans,
        reason=reason,
    )


def _raw_classify(scores: RegimeScores, th: FiveStateThresholds) -> str:
    vol = scores.vol_percentile
    er = scores.efficiency_ratio
    ts = scores.trend_score
    abs_ts = abs(ts)

    if (
        vol >= th.stress_vol
        or scores.spread_stress >= th.stress_spread
        or (vol >= th.high_vol_enter and scores.asset_correlation >= th.stress_corr)
    ):
        return FiveState.STRESS.value

    trending = abs_ts >= th.enter_trend_score and er >= th.enter_er_trend
    ranging = abs_ts <= th.enter_range_abs_trend and er <= th.enter_er_range_max

    if trending and vol >= th.high_vol_enter:
        return FiveState.HIGH_VOL_TREND.value
    if trending and vol < th.enter_vol_max_for_stable:
        return FiveState.STABLE_TREND.value
    if trending:
        # mid vol trend → treat as high-vol trend (size reduced)
        return FiveState.HIGH_VOL_TREND.value

    if ranging and vol >= th.high_vol_enter:
        return FiveState.HIGH_VOL_CHOP.value
    if ranging:
        return FiveState.STABLE_RANGE.value

    # Gray zone: high vol without clear trend → chop; else uncertain
    if vol >= th.high_vol_enter and er < th.enter_er_trend:
        return FiveState.HIGH_VOL_CHOP.value
    return FiveState.UNCERTAIN.value


def _apply_hysteresis(
    raw: str,
    previous: str | None,
    scores: RegimeScores,
    th: FiveStateThresholds,
) -> str:
    if previous is None or previous == FiveState.UNCERTAIN.value:
        return raw

    vol = scores.vol_percentile
    er = scores.efficiency_ratio
    abs_ts = abs(scores.trend_score)

    # Stress always overrides immediately
    if raw == FiveState.STRESS.value:
        return FiveState.STRESS.value

    # Stay in trend family until exit thresholds breach
    if previous in (FiveState.STABLE_TREND.value, FiveState.HIGH_VOL_TREND.value):
        still_trend = abs_ts >= th.exit_trend_score and er >= th.exit_er_trend
        if still_trend:
            if vol >= th.high_vol_enter:
                return FiveState.HIGH_VOL_TREND.value
            return FiveState.STABLE_TREND.value
        if raw in (FiveState.STABLE_TREND.value, FiveState.HIGH_VOL_TREND.value):
            return raw
        return FiveState.UNCERTAIN.value

    if previous == FiveState.STABLE_RANGE.value:
        still_range = abs_ts <= th.exit_range_abs_trend and er <= th.enter_er_trend
        if still_range and vol < th.high_vol_enter:
            return FiveState.STABLE_RANGE.value
        if still_range and vol >= th.high_vol_enter:
            return FiveState.HIGH_VOL_CHOP.value
        if raw == FiveState.STABLE_RANGE.value:
            return raw
        return FiveState.UNCERTAIN.value

    if previous == FiveState.HIGH_VOL_CHOP.value:
        if raw == FiveState.HIGH_VOL_CHOP.value:
            return raw
        # Require clear exit into A/C only
        if raw in (FiveState.STABLE_TREND.value, FiveState.STABLE_RANGE.value):
            return raw
        if raw == FiveState.HIGH_VOL_TREND.value:
            return raw
        return FiveState.HIGH_VOL_CHOP.value

    if previous == FiveState.STRESS.value:
        if raw == FiveState.STRESS.value:
            return raw
        # Leave stress only into uncertain first (confirm elsewhere)
        return FiveState.UNCERTAIN.value

    return raw


def _confidence(scores: RegimeScores, label: str) -> float:
    er = scores.efficiency_ratio
    vol = scores.vol_percentile
    abs_ts = abs(scores.trend_score)
    if label in (FiveState.STABLE_TREND.value, FiveState.HIGH_VOL_TREND.value):
        return float(min(1.0, 0.4 + 0.3 * abs_ts + 0.3 * er))
    if label == FiveState.STABLE_RANGE.value:
        return float(min(1.0, 0.4 + 0.4 * (1.0 - er) + 0.2 * (1.0 - abs_ts)))
    if label == FiveState.HIGH_VOL_CHOP.value:
        return float(min(1.0, 0.5 + 0.5 * vol))
    if label == FiveState.STRESS.value:
        return float(min(1.0, 0.6 + 0.4 * vol))
    return 0.35
