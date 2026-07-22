from __future__ import annotations

from dataclasses import dataclass, field

from src.regime.misjudgment import MisjudgmentPattern, diagnose_pattern
from src.regime.resilience import DegradationLevel, determine_degradation_level


@dataclass
class RegimeHealthReport:
    symbol: str
    switches_per_week: float
    avg_duration_days: float
    adx_boundary_oscillation: bool
    healthy: bool
    warnings: list[str] = field(default_factory=list)
    misjudgment_pattern: MisjudgmentPattern | None = None
    degradation_level: DegradationLevel = DegradationLevel.NORMAL


def detect_uncertain_triggers(
    *,
    max_probability: float,
    uncertain_prob_threshold: float,
    adx: float,
    adx_boundary_low: float,
    adx_boundary_high: float,
    annualized_volatility: float,
    ranging_vol_max: float,
    trending_adx_min: float,
    pending_switch: bool,
    post_switch_days: int,
    post_switch_uncertain_days: int,
) -> tuple[bool, list[str]]:
    """Lesson 13.4.2 uncertain triggers."""
    reasons: list[str] = []

    if max_probability < uncertain_prob_threshold:
        reasons.append(f"max probability {max_probability:.0%} < {uncertain_prob_threshold:.0%}")

    if adx_boundary_low <= adx <= adx_boundary_high:
        reasons.append(f"ADX {adx:.1f} in boundary band [{adx_boundary_low}, {adx_boundary_high}]")

    adx_trend = adx >= trending_adx_min
    vol_ranging = annualized_volatility <= ranging_vol_max
    if adx_trend and vol_ranging:
        reasons.append("conflicting: ADX trending but vol ranging")

    if pending_switch:
        reasons.append("pending regime switch confirmation")

    if 0 < post_switch_days <= post_switch_uncertain_days:
        reasons.append(f"within {post_switch_days}d after switch")

    return bool(reasons), reasons


def assess_regime_health(
    symbol: str,
    switch_stats: dict[str, float],
    *,
    max_switches_per_week: float,
    min_regime_duration_days: float,
    adx_oscillation: bool,
    max_probability: float,
    clear_prob_threshold: float,
    uncertain_prob_threshold: float,
    data_quality_ok: bool = True,
) -> RegimeHealthReport:
    """Lesson 13.5.2 regime agent health monitoring."""
    warnings: list[str] = []
    switches_per_week = 0.0
    avg_duration = switch_stats.get("avg_duration_days", 0.0)
    switches = switch_stats.get("switches", 0.0)
    observations = switch_stats.get("observations", 0.0)

    if observations > 0:
        span_weeks = max(switch_stats.get("avg_duration_days", 1) * max(switches, 1) / 7, 1)
        switches_per_week = switches / span_weeks

    if switches_per_week > max_switches_per_week:
        warnings.append(f"switches/week {switches_per_week:.1f} > {max_switches_per_week:.1f}")

    if avg_duration < min_regime_duration_days and switches > 0:
        warnings.append(f"avg duration {avg_duration:.1f}d < {min_regime_duration_days:.1f}d")

    if adx_oscillation:
        warnings.append("ADX oscillating near threshold")

    if max_probability < uncertain_prob_threshold:
        warnings.append(f"low regime confidence {max_probability:.0%}")

    pattern = diagnose_pattern(
        switches_per_week=switches_per_week,
        avg_duration_days=avg_duration,
        adx_oscillation=adx_oscillation,
        max_switches_per_week=max_switches_per_week,
        min_duration_days=min_regime_duration_days,
    )

    health_fail = bool(warnings)
    level = determine_degradation_level(
        max_probability=max_probability,
        clear_prob_threshold=clear_prob_threshold,
        uncertain_prob_threshold=uncertain_prob_threshold,
        health_fail=health_fail and pattern in (
            MisjudgmentPattern.OVERSENSITIVE,
            MisjudgmentPattern.BOUNDARY_OSCILLATION,
        ),
        data_quality_ok=data_quality_ok,
    )

    return RegimeHealthReport(
        symbol=symbol,
        switches_per_week=switches_per_week,
        avg_duration_days=avg_duration,
        adx_boundary_oscillation=adx_oscillation,
        healthy=not warnings,
        warnings=warnings,
        misjudgment_pattern=pattern,
        degradation_level=level,
    )
