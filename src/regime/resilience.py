from __future__ import annotations

from enum import IntEnum


class DegradationLevel(IntEnum):
    NORMAL = 0
    CAUTIOUS = 1
    DEFENSIVE = 2
    SAFE = 3


LEVEL_NAMES = {
    DegradationLevel.NORMAL: "normal",
    DegradationLevel.CAUTIOUS: "cautious",
    DegradationLevel.DEFENSIVE: "defensive",
    DegradationLevel.SAFE: "safe",
}


def determine_degradation_level(
    *,
    max_probability: float,
    clear_prob_threshold: float,
    uncertain_prob_threshold: float,
    health_fail: bool,
    data_quality_ok: bool,
    force_defensive: bool = False,
) -> DegradationLevel:
    """Meta Agent degradation ladder (Lesson 13.5.1)."""
    if not data_quality_ok:
        return DegradationLevel.SAFE
    if force_defensive or health_fail:
        return DegradationLevel.DEFENSIVE
    if max_probability >= clear_prob_threshold:
        return DegradationLevel.NORMAL
    if max_probability >= uncertain_prob_threshold:
        return DegradationLevel.CAUTIOUS
    return DegradationLevel.DEFENSIVE


def position_scale_multiplier(level: DegradationLevel, config_scales: dict[int, float]) -> float:
    return config_scales.get(int(level), 1.0)


def uncertain_position_scale(
    base_scale: float,
    uncertain_strategy: str,
    *,
    reduce_and_wait: float = 0.5,
    mix_scale: float = 0.7,
    worst_case_scale: float = 0.4,
) -> float:
    """Lesson 13.4.3 uncertain-state handling."""
    if uncertain_strategy == "reduce_and_wait":
        return base_scale * reduce_and_wait
    if uncertain_strategy == "strategy_mix":
        return base_scale * mix_scale
    if uncertain_strategy == "worst_case":
        return base_scale * worst_case_scale
    return base_scale * reduce_and_wait
