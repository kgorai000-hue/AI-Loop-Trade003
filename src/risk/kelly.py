from __future__ import annotations

import math


def full_kelly(win_rate: float, reward_risk_ratio: float) -> float:
    """Kelly fraction f* = (p*b - q) / b where b = reward/risk."""
    if reward_risk_ratio <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    b = reward_risk_ratio
    return max(0.0, (p * b - q) / b)


def _wilson_interval(
    wins: int,
    losses: int,
    confidence: float = 0.90,
) -> tuple[float, float, float]:
    n = wins + losses
    if n <= 0:
        return 0.5, 0.0, 1.0
    z = 1.645 if confidence >= 0.90 else 1.96
    p = wins / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def bayesian_kelly(
    wins: int,
    losses: int,
    avg_win: float,
    avg_loss: float,
    *,
    confidence: float = 0.90,
    apply_half_kelly: bool = True,
) -> dict[str, float | tuple[float, float] | int | str]:
    """
    Conservative Kelly using Wilson interval lower bound for win rate (Lesson 15.2).
    No scipy dependency.
    """
    n = wins + losses
    p_mean, p_lower, p_upper = _wilson_interval(wins, losses, confidence)
    odds = avg_win / max(avg_loss, 1e-9)

    kelly_mean = full_kelly(p_mean, odds)
    kelly_lower = full_kelly(p_lower, odds)
    kelly_upper = full_kelly(p_upper, odds)
    kelly_conservative = max(0.0, kelly_lower)
    recommendation = kelly_conservative / 2 if apply_half_kelly else kelly_conservative

    return {
        "p_estimate": p_mean,
        "p_interval": (p_lower, p_upper),
        "kelly_mean": kelly_mean,
        "kelly_conservative": kelly_conservative,
        "kelly_interval": (kelly_lower, kelly_upper),
        "sample_size": n,
        "recommendation": recommendation,
    }


def kelly_sample_discount(trade_count: int) -> float:
    """Kelly multiplier by sample size (Lesson 15.2 table)."""
    if trade_count < 30:
        return 0.0
    if trade_count < 100:
        return 0.25
    if trade_count < 300:
        return 0.5
    if trade_count < 1000:
        return 0.7
    return 0.8
