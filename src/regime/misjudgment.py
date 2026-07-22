from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MisjudgmentPattern(str, Enum):
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    DELAYED = "delayed"
    OVERSENSITIVE = "oversensitive"
    BOUNDARY_OSCILLATION = "boundary_oscillation"


@dataclass
class MisjudgmentCost:
    direct_loss: float
    opportunity_cost: float
    switch_cost: float

    @property
    def total(self) -> float:
        return self.direct_loss + self.opportunity_cost + self.switch_cost


# Lesson 13.3.1 monthly return matrix (actual_state, strategy) -> return
RETURN_MATRIX: dict[tuple[str, str], float] = {
    ("trending", "trend"): 0.05,
    ("trending", "mean_reversion"): -0.08,
    ("trending", "defensive"): -0.03,
    ("ranging", "trend"): -0.03,
    ("ranging", "mean_reversion"): 0.03,
    ("ranging", "defensive"): 0.00,
    ("crisis", "trend"): -0.15,
    ("crisis", "mean_reversion"): -0.25,
    ("crisis", "defensive"): -0.05,
}

STATE_DISTRIBUTION = {"trending": 0.30, "ranging": 0.50, "crisis": 0.20}


def misjudgment_cost(
    direct_loss: float,
    opportunity_cost: float,
    switch_count: int,
    switch_cost_pct: float,
) -> MisjudgmentCost:
    return MisjudgmentCost(
        direct_loss=direct_loss,
        opportunity_cost=opportunity_cost,
        switch_cost=switch_count * switch_cost_pct,
    )


def lag_cost_table() -> dict[int, dict[str, float | str]]:
    """Lesson 13.1.1 paper exercise: lag days vs loss saved."""
    return {
        1: {"loss_at_confirm_pct": 0.03, "saved_pct": 0.12, "note": "stop loss early"},
        3: {"loss_at_confirm_pct": 0.09, "saved_pct": 0.06, "note": "partial save"},
        5: {"loss_at_confirm_pct": 0.15, "saved_pct": 0.00, "note": "full drawdown"},
        10: {"loss_at_confirm_pct": 0.15, "saved_pct": 0.00, "note": "may miss rebound"},
    }


def expected_monthly_return(accuracy: float) -> dict[str, float]:
    """Lesson 13.3.3: return impact at given detection accuracy."""
    correct = accuracy
    wrong = 1.0 - accuracy

    correct_return = 0.0
    for state, prob in STATE_DISTRIBUTION.items():
        strategy = {"trending": "trend", "ranging": "mean_reversion", "crisis": "defensive"}[state]
        r = RETURN_MATRIX[(state, strategy)]
        correct_return += prob * correct * r

    wrong_return = 0.0
    mismatches = [
        ("trending", "mean_reversion", 0.5),
        ("trending", "defensive", 0.5),
        ("ranging", "trend", 0.5),
        ("ranging", "defensive", 0.5),
        ("crisis", "trend", 0.5),
        ("crisis", "mean_reversion", 0.5),
    ]
    for state, wrong_strategy, share in mismatches:
        prob = STATE_DISTRIBUTION[state] * wrong * share
        wrong_return += prob * RETURN_MATRIX[(state, wrong_strategy)]

    total = correct_return + wrong_return
    perfect = sum(
        STATE_DISTRIBUTION[s] * RETURN_MATRIX[(s, {"trending": "trend", "ranging": "mean_reversion", "crisis": "defensive"}[s])]
        for s in STATE_DISTRIBUTION
    )
    reduction = 1.0 - (total / perfect) if perfect else 0.0
    return {
        "monthly_return": total,
        "perfect_monthly_return": perfect,
        "return_reduction_pct": reduction,
    }


def diagnose_pattern(
    *,
    switches_per_week: float,
    avg_duration_days: float,
    adx_oscillation: bool,
    max_switches_per_week: float,
    min_duration_days: float,
) -> MisjudgmentPattern | None:
    if adx_oscillation:
        return MisjudgmentPattern.BOUNDARY_OSCILLATION
    if switches_per_week > max_switches_per_week:
        return MisjudgmentPattern.OVERSENSITIVE
    if avg_duration_days < min_duration_days and switches_per_week > 1:
        return MisjudgmentPattern.DELAYED
    return None
