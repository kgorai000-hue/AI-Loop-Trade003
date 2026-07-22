"""Market regime detection and resilience (Lessons 12-13)."""

from src.regime.detection import (
    RegimeDetectionResult,
    adjusted_strategy_weights,
    evaluate_regime_value,
    rule_based_detect,
    select_strategy_from_weights,
    volatility_cluster_label,
)
from src.regime.health import RegimeHealthReport, assess_regime_health, detect_uncertain_triggers
from src.regime.history import RegimeHistoryEntry, RegimeHistoryStore
from src.regime.misjudgment import (
    MisjudgmentCost,
    MisjudgmentPattern,
    expected_monthly_return,
    lag_cost_table,
    misjudgment_cost,
)
from src.regime.probabilistic import ProbabilisticRegimeDetector
from src.regime.resilience import DegradationLevel, determine_degradation_level, uncertain_position_scale

__all__ = [
    "DegradationLevel",
    "MisjudgmentCost",
    "MisjudgmentPattern",
    "ProbabilisticRegimeDetector",
    "RegimeDetectionResult",
    "RegimeHealthReport",
    "RegimeHistoryEntry",
    "RegimeHistoryStore",
    "adjusted_strategy_weights",
    "assess_regime_health",
    "detect_uncertain_triggers",
    "determine_degradation_level",
    "evaluate_regime_value",
    "expected_monthly_return",
    "lag_cost_table",
    "misjudgment_cost",
    "rule_based_detect",
    "select_strategy_from_weights",
    "uncertain_position_scale",
    "volatility_cluster_label",
]
