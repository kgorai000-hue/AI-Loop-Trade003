"""Online learning and strategy evolution (Lesson 17)."""

from src.online.alpha_decay import (
    expected_annual_return_from_ic,
    implied_monthly_decay,
    months_until_ic_below,
    projected_ic,
)
from src.online.decision import RetrainDecisionEngine
from src.online.drift import DDMDetector, calculate_psi, psi_severity, sliding_accuracy_drift
from src.online.ewm_model import ExponentialMovingModel, effective_lookback_days
from src.online.scheduler import AdaptiveUpdateScheduler
from src.online.signal_adaptation import dynamic_signal_threshold
from src.online.strategy_lifecycle import (
    build_strategy_lifecycle_states,
    inverse_vol_strategy_weights,
    sharpe_proportional_weights,
)
from src.online.types import (
    DriftSignal,
    EvolutionLevel,
    EvolutionReport,
    LifecycleStage,
    StrategyLifecycleState,
    UpdateAction,
    UpdateDecision,
)

__all__ = [
    "AdaptiveUpdateScheduler",
    "DDMDetector",
    "DriftSignal",
    "EvolutionLevel",
    "EvolutionReport",
    "ExponentialMovingModel",
    "LifecycleStage",
    "RetrainDecisionEngine",
    "StrategyLifecycleState",
    "UpdateAction",
    "UpdateDecision",
    "build_strategy_lifecycle_states",
    "calculate_psi",
    "dynamic_signal_threshold",
    "effective_lookback_days",
    "expected_annual_return_from_ic",
    "implied_monthly_decay",
    "inverse_vol_strategy_weights",
    "months_until_ic_below",
    "projected_ic",
    "sharpe_proportional_weights",
    "sliding_accuracy_drift",
]
