from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvolutionLevel(str, Enum):
    SIGNAL = "signal"
    MODEL = "model"
    STRATEGY = "strategy"
    ARCHITECTURE = "architecture"


class UpdateAction(str, Enum):
    RETRAIN = "retrain"
    PAUSE = "pause"
    CONTINUE = "continue"
    INVESTIGATE = "investigate"
    OBSERVE = "observe"


class LifecycleStage(str, Enum):
    INCUBATION = "incubation"
    VALIDATION = "validation"
    MATURITY = "maturity"
    DECAY = "decay"


@dataclass
class DriftSignal:
    metric: str
    value: float
    threshold: float
    detected: bool
    severity: str = "low"


@dataclass
class StrategyLifecycleState:
    strategy: str
    stage: LifecycleStage
    sharpe_proxy: float
    capital_weight: float
    note: str = ""


@dataclass
class UpdateDecision:
    action: UpdateAction
    confidence: float
    reason: str
    urgency: str = "none"
    evolution_level: EvolutionLevel = EvolutionLevel.MODEL


@dataclass
class EvolutionReport:
    enabled: bool = True
    mean_ic: float = 0.0
    ic_monthly_decay_rate: float = 0.0
    projected_ic_12m: float = 0.0
    effective_lookback_days: int = 0
    dynamic_threshold: float = 0.5
    drift_signals: list[DriftSignal] = field(default_factory=list)
    drift_detected: bool = False
    performance_drop_pct: float = 0.0
    update_decision: UpdateDecision | None = None
    strategy_states: list[StrategyLifecycleState] = field(default_factory=list)
    evolution_level: EvolutionLevel = EvolutionLevel.SIGNAL
    warnings: list[str] = field(default_factory=list)
