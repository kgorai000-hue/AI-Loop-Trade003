from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DrawdownLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    STOP = "stop"
    CIRCUIT = "circuit"


class DrawdownAction(str, Enum):
    NORMAL = "normal"
    REDUCE_RISK = "reduce_risk"
    STOP_NEW_POSITIONS = "stop_new_positions"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class DrawdownState:
    drawdown_pct: float
    level: DrawdownLevel
    action: DrawdownAction
    position_scale: float
    new_positions_allowed: bool
    circuit_breaker_active: bool
    message: str = ""


@dataclass
class RiskCheckContext:
    symbol: str
    requested_lots: float
    requested_exposure_pct: float
    symbol_exposure_pct: float
    sector_exposure_pct: float
    total_exposure_pct: float
    drawdown_pct: float
    drawdown_level: DrawdownLevel
    circuit_breaker_active: bool


@dataclass
class RiskControlReport:
    peak_equity: float
    current_equity: float
    drawdown_pct: float
    drawdown_level: str
    drawdown_action: str
    position_scale: float
    new_positions_allowed: bool
    circuit_breaker_active: bool
    total_exposure_pct: float
    warnings: list[str] = field(default_factory=list)
