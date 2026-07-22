from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GraduationStage(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    SMALL_LIVE = "small_live"
    SCALE_UP = "scale_up"


@dataclass
class ChecklistResult:
    category: str
    item: str
    passed: bool
    detail: str = ""


@dataclass
class AgentStatus:
    name: str
    role: str
    active: bool
    metric: str = ""


@dataclass
class IntegrationReport:
    graduation_stage: str
    modular_monolith: bool = True
    agents: list[AgentStatus] = field(default_factory=list)
    data_ready: bool = False
    pipeline_ok: bool = False
    backtest_gate_passed: bool | None = None
    pre_live_checklist: list[ChecklistResult] = field(default_factory=list)
    ready_for_paper: bool = False
    ready_for_live: bool = False
    warnings: list[str] = field(default_factory=list)
