from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeathModeInfo:
    """Appendix B death mode metadata."""

    mode_id: int
    name: str
    name_ja: str
    definition: str
    symptoms: tuple[str, ...]
    prevention: tuple[str, ...]
    source_module: str


@dataclass
class DeathModeStatus:
    mode_id: int
    name: str
    healthy: bool
    detail: str
    source_module: str
    diagnostic_priority: int


@dataclass
class WeeklyCheckItem:
    """Appendix B weekly health checklist row."""

    name: str
    passed: bool
    detail: str
    death_mode_id: int


@dataclass
class SurvivalContext:
    """Signals gathered from existing modules — no duplicate business logic."""

    mt5_connected: bool = False
    data_valid: bool = True
    data_warnings: list[str] = field(default_factory=list)
    resilience_level: str = "normal"
    resilience_warnings: list[str] = field(default_factory=list)
    avg_slippage_pct: float = 0.0
    avg_fill_ratio: float = 1.0
    slippage_threshold_pct: float = 0.05
    drawdown_pct: float = 0.0
    circuit_breaker_active: bool = False
    risk_circuit_configured: bool = True
    cost_blocked_count: int = 0
    portfolio_leverage: float = 0.0
    max_correlation: float = 0.0
    evolution_drift: bool = False
    decay_stage_count: int = 0
    projected_ic_12m: float | None = None
    ops_healthy: bool = True
    critical_alerts: int = 0
    manual_interventions: int = 0
    regulatory_flags: int = 0
    backtest_gate_configured: bool = True
    execution_enabled: bool = True


@dataclass
class SurvivalReport:
    modes: list[DeathModeStatus] = field(default_factory=list)
    weekly_checklist: list[WeeklyCheckItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_healthy(self) -> bool:
        return all(m.healthy for m in self.modes)

    @property
    def failed_modes(self) -> list[DeathModeStatus]:
        return [m for m in self.modes if not m.healthy]
