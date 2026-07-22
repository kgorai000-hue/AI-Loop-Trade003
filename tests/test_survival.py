from __future__ import annotations

from src.core.config import load_config
from src.survival.catalog import DEATH_MODES, DIAGNOSTIC_ORDER
from src.survival.diagnostics import SurvivalDiagnostics
from src.survival.types import SurvivalContext, SurvivalReport


def test_death_mode_catalog_has_twelve_modes() -> None:
    assert len(DEATH_MODES) == 12
    assert len(DIAGNOSTIC_ORDER) == 12
    assert set(DIAGNOSTIC_ORDER) == {m.mode_id for m in DEATH_MODES}


def test_survival_diagnostics_healthy_baseline() -> None:
    config = load_config()
    diag = SurvivalDiagnostics(config)
    ctx = SurvivalContext(
        mt5_connected=True,
        data_valid=True,
        resilience_level="normal",
        ops_healthy=True,
    )
    report = diag.assess(ctx)
    assert isinstance(report, SurvivalReport)
    assert len(report.modes) == 12
    assert len(report.weekly_checklist) == 12
    assert report.overall_healthy


def test_survival_flags_data_pollution() -> None:
    config = load_config()
    diag = SurvivalDiagnostics(config)
    ctx = SurvivalContext(data_valid=False, data_warnings=["EURUSD: gap detected"])
    report = diag.assess(ctx)
    data_mode = next(m for m in report.modes if m.mode_id == 1)
    assert not data_mode.healthy
    assert "gap" in data_mode.detail


def test_survival_flags_circuit_breaker() -> None:
    config = load_config()
    diag = SurvivalDiagnostics(config)
    ctx = SurvivalContext(circuit_breaker_active=True, drawdown_pct=12.0)
    report = diag.assess(ctx)
    risk_mode = next(m for m in report.modes if m.mode_id == 5)
    assert not risk_mode.healthy


def test_human_intervention_log_count(tmp_path) -> None:
    from src.extensions.human_intervention import HumanInterventionLog, InterventionRecord
    import time

    log = HumanInterventionLog(tmp_path)
    log.record(
        InterventionRecord(
            timestamp=int(time.time()),
            operator="ops",
            action="cancel_stop",
            symbol="EURUSD",
            justification="test",
        )
    )
    assert log.recent_count(days=1) == 1


def test_regulatory_monitor_stub() -> None:
    from src.extensions.regulatory import RegulatoryFlag, RegulatoryMonitor

    monitor = RegulatoryMonitor()
    monitor.add_flag(
        RegulatoryFlag(
            region="US",
            category="short_selling",
            summary="restricted list updated",
            severity="warn",
            effective_date="2020-01-01",
        )
    )
    assert len(monitor.active_flags()) == 1
    assert len(monitor.checklist()) >= 3
