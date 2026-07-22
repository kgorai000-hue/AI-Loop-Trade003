from __future__ import annotations

import time

import pytest

from src.core.config import OpsConfig
from src.ops.alerts import AlertManager
from src.ops.checklist import classify_recovery, reconcile_positions
from src.ops.deployment import AutoRollback, CanaryDeployment, ModelRecord, ModelRegistry
from src.ops.logging import StructuredTradeLogger, new_trace_id
from src.ops.monitoring import FourLayerMonitor, MonitoringContext
from src.ops.schedule import MarketSessionScheduler, SessionPhase
from src.ops.types import AlertSeverity, MonitorLevel


def _ops_cfg(**overrides) -> OpsConfig:
    defaults = dict(
        enabled=True,
        heartbeat_timeout_seconds=30.0,
        network_latency_warn_ms=200.0,
        cpu_warn_pct=80.0,
        memory_critical_pct=90.0,
        disk_free_warn_gb=10.0,
        data_disconnect_seconds=60.0,
        api_success_warn_pct=95.0,
        agent_latency_warn_ms=5000.0,
        task_queue_warn_count=100,
        daily_drawdown_warn_pct=3.0,
        weekly_drawdown_critical_pct=5.0,
        abnormal_trade_position_pct=10.0,
        trade_frequency_deviation_x=3.0,
        alert_suppress_seconds=300.0,
        quiet_hours_enabled=False,
        quiet_hours_start=22,
        quiet_hours_end=7,
        structured_log_dir="logs/structured",
        model_registry_path=":memory:",
        enforce_market_hours=False,
        allow_trading_in_dry_run=True,
        canary_initial_weight=0.05,
        rollback_max_error_rate=0.05,
        rollback_min_sharpe=0.5,
    )
    defaults.update(overrides)
    return OpsConfig(**defaults)


def test_four_layer_monitor_detects_drawdown() -> None:
    monitor = FourLayerMonitor(_ops_cfg())
    ctx = MonitoringContext(daily_drawdown_pct=4.0, weekly_drawdown_pct=6.0, mt5_connected=True)
    metrics = monitor.evaluate(ctx)
    dd_metrics = [m for m in metrics if "drawdown" in m.name]
    assert any(not m.healthy for m in dd_metrics)


def test_alert_manager_builds_actionable_alerts() -> None:
    monitor = FourLayerMonitor(_ops_cfg())
    ctx = MonitoringContext(daily_drawdown_pct=4.0, mt5_connected=False)
    unhealthy = [m for m in monitor.evaluate(ctx) if not m.healthy]
    alerts = AlertManager(_ops_cfg(alert_suppress_seconds=0)).from_metrics(unhealthy)
    assert alerts
    assert any(a.actionable for a in alerts)
    assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)


def test_structured_logger_writes_trace_id(tmp_path) -> None:
    logger = StructuredTradeLogger(tmp_path)
    trace = new_trace_id("test")
    record = logger.log_event(
        level="INFO",
        service="test",
        event="order_submitted",
        trace_id=trace,
        data={"symbol": "EURUSD"},
    )
    assert record["trace_id"] == trace
    files = list(tmp_path.glob("*.jsonl"))
    assert files


def test_market_scheduler_phases() -> None:
    scheduler = MarketSessionScheduler()
    phase = scheduler.current_phase()
    assert isinstance(phase, SessionPhase)
    assert scheduler.trading_allowed() in (True, False)


def test_canary_deployment_routing() -> None:
    canary = CanaryDeployment(
        stable_predict=lambda _: {"signal": 0.0},
        canary_predict=lambda _: {"signal": 1.0},
        initial_weight=1.0,
    )
    signal = canary.get_signal({})
    assert signal["model_version"] == "canary"
    canary.rollback()
    assert canary.canary_weight == 0.0


def test_auto_rollback_on_error_rate() -> None:
    registry = ModelRegistry(":memory:")
    registry.register(
        ModelRecord(
            model_id="signal_stable",
            version="v1.0.0",
            created_at=int(time.time()),
            created_by="test",
            metrics={"sharpe": 1.2},
            status="production",
            artifact_path="models/stable.pkl",
        )
    )
    rollback = AutoRollback({"max_error_rate": 0.05, "max_latency": 5000, "min_sharpe": 0.5}, registry)
    assert rollback.check_and_rollback({"error_rate": 0.10})
    assert rollback.rollback_triggered


def test_reconcile_positions_uses_broker_truth() -> None:
    warnings = reconcile_positions({"EURUSD": 1.0}, {"EURUSD": 0.5})
    assert warnings
    assert "broker" in warnings[0]


def test_recovery_classification() -> None:
    action = classify_recovery("data_source")
    assert action.failure_type == "data_source"
    assert "backup" in action.strategy or "pause" in action.strategy


def test_monitor_covers_four_levels() -> None:
    monitor = FourLayerMonitor(_ops_cfg())
    metrics = monitor.evaluate(MonitoringContext(mt5_connected=True))
    levels = {m.level for m in metrics}
    assert MonitorLevel.INFRASTRUCTURE in levels
    assert MonitorLevel.SERVICE in levels
    assert MonitorLevel.APPLICATION in levels
    assert MonitorLevel.BUSINESS in levels
