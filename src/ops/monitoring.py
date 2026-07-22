from __future__ import annotations

import shutil
import time
from dataclasses import dataclass

from src.core.config import OpsConfig
from src.core.mt5_connector import MT5Connector
from src.ops.types import AlertSeverity, HealthMetric, MonitorLevel


@dataclass
class MonitoringContext:
    mt5_connected: bool = False
    data_source_age_seconds: float = 0.0
    api_success_rate: float = 1.0
    agent_latency_ms: float = 0.0
    bus_backlog: int = 0
    daily_drawdown_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    max_trade_position_pct: float = 0.0
    trades_today: int = 0
    trades_daily_mean: float = 1.0
    heartbeat_age_seconds: float = 0.0
    network_latency_ms: float = 0.0


def collect_service_metrics() -> tuple[float, float, float]:
    """CPU %, memory %, free disk GB. Uses psutil when available."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        free_gb = psutil.disk_usage("/").free / (1024**3)
        return cpu, memory, free_gb
    except ImportError:
        free_gb = shutil.disk_usage("/").free / (1024**3)
        return 0.0, 0.0, free_gb


class FourLayerMonitor:
    """4-layer monitoring: infra / service / application / business (Lesson 20.1)."""

    def __init__(self, config: OpsConfig) -> None:
        self.cfg = config

    def evaluate(self, ctx: MonitoringContext) -> list[HealthMetric]:
        metrics: list[HealthMetric] = []

        metrics.append(
            self._metric(
                MonitorLevel.INFRASTRUCTURE,
                "heartbeat_age",
                ctx.heartbeat_age_seconds,
                self.cfg.heartbeat_timeout_seconds,
                "s",
                AlertSeverity.CRITICAL,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.INFRASTRUCTURE,
                "network_latency",
                ctx.network_latency_ms,
                self.cfg.network_latency_warn_ms,
                "ms",
                AlertSeverity.MEDIUM,
                op="gt",
            )
        )

        cpu, memory, free_gb = collect_service_metrics()
        metrics.append(
            self._metric(
                MonitorLevel.SERVICE,
                "cpu_usage",
                cpu,
                self.cfg.cpu_warn_pct,
                "%",
                AlertSeverity.MEDIUM,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.SERVICE,
                "memory_usage",
                memory,
                self.cfg.memory_critical_pct,
                "%",
                AlertSeverity.CRITICAL,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.SERVICE,
                "disk_free",
                free_gb,
                self.cfg.disk_free_warn_gb,
                "GB",
                AlertSeverity.MEDIUM,
                op="lt",
            )
        )

        metrics.append(
            self._metric(
                MonitorLevel.APPLICATION,
                "data_source_connected",
                0.0 if ctx.mt5_connected else 1.0,
                0.5,
                "flag",
                AlertSeverity.CRITICAL,
                op="gt",
                message="data source disconnected" if not ctx.mt5_connected else "connected",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.APPLICATION,
                "data_freshness",
                ctx.data_source_age_seconds,
                self.cfg.data_disconnect_seconds,
                "s",
                AlertSeverity.CRITICAL,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.APPLICATION,
                "api_success_rate",
                ctx.api_success_rate * 100.0,
                self.cfg.api_success_warn_pct,
                "%",
                AlertSeverity.MEDIUM,
                op="lt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.APPLICATION,
                "agent_latency",
                ctx.agent_latency_ms,
                self.cfg.agent_latency_warn_ms,
                "ms",
                AlertSeverity.MEDIUM,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.APPLICATION,
                "task_queue_backlog",
                float(ctx.bus_backlog),
                float(self.cfg.task_queue_warn_count),
                "items",
                AlertSeverity.MEDIUM,
                op="gt",
            )
        )

        metrics.append(
            self._metric(
                MonitorLevel.BUSINESS,
                "daily_drawdown",
                ctx.daily_drawdown_pct,
                self.cfg.daily_drawdown_warn_pct,
                "%",
                AlertSeverity.MEDIUM,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.BUSINESS,
                "weekly_drawdown",
                ctx.weekly_drawdown_pct,
                self.cfg.weekly_drawdown_critical_pct,
                "%",
                AlertSeverity.CRITICAL,
                op="gt",
            )
        )
        metrics.append(
            self._metric(
                MonitorLevel.BUSINESS,
                "abnormal_trade_size",
                ctx.max_trade_position_pct,
                self.cfg.abnormal_trade_position_pct,
                "%",
                AlertSeverity.CRITICAL,
                op="gt",
            )
        )
        if ctx.trades_daily_mean > 0:
            ratio = ctx.trades_today / ctx.trades_daily_mean
            metrics.append(
                self._metric(
                    MonitorLevel.BUSINESS,
                    "trade_frequency_ratio",
                    ratio,
                    self.cfg.trade_frequency_deviation_x,
                    "x",
                    AlertSeverity.MEDIUM,
                    op="gt",
                )
            )

        return metrics

    @staticmethod
    def _metric(
        level: MonitorLevel,
        name: str,
        value: float,
        threshold: float,
        unit: str,
        severity: AlertSeverity,
        *,
        op: str = "gt",
        message: str = "",
    ) -> HealthMetric:
        if op == "lt":
            healthy = value >= threshold
            breach = value < threshold
        else:
            healthy = value <= threshold
            breach = value > threshold

        if not message:
            message = f"{name}={value:.2f}{unit} threshold={threshold:.2f}{unit}"

        return HealthMetric(
            level=level,
            name=name,
            value=value,
            threshold=threshold,
            unit=unit,
            severity=severity,
            healthy=healthy if name != "data_source_connected" else value <= threshold,
            message=message if breach or name == "data_source_connected" else "",
        )


def build_monitoring_context(
    connector: MT5Connector,
    *,
    agent_latency_ms: float = 0.0,
    bus_backlog: int = 0,
    daily_drawdown_pct: float = 0.0,
    weekly_drawdown_pct: float = 0.0,
    max_trade_position_pct: float = 0.0,
    trades_today: int = 0,
    trades_daily_mean: float = 8.0,
    data_age_seconds: float = 0.0,
) -> MonitoringContext:
    return MonitoringContext(
        mt5_connected=connector.is_connected,
        data_source_age_seconds=data_age_seconds,
        agent_latency_ms=agent_latency_ms,
        bus_backlog=bus_backlog,
        daily_drawdown_pct=daily_drawdown_pct,
        weekly_drawdown_pct=weekly_drawdown_pct,
        max_trade_position_pct=max_trade_position_pct,
        trades_today=trades_today,
        trades_daily_mean=trades_daily_mean,
        heartbeat_age_seconds=0.0,
        network_latency_ms=0.0,
    )
