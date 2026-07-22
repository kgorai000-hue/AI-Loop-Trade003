from __future__ import annotations

import time
from collections import defaultdict

from src.core.config import OpsConfig
from src.ops.types import Alert, AlertChannel, AlertSeverity, HealthMetric


SEVERITY_CHANNEL: dict[AlertSeverity, AlertChannel] = {
    AlertSeverity.LOW: AlertChannel.EMAIL,
    AlertSeverity.MEDIUM: AlertChannel.SLACK,
    AlertSeverity.HIGH: AlertChannel.SMS,
    AlertSeverity.CRITICAL: AlertChannel.CIRCUIT_BREAKER,
}


class AlertManager:
    """Alert routing with suppression (Lesson 20.3)."""

    def __init__(self, config: OpsConfig) -> None:
        self.cfg = config
        self._last_sent: dict[str, float] = defaultdict(float)

    def from_metrics(self, metrics: list[HealthMetric], *, service: str = "ops") -> list[Alert]:
        alerts: list[Alert] = []
        for metric in metrics:
            if metric.healthy:
                continue
            alert = self._build_alert(metric, service)
            if self._should_suppress(alert):
                alert.suppressed = True
            else:
                self._last_sent[alert.event] = time.time()
            alerts.append(alert)
        return alerts

    def send(self, alert: Alert) -> None:
        if alert.suppressed:
            return
        # Production would dispatch to email/slack/sms; log locally for now.
        import logging

        logger = logging.getLogger("ops.alerts")
        logger.log(
            logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.WARNING,
            "[%s/%s] %s: %s | action: %s",
            alert.severity.value,
            alert.channel.value,
            alert.title,
            alert.message,
            alert.actionable,
        )

    def _build_alert(self, metric: HealthMetric, service: str) -> Alert:
        channel = SEVERITY_CHANNEL.get(metric.severity, AlertChannel.SLACK)
        actionable = self._actionable(metric)
        return Alert(
            severity=metric.severity,
            channel=channel,
            title=f"{metric.level.value}: {metric.name} breach",
            message=metric.message or f"{metric.name}={metric.value:.2f}{metric.unit}",
            service=service,
            event=f"{metric.level.value}.{metric.name}",
            actionable=actionable,
        )

    @staticmethod
    def _actionable(metric: HealthMetric) -> str:
        actions = {
            "data_source_connected": "Pause trading; verify MT5 terminal and reconnect",
            "daily_drawdown": "Review open positions; confirm circuit breaker engaged",
            "weekly_drawdown": "Stop new entries; manual review required",
            "abnormal_trade_size": "Cancel pending orders; verify position sizing logic",
            "agent_latency": "Check agent health; reduce parallel load",
        }
        return actions.get(metric.name, "Review dashboard and recent logs")

    def _should_suppress(self, alert: Alert) -> bool:
        now = time.time()
        last = self._last_sent.get(alert.event, 0.0)
        if now - last < self.cfg.alert_suppress_seconds:
            return True
        if self.cfg.quiet_hours_enabled and alert.severity in (AlertSeverity.LOW, AlertSeverity.MEDIUM):
            hour = time.localtime().tm_hour
            if hour < self.cfg.quiet_hours_start or hour >= self.cfg.quiet_hours_end:
                return True
        return False
