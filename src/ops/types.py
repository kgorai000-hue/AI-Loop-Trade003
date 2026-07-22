from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MonitorLevel(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    SERVICE = "service"
    APPLICATION = "application"
    BUSINESS = "business"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class HealthMetric:
    level: MonitorLevel
    name: str
    value: float
    threshold: float
    unit: str
    severity: AlertSeverity
    healthy: bool
    message: str = ""


@dataclass
class Alert:
    severity: AlertSeverity
    channel: AlertChannel
    title: str
    message: str
    service: str
    event: str
    actionable: str = ""
    suppressed: bool = False


@dataclass
class ChecklistItem:
    phase: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RecoveryAction:
    failure_type: str
    strategy: str
    status: str
    detail: str = ""


@dataclass
class OpsReport:
    enabled: bool = True
    session_phase: str = "unknown"
    trading_allowed: bool = True
    metrics: list[HealthMetric] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    checklist: list[ChecklistItem] = field(default_factory=list)
    recovery_actions: list[RecoveryAction] = field(default_factory=list)
    healthy: bool = True
    critical_count: int = 0
    warnings: list[str] = field(default_factory=list)
