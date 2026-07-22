from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.ops.alerts import AlertManager
from src.ops.deployment import AutoRollback, CanaryDeployment, ModelRegistry, ModelRecord
from src.ops.logging import StructuredTradeLogger, new_trace_id
from src.ops.monitoring import FourLayerMonitor, MonitoringContext
from src.ops.schedule import MarketSessionScheduler
from src.ops.types import AlertSeverity, MonitorLevel


def print_lesson_content() -> None:
    print("\n=== Lesson 20: Production Operations ===")
    print("  Goal: system runs correctly without you, alerts on failure")

    print("\n=== 4-Layer Monitoring (20.1) ===")
    for level in MonitorLevel:
        print(f"  {level.value}")

    print("\n=== Sample Metrics vs Thresholds ===")
    config = load_config()
    monitor = FourLayerMonitor(config.ops)
    ctx = MonitoringContext(
        mt5_connected=True,
        daily_drawdown_pct=2.5,
        weekly_drawdown_pct=4.0,
        agent_latency_ms=1200.0,
        api_success_rate=0.97,
    )
    for metric in monitor.evaluate(ctx):
        flag = "OK" if metric.healthy else metric.severity.value.upper()
        print(f"  [{flag:8}] {metric.level.value:15} {metric.name:22} {metric.value:.2f}{metric.unit}")

    print("\n=== Structured Log Example (20.2) ===")
    trace = new_trace_id()
    logger = StructuredTradeLogger("logs/structured")
    record = logger.log_event(
        level="INFO",
        service="execution_agent",
        event="order_submitted",
        trace_id=trace,
        data={"symbol": "EURUSD", "side": "BUY", "lots": 0.1, "order_type": "market"},
        context={"regime": "trending", "signal_strength": 0.72},
    )
    print(f"  trace_id={record['trace_id']} event={record['event']}")

    print("\n=== Alert Suppression (20.3) ===")
    manager = AlertManager(config.ops)
    unhealthy = [m for m in monitor.evaluate(ctx) if not m.healthy]
    alerts = manager.from_metrics(unhealthy)
    print(f"  alerts generated: {len(alerts)} (suppressed={sum(1 for a in alerts if a.suppressed)})")

    print("\n=== Market Session Phase (20.5) ===")
    scheduler = MarketSessionScheduler()
    phase = scheduler.current_phase()
    print(f"  current phase     : {phase.value}")
    print(f"  trading allowed   : {scheduler.trading_allowed()}")
    print(f"  new entries       : {scheduler.new_entries_allowed()}")

    print("\n=== Canary Deployment (20.7.3) ===")
    canary = CanaryDeployment(
        stable_predict=lambda d: {"signal": 0.1, "confidence": 0.6},
        canary_predict=lambda d: {"signal": 0.15, "confidence": 0.65},
        initial_weight=config.ops.canary_initial_weight,
    )
    versions = {"stable": 0, "canary": 0}
    for _ in range(100):
        sig = canary.get_signal({})
        versions[sig["model_version"]] += 1
    print(f"  canary weight     : {canary.canary_weight:.0%}")
    print(f"  sample split      : stable={versions['stable']} canary={versions['canary']}")
    print(f"  next promotion    : {canary.next_promotion_step()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Production operations report (Lesson 20)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    ops = config.ops
    print("\n=== Configured Operations ===")
    print(f"  enabled             : {ops.enabled}")
    print(f"  daily DD warn       : {ops.daily_drawdown_warn_pct}%")
    print(f"  weekly DD critical  : {ops.weekly_drawdown_critical_pct}%")
    print(f"  alert suppress      : {ops.alert_suppress_seconds}s")
    print(f"  structured logs     : {ops.structured_log_dir}")
    print(f"  model registry      : {ops.model_registry_path}")

    registry = ModelRegistry(ops.model_registry_path)
    prod = registry.get_production()
    if prod:
        print(f"\n=== Production Model ===")
        print(f"  model_id            : {prod.model_id}")
        print(f"  version             : {prod.version}")
        print(f"  sharpe              : {prod.metrics.get('sharpe', 0):.2f}")
    else:
        print("\n=== Production Model ===")
        print("  (none registered - register via ModelRegistry in deployment pipeline)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
