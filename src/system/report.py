from __future__ import annotations

from src.core.config import AppConfig, load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.system.architecture import AGENT_PIPELINE, DATA_FLOW, MODULAR_MONOLITH_NOTE, pipeline_stages
from src.system.integration import SystemIntegrator


def print_lesson_content() -> None:
    print("\n=== Lesson 21: Project Implementation ===")
    print("  Modular monolith: single process, clean agent boundaries")

    print("\n=== Agent Pipeline ===")
    for name in AGENT_PIPELINE:
        print(f"  - {name}")

    print("\n=== Pipeline Stages ===")
    for stage, desc in pipeline_stages():
        print(f"  {stage}: {desc}")

    print("\n=== Graduation Path (21.4) ===")
    stages = [
        ("Stage 1 Backtest", "2-4 weeks, quality gate must pass"),
        ("Stage 2 Paper", "2-4 weeks, dry_run=true, real-time data"),
        ("Stage 3 Small Live", "5-10% capital, verify execution"),
        ("Stage 4 Scale Up", "monthly review, gradual capital increase"),
    ]
    for name, desc in stages:
        print(f"  {name}: {desc}")

    print("\n=== Backtest Targets (21.3) ===")
    print("  Sharpe > 1.0 | Max DD < 20% | OOS Sharpe > IS x 0.7")


def print_system_status(config: AppConfig | None = None, *, paper_only: bool = False) -> int:
    print_lesson_content()
    if paper_only:
        return 0

    config = config or load_config()
    print("\n=== System Configuration ===")
    print(f"  graduation stage  : {config.project.graduation_stage}")
    print(f"  dry_run           : {config.trading.dry_run}")
    print(f"  symbols           : {len(config.symbols)}")
    print(f"  broker            : {config.broker_name} ({config.account_type})")
    print(f"  modular monolith  : {MODULAR_MONOLITH_NOTE[:60]}...")

    print("\n=== Data Flow ===")
    for line in DATA_FLOW.strip().splitlines():
        print(f"  {line}")

    store = OHLCVStore(config.storage.path)
    connector = MT5Connector(config)
    integrator = SystemIntegrator(config, connector, store)

    try:
        connector.connect()
        report = integrator.assess(pipeline_ok=False)
    except Exception as exc:
        print(f"\n  MT5 not connected: {exc}")
        report = integrator.assess(pipeline_ok=False)
    finally:
        if connector.is_connected:
            connector.disconnect()

    print("\n=== Pre-Live Checklist ===")
    for item in report.pre_live_checklist:
        flag = "OK" if item.passed else "PENDING"
        print(f"  [{flag:7}] {item.category:10} {item.item:22} | {item.detail}")

    print(f"\n  Ready for paper   : {report.ready_for_paper}")
    print(f"  Ready for live    : {report.ready_for_live}")
    return 0
