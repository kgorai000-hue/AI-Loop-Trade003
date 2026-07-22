from __future__ import annotations

import pytest

from src.core.data_manager import DataManager
from src.system.architecture import AGENT_PIPELINE, pipeline_stages
from src.system.integration import build_pre_live_checklist, SystemIntegrator
from src.system.types import GraduationStage


def test_agent_pipeline_includes_core_agents() -> None:
    registered = set(AGENT_PIPELINE)
    assert "RiskAgent" in registered
    assert "ExecutionAgent" in registered
    assert "OperationsAgent" in registered
    assert "PortfolioAgent" in registered
    assert len(pipeline_stages()) >= 8


def test_pre_live_checklist_demo_live_mode() -> None:
    from src.core.config import load_config

    config = load_config()
    checklist = build_pre_live_checklist(
        config,
        connector_connected=True,
        data_ready=True,
        pipeline_ok=True,
        backtest_passed=True,
        ops_healthy=True,
        dry_run=False,
    )
    assert len(checklist) >= 8
    mode = next(c for c in checklist if c.item == "dry_run_enabled")
    assert mode.passed
    demo = next(c for c in checklist if c.item == "require_demo")
    assert demo.passed


def test_data_manager_validate_empty_store(tmp_path) -> None:
    from src.core.config import load_config

    config = load_config()
    db_path = tmp_path / "test.db"
    from src.data.store import OHLCVStore

    store = OHLCVStore(db_path)
    dm = DataManager(config, store)
    valid, errors = dm.validate("EURUSD")
    assert not valid
    assert errors


def test_data_manager_indicators_with_synthetic_bars(tmp_path) -> None:
    from src.core.config import load_config
    from src.data.store import BarRecord, OHLCVStore

    config = load_config()
    store = OHLCVStore(tmp_path / "test.db")
    base_time = 1_700_000_000
    bars = []
    price = 1.08
    for idx in range(80):
        p = price + idx * 0.0001
        bars.append(
            BarRecord(
                symbol="EURUSD",
                timeframe="H1",
                time=base_time + idx * 3600,
                open=p,
                high=p + 0.0005,
                low=p - 0.0005,
                close=p + 0.0001,
                tick_volume=100,
                spread=10,
                real_volume=0,
            )
        )
    store.upsert_bars(bars)

    dm = DataManager(config, store)
    indicators = dm.calculate_indicators("EURUSD", "H1", 80)
    assert "rsi" in indicators
    assert "adx" in indicators
    assert indicators["close"] > 0

    latest = dm.get_latest(["EURUSD"], "H1")
    assert "EURUSD" in latest


def test_graduation_stages() -> None:
    assert GraduationStage.PAPER.value == "paper"
    assert GraduationStage.SMALL_LIVE.value == "small_live"


def test_system_integrator_offline_assessment(tmp_path) -> None:
    from src.core.config import load_config
    from src.data.store import OHLCVStore

    config = load_config()
    store = OHLCVStore(tmp_path / "test.db")

    class FakeConnector:
        is_connected = False

    integrator = SystemIntegrator(config, FakeConnector(), store)  # type: ignore[arg-type]
    report = integrator.assess(pipeline_ok=False, backtest_passed=None)
    assert report.graduation_stage == config.project.graduation_stage
    assert len(report.agents) >= 10
    assert not report.pipeline_ok
