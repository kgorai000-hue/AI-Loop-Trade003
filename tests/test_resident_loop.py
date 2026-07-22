"""Tests for resident loop scheduling and process lock naming."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from src.core.config import load_config
from src.intelligence.persistence import StateStore
from src.intelligence.process_lock import ProcessLock
from src.intelligence.resident import ResidentLoopEngine


def test_intelligence_loop_config_loaded():
    config = load_config()
    assert config.intelligence.loop.poll_seconds == 30
    assert config.intelligence.loop.review_weekday == 5
    assert config.intelligence.loop.sync_on_poll is True
    assert config.intelligence.loop.check_all_asset_groups is True


def test_engineering_universe_covers_all_asset_groups():
    config = load_config()
    engine = ResidentLoopEngine(config)
    assert set(engine.engineering_symbols) == set(config.tradeable_symbols_all_groups())
    assert len(engine.engineering_symbols) == 12
    groups = {engine._group_name(s) for s in engine.engineering_symbols}
    assert groups == {"Group1", "Group2", "Group3", "Group4", "Group5"}
    # CLI trade subset must not shrink engineering when check_all_asset_groups=True
    engine_subset = ResidentLoopEngine(config, symbols=["GOLD", "SILVER"])
    assert set(engine_subset.trade_symbols) == {"GOLD", "SILVER"}
    assert len(engine_subset.engineering_symbols) == 12
    assert "EURUSD" in engine_subset.engineering_symbols
    assert "WTI" in engine_subset.engineering_symbols


def test_should_review_persisted_across_restart(tmp_path: Path, monkeypatch):
    config = load_config()
    # Point state into tmp and avoid MT5/real store side effects.
    monkeypatch.setattr(config.intelligence, "state_dir", str(tmp_path))
    monkeypatch.setattr(config, "symbols", ["#US30"])

    engine = ResidentLoopEngine.__new__(ResidentLoopEngine)
    engine.config = config
    engine.symbols = ["#US30"]
    engine.trade_symbols = ["#US30"]
    engine.engineering_symbols = ["#US30"]
    engine.check_all_asset_groups = False
    engine.strategy = "feature_score"
    engine.timeframe = "M30"
    engine.poll_seconds = 30
    engine.review_weekday = 5
    engine.review_hour_utc = 6
    engine.sharpe_degrade_trigger = 0.2
    engine.run_pipeline_on_bar = True
    engine.sync_on_poll = True
    engine.pretrade_optimize = False
    engine.require_adopted_params = False
    engine.optimize_pairs = False
    engine.state_dir = str(tmp_path)
    engine.stores = {"#US30": StateStore(tmp_path, "#US30")}
    engine.pair_stores = {}
    engine.pair_ids = []
    engine._pair_legs = {}
    engine._last_review_date = engine._load_last_review_date()
    engine.connector = MagicMock()
    engine.ohlcv = MagicMock()
    engine._lock = MagicMock()
    engine._pretrade_ready = True

    saturday = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
    assert engine.should_review(saturday) is True

    engine._persist_last_review_date("2026-07-11")
    assert engine.stores["#US30"].read_state().get("last_review_date") == "2026-07-11"

    engine2 = ResidentLoopEngine.__new__(ResidentLoopEngine)
    engine2.config = config
    engine2.symbols = ["#US30"]
    engine2.engineering_symbols = ["#US30"]
    engine2.review_weekday = 5
    engine2.review_hour_utc = 6
    engine2.stores = {"#US30": StateStore(tmp_path, "#US30")}
    engine2.pair_stores = {}
    engine2._last_review_date = engine2._load_last_review_date()
    assert engine2._last_review_date == "2026-07-11"
    assert engine2.should_review(saturday) is False


def test_process_lock_name_for_trade003(tmp_path: Path):
    lock = ProcessLock.for_project(tmp_path, tmp_path)
    assert "AILoopTrade003" in lock.mutex_name
    assert lock.lock_path.name == ".ai_loop_trade003.lock"
