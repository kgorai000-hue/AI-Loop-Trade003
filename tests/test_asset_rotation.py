"""Tests for multi-group concurrency and gradual asset migration."""

from __future__ import annotations

from src.core.config import load_config
from src.core.types import SignalMode, SignalSide, StrategyKind, TradeSignal
from src.portfolio.asset_rotation import (
    apply_asset_rotation,
    resolve_scan_symbols,
    save_rotation_state,
    AssetRotationPlan,
)


def _sig(symbol: str, group: str, strength: float) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        side=SignalSide.BUY,
        timeframe="M30",
        strength=strength,
        reason="test",
        mode=SignalMode.MOMENTUM,
        strategy=StrategyKind.TREND_FOLLOWING,
        group_id=group,
        trade_mode="single",
        portfolio_weight=1.0,
    )


def test_resolve_scan_symbols_expands_all_groups():
    config = load_config()
    symbols = resolve_scan_symbols(config, ["GOLD"])
    assert len(symbols) == 12
    assert "EURUSD" in symbols
    assert "WTI" in symbols


def test_multi_group_concurrent_selection(tmp_path, monkeypatch):
    config = load_config()
    monkeypatch.setattr(config.intelligence, "state_dir", str(tmp_path))
    signals = [
        _sig("#US30", "Group1", 0.9),
        _sig("#USSPX500", "Group1", 0.7),
        _sig("GOLD", "Group3", 0.8),
        _sig("EURUSD", "Group4", 0.6),
        _sig("WTI", "Group5", 0.5),
    ]
    out, plan = apply_asset_rotation(config, signals)
    assert config.asset_rotation.multi_group_enabled is True
    assert len(plan.active_groups) >= 2
    assert len(plan.selected_symbols) >= 2
    assert len(out) >= 2
    # Multiple groups represented when signals span groups.
    groups = {s.group_id for s in out}
    assert len(groups) >= 2
    assert all(s.portfolio_weight is not None and s.portfolio_weight > 0 for s in out)


def test_migration_shifts_weight_toward_stronger_group(tmp_path, monkeypatch):
    config = load_config()
    monkeypatch.setattr(config.intelligence, "state_dir", str(tmp_path))
    # Seed previous weights favoring Group1.
    prior = AssetRotationPlan(
        group_weights={
            "Group1": 0.45,
            "Group2": 0.10,
            "Group3": 0.15,
            "Group4": 0.15,
            "Group5": 0.15,
        },
        active_groups=["Group1", "Group3", "Group4", "Group5", "Group2"],
    )
    save_rotation_state(config, config.asset_rotation, prior)

    # Now Group3 dominates signals → migration room toward metals.
    signals = [
        _sig("#US30", "Group1", 0.2),
        _sig("GOLD", "Group3", 0.95),
        _sig("SILVER", "Group3", 0.90),
    ]
    out, plan = apply_asset_rotation(config, signals)
    assert plan.group_weights.get("Group3", 0) > plan.group_weights.get("Group1", 0)
    assert "Group3" in plan.active_groups
    # Floor remains so Group1 can migrate back later.
    assert plan.group_weights.get("Group1", 0) >= config.asset_rotation.min_group_weight * 0.5
    assert any(s.group_id == "Group3" for s in out)
