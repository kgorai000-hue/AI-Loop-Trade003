"""Phase 1: asset groups, within-group pairs, arbitration."""

from __future__ import annotations

from src.core.config import load_config
from src.core.types import SignalMode, SignalSide, StrategyKind, TradeSignal
from src.agents.portfolio_agent import PortfolioAgent


def test_asset_groups_twelve_symbols():
    config = load_config()
    assert config.project.name == "AI-Loop-Trade003"
    assert len(config.symbols) == 12
    assert set(config.asset_groups) == {
        "Group1",
        "Group2",
        "Group3",
        "Group4",
        "Group5",
    }
    assert config.group_for_symbol("WTI").lot_multiplier == 0.5
    assert config.group_for_symbol("GOLD").strategy == "breakout_high_vol"
    assert config.trading.dry_run is False
    assert config.project.graduation_stage == "demo_live"
    assert config.mt5.require_demo is True
    assert len(config.tradeable_symbols_all_groups()) == 12
    assert len(config.pairs_all_groups()) == 9


def test_pairs_are_within_groups():
    config = load_config()
    for pair in config.strategies.pairs:
        left = config.group_for_symbol(pair[0])
        right = config.group_for_symbol(pair[1])
        assert left is not None and right is not None
        assert left.name == right.name


def test_arbitration_prefers_pair():
    config = load_config()
    agent = PortfolioAgent.__new__(PortfolioAgent)
    agent.config = config

    singles = [
        TradeSignal(
            symbol="GOLD",
            side=SignalSide.BUY,
            timeframe="M30",
            strength=0.4,
            reason="single",
            mode=SignalMode.MOMENTUM,
            strategy=StrategyKind.TREND_FOLLOWING,
            trade_mode="single",
            group_id="Group3",
        ),
        TradeSignal(
            symbol="SILVER",
            side=SignalSide.BUY,
            timeframe="M30",
            strength=0.3,
            reason="single",
            mode=SignalMode.MOMENTUM,
            strategy=StrategyKind.TREND_FOLLOWING,
            trade_mode="single",
            group_id="Group3",
        ),
    ]
    pairs = [
        TradeSignal(
            symbol="GOLD",
            side=SignalSide.BUY,
            timeframe="M30",
            strength=0.8,
            reason="pair",
            mode=SignalMode.MEAN_REVERSION,
            strategy=StrategyKind.PAIRS,
            trade_mode="pair",
            group_id="Group3",
            pair_id="pairs/GOLD__SILVER",
        ),
        TradeSignal(
            symbol="SILVER",
            side=SignalSide.SELL,
            timeframe="M30",
            strength=0.8,
            reason="pair",
            mode=SignalMode.MEAN_REVERSION,
            strategy=StrategyKind.PAIRS,
            trade_mode="pair",
            group_id="Group3",
            pair_id="pairs/GOLD__SILVER",
        ),
    ]
    out = agent._arbitrate_single_vs_pair(singles, pairs)
    assert all(s.trade_mode == "pair" for s in out)
    assert {s.symbol for s in out} == {"GOLD", "SILVER"}
