from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.backtest_agent import StrategyValidationResult, ValidationReport
from src.agents.portfolio_agent import PortfolioAgent
from src.backtest.gate_registry import (
    GateEntry,
    GateRegistry,
    build_gate_registry,
    load_or_build_gate_registry,
)
from src.backtest.types import BacktestResult, QualityGateCheck, QualityGateReport
from src.core.config import load_config
from src.core.types import MarketRegime, RegimeAssessment, SignalMode, SignalSide, StrategyKind, TradeSignal
from src.stats.performance import PerformanceReport


def _perf(sharpe: float = 0.0) -> PerformanceReport:
    return PerformanceReport(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=sharpe,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        trades=0,
        win_rate=0.0,
    )


def _strategy_result(name: str, *, passed: bool, sharpe: float) -> StrategyValidationResult:
    gate = QualityGateReport(
        strategy_name=name,
        checks=[QualityGateCheck("L1", "t1", "test", passed, "ok")],
    )
    backtest = BacktestResult(
        strategy_name=name,
        returns=[],
        positions=[],
        trades=[],
        performance=_perf(sharpe),
        cost_per_trade_pct=0.0,
        total_cost_pct=0.0,
    )
    return StrategyValidationResult(
        strategy_name=name,
        backtest=backtest,
        oos=MagicMock(),
        walk_forward=[],
        walk_forward_summary={},
        monte_carlo=MagicMock(),
        param_sensitivity=[],
        quality_gate=gate,
    )


def _validation_report(symbol: str, timeframe: str, trend_pass: bool) -> ValidationReport:
    return ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        cost_per_trade_pct=0.01,
        strategies=[
            _strategy_result("trend_following", passed=trend_pass, sharpe=1.2 if trend_pass else -0.5),
            _strategy_result("mean_reversion", passed=True, sharpe=0.3),
            _strategy_result("feature_score", passed=False, sharpe=-1.0),
        ],
    )


class GateRegistryTests(unittest.TestCase):
    def test_allows_unknown_symbol_strategy(self) -> None:
        registry = GateRegistry(
            timeframe="H1",
            entries={( "EURUSD", "trend_following"): GateEntry("EURUSD", "H1", "trend_following", False)},
        )
        self.assertTrue(registry.allows("GBPUSD", "trend_following"))
        self.assertFalse(registry.allows("EURUSD", "trend_following"))

    def test_check_increments_blocked_count(self) -> None:
        registry = GateRegistry(
            timeframe="H1",
            entries={( "EURUSD", "mean_reversion"): GateEntry("EURUSD", "H1", "mean_reversion", False)},
        )
        self.assertFalse(registry.check("EURUSD", "mean_reversion"))
        self.assertEqual(registry.blocked_count, 1)
        self.assertTrue(registry.check("EURUSD", "trend_following"))

    def test_save_and_load_roundtrip(self) -> None:
        registry = GateRegistry.from_validation_report(_validation_report("USDJPY", "H1", True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            registry.save(path)
            loaded = GateRegistry.load(path, "H1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded.entries), 3)
        self.assertTrue(loaded.allows("USDJPY", "trend_following"))
        self.assertFalse(loaded.allows("USDJPY", "feature_score"))

    def test_load_rejects_mismatched_timeframe(self) -> None:
        registry = GateRegistry(timeframe="H1", entries={}, updated_at=int(time.time()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            registry.save(path)
            self.assertIsNone(GateRegistry.load(path, "H4"))

    def test_is_fresh(self) -> None:
        registry = GateRegistry(timeframe="H1", entries={}, updated_at=int(time.time()))
        self.assertTrue(registry.is_fresh(24.0))
        registry.updated_at = int(time.time()) - 48 * 3600
        self.assertFalse(registry.is_fresh(24.0))

    def test_merge_reports(self) -> None:
        reports = [
            _validation_report("EURUSD", "H1", False),
            _validation_report("USDJPY", "H1", True),
        ]
        merged = GateRegistry.merge_reports(reports, "H1")
        self.assertEqual(len(merged.entries), 6)
        self.assertFalse(merged.allows("EURUSD", "trend_following"))
        self.assertTrue(merged.allows("USDJPY", "trend_following"))

    def test_load_or_build_skips_rebuild_when_build_on_miss_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_gate.json"
            registry, from_cache = load_or_build_gate_registry(
                MagicMock(),
                MagicMock(),
                ["GOLD"],
                "M30",
                str(path),
                24.0,
                build_on_miss=False,
            )
            self.assertFalse(from_cache)
            self.assertEqual(registry.entries, {})
            self.assertTrue(registry.allows("GOLD", "trend_following"))

    def test_build_gate_registry_skips_value_errors(self) -> None:
        backtest = MagicMock()
        backtest.validate_symbol.side_effect = [
            _validation_report("EURUSD", "H1", True),
            ValueError("no bars"),
        ]
        regime_agent = MagicMock()
        regime_agent.assess.return_value = None

        registry = build_gate_registry(backtest, regime_agent, ["EURUSD", "GBPUSD"], "H1")
        self.assertEqual(len(registry.entries), 3)
        backtest.validate_symbol.assert_called()


class PortfolioGateFilterTests(unittest.TestCase):
    def test_scan_skips_blocked_trend_strategy(self) -> None:
        config = load_config()
        store = MagicMock()
        agent = PortfolioAgent(config, store)

        trend_sig = TradeSignal(
            symbol="#UK100",
            side=SignalSide.BUY,
            timeframe="H1",
            strength=0.8,
            reason="trend",
            strategy=StrategyKind.TREND_FOLLOWING,
        )
        mr_sig = TradeSignal(
            symbol="#UK100",
            side=SignalSide.SELL,
            timeframe="H1",
            strength=0.7,
            reason="mr",
            strategy=StrategyKind.MEAN_REVERSION,
        )

        agent.trend_agent.generate = MagicMock(return_value=trend_sig)
        agent.mr_agent.generate = MagicMock(return_value=mr_sig)
        agent.pair_agent.scan = MagicMock(return_value=[])
        agent.ml_agent.generate = MagicMock(return_value=None)

        regime = RegimeAssessment(
            symbol="#UK100",
            regime=MarketRegime.BULL,
            annualized_volatility=0.1,
            recent_return=0.01,
            recommended_mode=SignalMode.MOMENTUM,
            reason="test",
            adx=30.0,
            selected_strategy=StrategyKind.TREND_FOLLOWING,
            regime_label="stable_trend",
            position_scale=1.0,
            strategy_weights={"trend": 1.0, "mean_reversion": 0.0, "defensive": 0.0},
        )
        registry = GateRegistry(
            timeframe="H1",
            entries={
                ("#UK100", "trend_following"): GateEntry(
                    "#UK100", "H1", "trend_following", False
                ),
            },
        )

        signals = agent.scan(["#UK100"], {"#UK100": regime}, gate_registry=registry)
        strategies = {sig.strategy for sig in signals}
        self.assertNotIn(StrategyKind.TREND_FOLLOWING, strategies)
        self.assertNotIn(StrategyKind.MEAN_REVERSION, strategies)
        self.assertEqual(registry.blocked_count, 1)


if __name__ == "__main__":
    unittest.main()
