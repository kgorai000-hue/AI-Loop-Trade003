from __future__ import annotations

import unittest

from src.agents.arbitration import resolve_signal_conflicts, vote_on_proposal
from src.core.types import (
    ArbitrationMode,
    MarketRegime,
    RegimeAssessment,
    SignalMode,
    SignalSide,
    StrategyKind,
    TradeSignal,
)


class ArbitrationTests(unittest.TestCase):
    def test_hierarchy_picks_strongest(self) -> None:
        signals = [
            TradeSignal("EURUSD", SignalSide.BUY, "H1", 0.4, "weak"),
            TradeSignal("EURUSD", SignalSide.SELL, "H1", 0.9, "strong"),
        ]
        resolved, notes = resolve_signal_conflicts(signals, ArbitrationMode.HIERARCHY)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].side, SignalSide.SELL)
        self.assertEqual(len(notes), 1)

    def test_voting_lesson_scenario_approved(self) -> None:
        signal = TradeSignal(
            symbol="AAPL",
            side=SignalSide.BUY,
            timeframe="H1",
            strength=0.8,
            reason="demo",
            mode=SignalMode.MOMENTUM,
            strategy=StrategyKind.TREND_FOLLOWING,
        )
        regime = RegimeAssessment(
            symbol="AAPL",
            regime=MarketRegime.BULL,
            annualized_volatility=0.2,
            recent_return=0.05,
            recommended_mode=SignalMode.MOMENTUM,
            reason="trend",
            selected_strategy=StrategyKind.TREND_FOLLOWING,
        )
        result = vote_on_proposal(signal, regime, exposure_pct=50.0, max_exposure_pct=60.0)
        self.assertTrue(result.approved)
        self.assertGreater(result.net_score, 0)

    def test_voting_risk_rejects_high_exposure(self) -> None:
        signal = TradeSignal(
            symbol="AAPL",
            side=SignalSide.BUY,
            timeframe="H1",
            strength=0.8,
            reason="demo",
            mode=SignalMode.MOMENTUM,
        )
        regime = RegimeAssessment(
            symbol="AAPL",
            regime=MarketRegime.BULL,
            annualized_volatility=0.2,
            recent_return=0.05,
            recommended_mode=SignalMode.MOMENTUM,
            reason="trend",
        )
        result = vote_on_proposal(signal, regime, exposure_pct=65.0, max_exposure_pct=60.0)
        self.assertFalse(result.approved)


if __name__ == "__main__":
    unittest.main()
