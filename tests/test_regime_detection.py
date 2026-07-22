from __future__ import annotations

import unittest

import numpy as np

from src.regime.detection import evaluate_regime_value, rule_based_detect
from src.regime.five_state import FiveStateThresholds, classify_five_state
from src.regime.scores import RegimeScores, efficiency_ratio
from src.regime.series import mask_signals_for_strategy
from src.regime.states import FiveState, strategy_for_five_state
from src.core.types import StrategyKind


class LegacyRuleBasedTests(unittest.TestCase):
    def test_period_a_trending(self) -> None:
        result = rule_based_detect(adx=35, annualized_volatility=0.18, recent_return=0.12, asset_correlation=0.5)
        self.assertEqual(result.label, "trending")

    def test_period_b_ranging(self) -> None:
        result = rule_based_detect(adx=15, annualized_volatility=0.08, recent_return=-0.02, asset_correlation=0.5)
        self.assertEqual(result.label, "mean_reverting")

    def test_period_c_crisis(self) -> None:
        result = rule_based_detect(adx=28, annualized_volatility=0.45, recent_return=-0.25, asset_correlation=0.5)
        self.assertEqual(result.label, "crisis")

    def test_regime_value_lesson_example(self) -> None:
        value = evaluate_regime_value(0.08, 0.15, 24, 0.005)
        self.assertAlmostEqual(value["return_improvement"], 0.07, places=3)
        self.assertAlmostEqual(value["switch_cost_total"], 0.12, places=3)
        self.assertAlmostEqual(value["net_value"], -0.05, places=3)


class FiveStateTests(unittest.TestCase):
    def _scores(self, **kwargs) -> RegimeScores:
        base = dict(
            trend_score=0.0,
            vol_percentile=0.5,
            efficiency_ratio=0.3,
            slope=0.0,
            slope_t=0.0,
            asset_correlation=0.3,
            spread_stress=0.0,
            reason_bits=[],
        )
        base.update(kwargs)
        return RegimeScores(**base)

    def test_stable_trend(self) -> None:
        result = classify_five_state(
            self._scores(trend_score=0.8, efficiency_ratio=0.6, vol_percentile=0.4)
        )
        self.assertEqual(result.label, FiveState.STABLE_TREND.value)
        self.assertEqual(strategy_for_five_state(result.label), StrategyKind.TREND_FOLLOWING)

    def test_high_vol_trend(self) -> None:
        result = classify_five_state(
            self._scores(trend_score=0.8, efficiency_ratio=0.6, vol_percentile=0.8)
        )
        self.assertEqual(result.label, FiveState.HIGH_VOL_TREND.value)

    def test_stable_range(self) -> None:
        result = classify_five_state(
            self._scores(trend_score=0.1, efficiency_ratio=0.2, vol_percentile=0.4)
        )
        self.assertEqual(result.label, FiveState.STABLE_RANGE.value)
        self.assertEqual(strategy_for_five_state(result.label), StrategyKind.MEAN_REVERSION)

    def test_high_vol_chop_halts(self) -> None:
        result = classify_five_state(
            self._scores(trend_score=0.1, efficiency_ratio=0.2, vol_percentile=0.85)
        )
        self.assertEqual(result.label, FiveState.HIGH_VOL_CHOP.value)
        self.assertEqual(strategy_for_five_state(result.label), StrategyKind.CRISIS_HALT)

    def test_stress(self) -> None:
        result = classify_five_state(
            self._scores(trend_score=0.2, efficiency_ratio=0.3, vol_percentile=0.95)
        )
        self.assertEqual(result.label, FiveState.STRESS.value)

    def test_hysteresis_holds_trend(self) -> None:
        th = FiveStateThresholds(exit_trend_score=0.35, exit_er_trend=0.30)
        # Weaker than enter but above exit → stay in trend
        result = classify_five_state(
            self._scores(trend_score=0.4, efficiency_ratio=0.35, vol_percentile=0.4),
            previous=FiveState.STABLE_TREND.value,
            thresholds=th,
        )
        self.assertEqual(result.label, FiveState.STABLE_TREND.value)

    def test_efficiency_ratio_bounds(self) -> None:
        closes = np.linspace(100, 110, 21)
        self.assertGreater(efficiency_ratio(closes, 20), 0.9)
        zig = np.array([100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 100], dtype=float)
        self.assertLess(efficiency_ratio(zig, 10), 0.2)

    def test_mask_signals(self) -> None:
        signals = np.array([1.0, 1.0, 1.0, 1.0])
        labels = np.array(
            [
                FiveState.STABLE_TREND.value,
                FiveState.STABLE_RANGE.value,
                FiveState.HIGH_VOL_CHOP.value,
                FiveState.HIGH_VOL_TREND.value,
            ],
            dtype=object,
        )
        masked = mask_signals_for_strategy(signals, labels, "trend_following")
        self.assertEqual(list(masked), [1.0, 0.0, 0.0, 1.0])
        masked_mr = mask_signals_for_strategy(signals, labels, "mean_reversion")
        self.assertEqual(list(masked_mr), [0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
