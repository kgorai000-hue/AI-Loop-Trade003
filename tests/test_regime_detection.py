from __future__ import annotations

import unittest

from src.regime.detection import evaluate_regime_value, rule_based_detect


class RegimeDetectionTests(unittest.TestCase):
    def test_period_a_trending(self) -> None:
        result = rule_based_detect(adx=35, annualized_volatility=0.18, recent_return=0.12, asset_correlation=0.5)
        self.assertEqual(result.label, "trending")

    def test_period_b_ranging(self) -> None:
        result = rule_based_detect(adx=15, annualized_volatility=0.08, recent_return=-0.02, asset_correlation=0.5)
        self.assertEqual(result.label, "mean_reverting")

    def test_period_c_crisis(self) -> None:
        result = rule_based_detect(adx=28, annualized_volatility=0.45, recent_return=-0.25, asset_correlation=0.5)
        self.assertEqual(result.label, "crisis")

    def test_period_d_transition(self) -> None:
        result = rule_based_detect(adx=22, annualized_volatility=0.12, recent_return=0.03, asset_correlation=0.5)
        self.assertEqual(result.label, "transition")
        self.assertTrue(result.is_transition)

    def test_scenario_3_crisis_correlation(self) -> None:
        result = rule_based_detect(
            adx=25,
            annualized_volatility=0.38,
            recent_return=-0.15,
            asset_correlation=0.85,
        )
        self.assertEqual(result.label, "crisis")

    def test_scenario_4_transition(self) -> None:
        result = rule_based_detect(
            adx=23,
            annualized_volatility=0.18,
            recent_return=0.03,
            asset_correlation=0.5,
        )
        self.assertEqual(result.label, "transition")

    def test_regime_value_lesson_example(self) -> None:
        value = evaluate_regime_value(0.08, 0.15, 24, 0.005)
        self.assertAlmostEqual(value["return_improvement"], 0.07, places=3)
        self.assertAlmostEqual(value["switch_cost_total"], 0.12, places=3)
        self.assertAlmostEqual(value["net_value"], -0.05, places=3)


if __name__ == "__main__":
    unittest.main()
