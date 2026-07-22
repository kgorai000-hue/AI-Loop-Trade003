from __future__ import annotations

import unittest

from src.regime.health import detect_uncertain_triggers
from src.regime.misjudgment import expected_monthly_return, lag_cost_table
from src.regime.resilience import DegradationLevel, determine_degradation_level


class ResilienceTests(unittest.TestCase):
    def test_degradation_normal_when_clear(self) -> None:
        level = determine_degradation_level(
            max_probability=0.80,
            clear_prob_threshold=0.70,
            uncertain_prob_threshold=0.50,
            health_fail=False,
            data_quality_ok=True,
        )
        self.assertEqual(level, DegradationLevel.NORMAL)

    def test_degradation_safe_when_data_bad(self) -> None:
        level = determine_degradation_level(
            max_probability=0.90,
            clear_prob_threshold=0.70,
            uncertain_prob_threshold=0.50,
            health_fail=False,
            data_quality_ok=False,
        )
        self.assertEqual(level, DegradationLevel.SAFE)

    def test_uncertain_low_probability(self) -> None:
        uncertain, reasons = detect_uncertain_triggers(
            max_probability=0.40,
            uncertain_prob_threshold=0.50,
            adx=24.0,
            adx_boundary_low=22.0,
            adx_boundary_high=28.0,
            annualized_volatility=0.18,
            ranging_vol_max=0.15,
            trending_adx_min=25.0,
            pending_switch=False,
            post_switch_days=10,
            post_switch_uncertain_days=2,
        )
        self.assertTrue(uncertain)
        self.assertTrue(any("probability" in r for r in reasons))

    def test_accuracy_impact_70pct(self) -> None:
        impact = expected_monthly_return(0.70)
        self.assertLess(impact["monthly_return"], impact["perfect_monthly_return"])
        self.assertGreater(impact["return_reduction_pct"], 0.0)

    def test_lag_cost_table_has_entries(self) -> None:
        table = lag_cost_table()
        self.assertIn(3, table)
        self.assertGreater(table[3]["loss_at_confirm_pct"], table[1]["loss_at_confirm_pct"])


if __name__ == "__main__":
    unittest.main()
