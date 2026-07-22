from __future__ import annotations

import unittest

from src.execution.position_sizing import conservative_kelly_cap, half_kelly
from src.risk.budget import inverse_drawdown_weights, scale_weights_to_budget
from src.risk.drawdown import evaluate_drawdown
from src.risk.kelly import bayesian_kelly, full_kelly, kelly_sample_discount
from src.risk.stops import atr_stop_distance, fixed_stop_distance, vol_stop_distance
from src.risk.types import DrawdownAction, DrawdownLevel


class RiskControlTests(unittest.TestCase):
    def test_full_kelly_lesson_table(self) -> None:
        self.assertAlmostEqual(full_kelly(0.60, 1.0), 0.20, places=3)
        self.assertAlmostEqual(full_kelly(0.55, 1.5), 0.25, places=3)
        self.assertAlmostEqual(full_kelly(0.50, 1.0), 0.0, places=3)

    def test_bayesian_kelly_conservative_below_mean(self) -> None:
        result = bayesian_kelly(60, 40, 0.02, 0.015)
        self.assertLess(float(result["kelly_conservative"]), float(result["kelly_mean"]))
        self.assertLess(float(result["recommendation"]), float(result["kelly_conservative"]))

    def test_kelly_sample_discount_small_sample(self) -> None:
        self.assertEqual(kelly_sample_discount(20), 0.0)
        self.assertEqual(kelly_sample_discount(50), 0.25)
        self.assertEqual(kelly_sample_discount(200), 0.5)

    def test_conservative_kelly_uses_bayesian_when_history(self) -> None:
        cap, method = conservative_kelly_cap(
            win_rate=0.55,
            reward_risk_ratio=1.5,
            trade_wins=60,
            trade_losses=40,
            avg_win_pct=0.02,
            avg_loss_pct=0.015,
        )
        self.assertEqual(method, "bayesian_kelly")
        self.assertGreater(cap, 0.0)
        self.assertLess(cap, half_kelly(0.55, 1.5))

    def test_drawdown_levels(self) -> None:
        normal = evaluate_drawdown(3.0)
        self.assertEqual(normal.level, DrawdownLevel.NORMAL)

        warning = evaluate_drawdown(6.0)
        self.assertEqual(warning.level, DrawdownLevel.WARNING)
        self.assertEqual(warning.action, DrawdownAction.REDUCE_RISK)
        self.assertAlmostEqual(warning.position_scale, 0.7)

        stop = evaluate_drawdown(12.0)
        self.assertEqual(stop.level, DrawdownLevel.STOP)
        self.assertFalse(stop.new_positions_allowed)

        circuit = evaluate_drawdown(16.0)
        self.assertEqual(circuit.level, DrawdownLevel.CIRCUIT)
        self.assertTrue(circuit.circuit_breaker_active)

    def test_inverse_drawdown_weights(self) -> None:
        drawdowns = {"A": 0.25, "B": 0.12, "C": 0.08}
        weights = inverse_drawdown_weights(drawdowns)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["C"], weights["A"])

    def test_scale_weights_to_budget(self) -> None:
        drawdowns = {"A": 0.25, "B": 0.12, "C": 0.08}
        weights = inverse_drawdown_weights(drawdowns)
        scaled = scale_weights_to_budget(
            weights,
            max_portfolio_drawdown=0.15,
            strategy_drawdowns=drawdowns,
        )
        expected_dd = sum(scaled[name] * drawdowns[name] for name in scaled)
        self.assertLessEqual(expected_dd, 0.15 + 1e-9)

    def test_stop_distances(self) -> None:
        self.assertAlmostEqual(fixed_stop_distance(100.0, 5.0), 5.0)
        self.assertAlmostEqual(atr_stop_distance(2.0, 2.0), 4.0)
        self.assertAlmostEqual(vol_stop_distance(100.0, 0.02, 2.5), 5.0)


if __name__ == "__main__":
    unittest.main()
