from __future__ import annotations

import unittest

import numpy as np

from src.online.alpha_decay import expected_annual_return_from_ic, projected_ic
from src.online.decision import RetrainDecisionEngine
from src.online.drift import calculate_psi, sliding_accuracy_drift
from src.online.ewm_model import ExponentialMovingModel, effective_lookback_days
from src.online.scheduler import AdaptiveUpdateScheduler
from src.online.signal_adaptation import dynamic_signal_threshold
from src.online.strategy_lifecycle import (
    build_strategy_lifecycle_states,
    classify_lifecycle_stage,
    inverse_vol_strategy_weights,
    sharpe_proportional_weights,
)
from src.online.types import LifecycleStage, UpdateAction


class OnlineLearningTests(unittest.TestCase):
    def test_alpha_decay_lesson_table(self) -> None:
        ic6 = projected_ic(0.05, 0.05, 6)
        ic12 = projected_ic(0.05, 0.05, 12)
        self.assertAlmostEqual(ic6, 0.037, places=2)
        self.assertAlmostEqual(ic12, 0.027, places=2)

    def test_expected_return_from_ic(self) -> None:
        ann = expected_annual_return_from_ic(0.05)
        self.assertAlmostEqual(ann, 0.158, places=2)

    def test_effective_lookback(self) -> None:
        self.assertAlmostEqual(effective_lookback_days(0.95), 44, delta=2)
        self.assertAlmostEqual(effective_lookback_days(0.90), 21, delta=2)

    def test_exponential_moving_model_updates(self) -> None:
        model = ExponentialMovingModel(0.95)
        x = np.array([1.0, 0.0])
        pred, err = model.update(x, 1.0, learning_rate=0.1)
        self.assertIsInstance(pred, float)
        self.assertIsInstance(err, float)
        self.assertGreater(model.get_effective_lookback(), 0)

    def test_dynamic_threshold_high_vol(self) -> None:
        np.random.seed(0)
        normal = list(np.random.normal(0.30, 0.15, 30))
        high_vol = list(np.random.normal(0.35, 0.25, 30))
        self.assertGreater(dynamic_signal_threshold(high_vol), dynamic_signal_threshold(normal))

    def test_psi_detects_shift(self) -> None:
        baseline = np.random.normal(0, 1, 200)
        shifted = np.random.normal(2, 1, 200)
        psi = calculate_psi(baseline, shifted)
        self.assertGreater(psi, 0.25)

    def test_sliding_accuracy_drift(self) -> None:
        detected, avg = sliding_accuracy_drift([0.55, 0.52, 0.48, 0.45, 0.42], window=5, threshold=0.5)
        self.assertTrue(detected)
        self.assertLess(avg, 0.5)

    def test_retrain_decision_matrix(self) -> None:
        engine = RetrainDecisionEngine(performance_drop_threshold=0.3)
        retrain = engine.decide(drift_detected=True, ic_change=-0.40)
        pause = engine.decide(drift_detected=False, ic_change=-0.35)
        cont = engine.decide(drift_detected=False, ic_change=-0.05)
        self.assertEqual(retrain.action, UpdateAction.RETRAIN)
        self.assertEqual(pause.action, UpdateAction.PAUSE)
        self.assertEqual(cont.action, UpdateAction.CONTINUE)

    def test_strategy_lifecycle_stages(self) -> None:
        self.assertEqual(classify_lifecycle_stage(1.6), LifecycleStage.MATURITY)
        self.assertEqual(classify_lifecycle_stage(0.5), LifecycleStage.DECAY)

    def test_strategy_weight_methods(self) -> None:
        inv = inverse_vol_strategy_weights({"A": 0.05, "B": 0.08, "C": 0.08})
        sh = sharpe_proportional_weights({"A": 1.6, "B": 0.75, "C": 1.25})
        self.assertAlmostEqual(inv["A"], 0.44, places=1)
        self.assertAlmostEqual(sh["A"], 0.44, places=1)

    def test_build_strategy_lifecycle_states(self) -> None:
        states = build_strategy_lifecycle_states(
            {"trend": (0.12, 0.08), "mean_reversion": (0.04, 0.10)}
        )
        self.assertEqual(len(states), 2)


if __name__ == "__main__":
    unittest.main()
