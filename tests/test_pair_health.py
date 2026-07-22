from __future__ import annotations

import unittest

import numpy as np

from src.core.types import MarketRegime, RegimeAssessment, SignalMode, StrategyKind
from src.pairs.health import PairHealthThresholds, classify_pair_health
from src.pairs.spread import build_spread_snapshot, rolling_ols_beta
from src.pairs.states import PairRegime
from src.strategies.pairs import compute_spread_zscore, evaluate_pair


class SpreadBetaTests(unittest.TestCase):
    def test_rolling_beta_recovers_known_ratio(self) -> None:
        rng = np.random.default_rng(0)
        t = np.arange(200, dtype=float)
        # B random walk; A = 1.5 * B in log space + noise → beta ≈ 1.5
        log_b = np.cumsum(rng.normal(0, 0.01, size=200))
        log_a = 0.1 + 1.5 * log_b + rng.normal(0, 0.001, size=200)
        a = np.exp(log_a)
        b = np.exp(log_b)
        beta, se = rolling_ols_beta(np.log(a), np.log(b), 80)
        self.assertAlmostEqual(beta, 1.5, delta=0.05)
        self.assertGreater(se, 0.0)

    def test_mean_reverting_spread_has_finite_half_life(self) -> None:
        rng = np.random.default_rng(1)
        # Cointegrated-ish: common factor + MR residual
        common = np.cumsum(rng.normal(0, 0.01, size=300))
        resid = np.zeros(300)
        for i in range(1, 300):
            resid[i] = 0.7 * resid[i - 1] + rng.normal(0, 0.005)
        a = np.exp(common + resid)
        b = np.exp(common)
        snap = build_spread_snapshot(a, b, z_lookback=40, beta_window=60)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertIsNotNone(snap.half_life)
        self.assertLess(snap.half_life or 999, 40)

    def test_evaluate_pair_with_beta(self) -> None:
        rng = np.random.default_rng(2)
        common = np.cumsum(rng.normal(0, 0.01, size=250))
        resid = np.zeros(250)
        for i in range(1, 250):
            resid[i] = 0.85 * resid[i - 1] + rng.normal(0, 0.01)
        # Force a large positive residual at the end
        resid[-1] = 0.15
        a = np.exp(common + resid)
        b = np.exp(common)
        z = compute_spread_zscore(a, b, 30, beta_window=60)
        self.assertIsNotNone(z)
        leg_a, leg_b = evaluate_pair(a, b, 30, 1.5, 0.5, "A", "B", beta_window=60)
        # Large positive z → short A / long B
        if z is not None and z >= 1.5:
            self.assertIsNotNone(leg_a)
            self.assertIsNotNone(leg_b)


class PairHealthTests(unittest.TestCase):
    def _snap_from_series(self, a: np.ndarray, b: np.ndarray):
        return build_spread_snapshot(a, b, z_lookback=40, beta_window=60)

    def test_r4_on_extreme_beta_drift(self) -> None:
        rng = np.random.default_rng(3)
        # First half β≈1, second half β≈3 → large drift
        n = 200
        b = np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
        a = np.empty(n)
        a[:100] = b[:100] * np.exp(rng.normal(0, 0.001, size=100))
        a[100:] = (b[100:] ** 3) * np.exp(rng.normal(0, 0.001, size=100))
        snap = self._snap_from_series(a, b)
        self.assertIsNotNone(snap)
        assert snap is not None
        th = PairHealthThresholds(break_beta_drift=0.25, max_beta_drift=0.1)
        health = classify_pair_health(
            snap,
            closes_a=a,
            closes_b=b,
            thresholds=th,
            beta_short_window=40,
            beta_long_window=120,
        )
        self.assertIn(
            health.regime,
            {PairRegime.R4_STRUCTURAL_BREAK.value, PairRegime.R3_WEAKENING.value},
        )
        self.assertFalse(health.allow_entry)

    def test_r5_on_leg_stress(self) -> None:
        rng = np.random.default_rng(4)
        common = np.cumsum(rng.normal(0, 0.01, size=200))
        a = np.exp(common)
        b = np.exp(common + rng.normal(0, 0.001, size=200))
        snap = self._snap_from_series(a, b)
        assert snap is not None
        stress = RegimeAssessment(
            symbol="A",
            regime=MarketRegime.CRISIS,
            annualized_volatility=0.5,
            recent_return=-0.1,
            recommended_mode=SignalMode.NONE,
            reason="stress",
            regime_label="stress",
            selected_strategy=StrategyKind.CRISIS_HALT,
            position_scale=0.0,
        )
        health = classify_pair_health(
            snap,
            closes_a=a,
            closes_b=b,
            leg_a_regime=stress,
            leg_b_regime=None,
        )
        self.assertEqual(health.regime, PairRegime.R5_EVENT_EXEC.value)
        self.assertFalse(health.allow_entry)

    def test_single_regime_is_reference_not_hard_block(self) -> None:
        """Different leg families alone must not force skip if relationship is healthy."""
        rng = np.random.default_rng(5)
        common = np.cumsum(rng.normal(0, 0.008, size=250))
        resid = np.zeros(250)
        for i in range(1, 250):
            resid[i] = 0.6 * resid[i - 1] + rng.normal(0, 0.003)
        a = np.exp(common + resid)
        b = np.exp(common)
        snap = self._snap_from_series(a, b)
        assert snap is not None
        trend_leg = RegimeAssessment(
            symbol="A",
            regime=MarketRegime.BULL,
            annualized_volatility=0.15,
            recent_return=0.05,
            recommended_mode=SignalMode.MOMENTUM,
            reason="trend",
            regime_label="stable_trend",
            selected_strategy=StrategyKind.TREND_FOLLOWING,
            position_scale=1.0,
        )
        range_leg = RegimeAssessment(
            symbol="B",
            regime=MarketRegime.SIDEWAYS,
            annualized_volatility=0.1,
            recent_return=0.0,
            recommended_mode=SignalMode.MEAN_REVERSION,
            reason="range",
            regime_label="stable_range",
            selected_strategy=StrategyKind.MEAN_REVERSION,
            position_scale=1.0,
        )
        health = classify_pair_health(
            snap,
            closes_a=a,
            closes_b=b,
            thresholds=PairHealthThresholds(
                max_half_life_bars=80,
                max_beta_drift=0.5,
                max_abs_trend_slope=0.01,
                min_zero_cross_rate=0.0,
            ),
            leg_a_regime=trend_leg,
            leg_b_regime=range_leg,
        )
        self.assertIn("stable_trend/stable_range", health.single_regime_note)
        # Should not be R5 merely from mixed families
        self.assertNotEqual(health.regime, PairRegime.R5_EVENT_EXEC.value)


if __name__ == "__main__":
    unittest.main()
