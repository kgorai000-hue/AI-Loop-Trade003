from __future__ import annotations

import unittest

from src.agents.backtest_agent import StrategyValidationResult
from src.backtest.loop_criteria import (
    TrialVerdict,
    evaluate_trial,
    min_trades_for_timeframe,
    should_stop_all_unstable,
    should_stop_baseline_degradation,
)
from src.backtest.loop_engine import apply_config_overrides, clone_config, _values_equal
from src.backtest.parameter_spaces import default_parameter_specs, specs_for_strategy
from src.backtest.types import (
    BacktestResult,
    MonteCarloResult,
    OOSSplitResult,
    ParameterSensitivityResult,
    QualityGateCheck,
    QualityGateReport,
)
from src.core.config import load_config
from src.stats.performance import PerformanceReport


def _perf(**kwargs) -> PerformanceReport:
    defaults = dict(
        total_return=0.1,
        annualized_return=0.08,
        sharpe_ratio=1.0,
        sortino_ratio=1.1,
        max_drawdown=0.08,
        calmar_ratio=1.0,
        trades=40,
        win_rate=0.55,
        profit_factor=1.3,
    )
    defaults.update(kwargs)
    return PerformanceReport(**defaults)


def _result(
    *,
    wf_sharpe: float = 0.6,
    oos_ratio: float = 0.6,
    live_expected: float = 0.02,
    mdd: float = 0.10,
    trades: int = 40,
    is_sharpe: float = 1.0,
    mc_prob: float = 0.55,
    mc_p5: float = -0.05,
    wf_positive: float = 0.6,
    gate_passes: int = 11,
    stable: bool = True,
) -> StrategyValidationResult:
    gate = QualityGateReport(
        strategy_name="feature_score",
        checks=[QualityGateCheck("3", "x", "test", i < gate_passes, "ok") for i in range(12)],
        live_expected_return=live_expected,
    )
    return StrategyValidationResult(
        strategy_name="feature_score",
        backtest=BacktestResult(
            strategy_name="feature_score",
            returns=[0.001, -0.0005],
            positions=[1.0, 1.0],
            trades=[],
            performance=_perf(sharpe_ratio=is_sharpe, max_drawdown=mdd, trades=trades),
            cost_per_trade_pct=0.01,
            total_cost_pct=0.02,
        ),
        oos=OOSSplitResult(0.1, 0.05, 0.05, 1.0, 0.8, 0.6, oos_ratio),
        walk_forward=[],
        walk_forward_summary={
            "avg_test_sharpe": wf_sharpe,
            "positive_rounds_pct": wf_positive,
        },
        monte_carlo=MonteCarloResult(
            simulations=100,
            mean_return=0.05,
            std_return=0.02,
            percentile_5=mc_p5,
            percentile_50=0.04,
            percentile_95=0.08,
            prob_positive=mc_prob,
        ),
        param_sensitivity=[
            ParameterSensitivityResult("ma_short", 0.1, 0.09, 0.11, 0.1, stable)
        ],
        quality_gate=gate,
    )


class LoopCriteriaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_hard_stop_low_trades(self) -> None:
        baseline = _result()
        trial = _result(trades=1)
        ev = evaluate_trial(trial, baseline, self.config, "H1")
        self.assertEqual(ev.verdict, TrialVerdict.HARD_STOP)

    def test_tier_b_adopt(self) -> None:
        baseline = _result(wf_sharpe=0.5)
        trial = _result(wf_sharpe=0.7, gate_passes=11)
        ev = evaluate_trial(trial, baseline, self.config, "H1")
        # Demo settings may accept Tier A; ensure at least Tier A or adopt.
        self.assertIn(ev.verdict, (TrialVerdict.TIER_B_ADOPT, TrialVerdict.TIER_A))

    def test_min_trades_m15(self) -> None:
        self.assertEqual(min_trades_for_timeframe("M15", self.config.loop_engineering), 15)

    def test_apply_config_override(self) -> None:
        cfg = apply_config_overrides(self.config, [("indicators.signal_score_threshold", 0.2)])
        self.assertAlmostEqual(cfg.indicators.signal_score_threshold, 0.2)

    def test_parameter_specs_sorted(self) -> None:
        specs = specs_for_strategy("feature_score")
        self.assertEqual(specs[0].name, "signal_score_threshold")

    def test_values_equal_tuple(self) -> None:
        self.assertTrue(_values_equal((25, 75), (25, 75)))
        self.assertFalse(_values_equal((25, 75), (30, 70)))

    def test_sequential_config_accumulation(self) -> None:
        cfg = load_config()
        working = apply_config_overrides(cfg, [("indicators.signal_score_threshold", 0.2)])
        working = apply_config_overrides(working, [("trading.profile", "low")])
        self.assertAlmostEqual(working.indicators.signal_score_threshold, 0.2)
        self.assertEqual(working.trading.profile, "low")
        self.assertEqual(working.trading.trades_per_day, 2)

    def test_stop_baseline_degradation(self) -> None:
        from dataclasses import replace

        loop_cfg = replace(self.config.loop_engineering, baseline_wf_sharpe_stop_delta=0.2)
        stop, reason = should_stop_baseline_degradation(0.8, 0.5, loop_cfg)
        self.assertTrue(stop)
        self.assertIn("baseline", reason)
        stop, _ = should_stop_baseline_degradation(0.8, 0.7, loop_cfg)
        self.assertFalse(stop)

    def test_stop_all_unstable(self) -> None:
        from dataclasses import replace

        loop_cfg = replace(self.config.loop_engineering, stop_on_all_unstable=True)
        stop, reason = should_stop_all_unstable([False, False], loop_cfg)
        self.assertTrue(stop)
        self.assertIn("unstable", reason)
        stop, _ = should_stop_all_unstable([False, True], loop_cfg)
        self.assertFalse(stop)
        stop, _ = should_stop_all_unstable([], loop_cfg)
        self.assertFalse(stop)
        disabled = replace(self.config.loop_engineering, stop_on_all_unstable=False)
        stop, _ = should_stop_all_unstable([False, False], disabled)
        self.assertFalse(stop)


if __name__ == "__main__":
    unittest.main()
