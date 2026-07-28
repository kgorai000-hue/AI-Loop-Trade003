from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.agents.backtest_agent import StrategyValidationResult
from src.backtest.overfitting import expected_live_return
from src.core.config import AppConfig, LoopEngineeringConfig


class TrialVerdict(str, Enum):
    HARD_STOP = "hard_stop"
    REJECT = "reject"
    TIER_A = "tier_a"
    TIER_B_ADOPT = "adopt"
    BASELINE = "baseline"


@dataclass
class TrialEvaluation:
    verdict: TrialVerdict
    tier_a: bool
    tier_b: bool
    hard_stop: bool
    reasons: list[str]


def min_trades_for_timeframe(timeframe: str, loop_cfg: LoopEngineeringConfig) -> int:
    if timeframe.upper() in ("M15", "M5", "M1"):
        return loop_cfg.min_trades_m15
    return loop_cfg.min_trades_h1


def _wf_avg_test_sharpe(result: StrategyValidationResult) -> float:
    return float(result.walk_forward_summary.get("avg_test_sharpe", 0.0))


def _wf_positive_pct(result: StrategyValidationResult) -> float:
    return float(result.walk_forward_summary.get("positive_rounds_pct", 0.0))


def _gate_pass_count(result: StrategyValidationResult) -> int:
    return sum(1 for c in result.quality_gate.checks if c.passed)


def _param_all_stable(result: StrategyValidationResult) -> bool:
    if not result.param_sensitivity:
        return True
    return all(p.stable for p in result.param_sensitivity)


def evaluate_trial(
    trial: StrategyValidationResult,
    baseline: StrategyValidationResult,
    config: AppConfig,
    timeframe: str,
) -> TrialEvaluation:
    """Loop criteria PDF: Tier A/B and hard stops."""
    loop_cfg = config.loop_engineering
    bt_cfg = config.backtest
    reasons: list[str] = []

    perf = trial.backtest.performance
    base_perf = baseline.backtest.performance
    wf_sharpe = _wf_avg_test_sharpe(trial)
    base_wf_sharpe = _wf_avg_test_sharpe(baseline)
    expected_live = trial.quality_gate.live_expected_return
    mdd_pct = perf.max_drawdown * 100.0
    base_mdd_pct = base_perf.max_drawdown * 100.0
    min_trades = min_trades_for_timeframe(timeframe, loop_cfg)

    # Hard stops (immediate discard)
    if perf.trades < loop_cfg.hard_stop_min_trades:
        reasons.append(f"trades {perf.trades} < {loop_cfg.hard_stop_min_trades}")
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    if mdd_pct > loop_cfg.hard_stop_mdd_pct:
        reasons.append(f"MDD {mdd_pct:.1f}% > {loop_cfg.hard_stop_mdd_pct}%")
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    if trial.oos.oos_ratio < loop_cfg.hard_stop_oos_ratio and trial.oos.test_return < 0:
        reasons.append(
            f"OOS collapse ratio={trial.oos.oos_ratio:.2f} test={trial.oos.test_return:.2%}"
        )
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    if expected_live < loop_cfg.hard_stop_expected_live_pct / 100.0:
        reasons.append(f"expected live {expected_live:.2%} < {loop_cfg.hard_stop_expected_live_pct}%")
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    mc_p5 = trial.monte_carlo.percentile_5
    if mc_p5 < loop_cfg.hard_stop_mc_p5_pct / 100.0:
        reasons.append(f"MC P5 {mc_p5:.2%} < {loop_cfg.hard_stop_mc_p5_pct}%")
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    if (
        perf.sharpe_ratio > loop_cfg.overfit_is_sharpe
        and wf_sharpe < loop_cfg.overfit_wf_sharpe
    ):
        reasons.append(f"overfit IS Sharpe={perf.sharpe_ratio:.2f} WF={wf_sharpe:.2f}")
        return TrialEvaluation(TrialVerdict.HARD_STOP, False, False, True, reasons)

    # Tier A (adoption candidate)
    tier_a = True
    if wf_sharpe < loop_cfg.min_wf_test_sharpe:
        tier_a = False
        reasons.append(f"WF Sharpe {wf_sharpe:.2f} < {loop_cfg.min_wf_test_sharpe}")
    if wf_sharpe < base_wf_sharpe + loop_cfg.wf_sharpe_improvement:
        tier_a = False
        reasons.append(
            f"WF Sharpe not improved by {loop_cfg.wf_sharpe_improvement} "
            f"({wf_sharpe:.2f} vs base {base_wf_sharpe:.2f})"
        )
    if trial.oos.oos_ratio < bt_cfg.min_oos_ratio:
        tier_a = False
        reasons.append(f"OOS ratio {trial.oos.oos_ratio:.2f} < {bt_cfg.min_oos_ratio}")
    if expected_live <= loop_cfg.min_expected_live_pct / 100.0:
        tier_a = False
        reasons.append(
            f"expected live {expected_live:.2%} <= {loop_cfg.min_expected_live_pct}%"
        )
    if mdd_pct > loop_cfg.max_mdd_pct:
        tier_a = False
        reasons.append(f"MDD {mdd_pct:.1f}% > {loop_cfg.max_mdd_pct}%")
    if mdd_pct > base_mdd_pct + loop_cfg.mdd_worsen_tolerance_pct:
        tier_a = False
        reasons.append(f"MDD worsened vs baseline ({mdd_pct:.1f}% vs {base_mdd_pct:.1f}%)")
    if perf.trades < min_trades:
        tier_a = False
        reasons.append(f"trades {perf.trades} < {min_trades}")

    if not tier_a:
        return TrialEvaluation(TrialVerdict.REJECT, False, False, False, reasons)

    # Tier B (robustness)
    tier_b = True
    if trial.monte_carlo.prob_positive < loop_cfg.tier_b_mc_prob_positive:
        tier_b = False
        reasons.append(f"MC prob+ {trial.monte_carlo.prob_positive:.1%}")
    if mc_p5 < loop_cfg.tier_b_mc_p5_pct / 100.0:
        tier_b = False
        reasons.append(f"MC P5 {mc_p5:.2%} < {loop_cfg.tier_b_mc_p5_pct}%")
    if _wf_positive_pct(trial) < loop_cfg.tier_b_wf_positive_pct:
        tier_b = False
        reasons.append(f"WF positive rounds {_wf_positive_pct(trial):.0%}")
    if not _param_all_stable(trial):
        tier_b = False
        reasons.append("parameter sensitivity unstable")
    gate_passes = _gate_pass_count(trial)
    if gate_passes < loop_cfg.tier_b_min_gate_passes:
        tier_b = False
        reasons.append(f"quality gate {gate_passes}/{len(trial.quality_gate.checks)}")

    if tier_b:
        reasons.append("Tier B passed")
        return TrialEvaluation(TrialVerdict.TIER_B_ADOPT, True, True, False, reasons)

    reasons.append("Tier A only (Tier B failed)")
    return TrialEvaluation(TrialVerdict.TIER_A, True, False, False, reasons)


def wf_avg_test_sharpe(result: StrategyValidationResult) -> float:
    return _wf_avg_test_sharpe(result)


def param_all_stable(result: StrategyValidationResult) -> bool:
    return _param_all_stable(result)


def should_stop_baseline_degradation(
    initial_baseline_wf_sharpe: float,
    spec_best_wf_sharpe: float,
    loop_cfg: LoopEngineeringConfig,
) -> tuple[bool, str]:
    """PDF: stop parameter loop if best value is still far below initial baseline."""
    threshold = initial_baseline_wf_sharpe - loop_cfg.baseline_wf_sharpe_stop_delta
    if spec_best_wf_sharpe < threshold:
        return (
            True,
            f"best WF Sharpe {spec_best_wf_sharpe:.2f} < initial baseline "
            f"{initial_baseline_wf_sharpe:.2f} - {loop_cfg.baseline_wf_sharpe_stop_delta}",
        )
    return False, ""


def should_stop_all_unstable(
    stabilities: list[bool],
    loop_cfg: LoopEngineeringConfig,
) -> tuple[bool, str]:
    """PDF: stop when every tested value has unstable parameter sensitivity."""
    if not loop_cfg.stop_on_all_unstable:
        return False, ""
    if stabilities and not any(stabilities):
        return True, "all tested values had unstable parameter sensitivity"
    return False, ""


def summarize_metrics(result: StrategyValidationResult, timeframe: str, config: AppConfig) -> dict[str, float]:
    perf = result.backtest.performance
    return {
        "wf_avg_test_sharpe": _wf_avg_test_sharpe(result),
        "oos_ratio": result.oos.oos_ratio,
        "expected_live_return": result.quality_gate.live_expected_return,
        "max_drawdown_pct": perf.max_drawdown * 100.0,
        "trades": float(perf.trades),
        "in_sample_sharpe": perf.sharpe_ratio,
        "profit_factor": perf.profit_factor,
        "mc_prob_positive": result.monte_carlo.prob_positive,
        "mc_p5": result.monte_carlo.percentile_5,
        "gate_passes": float(_gate_pass_count(result)),
        "min_trades_required": float(min_trades_for_timeframe(timeframe, config.loop_engineering)),
        "param_sensitivity_stable": 1.0 if _param_all_stable(result) else 0.0,
    }
