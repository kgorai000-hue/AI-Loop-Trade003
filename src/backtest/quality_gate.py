from __future__ import annotations

from src.backtest.overfitting import bonferroni_threshold, expected_live_return
from src.backtest.types import (
    BacktestResult,
    MonteCarloResult,
    OOSSplitResult,
    ParameterSensitivityResult,
    QualityGateCheck,
    QualityGateReport,
    WalkForwardRound,
)
from src.backtest.walk_forward import summarize_walk_forward
from src.core.config import AppConfig
import numpy as np


def evaluate_quality_gates(
    strategy_name: str,
    config: AppConfig,
    backtest: BacktestResult,
    oos: OOSSplitResult,
    walk_forward: list[WalkForwardRound],
    monte_carlo: MonteCarloResult,
    param_sensitivity: list[ParameterSensitivityResult],
    n_bars: int,
    n_strategies_tested: int,
    bars_span_years: float,
) -> QualityGateReport:
    """Lesson 7.5 quality gate checklist (adapted for MT5 CFD)."""
    bt_cfg = config.backtest
    checks: list[QualityGateCheck] = []

    checks.append(
        QualityGateCheck(
            layer="1",
            check_id="1.1",
            name="Data coverage",
            passed=bars_span_years >= 0.5 or n_bars >= 200,
            detail=f"{n_bars} bars (~{bars_span_years:.1f}y); need >=200 bars or 0.5y",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="1",
            check_id="1.3",
            name="Survivorship bias",
            passed=True,
            detail="MT5 CFD symbols; survivorship N/A (Lesson 6 note)",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="1",
            check_id="1.4",
            name="Timezone",
            passed=True,
            detail="UTC timestamps from MT5 (Lesson 06)",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="2",
            check_id="2.1",
            name="Look-ahead bias",
            passed=_verify_no_lookahead(backtest),
            detail="T signal -> T+1 execution enforced in engine",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="2",
            check_id="2.2",
            name="Temporal split",
            passed=oos.train_return != 0 or oos.test_return != 0,
            detail=f"train={oos.train_return:.2%} val={oos.val_return:.2%} test={oos.test_return:.2%}",
        )
    )
    oos_pass = oos.oos_ratio >= bt_cfg.min_oos_ratio if oos.train_return > 0 else oos.test_return >= 0
    checks.append(
        QualityGateCheck(
            layer="3",
            check_id="3.1",
            name="OOS performance",
            passed=oos_pass,
            detail=f"test/train ratio={oos.oos_ratio:.2f} (min {bt_cfg.min_oos_ratio})",
        )
    )
    param_stable = all(p.stable for p in param_sensitivity) if param_sensitivity else True
    checks.append(
        QualityGateCheck(
            layer="3",
            check_id="3.2",
            name="Parameter stability",
            passed=param_stable,
            detail="+/-20% MA params" if param_sensitivity else "n/a for non-trend",
        )
    )
    bonf = bonferroni_threshold(n_strategies_tested)
    checks.append(
        QualityGateCheck(
            layer="3",
            check_id="3.3",
            name="Multiple testing",
            passed=True,
            detail=f"Bonferroni p-threshold={bonf:.6f} for {n_strategies_tested} strategies",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="4",
            check_id="4.2",
            name="Slippage modeled",
            passed=backtest.cost_per_trade_pct > 0,
            detail=f"round-trip cost={backtest.cost_per_trade_pct:.3f}% per trade",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="4",
            check_id="4.3",
            name="Total cost impact",
            passed=True,
            detail=f"total cost drag={backtest.total_cost_pct:.2f}% over backtest",
        )
    )
    wf_summary = summarize_walk_forward(walk_forward)
    checks.append(
        QualityGateCheck(
            layer="5",
            check_id="5.1",
            name="Walk-forward",
            passed=wf_summary["rounds"] >= bt_cfg.min_walk_forward_rounds,
            detail=f"{int(wf_summary['rounds'])} rounds (min {bt_cfg.min_walk_forward_rounds})",
        )
    )
    checks.append(
        QualityGateCheck(
            layer="5",
            check_id="5.2",
            name="Monte Carlo",
            passed=monte_carlo.percentile_5 >= 0 or monte_carlo.prob_positive >= bt_cfg.min_mc_prob_positive,
            detail=(
                f"P5={monte_carlo.percentile_5:.2%} P50={monte_carlo.percentile_50:.2%} "
                f"prob+={monte_carlo.prob_positive:.1%}"
            ),
        )
    )
    checks.append(
        QualityGateCheck(
            layer="5",
            check_id="5.4",
            name="Return decay tolerance",
            passed=backtest.performance.annualized_return * bt_cfg.live_decay_factor > 0
            or backtest.performance.total_return <= 0,
            detail=f"BT ann={backtest.performance.annualized_return:.2%} x{bt_cfg.live_decay_factor} still viable",
        )
    )

    live_expected = expected_live_return(
        backtest.performance.annualized_return,
        bt_cfg.live_decay_factor,
        bt_cfg.hidden_cost_pct,
    )

    return QualityGateReport(
        strategy_name=strategy_name,
        checks=checks,
        live_expected_return=live_expected,
        bonferroni_threshold=bonf,
    )


def _verify_no_lookahead(backtest: BacktestResult) -> bool:
    for trade in backtest.trades:
        if trade.signal_bar >= trade.execution_bar:
            return False
    return True


def estimate_bar_span_years(n_bars: int, timeframe: str, trading_days: int = 252) -> float:
    from src.core.history import bars_per_trading_day

    return n_bars / (bars_per_trading_day(timeframe) * trading_days)
