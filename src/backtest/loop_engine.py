from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.backtest_agent import BacktestAgent, StrategyValidationResult
from src.agents.regime_agent import RegimeAgent
from src.backtest.loop_criteria import (
    TrialVerdict,
    evaluate_trial,
    param_all_stable,
    should_stop_all_unstable,
    should_stop_baseline_degradation,
    summarize_metrics,
    wf_avg_test_sharpe,
)
from src.backtest.parameter_spaces import ParameterSpec, default_parameter_specs, specs_for_strategy
from src.core.config import TRADING_PROFILES, AppConfig, PROJECT_ROOT
from src.core.types import MarketRegime
from src.data.store import OHLCVStore

logger = logging.getLogger(__name__)

DEFAULT_LOOP_STRATEGIES = ("trend_following", "mean_reversion", "feature_score")


@dataclass
class LoopTrialRecord:
    parameter: str
    value: Any
    verdict: str
    reasons: list[str]
    metrics: dict[str, float]
    adopted: bool = False


@dataclass
class ParameterLoopReport:
    symbol: str
    timeframe: str
    strategy: str
    baseline_metrics: dict[str, float]
    trials: list[LoopTrialRecord] = field(default_factory=list)
    adopted: dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False
    stop_reason: str = ""


def clone_config(config: AppConfig) -> AppConfig:
    return copy.deepcopy(config)


def apply_config_overrides(config: AppConfig, overrides: list[tuple[str, Any]]) -> AppConfig:
    """Apply dot-path overrides to a cloned AppConfig."""
    cfg = clone_config(config)
    for path, value in overrides:
        parts = path.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        if path == "trading.profile" and value in TRADING_PROFILES:
            profile = TRADING_PROFILES[str(value)]
            cfg.trading.trades_per_day = int(profile["trades_per_day"])
            cfg.trading.primary_timeframe = str(profile["primary_timeframe"])
    return cfg


def _baseline_value(config: AppConfig, spec: ParameterSpec) -> Any:
    paths = spec.apply(spec.values[0])
    if len(paths) > 1:
        return tuple(_read_path(config, path) for path, _ in paths)
    return _read_path(config, paths[0][0])


def _read_path(config: AppConfig, path: str) -> Any:
    obj: Any = config
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _check_consecutive_hard_stops(
    consecutive_hard_stops: int,
    spec_name: str,
    loop_cfg,
    report: ParameterLoopReport,
) -> bool:
    if consecutive_hard_stops >= loop_cfg.consecutive_hard_stop_limit:
        report.stopped_early = True
        report.stop_reason = (
            f"{spec_name}: {loop_cfg.consecutive_hard_stop_limit} consecutive hard stops"
        )
        return True
    return False


def _evaluate_spec_stop_conditions(
    report: ParameterLoopReport,
    spec: ParameterSpec,
    initial_baseline_wf_sharpe: float,
    spec_trial_wf_sharpes: list[float],
    spec_stabilities: list[bool],
    loop_cfg,
) -> bool:
    """Return True if the outer parameter loop should stop after this spec."""
    if not spec_trial_wf_sharpes:
        return False

    spec_best_wf = max(spec_trial_wf_sharpes)
    stop, reason = should_stop_baseline_degradation(
        initial_baseline_wf_sharpe,
        spec_best_wf,
        loop_cfg,
    )
    if stop:
        report.stopped_early = True
        report.stop_reason = f"{spec.name}: {reason}"
        logger.warning("Loop stop after %s: %s", spec.name, reason)
        return True

    stop, reason = should_stop_all_unstable(spec_stabilities, loop_cfg)
    if stop:
        report.stopped_early = True
        report.stop_reason = f"{spec.name}: {reason}"
        logger.warning("Loop stop after %s: %s", spec.name, reason)
        return True

    return False


def run_parameter_loop(
    config: AppConfig,
    store: OHLCVStore,
    symbol: str,
    strategy_name: str,
    timeframe: str | None = None,
    *,
    specs: list[ParameterSpec] | None = None,
    regime: MarketRegime | None = None,
) -> ParameterLoopReport:
    """One-parameter-at-a-time optimization per loop criteria PDF."""
    timeframe = timeframe or config.stats.signal_timeframe
    loop_cfg = config.loop_engineering
    specs = specs or specs_for_strategy(strategy_name)
    if not specs:
        specs = [s for s in default_parameter_specs() if s.strategy == strategy_name]

    agent = BacktestAgent(config, store)
    baseline = agent.validate_strategy(symbol, strategy_name, timeframe, regime)
    initial_baseline_wf_sharpe = wf_avg_test_sharpe(baseline)
    report = ParameterLoopReport(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy_name,
        baseline_metrics=summarize_metrics(baseline, timeframe, config),
    )

    working_config = clone_config(config)
    best_result = baseline

    for spec in sorted(specs, key=lambda s: s.priority):
        consecutive_hard_stops = 0
        spec_trial_wf_sharpes: list[float] = []
        spec_stabilities: list[bool] = []
        current_val = _baseline_value(working_config, spec)
        logger.info("Loop parameter %s (current=%s)", spec.name, current_val)

        for value in spec.values:
            if _values_equal(value, current_val):
                continue

            trial_config = apply_config_overrides(working_config, spec.apply(value))
            trial_agent = BacktestAgent(trial_config, store)
            try:
                trial = trial_agent.validate_strategy(symbol, strategy_name, timeframe, regime)
            except ValueError as exc:
                report.trials.append(
                    LoopTrialRecord(
                        parameter=spec.name,
                        value=value,
                        verdict=TrialVerdict.HARD_STOP.value,
                        reasons=[str(exc)],
                        metrics={},
                    )
                )
                consecutive_hard_stops += 1
                if _check_consecutive_hard_stops(
                    consecutive_hard_stops, spec.name, loop_cfg, report
                ):
                    break
                continue

            evaluation = evaluate_trial(trial, best_result, trial_config, timeframe)
            metrics = summarize_metrics(trial, timeframe, trial_config)
            adopted = evaluation.verdict == TrialVerdict.TIER_B_ADOPT or (
                bool(getattr(loop_cfg, "adopt_tier_a", False))
                and evaluation.verdict == TrialVerdict.TIER_A
            )
            spec_trial_wf_sharpes.append(metrics["wf_avg_test_sharpe"])
            spec_stabilities.append(param_all_stable(trial))

            report.trials.append(
                LoopTrialRecord(
                    parameter=spec.name,
                    value=value,
                    verdict=evaluation.verdict.value,
                    reasons=evaluation.reasons,
                    metrics=metrics,
                    adopted=adopted,
                )
            )

            if evaluation.hard_stop:
                consecutive_hard_stops += 1
                if _check_consecutive_hard_stops(
                    consecutive_hard_stops, spec.name, loop_cfg, report
                ):
                    break
                continue

            consecutive_hard_stops = 0
            if adopted:
                best_result = trial
                working_config = apply_config_overrides(working_config, spec.apply(value))
                report.adopted[spec.name] = value
                current_val = value
                logger.info(
                    "Adopted %s=%s WF Sharpe=%.2f",
                    spec.name,
                    value,
                    metrics["wf_avg_test_sharpe"],
                )

        if report.stopped_early:
            break

        if _evaluate_spec_stop_conditions(
            report,
            spec,
            initial_baseline_wf_sharpe,
            spec_trial_wf_sharpes,
            spec_stabilities,
            loop_cfg,
        ):
            break

    return report


def save_loop_report(report: ParameterLoopReport, config: AppConfig) -> Path:
    out_dir = PROJECT_ROOT / config.loop_engineering.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{report.symbol}_{report.timeframe}_{report.strategy}_{stamp}.json"

    payload = {
        "symbol": report.symbol,
        "timeframe": report.timeframe,
        "strategy": report.strategy,
        "baseline_metrics": report.baseline_metrics,
        "adopted": report.adopted,
        "stopped_early": report.stopped_early,
        "stop_reason": report.stop_reason,
        "trials": [asdict(t) for t in report.trials],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _all_strategies_failed_baseline_gate(
    config: AppConfig,
    store: OHLCVStore,
    symbol: str,
    strategies: list[str],
    timeframe: str,
    regime: MarketRegime | None,
) -> bool:
    """PDF: skip symbol when every strategy fails quality gate at baseline."""
    agent = BacktestAgent(config, store)
    results: list[bool] = []

    for strategy in strategies:
        try:
            result = agent.validate_strategy(symbol, strategy, timeframe, regime)
        except ValueError:
            results.append(False)
            continue
        results.append(result.quality_gate.passed)

    return bool(results) and not any(results)


def run_full_loop_matrix(
    config: AppConfig,
    store: OHLCVStore,
    symbols: list[str],
    strategies: list[str],
    timeframe: str | None = None,
) -> list[ParameterLoopReport]:
    timeframe = timeframe or config.stats.signal_timeframe
    loop_cfg = config.loop_engineering
    regime_agent = RegimeAgent(config, store)
    reports: list[ParameterLoopReport] = []

    for symbol in symbols:
        regime = regime_agent.assess(symbol)
        market_regime = regime.regime if regime else None

        if loop_cfg.stop_on_all_strategies_gate_fail and _all_strategies_failed_baseline_gate(
            config,
            store,
            symbol,
            strategies,
            timeframe,
            market_regime,
        ):
            logger.warning(
                "Skip loop for %s %s: all strategies failed quality gate at baseline",
                symbol,
                timeframe,
            )
            continue

        for strategy in strategies:
            try:
                reports.append(
                    run_parameter_loop(
                        config,
                        store,
                        symbol,
                        strategy,
                        timeframe,
                        regime=market_regime,
                    )
                )
            except ValueError as exc:
                logger.warning("Loop skip %s %s %s: %s", symbol, timeframe, strategy, exc)
    return reports


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return a == b
    return a == b
