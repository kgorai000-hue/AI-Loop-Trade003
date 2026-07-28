"""Validator adapter: BacktestAgent + loop_criteria evaluate_trial."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.backtest_agent import BacktestAgent, StrategyValidationResult
from src.agents.regime_agent import RegimeAgent
from src.backtest.loop_criteria import TrialEvaluation, TrialVerdict, evaluate_trial
from src.backtest.loop_engine import apply_config_overrides
from src.core.config import AppConfig
from src.data.store import OHLCVStore
from src.intelligence.params import LoopParams, overrides_to_tuples

logger = logging.getLogger(__name__)


@dataclass
class ValidationOutcome:
    accepted: bool
    verdict: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    trial: StrategyValidationResult | None = None


class ParamValidator:
    """Mathematical validator using Trade003 backtest quality gates."""

    def __init__(
        self,
        config: AppConfig,
        store: OHLCVStore,
        *,
        symbol: str,
        strategy: str,
        timeframe: str = "M30",
    ) -> None:
        self.base_config = config
        self.store = store
        self.symbol = symbol
        self.strategy = strategy
        self.timeframe = timeframe
        self._regime = None
        try:
            assessment = RegimeAgent(config, store).assess(symbol)
            self._regime = assessment.regime if assessment else None
        except Exception as exc:
            logger.warning("Regime assess failed for %s: %s", symbol, exc)

    def baseline(self, params: LoopParams | None = None) -> StrategyValidationResult:
        cfg = self._config_with(params or LoopParams())
        return BacktestAgent(cfg, self.store).validate_strategy(
            self.symbol,
            self.strategy,
            self.timeframe,
            self._regime,
        )

    def validate(
        self,
        candidate: LoopParams,
        baseline: StrategyValidationResult,
        working: LoopParams,
    ) -> ValidationOutcome:
        """Validate working∪candidate against baseline using Tier A/B criteria."""
        merged = working.merge(candidate.as_dict())
        trial_cfg = self._config_with(merged)
        try:
            trial = BacktestAgent(trial_cfg, self.store).validate_strategy(
                self.symbol,
                self.strategy,
                self.timeframe,
                self._regime,
            )
        except Exception as exc:
            logger.error("Validation backtest failed: %s", exc)
            return ValidationOutcome(
                accepted=False,
                verdict="error",
                reasons=[str(exc)],
            )

        evaluation: TrialEvaluation = evaluate_trial(
            trial, baseline, trial_cfg, self.timeframe
        )
        metrics = _metrics_from_result(trial)
        accepted = evaluation.verdict == TrialVerdict.TIER_B_ADOPT or (
            bool(getattr(config.loop_engineering, "adopt_tier_a", False))
            and evaluation.verdict == TrialVerdict.TIER_A
        )
        return ValidationOutcome(
            accepted=accepted,
            verdict=evaluation.verdict.value,
            reasons=list(evaluation.reasons),
            metrics=metrics,
            trial=trial,
        )

    def _config_with(self, params: LoopParams) -> AppConfig:
        if not params.overrides:
            return self.base_config
        return apply_config_overrides(
            self.base_config,
            overrides_to_tuples(params.overrides),
        )


def _metrics_from_result(result: StrategyValidationResult) -> dict[str, Any]:
    perf = result.backtest.performance
    return {
        "sharpe": float(perf.sharpe_ratio),
        "max_drawdown": float(perf.max_drawdown),
        "trades": int(perf.trades),
        "wf_avg_test_sharpe": float(
            result.walk_forward_summary.get("avg_test_sharpe", 0.0)
        ),
        "oos_ratio": float(result.oos.oos_ratio),
        "expected_live": float(result.quality_gate.live_expected_return),
        "gate_passed": bool(result.quality_gate.passed),
    }
