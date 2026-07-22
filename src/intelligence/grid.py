"""Grid fallback when Anthropic is unavailable or LLM path finds no adopt."""

from __future__ import annotations

import logging
from typing import Any

from src.backtest.parameter_spaces import specs_for_strategy
from src.intelligence.params import LoopParams, llm_parameter_specs
from src.intelligence.validator import ParamValidator

logger = logging.getLogger(__name__)


def run_grid_search(
    validator: ParamValidator,
    working: LoopParams,
    baseline_result: Any,
) -> tuple[LoopParams, dict[str, Any], list[dict[str, Any]]]:
    """
    One-parameter-at-a-time search using parameter_spaces.
    Returns (best_params, best_metrics, trial_log).
    """
    allowed_names = {s.name for s in llm_parameter_specs()}
    specs = [
        s
        for s in specs_for_strategy(validator.strategy)
        if s.name in allowed_names
    ]
    if not specs:
        specs = [s for s in llm_parameter_specs()]

    best = LoopParams(overrides=dict(working.overrides))
    best_metrics: dict[str, Any] = {}
    trials: list[dict[str, Any]] = []
    current_baseline = baseline_result

    for spec in sorted(specs, key=lambda s: s.priority):
        for value in spec.values:
            candidate_overrides = {path: val for path, val in spec.apply(value)}
            candidate = LoopParams(overrides=candidate_overrides)
            outcome = validator.validate(candidate, current_baseline, best)
            trials.append(
                {
                    "parameter": spec.name,
                    "value": value,
                    "verdict": outcome.verdict,
                    "reasons": outcome.reasons,
                    "metrics": outcome.metrics,
                    "adopted": outcome.accepted,
                }
            )
            if outcome.accepted:
                best = best.merge(candidate_overrides)
                best_metrics = outcome.metrics
                if outcome.trial is not None:
                    current_baseline = outcome.trial
                logger.info(
                    "Grid adopted %s=%s for %s (%s)",
                    spec.name,
                    value,
                    validator.symbol,
                    outcome.verdict,
                )
                break  # next parameter after first adopt for this spec

    return best, best_metrics, trials
