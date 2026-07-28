"""IntelligenceLoop: Maker → Checker → Validator (grid fallback)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import AppConfig
from src.data.store import OHLCVStore
from src.intelligence.google_ai_studio_client import GoogleAIClient
from src.intelligence.checker import StrategyChecker
from src.intelligence.grid import run_grid_search
from src.intelligence.maker import StrategyMaker
from src.intelligence.params import LoopParams, params_from_config
from src.intelligence.persistence import StateStore
from src.intelligence.validator import ParamValidator, _metrics_from_result

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceOutcome:
    symbol: str
    strategy: str
    timeframe: str
    path: str
    accepted: bool
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


class IntelligenceLoop:
    def __init__(
        self,
        config: AppConfig,
        store: OHLCVStore,
        *,
        symbol: str,
        strategy: str = "feature_score",
        timeframe: str | None = None,
        state_dir: str | Path | None = None,
        state_key: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.symbol = symbol
        self.strategy = strategy
        self.timeframe = (timeframe or config.trading.primary_timeframe or "M30").upper()

        intel = getattr(config, "intelligence", None)
        state_root = state_dir
        if state_root is None:
            if intel is not None:
                state_root = intel.state_dir
            else:
                state_root = "state"
        from src.intelligence.params import strategy_state_key

        default_key = strategy_state_key(symbol, strategy) if strategy else symbol
        self.state = StateStore(state_root, state_key or default_key)

        maker_model = "gemini-1.5-pro"
        checker_model = "gemini-1.5-pro"
        n_candidates = 6
        max_retries = 5
        enable_cache = True
        if intel is not None:
            maker_model = intel.maker_model
            checker_model = intel.checker_model
            n_candidates = intel.maker_candidates
            max_retries = intel.max_retries
            enable_cache = intel.enable_prompt_cache

        self.client = GoogleAIClient(
            max_retries=max_retries,
        )
        self.maker = StrategyMaker(
            self.client, model=maker_model, n_candidates=n_candidates
        )
        self.checker = StrategyChecker(self.client, model=checker_model)
        self.validator = ParamValidator(
            config,
            store,
            symbol=symbol,
            strategy=strategy,
            timeframe=self.timeframe,
        )
        loop_cfg = getattr(intel, "loop", None) if intel is not None else None
        self.no_api_seed_baseline = bool(
            getattr(loop_cfg, "no_api_seed_baseline", True)
        )
        self.seed_baseline_if_no_adopt = bool(
            getattr(loop_cfg, "seed_baseline_if_no_adopt", True)
        )

    def run(self) -> IntelligenceOutcome:
        defaults = params_from_config(self.config)
        stored = self.state.get_params()
        working = LoopParams(overrides={**defaults.overrides, **stored.overrides})
        state = self.state.read_state()
        last_metrics = state.get("last_metrics") or {}

        # Demo path: no valid Gemini key → seed settings defaults (seconds, not hours).
        if not self.client.available() and self.no_api_seed_baseline:
            logger.info(
                "No valid GEMINI_API_KEY — seeding config defaults for %s (%s)",
                self.symbol,
                self.strategy,
            )
            return self._seed_baseline(
                working,
                metrics={},
                message="seeded config defaults (no valid API key)",
            )

        try:
            baseline = self.validator.baseline(working)
        except Exception as exc:
            msg = f"baseline validation failed: {exc}"
            logger.error(msg)
            if self.seed_baseline_if_no_adopt:
                return self._seed_baseline(
                    working,
                    metrics={},
                    message=f"seeded config defaults after baseline error: {exc}",
                )
            return IntelligenceOutcome(
                symbol=self.symbol,
                strategy=self.strategy,
                timeframe=self.timeframe,
                path="error",
                accepted=False,
                params=working.as_dict(),
                message=msg,
            )

        if self.client.available():
            outcome = self._run_llm(working, baseline, last_metrics)
            if outcome.accepted:
                return outcome
            logger.info(
                "LLM path did not adopt for %s (%s); trying grid fallback",
                self.symbol,
                outcome.message,
            )

        return self._run_grid(working, baseline)

    def _run_llm(
        self,
        working: LoopParams,
        baseline: Any,
        last_metrics: dict[str, Any],
    ) -> IntelligenceOutcome:
        skills = self.state.skills_text()
        candidates = self.maker.propose(
            current_params=working,
            last_metrics=last_metrics,
            skills_text=skills,
            symbol=self.symbol,
            strategy=self.strategy,
        )
        if not candidates:
            return IntelligenceOutcome(
                symbol=self.symbol,
                strategy=self.strategy,
                timeframe=self.timeframe,
                path="llm",
                accepted=False,
                params=working.as_dict(),
                message="maker returned no valid candidates",
            )

        reviews = self.checker.review(candidates, skills_text=skills)
        trials: list[dict[str, Any]] = []
        best_params = working
        best_metrics: dict[str, Any] = {}
        adopted = False

        for review in reviews:
            if not review.approved:
                self.state.append_lesson(
                    f"Checker reject {review.params.as_dict()}: {review.reason}"
                )
                trials.append(
                    {
                        "overrides": review.params.as_dict(),
                        "verdict": "checker_reject",
                        "reasons": [review.reason],
                        "adopted": False,
                    }
                )
                continue

            outcome = self.validator.validate(review.params, baseline, best_params)
            trials.append(
                {
                    "overrides": review.params.as_dict(),
                    "verdict": outcome.verdict,
                    "reasons": outcome.reasons,
                    "metrics": outcome.metrics,
                    "adopted": outcome.accepted,
                }
            )
            if outcome.accepted:
                best_params = best_params.merge(review.params.as_dict())
                best_metrics = outcome.metrics
                adopted = True
                if outcome.trial is not None:
                    baseline = outcome.trial
            else:
                self.state.append_lesson(
                    f"Validator reject {review.params.as_dict()}: "
                    f"{outcome.verdict} ({'; '.join(outcome.reasons[:3])})"
                )

        if adopted:
            self._persist(best_params, best_metrics, path="llm", accepted=True)
            return IntelligenceOutcome(
                symbol=self.symbol,
                strategy=self.strategy,
                timeframe=self.timeframe,
                path="llm",
                accepted=True,
                params=best_params.as_dict(),
                metrics=best_metrics,
                trials=trials,
                message="adopted via Maker→Checker→Validator",
            )

        return IntelligenceOutcome(
            symbol=self.symbol,
            strategy=self.strategy,
            timeframe=self.timeframe,
            path="llm",
            accepted=False,
            params=working.as_dict(),
            metrics={},
            trials=trials,
            message="no candidate adopted by validator",
        )

    def _run_grid(self, working: LoopParams, baseline: Any) -> IntelligenceOutcome:
        best, metrics, trials = run_grid_search(self.validator, working, baseline)
        accepted = bool(metrics) and best.as_dict() != working.as_dict()
        if accepted:
            self._persist(best, metrics, path="grid", accepted=True)
            message = "adopted via grid fallback"
            return IntelligenceOutcome(
                symbol=self.symbol,
                strategy=self.strategy,
                timeframe=self.timeframe,
                path="grid" if not self.client.available() else "grid_fallback",
                accepted=True,
                params=best.as_dict(),
                metrics=metrics,
                trials=trials,
                message=message,
            )

        if self.seed_baseline_if_no_adopt:
            baseline_metrics: dict[str, Any] = {}
            try:
                baseline_metrics = _metrics_from_result(baseline)
            except Exception:
                baseline_metrics = {}
            seeded = self._seed_baseline(
                working,
                metrics=baseline_metrics,
                message="seeded config defaults (grid found no improvement)",
                trials=trials,
            )
            return seeded

        self._persist(working, {}, path="grid", accepted=False)
        return IntelligenceOutcome(
            symbol=self.symbol,
            strategy=self.strategy,
            timeframe=self.timeframe,
            path="grid" if not self.client.available() else "grid_fallback",
            accepted=False,
            params=working.as_dict(),
            metrics={},
            trials=trials,
            message="grid found no improvement",
        )

    def _seed_baseline(
        self,
        working: LoopParams,
        *,
        metrics: dict[str, Any],
        message: str,
        trials: list[dict[str, Any]] | None = None,
    ) -> IntelligenceOutcome:
        self._persist(working, metrics, path="baseline_seed", accepted=True)
        return IntelligenceOutcome(
            symbol=self.symbol,
            strategy=self.strategy,
            timeframe=self.timeframe,
            path="baseline_seed",
            accepted=True,
            params=working.as_dict(),
            metrics=metrics,
            trials=list(trials or []),
            message=message,
        )

    def _persist(
        self,
        params: LoopParams,
        metrics: dict[str, Any],
        *,
        path: str,
        accepted: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "params": params.as_dict(),
            "accepted": accepted,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "path": path,
            "last_maker_run": now,
        }
        if metrics:
            payload["last_metrics"] = metrics
        self.state.update_state(**payload)


def apply_state_overrides(
    config: AppConfig,
    symbol: str,
    *,
    strategy: str | None = None,
) -> AppConfig:
    """Apply adopted STATE params for a symbol (optionally strategy-scoped) onto config."""
    from src.backtest.loop_engine import apply_config_overrides
    from src.intelligence.params import overrides_to_tuples, strategy_state_key

    intel = getattr(config, "intelligence", None)
    state_dir = intel.state_dir if intel is not None else "state"
    key = strategy_state_key(symbol, strategy) if strategy else symbol
    store = StateStore(state_dir, key)
    params = store.get_params()
    if not params.overrides:
        return config
    return apply_config_overrides(config, overrides_to_tuples(params.overrides))
