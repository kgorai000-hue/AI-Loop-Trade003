from __future__ import annotations

import logging

import numpy as np

from src.core.config import AppConfig
from src.core.types import StrategyKind, SymbolStatsReport, TradeSignal
from src.online.alpha_decay import projected_ic
from src.online.decision import RetrainDecisionEngine
from src.online.drift import calculate_psi, psi_severity, sliding_accuracy_drift
from src.online.ewm_model import effective_lookback_days
from src.online.signal_adaptation import dynamic_signal_threshold
from src.online.strategy_lifecycle import build_strategy_lifecycle_states
from src.online.types import DriftSignal, EvolutionLevel, EvolutionReport, UpdateAction
from src.portfolio.types import PortfolioReport

logger = logging.getLogger(__name__)


class EvolutionAgent:
    """Monitor drift, schedule updates, and strategy lifecycle (Lesson 17)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.online_learning
        self.decision_engine = RetrainDecisionEngine(
            psi_threshold=self.cfg.psi_threshold,
            performance_drop_threshold=self.cfg.performance_drop_threshold,
        )

    def assess(
        self,
        research_reports: list[SymbolStatsReport] | None = None,
        signals: list[TradeSignal] | None = None,
        portfolio_report: PortfolioReport | None = None,
    ) -> EvolutionReport:
        if not self.cfg.enabled:
            return EvolutionReport(enabled=False)

        research_reports = research_reports or []
        signals = signals or []
        warnings: list[str] = []
        drift_signals: list[DriftSignal] = []

        mean_ic = self._estimate_mean_ic(research_reports)
        monthly_decay = self.cfg.default_monthly_ic_decay
        projected = projected_ic(mean_ic, monthly_decay, 12)
        lookback = effective_lookback_days(self.cfg.decay_factor)

        if projected < self.cfg.ic_viability_threshold:
            warnings.append(
                f"projected IC {projected:.3f} below viability {self.cfg.ic_viability_threshold:.3f}"
            )

        vol_drift = self._volatility_psi_drift(research_reports)
        if vol_drift:
            drift_signals.append(vol_drift)

        signal_strengths = [s.strength for s in signals]
        dynamic_threshold = dynamic_signal_threshold(
            signal_strengths,
            k=self.cfg.dynamic_threshold_k,
            default=self.cfg.signal_threshold_default,
        )

        accuracy_drift, avg_acc = sliding_accuracy_drift(
            self._proxy_accuracies(research_reports),
            window=self.cfg.accuracy_window,
            threshold=self.cfg.accuracy_warning_threshold,
        )
        if accuracy_drift:
            drift_signals.append(
                DriftSignal(
                    metric="sliding_accuracy",
                    value=avg_acc,
                    threshold=self.cfg.accuracy_warning_threshold,
                    detected=True,
                    severity="medium",
                )
            )

        ic_change = self._ic_change_proxy(research_reports)
        drift_detected = any(d.detected for d in drift_signals)
        update_decision = self.decision_engine.decide(
            drift_detected=drift_detected,
            ic_change=ic_change,
            data_quality_ok=len(research_reports) >= 3,
            sample_size_ok=sum(r.bars for r in research_reports) >= self.cfg.min_total_bars,
        )

        strategy_metrics = self._strategy_metrics(signals, research_reports)
        strategy_states = build_strategy_lifecycle_states(
            strategy_metrics,
            incubation_min=self.cfg.lifecycle_incubation_sharpe_min,
            maturity_min=self.cfg.lifecycle_maturity_sharpe_min,
            decay_max=self.cfg.lifecycle_decay_sharpe_max,
        )

        evolution_level = EvolutionLevel.SIGNAL
        if update_decision.action in (UpdateAction.RETRAIN, UpdateAction.INVESTIGATE):
            evolution_level = EvolutionLevel.MODEL
        elif update_decision.action == UpdateAction.PAUSE:
            evolution_level = EvolutionLevel.STRATEGY

        if portfolio_report and portfolio_report.avg_correlation >= self.cfg.correlation_crisis_threshold:
            warnings.append(
                f"high avg correlation {portfolio_report.avg_correlation:.2f} — diversification illusion risk"
            )
            evolution_level = EvolutionLevel.ARCHITECTURE

        report = EvolutionReport(
            enabled=True,
            mean_ic=mean_ic,
            ic_monthly_decay_rate=monthly_decay,
            projected_ic_12m=projected,
            effective_lookback_days=lookback,
            dynamic_threshold=dynamic_threshold,
            drift_signals=drift_signals,
            drift_detected=drift_detected,
            performance_drop_pct=abs(ic_change) if ic_change < 0 else 0.0,
            update_decision=update_decision,
            strategy_states=strategy_states,
            evolution_level=evolution_level,
            warnings=warnings,
        )

        if update_decision.action != UpdateAction.CONTINUE:
            logger.info(
                "EvolutionAgent: action=%s level=%s reason=%s",
                update_decision.action.value,
                evolution_level.value,
                update_decision.reason,
            )
        return report

    def apply_signal_adaptation(self, signals: list[TradeSignal], report: EvolutionReport) -> list[TradeSignal]:
        """Level 1: filter signals below dynamic threshold."""
        if not self.cfg.apply_dynamic_threshold or not signals:
            return signals

        threshold = report.dynamic_threshold
        adapted: list[TradeSignal] = []
        for signal in signals:
            if signal.strength < threshold:
                continue
            adapted.append(
                TradeSignal(
                    symbol=signal.symbol,
                    side=signal.side,
                    timeframe=signal.timeframe,
                    strength=signal.strength,
                    reason=f"{signal.reason} | dynamic threshold {threshold:.3f}",
                    mode=signal.mode,
                    strategy=signal.strategy,
                    predicted_return=signal.predicted_return,
                    confidence=signal.confidence,
                    requested_lots=signal.requested_lots,
                    portfolio_weight=signal.portfolio_weight,
                )
            )
        return adapted

    @staticmethod
    def _estimate_mean_ic(reports: list[SymbolStatsReport]) -> float:
        if not reports:
            return 0.05
        ics = []
        for report in reports:
            ic_proxy = report.autocorr_lag1 * 0.5 + (0.05 if report.annualized_return > 0 else -0.02)
            ics.append(ic_proxy)
        return float(np.mean(ics))

    def _volatility_psi_drift(self, reports: list[SymbolStatsReport]) -> DriftSignal | None:
        if len(reports) < 4:
            return None
        vols = np.array([r.annualized_volatility for r in reports], dtype=float)
        mid = len(vols) // 2
        baseline = vols[:mid]
        recent = vols[mid:]
        psi = calculate_psi(baseline, recent)
        detected = psi >= self.cfg.psi_threshold
        return DriftSignal(
            metric="volatility_psi",
            value=psi,
            threshold=self.cfg.psi_threshold,
            detected=detected,
            severity=psi_severity(psi),
        )

    @staticmethod
    def _proxy_accuracies(reports: list[SymbolStatsReport]) -> list[float]:
        accuracies: list[float] = []
        for report in reports:
            hit = 0.55 if report.annualized_return > 0 else 0.45
            if report.regime.value == "crisis":
                hit -= 0.05
            accuracies.append(hit)
        return accuracies

    @staticmethod
    def _ic_change_proxy(reports: list[SymbolStatsReport]) -> float:
        if len(reports) < 4:
            return 0.0
        returns = [r.annualized_return for r in reports]
        mid = len(returns) // 2
        early = float(np.mean(returns[:mid]))
        recent = float(np.mean(returns[mid:]))
        if abs(early) < 1e-9:
            return 0.0
        return (recent - early) / abs(early)

    @staticmethod
    def _strategy_metrics(
        signals: list[TradeSignal],
        reports: list[SymbolStatsReport],
    ) -> dict[str, tuple[float, float]]:
        report_map = {r.symbol: r for r in reports}
        metrics: dict[str, tuple[float, float]] = {}

        strategies = {s.strategy.value for s in signals} if signals else {
            StrategyKind.TREND_FOLLOWING.value,
            StrategyKind.MEAN_REVERSION.value,
            StrategyKind.PAIRS.value,
        }

        for strategy in strategies:
            related = [report_map[s.symbol] for s in signals if s.strategy.value == strategy and s.symbol in report_map]
            if not related:
                related = reports[:3]
            if not related:
                metrics[strategy] = (0.05, 0.20)
                continue
            ann_ret = float(np.mean([r.annualized_return for r in related]))
            ann_vol = float(np.mean([max(r.annualized_volatility, 0.05) for r in related]))
            metrics[strategy] = (ann_ret, ann_vol)

        return metrics
