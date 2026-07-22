from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, ResilienceReport
from src.regime.health import RegimeHealthReport
from src.regime.misjudgment import MisjudgmentPattern
from src.regime.resilience import LEVEL_NAMES, DegradationLevel, position_scale_multiplier

logger = logging.getLogger(__name__)


class ResilienceAgent:
    """Meta Agent degradation and regime health (Lesson 13.5)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.resilience = config.resilience

    def build_report(
        self,
        market_regime: RegimeAssessment | None,
        regime_map: dict[str, RegimeAssessment],
        data_quality_ok: bool = True,
    ) -> ResilienceReport:
        health_reports = [self._health_from_assessment(a) for a in regime_map.values()]
        proxy_health = self._health_from_assessment(market_regime) if market_regime else None

        level = DegradationLevel.NORMAL
        if proxy_health is not None:
            level = max(level, proxy_health.degradation_level)
        for report in health_reports:
            level = max(level, report.degradation_level)

        if not data_quality_ok:
            level = DegradationLevel.SAFE

        multiplier = position_scale_multiplier(level, self._scale_map())
        warnings: list[str] = []
        if proxy_health:
            warnings.extend(proxy_health.warnings)
        for report in health_reports:
            if report.warnings:
                warnings.extend(f"{report.symbol}: {w}" for w in report.warnings[:2])

        return ResilienceReport(
            degradation_level=int(level),
            level_name=LEVEL_NAMES[level],
            position_scale_multiplier=multiplier,
            data_quality_ok=data_quality_ok,
            market_health=proxy_health,
            symbol_health=health_reports,
            warnings=list(dict.fromkeys(warnings))[:10],
        )

    def apply_degradation(self, regime_map: dict[str, RegimeAssessment], report: ResilienceReport) -> None:
        if not self.resilience.enabled:
            return
        multiplier = report.position_scale_multiplier
        for assessment in regime_map.values():
            assessment.position_scale = round(assessment.position_scale * multiplier, 4)
            if assessment.uncertain:
                assessment.position_scale = round(
                    assessment.position_scale * self.resilience.uncertain_position_scale,
                    4,
                )

        if report.degradation_level >= DegradationLevel.SAFE:
            logger.warning("ResilienceAgent SAFE mode: blocking new risk via zero scale")
            for assessment in regime_map.values():
                assessment.position_scale = 0.0

    def _health_from_assessment(self, assessment: RegimeAssessment | None) -> RegimeHealthReport | None:
        if assessment is None:
            return None
        pattern = None
        if assessment.misjudgment_pattern:
            try:
                pattern = MisjudgmentPattern(assessment.misjudgment_pattern)
            except ValueError:
                pattern = None
        return RegimeHealthReport(
            symbol=assessment.symbol,
            switches_per_week=assessment.switches_per_week,
            avg_duration_days=assessment.avg_regime_duration_days,
            adx_boundary_oscillation=assessment.adx_oscillation,
            healthy=not assessment.health_warnings,
            warnings=list(assessment.health_warnings),
            misjudgment_pattern=pattern,
            degradation_level=DegradationLevel(assessment.degradation_level),
        )

    def _scale_map(self) -> dict[int, float]:
        r = self.resilience
        return {
            DegradationLevel.NORMAL: 1.0,
            DegradationLevel.CAUTIOUS: r.cautious_position_scale,
            DegradationLevel.DEFENSIVE: r.defensive_position_scale,
            DegradationLevel.SAFE: 0.0,
        }
