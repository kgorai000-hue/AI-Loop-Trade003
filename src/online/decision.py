from __future__ import annotations

from src.online.types import UpdateAction, UpdateDecision, EvolutionLevel


class RetrainDecisionEngine:
    """Retrain vs pause decision matrix (Lesson 17.5)."""

    def __init__(
        self,
        *,
        psi_threshold: float = 0.2,
        performance_drop_threshold: float = 0.3,
        min_confidence_for_retrain: float = 0.6,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.perf_threshold = performance_drop_threshold
        self.min_confidence = min_confidence_for_retrain

    def decide(
        self,
        *,
        drift_detected: bool,
        ic_change: float,
        data_quality_ok: bool = True,
        sample_size_ok: bool = True,
    ) -> UpdateDecision:
        perf_drop = abs(ic_change) if ic_change < 0 else 0.0

        if drift_detected:
            if perf_drop > self.perf_threshold:
                if data_quality_ok and sample_size_ok:
                    return UpdateDecision(
                        action=UpdateAction.RETRAIN,
                        confidence=0.9,
                        reason="drift detected with performance drop, data quality OK",
                        urgency="high",
                        evolution_level=EvolutionLevel.MODEL,
                    )
                return UpdateDecision(
                    action=UpdateAction.INVESTIGATE,
                    confidence=0.5,
                    reason="retrain needed but data quality insufficient",
                    urgency="high",
                    evolution_level=EvolutionLevel.MODEL,
                )
            return UpdateDecision(
                action=UpdateAction.OBSERVE,
                confidence=0.6,
                reason="drift detected but performance still acceptable",
                urgency="low",
                evolution_level=EvolutionLevel.SIGNAL,
            )

        if perf_drop > self.perf_threshold:
            return UpdateDecision(
                action=UpdateAction.PAUSE,
                confidence=0.7,
                reason="performance drop without clear drift, pause and investigate",
                urgency="medium",
                evolution_level=EvolutionLevel.STRATEGY,
            )

        return UpdateDecision(
            action=UpdateAction.CONTINUE,
            confidence=0.95,
            reason="all normal",
            urgency="none",
            evolution_level=EvolutionLevel.SIGNAL,
        )
