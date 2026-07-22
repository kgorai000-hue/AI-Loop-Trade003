"""Appendix B: 12 ways quant systems die — diagnostic aggregation."""

from src.survival.catalog import DEATH_MODES, DEATH_MODE_BY_ID, DIAGNOSTIC_ORDER
from src.survival.diagnostics import SurvivalDiagnostics
from src.survival.types import DeathModeInfo, DeathModeStatus, SurvivalContext, SurvivalReport, WeeklyCheckItem

__all__ = [
    "DEATH_MODES",
    "DEATH_MODE_BY_ID",
    "DIAGNOSTIC_ORDER",
    "DeathModeInfo",
    "DeathModeStatus",
    "SurvivalContext",
    "SurvivalDiagnostics",
    "SurvivalReport",
    "WeeklyCheckItem",
]
