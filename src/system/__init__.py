"""End-to-end multi-agent trading system integration (Lesson 21)."""

from src.system.architecture import AGENT_PIPELINE, DATA_FLOW, MODULAR_MONOLITH_NOTE, pipeline_stages
from src.system.integration import SystemIntegrator, build_pre_live_checklist
from src.system.types import GraduationStage, IntegrationReport

__all__ = [
    "AGENT_PIPELINE",
    "DATA_FLOW",
    "GraduationStage",
    "IntegrationReport",
    "MODULAR_MONOLITH_NOTE",
    "SystemIntegrator",
    "build_pre_live_checklist",
    "pipeline_stages",
]


def __getattr__(name: str):
    if name == "TradingSystem":
        from src.system.runner import TradingSystem

        return TradingSystem
    if name == "SystemRunResult":
        from src.system.runner import SystemRunResult

        return SystemRunResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
