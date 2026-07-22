"""Future extension stubs (Lesson 22.4). Not wired into the live pipeline."""

from src.extensions.alt_data import AltDataEvent, AltDataProvider
from src.extensions.distributed import (
    EXTRACTION_CRITERIA,
    AgentServiceSpec,
    RemoteAgentStub,
    ServiceBoundary,
)
from src.extensions.hft import LatencyBudget, MicrostructureAnalyzer, OrderBookSnapshot
from src.extensions.human_intervention import HumanInterventionLog, InterventionRecord
from src.extensions.regulatory import RegulatoryFlag, RegulatoryMonitor

__all__ = [
    "EXTRACTION_CRITERIA",
    "AgentServiceSpec",
    "AltDataEvent",
    "AltDataProvider",
    "HumanInterventionLog",
    "InterventionRecord",
    "LatencyBudget",
    "MicrostructureAnalyzer",
    "OrderBookSnapshot",
    "RegulatoryFlag",
    "RegulatoryMonitor",
    "RemoteAgentStub",
    "ServiceBoundary",
]
