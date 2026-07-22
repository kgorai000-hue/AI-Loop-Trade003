from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ServiceBoundary(str, Enum):
    """When to extract agents to separate services (Lesson 21.1, 22.4)."""

    RESEARCH_GPU = "research_gpu"
    RISK_RUST = "risk_rust"
    EXECUTION_GO = "execution_go"
    DATA_STREAM = "data_stream"


@dataclass
class AgentServiceSpec:
    agent_name: str
    boundary: ServiceBoundary | None
    protocol: str = "in_process"
    notes: str = ""


EXTRACTION_CRITERIA = [
    "Independent scaling needed (e.g. GPU research vs CPU execution)",
    "Latency tier mismatch (sub-ms risk vs second-level research)",
    "Team needs independent release cycles",
]


class RemoteAgentStub(ABC):
    """Placeholder for gRPC/Protobuf agent extraction (Lesson 22.4)."""

    @abstractmethod
    def health_check(self) -> bool:
        ...

    @abstractmethod
    def invoke(self, payload: dict) -> dict:
        ...
