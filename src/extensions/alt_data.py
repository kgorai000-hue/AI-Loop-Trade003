from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class AltDataEvent:
    symbol: str
    source: str
    metric: str
    value: float
    timestamp: int


class AltDataProvider(ABC):
    """Future: satellite, social, supply-chain feeds (Lesson 22.4 Path 2). Stub only."""

    @abstractmethod
    def connect(self, config: dict) -> None:
        ...

    @abstractmethod
    def stream(self, symbols: list[str]) -> Iterator[AltDataEvent]:
        ...
