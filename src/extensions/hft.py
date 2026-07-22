from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp_ns: int


class MicrostructureAnalyzer(ABC):
    """Future: HFT order-book analysis (Lesson 22.4 Path 1). Not used in MT5 daily pipeline."""

    @abstractmethod
    def estimate_slippage(self, book: OrderBookSnapshot, order_size: float) -> float:
        ...


class LatencyBudget:
    """Production targets vs current Python/MT5 stack (Lesson 22.4 benchmarks)."""

    TARGETS = {
        "risk_check_ms": 1.0,
        "order_submit_ms": 10.0,
        "fill_rate_pct": 94.0,
    }
    CURRENT_ESTIMATE = {
        "risk_check_ms": 10.0,
        "order_submit_ms": 100.0,
        "fill_rate_pct": None,
    }

    @classmethod
    def gap_report(cls) -> dict[str, str]:
        rows: dict[str, str] = {}
        for key, target in cls.TARGETS.items():
            current = cls.CURRENT_ESTIMATE.get(key)
            rows[key] = f"target={target} current~{current}"
        return rows
