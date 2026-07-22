from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionExceptionKind(str, Enum):
    ORDER_REJECTED = "order_rejected"
    PARTIAL_FILL = "partial_fill"
    PRICE_GAP = "price_gap"
    API_TIMEOUT = "api_timeout"
    DATA_MISSING = "data_missing"


class ExecutionError(Exception):
    """Base execution exception (Lesson 10.5)."""

    kind: ExecutionExceptionKind

    def __init__(self, message: str, kind: ExecutionExceptionKind) -> None:
        super().__init__(message)
        self.kind = kind


class OrderRejectedError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ExecutionExceptionKind.ORDER_REJECTED)


class PartialFillError(ExecutionError):
    def __init__(self, message: str, filled_lots: float, requested_lots: float) -> None:
        super().__init__(message, ExecutionExceptionKind.PARTIAL_FILL)
        self.filled_lots = filled_lots
        self.requested_lots = requested_lots


class PriceGapError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ExecutionExceptionKind.PRICE_GAP)


class ApiTimeoutError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ExecutionExceptionKind.API_TIMEOUT)


class DataMissingError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ExecutionExceptionKind.DATA_MISSING)


@dataclass
class ExecutionAttempt:
    attempt: int
    success: bool
    message: str
    exception_kind: ExecutionExceptionKind | None = None
