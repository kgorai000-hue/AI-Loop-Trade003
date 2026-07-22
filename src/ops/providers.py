from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class MarketDataEvent:
    symbol: str
    bid: float
    ask: float
    timestamp: int


@dataclass
class OrderAck:
    order_id: str
    symbol: str
    status: str
    message: str = ""


@dataclass
class PositionSnapshot:
    symbol: str
    lots: float
    side: str


class DataProvider(ABC):
    """Data provider interface (Lesson 20.6)."""

    @abstractmethod
    def connect(self, config: Any) -> None:
        ...

    @abstractmethod
    def subscribe(self, symbols: list[str]) -> None:
        ...

    @abstractmethod
    def stream(self) -> Iterator[MarketDataEvent]:
        ...


class ExecutionVenue(ABC):
    """Execution venue interface (Lesson 20.6)."""

    @abstractmethod
    def submit_order(self, order: dict[str, Any]) -> OrderAck:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderAck:
        ...

    @abstractmethod
    def get_positions(self) -> list[PositionSnapshot]:
        ...


class MT5DataProvider(DataProvider):
    def __init__(self, connector: Any) -> None:
        self.connector = connector
        self._symbols: list[str] = []

    def connect(self, config: Any) -> None:
        self.connector.connect()

    def subscribe(self, symbols: list[str]) -> None:
        self._symbols = symbols
        for symbol in symbols:
            self.connector.ensure_symbol(symbol)

    def stream(self) -> Iterator[MarketDataEvent]:
        import MetaTrader5 as mt5
        import time

        for symbol in self._symbols:
            resolved = self.connector.ensure_symbol(symbol)
            tick = mt5.symbol_info_tick(resolved)
            if tick is None:
                continue
            yield MarketDataEvent(
                symbol=resolved,
                bid=float(tick.bid),
                ask=float(tick.ask),
                timestamp=int(time.time()),
            )


class MT5ExecutionVenue(ExecutionVenue):
    def __init__(self, connector: Any, *, dry_run: bool = True) -> None:
        self.connector = connector
        self.dry_run = dry_run

    def submit_order(self, order: dict[str, Any]) -> OrderAck:
        if self.dry_run:
            return OrderAck(
                order_id=f"dry_{order.get('symbol', 'unknown')}",
                symbol=str(order.get("symbol", "")),
                status="simulated",
                message="dry_run order not sent to broker",
            )
        raise RuntimeError("Live MT5 order submission not enabled")

    def cancel_order(self, order_id: str) -> OrderAck:
        return OrderAck(order_id=order_id, symbol="", status="cancelled", message="dry_run")

    def get_positions(self) -> list[PositionSnapshot]:
        import MetaTrader5 as mt5

        positions = mt5.positions_get()
        if positions is None:
            return []
        result: list[PositionSnapshot] = []
        for pos in positions:
            side = "buy" if pos.type == 0 else "sell"
            result.append(
                PositionSnapshot(symbol=pos.symbol, lots=float(pos.volume), side=side)
            )
        return result
