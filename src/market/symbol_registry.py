from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.core.config import AppConfig


@dataclass(frozen=True)
class CanonicalSymbol:
    """Internal symbol format SYMBOL.VENUE (Lesson 19.9)."""

    symbol: str
    venue: str

    def __str__(self) -> str:
        return f"{self.symbol}.{self.venue}"

    @classmethod
    def parse(cls, value: str) -> CanonicalSymbol:
        if "." not in value:
            return cls(symbol=value, venue="MT5")
        base, venue = value.rsplit(".", 1)
        return cls(symbol=base, venue=venue)


class MT5SymbolAdapter:
    """Translate between MT5 broker symbols and canonical internal form."""

    VENUE = "MT5"

    INDEX_PREFIX = "#"
    ALIAS_MAP = {
        "#USSPX500": "SPX",
        "#USNDAQ100": "NDX",
        "#US30": "DJI",
        "#Japan225": "N225",
        "#Germany40": "DAX",
        "#UK100": "FTSE",
    }

    def to_canonical(self, broker_symbol: str) -> CanonicalSymbol:
        clean = broker_symbol.strip()
        display = self.ALIAS_MAP.get(clean, clean.lstrip(self.INDEX_PREFIX))
        display = re.sub(r"[^A-Za-z0-9\-]", "-", display)
        return CanonicalSymbol(symbol=display, venue=self.VENUE)

    def to_broker(self, canonical: CanonicalSymbol | str) -> str:
        if isinstance(canonical, str):
            canonical = CanonicalSymbol.parse(canonical)
        reverse = {v: k for k, v in self.ALIAS_MAP.items()}
        if canonical.symbol in reverse:
            return reverse[canonical.symbol]
        if canonical.venue != self.VENUE:
            return canonical.symbol
        for broker, alias in self.ALIAS_MAP.items():
            if alias == canonical.symbol:
                return broker
        return canonical.symbol


class SymbolRegistry:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config
        self._adapter = MT5SymbolAdapter()

    def to_canonical(self, broker_symbol: str) -> CanonicalSymbol:
        return self._adapter.to_canonical(broker_symbol)

    def to_broker(self, canonical: CanonicalSymbol | str) -> str:
        return self._adapter.to_broker(canonical)

    def validate_quote_age(self, quote_timestamp: int | float | None, max_age_seconds: float) -> bool:
        if quote_timestamp is None:
            return False
        age = time.time() - float(quote_timestamp)
        return age <= max_age_seconds
