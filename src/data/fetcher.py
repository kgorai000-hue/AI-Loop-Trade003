from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.mt5_connector import MT5Connector
from src.data.quality import DataQualityReport
from src.data.store import BarRecord


@dataclass
class FetchResult:
    symbol: str
    timeframe: str
    fetched: int
    stored: int
    rejected: int
    mode: str
    quality: DataQualityReport | None = None
    rejection_notes: list[str] = field(default_factory=list)


@dataclass
class SyncSummary:
    results: list[FetchResult]
    total_stored: int
    total_rejected: int
    errors: list[str]
    run_id: int | None = None
    quality_reports: list[DataQualityReport] = field(default_factory=list)


def rates_to_bars(symbol: str, timeframe: str, rates) -> list[BarRecord]:
    bars: list[BarRecord] = []
    for row in rates:
        bars.append(
            BarRecord(
                symbol=symbol,
                timeframe=timeframe,
                time=int(row["time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=int(row["tick_volume"]),
                spread=int(row["spread"]),
                real_volume=int(row["real_volume"]),
            )
        )
    return bars


class OHLCVFetcher:
    """Simple MT5 fetcher (used by RobustOHLCVFetcher internals)."""

    def __init__(self, connector: MT5Connector) -> None:
        self.connector = connector

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        last_bar_time: int | None = None,
    ) -> tuple[list[BarRecord], str]:
        resolved = self.connector.ensure_symbol(symbol)

        if last_bar_time is None:
            rates = self.connector.get_rates(resolved, timeframe, count)
            return rates_to_bars(resolved, timeframe, rates), "initial"

        date_from = datetime.fromtimestamp(last_bar_time, tz=timezone.utc)
        rates = self.connector.get_rates_from(resolved, timeframe, date_from, count)
        bars = rates_to_bars(resolved, timeframe, rates)
        return bars, "incremental"
