from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.core.config import DataFetchConfig
from src.core.history import BARS_PER_TRADING_DAY
from src.core.mt5_connector import MT5Connector
from src.data.fetcher import rates_to_bars
from src.data.store import BarRecord

logger = logging.getLogger(__name__)


@dataclass
class RobustFetchResult:
    bars: list[BarRecord]
    mode: str
    attempts: int


def _dedupe_bars(bars: list[BarRecord]) -> list[BarRecord]:
    by_time: dict[int, BarRecord] = {}
    for bar in bars:
        by_time[bar.time] = bar
    return [by_time[t] for t in sorted(by_time)]


class RobustOHLCVFetcher:
    """MT5 fetch with retry/backoff, deep history, and empty-data guards (Lesson 6.2)."""

    def __init__(
        self,
        connector: MT5Connector,
        fetch_cfg: DataFetchConfig,
    ) -> None:
        self.connector = connector
        self.fetch_cfg = fetch_cfg

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> RobustFetchResult:
        """Fetch the newest bars from the present (copy_rates_from_pos)."""
        cfg = self.fetch_cfg
        last_error: Exception | None = None
        count = max(int(count), 1)

        for attempt in range(cfg.max_retries):
            try:
                if not self.connector.is_connected:
                    self.connector.connect()

                resolved = self.connector.ensure_symbol(symbol)
                rates = self.connector.get_rates(resolved, timeframe, count)
                bars = rates_to_bars(resolved, timeframe, rates)
                if not bars:
                    raise RuntimeError(f"Empty tip data for {symbol} {timeframe}")
                return RobustFetchResult(bars=bars, mode="tip", attempts=attempt + 1)
            except (ConnectionError, RuntimeError, ValueError) as exc:
                last_error = exc
                wait = cfg.backoff_base ** attempt
                logger.warning(
                    "Tip fetch attempt %d/%d failed for %s %s: %s",
                    attempt + 1,
                    cfg.max_retries,
                    symbol,
                    timeframe,
                    exc,
                )
                if attempt < cfg.max_retries - 1:
                    try:
                        self.connector.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(wait)
                continue

        raise RuntimeError(
            f"Tip fetch failed after {cfg.max_retries} attempts for {symbol} {timeframe}: {last_error}"
        )

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        last_bar_time: int | None = None,
    ) -> RobustFetchResult:
        cfg = self.fetch_cfg
        last_error: Exception | None = None

        for attempt in range(cfg.max_retries):
            try:
                if not self.connector.is_connected:
                    self.connector.connect()

                if last_bar_time is None:
                    return self.fetch_deep(symbol, timeframe, count)

                resolved = self.connector.ensure_symbol(symbol)
                # Prefer range from last bar → now so a stale MAX(time) cannot stall.
                date_from = datetime.fromtimestamp(last_bar_time, tz=timezone.utc)
                date_to = datetime.now(tz=timezone.utc) + timedelta(minutes=1)
                rates = self.connector.get_rates_range(resolved, timeframe, date_from, date_to)
                bars = rates_to_bars(resolved, timeframe, rates)
                mode = "incremental"

                if not bars:
                    # Fallback: tip from present (from_pos), then classic from-date.
                    tip = self.fetch_latest(symbol, timeframe, min(count, 200))
                    if tip.bars:
                        return tip
                    rates = self.connector.get_rates_from(resolved, timeframe, date_from, count)
                    bars = rates_to_bars(resolved, timeframe, rates)

                if not bars:
                    raise RuntimeError(f"Empty data for {symbol} {timeframe}")

                return RobustFetchResult(bars=bars, mode=mode, attempts=attempt + 1)

            except (ConnectionError, RuntimeError, ValueError) as exc:
                last_error = exc
                wait = cfg.backoff_base ** attempt
                logger.warning(
                    "Fetch attempt %d/%d failed for %s %s: %s",
                    attempt + 1,
                    cfg.max_retries,
                    symbol,
                    timeframe,
                    exc,
                )
                if attempt < cfg.max_retries - 1:
                    try:
                        self.connector.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(wait)
                continue

        raise RuntimeError(
            f"Fetch failed after {cfg.max_retries} attempts for {symbol} {timeframe}: {last_error}"
        )

    def fetch_deep(
        self,
        symbol: str,
        timeframe: str,
        target_count: int,
    ) -> RobustFetchResult:
        """Fetch up to target_count bars using pos/range/pagination fallbacks."""
        cfg = self.fetch_cfg
        last_error: Exception | None = None

        for attempt in range(cfg.max_retries):
            try:
                if not self.connector.is_connected:
                    self.connector.connect()

                resolved = self.connector.ensure_symbol(symbol)
                bars = self._fetch_deep_resolved(resolved, timeframe, target_count)

                if not bars:
                    raise RuntimeError(f"Empty data for {symbol} {timeframe}")

                if len(bars) < target_count * cfg.min_completeness_ratio:
                    logger.warning(
                        "Deep fetch incomplete for %s %s: got %d, target %d",
                        symbol,
                        timeframe,
                        len(bars),
                        target_count,
                    )

                return RobustFetchResult(bars=bars, mode="backfill", attempts=attempt + 1)

            except (ConnectionError, RuntimeError, ValueError) as exc:
                last_error = exc
                wait = cfg.backoff_base ** attempt
                logger.warning(
                    "Deep fetch attempt %d/%d failed for %s %s: %s",
                    attempt + 1,
                    cfg.max_retries,
                    symbol,
                    timeframe,
                    exc,
                )
                if attempt < cfg.max_retries - 1:
                    try:
                        self.connector.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(wait)
                continue

        raise RuntimeError(
            f"Deep fetch failed after {cfg.max_retries} attempts for {symbol} {timeframe}: {last_error}"
        )

    def _fetch_deep_resolved(
        self,
        resolved: str,
        timeframe: str,
        target_count: int,
    ) -> list[BarRecord]:
        cfg = self.fetch_cfg
        bars: list[BarRecord] = []

        rates = self.connector.get_rates(resolved, timeframe, target_count)
        bars = _dedupe_bars(rates_to_bars(resolved, timeframe, rates))

        if len(bars) >= target_count * cfg.min_completeness_ratio:
            return bars[-target_count:]

        now = datetime.now(tz=timezone.utc)
        bars_per_day = BARS_PER_TRADING_DAY.get(timeframe.upper(), 24)
        lookback_days = int((target_count / max(bars_per_day, 1)) * (365 / 252) * 1.15) + 30
        date_from = now - timedelta(days=lookback_days)

        range_bars = self._fetch_range_paginated(resolved, timeframe, date_from, now)
        bars = _dedupe_bars(bars + range_bars)

        if len(bars) >= target_count * cfg.min_completeness_ratio:
            return bars[-target_count:]

        bars = self._paginate_backward(resolved, timeframe, target_count, bars)
        bars = _dedupe_bars(bars)
        if len(bars) > target_count:
            bars = bars[-target_count:]
        return bars

    def _fetch_range_paginated(
        self,
        resolved: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> list[BarRecord]:
        cfg = self.fetch_cfg
        chunk_days = max(cfg.range_chunk_days, 1)
        cursor = date_from
        bars: list[BarRecord] = []

        while cursor < date_to:
            chunk_end = min(cursor + timedelta(days=chunk_days), date_to)
            try:
                rates = self.connector.get_rates_range(resolved, timeframe, cursor, chunk_end)
            except RuntimeError as exc:
                logger.warning(
                    "Range chunk failed for %s %s (%s -> %s): %s",
                    resolved,
                    timeframe,
                    cursor.date(),
                    chunk_end.date(),
                    exc,
                )
                cursor = chunk_end
                continue

            if rates is not None and len(rates) > 0:
                bars.extend(rates_to_bars(resolved, timeframe, rates))
            cursor = chunk_end

        return bars

    def _paginate_backward(
        self,
        resolved: str,
        timeframe: str,
        target_count: int,
        existing_bars: list[BarRecord],
    ) -> list[BarRecord]:
        cfg = self.fetch_cfg
        chunk = cfg.chunk_size
        bars_per_day = BARS_PER_TRADING_DAY.get(timeframe.upper(), 24)
        step_days = max(int(chunk / max(bars_per_day, 1)) + 7, 30)

        all_bars = list(existing_bars)
        seen = {bar.time for bar in all_bars}

        if all_bars:
            oldest = min(bar.time for bar in all_bars)
            date_from = datetime.fromtimestamp(oldest, tz=timezone.utc) - timedelta(days=step_days)
        else:
            date_from = datetime.now(tz=timezone.utc) - timedelta(days=step_days)

        stagnant_rounds = 0
        while len(all_bars) < target_count and stagnant_rounds < 3:
            rates = self.connector.get_rates_from(resolved, timeframe, date_from, chunk)
            if rates is None or len(rates) == 0:
                stagnant_rounds += 1
                date_from -= timedelta(days=step_days)
                continue

            new_bars = [bar for bar in rates_to_bars(resolved, timeframe, rates) if bar.time not in seen]
            if not new_bars:
                stagnant_rounds += 1
                date_from -= timedelta(days=step_days)
                continue

            stagnant_rounds = 0
            for bar in new_bars:
                seen.add(bar.time)
            all_bars.extend(new_bars)

            oldest = min(bar.time for bar in new_bars)
            date_from = datetime.fromtimestamp(oldest, tz=timezone.utc) - timedelta(days=step_days)

        return all_bars
