from __future__ import annotations

import logging
import time

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.data.fetcher import FetchResult, SyncSummary
from src.data.lineage import SyncRunStore
from src.data.quality import check_data_quality, filter_valid_bars
from src.data.robust_fetch import RobustOHLCVFetcher
from src.data.store import BarRecord, OHLCVStore

logger = logging.getLogger(__name__)


class DataAgent:
    """Data layer agent: robust MT5 fetch, validation, lineage (Lesson 06)."""

    def __init__(self, config: AppConfig, connector: MT5Connector, store: OHLCVStore) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.fetcher = RobustOHLCVFetcher(connector, config.data.fetch)
        self.lineage = SyncRunStore(config.storage.path)
        self.quality_cfg = config.data.quality
        self.data_source = config.data.source

    def sync_all(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        history_bars: int | None = None,
    ) -> SyncSummary:
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes

        broker_server: str | None = None
        try:
            account = self.connector.get_account_info()
            broker_server = str(account.get("server"))
        except Exception:  # noqa: BLE001
            pass

        run_id = self.lineage.start_run(
            source=self.data_source,
            broker_server=broker_server,
            symbols_count=len(symbols),
            timeframes=timeframes,
        )

        results: list[FetchResult] = []
        errors: list[str] = []
        quality_reports: list = []
        total_stored = 0
        total_rejected = 0
        ingested_at = int(time.time())

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    target_bars = history_bars or self.config.history_bars_for(timeframe)
                    result = self.sync_symbol(symbol, timeframe, target_bars, ingested_at)
                    results.append(result)
                    total_stored += result.stored
                    total_rejected += result.rejected
                    if result.quality is not None:
                        quality_reports.append(result.quality)
                except Exception as exc:  # noqa: BLE001
                    message = f"{symbol} {timeframe}: {exc}"
                    logger.error(message)
                    errors.append(message)

        status = "completed_with_errors" if errors else "completed"
        self.lineage.finish_run(
            run_id=run_id,
            fetched_total=sum(r.fetched for r in results),
            stored_total=total_stored,
            rejected_total=total_rejected,
            errors=errors,
            status=status,
        )

        return SyncSummary(
            results=results,
            total_stored=total_stored,
            total_rejected=total_rejected,
            errors=errors,
            run_id=run_id,
            quality_reports=quality_reports,
        )

    def sync_symbol(
        self,
        symbol: str,
        timeframe: str,
        history_bars: int,
        ingested_at: int | None = None,
    ) -> FetchResult:
        resolved = self.connector.ensure_symbol(symbol)
        stored_count = self.store.count_bars(resolved, timeframe)
        ingested_at = ingested_at or int(time.time())

        fetched_batches: list[tuple[list[BarRecord], str, int]] = []

        if stored_count < history_bars:
            backfill = self.fetcher.fetch_deep(symbol, timeframe, history_bars)
            fetched_batches.append((backfill.bars, backfill.mode, backfill.attempts))
            logger.info(
                "Backfill %s %s: target=%d stored_before=%d fetched=%d",
                resolved,
                timeframe,
                history_bars,
                stored_count,
                len(backfill.bars),
            )

        last_bar_time = self.store.get_last_bar_time(resolved, timeframe)
        if last_bar_time is not None:
            # Resident polls often: pull a short tail, not a full 1000-bar window.
            incremental_count = 50 if stored_count >= history_bars else min(history_bars, 1000)
            incremental = self.fetcher.fetch(
                symbol=symbol,
                timeframe=timeframe,
                count=incremental_count,
                last_bar_time=last_bar_time,
            )
            if incremental.bars:
                fetched_batches.append(
                    (incremental.bars, incremental.mode, incremental.attempts)
                )

        if not fetched_batches:
            fetch_result = self.fetcher.fetch_deep(symbol, timeframe, history_bars)
            fetched_batches.append((fetch_result.bars, fetch_result.mode, fetch_result.attempts))

        all_bars: list[BarRecord] = []
        mode = fetched_batches[-1][1]
        attempts = max(batch[2] for batch in fetched_batches)
        seen_times: set[int] = set()
        for batch_bars, batch_mode, _ in fetched_batches:
            for bar in batch_bars:
                if bar.time in seen_times:
                    continue
                seen_times.add(bar.time)
                all_bars.append(bar)
            if batch_mode == "incremental":
                mode = "incremental"
            elif mode != "incremental":
                mode = batch_mode

        for bar in all_bars:
            bar.source = self.data_source
            bar.ingested_at = ingested_at

        valid_bars, rejections = filter_valid_bars(all_bars, self.quality_cfg)
        rejected = len(all_bars) - len(valid_bars)

        if rejections:
            for note in rejections[:5]:
                logger.warning("Rejected bar: %s", note)
            if len(rejections) > 5:
                logger.warning("... and %d more rejected bars", len(rejections) - 5)

        stored = self.store.upsert_bars(valid_bars)
        quality = check_data_quality(
            valid_bars,
            resolved,
            timeframe,
            self.quality_cfg,
            rejected_count=rejected,
        )

        result = FetchResult(
            symbol=resolved,
            timeframe=timeframe,
            fetched=len(all_bars),
            stored=stored,
            rejected=rejected,
            mode=mode,
            quality=quality,
            rejection_notes=rejections,
        )
        final_count = self.store.count_bars(resolved, timeframe)
        logger.info(
            "Synced %s %s [%s]: fetched=%d stored=%d rejected=%d attempts=%d "
            "total_bars=%d target=%d valid=%s",
            resolved,
            timeframe,
            mode,
            result.fetched,
            result.stored,
            result.rejected,
            attempts,
            final_count,
            history_bars,
            quality.is_valid,
        )
        return result

    def audit_stored(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> list:
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        reports = []

        for symbol in symbols:
            try:
                resolved = self.connector.ensure_symbol(symbol)
            except Exception:  # noqa: BLE001
                resolved = symbol

            for timeframe in timeframes:
                bars = self.store.get_all_bars(resolved, timeframe)
                if not bars:
                    continue
                bar_records = [
                    BarRecord(
                        symbol=resolved,
                        timeframe=timeframe,
                        time=int(b["time"]),
                        open=float(b["open"]),
                        high=float(b["high"]),
                        low=float(b["low"]),
                        close=float(b["close"]),
                        tick_volume=int(b["tick_volume"]),
                        spread=int(b["spread"]),
                        real_volume=int(b["real_volume"]),
                        source=str(b.get("source", "mt5")),
                        ingested_at=int(b["ingested_at"]) if b.get("ingested_at") else None,
                    )
                    for b in bars
                ]
                reports.append(
                    check_data_quality(bar_records, resolved, timeframe, self.quality_cfg)
                )

        return reports
