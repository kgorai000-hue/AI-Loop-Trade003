from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_agent import DataAgent
from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.lineage import SyncRunStore
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from MT5 into SQLite")
    parser.add_argument("--timeframe", action="append", help="Timeframe to fetch (repeatable)")
    parser.add_argument("--symbol", action="append", help="Symbol to fetch (repeatable)")
    parser.add_argument("--bars", type=int, help="Number of bars to fetch per symbol/timeframe")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)

    symbols = args.symbol or config.symbols
    timeframes = args.timeframe or config.timeframes
    history_bars = args.bars

    store = OHLCVStore(config.storage.path)
    lineage = SyncRunStore(config.storage.path)
    connector = MT5Connector(config)
    agent = DataAgent(config, connector, store)

    try:
        connector.connect()

        print_section("Fetch Start")
        print(f"  source: {config.data.source} (MT5 only, Lesson 06)")
        print(f"  symbols: {len(symbols)}")
        print(f"  timeframes: {', '.join(timeframes)}")
        if history_bars:
            print(f"  bars: {history_bars} (override, all timeframes)")
        else:
            print("  bars per timeframe:")
            for tf in timeframes:
                print(f"    {tf:4} -> {config.history_bars_for(tf)}")
        print(f"  database: {store.db_path}")

        summary = agent.sync_all(
            symbols=symbols,
            timeframes=timeframes,
            history_bars=history_bars,
        )

        print_section("Fetch Results")
        for result in summary.results:
            valid = "OK" if result.quality and result.quality.is_valid else "WARN"
            print(
                f"  {result.symbol:12} {result.timeframe:4} "
                f"[{result.mode:11}] fetched={result.fetched:4d} "
                f"stored={result.stored:4d} rejected={result.rejected:3d} [{valid}]"
            )
            if result.quality and result.quality.anomalies:
                for note in result.quality.anomalies[:2]:
                    print(f"      ! {note}")

        if summary.errors:
            print_section("Errors")
            for error in summary.errors:
                print(f"  {error}")

        print_section("Quality Summary")
        print(f"  sync run id     : {summary.run_id}")
        print(f"  rows rejected   : {summary.total_rejected}")
        warn_count = sum(1 for r in summary.quality_reports if not r.is_valid)
        print(f"  series warnings : {warn_count}/{len(summary.quality_reports)}")

        print_section("Database Summary")
        for row in store.get_summary():
            ingested = (
                row["last_ingested_at"].strftime("%Y-%m-%d %H:%M")
                if row["last_ingested_at"]
                else "n/a"
            )
            print(
                f"  {row['symbol']:12} {row['timeframe']:4} "
                f"bars={row['bars']:4d} "
                f"{row['first_time'].strftime('%Y-%m-%d %H:%M')} -> "
                f"{row['last_time'].strftime('%Y-%m-%d %H:%M')} UTC "
                f"(ingested {ingested} UTC)"
            )

        print_section("Total")
        print(f"  rows upserted this run: {summary.total_stored}")
        print(f"  total rows in database: {store.count_bars()}")

        if summary.run_id:
            runs = lineage.get_recent_runs(1)
            if runs:
                run = runs[0]
                print(f"  lineage status  : {run['status']} (broker={run.get('broker_server', 'n/a')})")

        return 1 if summary.errors else 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
