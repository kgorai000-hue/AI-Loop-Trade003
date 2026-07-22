from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.data.lineage import SyncRunStore
from src.data.quality import DataQualityReport
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_report(report: DataQualityReport) -> None:
    status = "OK" if report.is_valid else "WARN"
    print(
        f"  [{status}] {report.symbol:12} {report.timeframe:4} "
        f"rows={report.total_rows:4d} gaps={report.gap_count:3d} "
        f"missing={report.missing_rate_pct:5.1f}% rejected={report.rejected_bars:3d}"
    )
    for anomaly in report.anomalies:
        print(f"         - {anomaly}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data quality audit for stored OHLCV (Lesson 06, MT5 source only)"
    )
    parser.add_argument("--symbol", action="append", help="Limit to specific symbols")
    parser.add_argument("--timeframe", action="append", help="Limit to specific timeframes")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    lineage = SyncRunStore(config.storage.path)

    symbols = args.symbol or config.symbols
    timeframes = args.timeframe or config.timeframes

    from src.agents.data_agent import DataAgent
    from src.core.mt5_connector import MT5Connector

    connector = MT5Connector(config)
    agent = DataAgent(config, connector, store)

    try:
        connector.connect()
        reports = agent.audit_stored(symbols, timeframes)
    except Exception:
        reports = []
        for symbol in symbols:
            for timeframe in timeframes:
                bars = store.get_all_bars(symbol, timeframe)
                if not bars:
                    continue
                from src.data.quality import check_data_quality
                from src.data.store import BarRecord

                records = [
                    BarRecord(
                        symbol=symbol,
                        timeframe=timeframe,
                        time=int(b["time"]),
                        open=float(b["open"]),
                        high=float(b["high"]),
                        low=float(b["low"]),
                        close=float(b["close"]),
                        tick_volume=int(b["tick_volume"]),
                        spread=int(b["spread"]),
                        real_volume=int(b["real_volume"]),
                    )
                    for b in bars
                ]
                reports.append(
                    check_data_quality(records, symbol, timeframe, config.data.quality)
                )
    finally:
        if connector.is_connected:
            connector.disconnect()

    print("\n=== Data Quality Report (Lesson 06) ===")
    print(f"  source: MT5 only ({config.data.source})")
    print(f"  database: {store.db_path}")
    print("  Note: index CFDs may show weekend gaps; survivorship bias N/A for MT5 CFDs.\n")

    if not reports:
        print("  No stored data found for requested symbols/timeframes.")
        return 1

    valid_count = sum(1 for r in reports if r.is_valid)
    for report in reports:
        print_report(report)

    print(f"\n=== Summary ===")
    print(f"  series checked : {len(reports)}")
    print(f"  valid          : {valid_count}")
    print(f"  warnings       : {len(reports) - valid_count}")

    recent_runs = lineage.get_recent_runs(3)
    if recent_runs:
        print(f"\n=== Recent Sync Runs (lineage) ===")
        for run in recent_runs:
            print(
                f"  run #{run['id']} [{run['status']}] "
                f"fetched={run['fetched_total']} stored={run['stored_total']} "
                f"rejected={run['rejected_total']} source={run['source']}"
            )

    return 0 if valid_count == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
