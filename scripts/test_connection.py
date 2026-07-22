from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import LOCAL_SETTINGS_PATH, load_config
from src.core.mt5_connector import MT5Connector


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="MT5 connection and symbol check")
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    try:
        config = load_config(settings_path=args.settings)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    setup_logging(config.log_level)

    if not LOCAL_SETTINGS_PATH.exists():
        print("WARNING: config/settings.local.yaml not found.")
        print("Copy config/settings.local.yaml.example and set PxPro demo credentials.")

    connector = MT5Connector(config)

    try:
        connector.connect()

        print_section("Health Check")
        health = connector.health_check()
        for key, value in health.items():
            print(f"  {key}: {value}")

        print_section("Account Info")
        account = connector.get_account_info()
        for key in ("login", "server", "balance", "equity", "margin_free", "currency", "trade_mode"):
            print(f"  {key}: {account.get(key)}")

        print_section("Symbol Check")
        statuses = connector.check_symbols(config.symbols)
        available_count = 0
        for status in statuses:
            if status.available:
                available_count += 1
                label = status.resolved if status.resolved != status.requested else status.requested
                print(
                    f"  OK  {label}: bid={status.bid} ask={status.ask} spread={status.spread_points}pt"
                )
            else:
                print(f"  NG  {status.requested}: {status.error or 'unavailable'}")

        print_section("Sample OHLCV")
        sample_symbol = next((s.resolved for s in statuses if s.available), None)
        if sample_symbol:
            rates = connector.get_rates(sample_symbol, config.default_timeframe, 3)
            for row in rates:
                print(
                    f"  {sample_symbol} {config.default_timeframe} "
                    f"time={row['time']} O={row['open']} H={row['high']} "
                    f"L={row['low']} C={row['close']} V={row['tick_volume']}"
                )
        else:
            print("  No available symbols to fetch sample rates.")

        print_section("Summary")
        print(f"  Symbols available: {available_count}/{len(statuses)}")
        if available_count == 0:
            print("  MT5 connected, but no configured symbols were found.")
            return 2
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI should report any connection failure
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
