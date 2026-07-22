from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.market.symbol_info import fetch_market_symbol_info


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MT5 symbol_info (commission, spread, contract)")
    parser.add_argument("--symbol", action="append", help="Symbol to inspect")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    symbols = args.symbol or config.symbols

    connector = MT5Connector(config)
    try:
        connector.connect()
        print("\n=== Market Info (symbol_info) ===\n")

        commission_zero_count = 0
        for symbol in symbols:
            info = fetch_market_symbol_info(connector, config, symbol)
            if info.commission_is_zero:
                commission_zero_count += 1

            print(f"[{info.symbol}] market={info.market_type.value}")
            print(f"  bid/ask       : {info.bid} / {info.ask}")
            print(f"  spread        : {info.spread_points} pt ({info.spread_price})")
            print(f"  commission    : {info.commission} (mode={info.commission_mode})")
            print(f"  contract_size : {info.contract_size}")
            print(f"  tick_value    : {info.tick_value_profit} {info.currency_profit}")
            print(f"  volume        : min={info.volume_min} step={info.volume_step} max={info.volume_max}")
            print(f"  currencies    : base={info.currency_base} profit={info.currency_profit}")
            print()

        print("=== Commission Summary ===")
        print(f"  symbols checked     : {len(symbols)}")
        print(f"  commission == 0     : {commission_zero_count}/{len(symbols)}")
        if commission_zero_count == len(symbols):
            print("  result              : all symbols report zero commission via symbol_info")
        else:
            print("  result              : some symbols report non-zero commission - review above")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
