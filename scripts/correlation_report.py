from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.data.store import OHLCVStore
from src.stats.returns import log_returns
from src.stats.risk import correlation


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_group_returns(
    store: OHLCVStore,
    symbols: list[str],
    timeframe: str,
    bars: int,
) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for symbol in symbols:
        data = store.get_recent_bars(symbol, timeframe, bars)
        if len(data) < 30:
            continue
        closes = [float(row["close"]) for row in data]
        rets = log_returns(closes)
        if len(rets) >= 20:
            series[symbol] = list(rets)
    return series


def print_group(title: str, series: dict[str, list[float]]) -> None:
    symbols = sorted(series.keys())
    if len(symbols) < 2:
        print(f"\n=== {title} ===")
        print("  insufficient symbols")
        return

    print(f"\n=== {title} ===")
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1 :]:
            corr = correlation(series[sym_a], series[sym_b])
            label = "high positive" if corr > 0.7 else "negative hedge" if corr < -0.3 else "low/mixed"
            print(f"  {sym_a:12} vs {sym_b:12}: {corr:+.3f} ({label})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Group correlation report (Lesson 03)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    timeframe = config.stats.analysis_timeframe
    bars = config.history_bars_for(timeframe)

    groups = config.symbol_groups
    indices = load_group_returns(store, groups.get("indices", []), timeframe, bars)
    commodities = load_group_returns(store, groups.get("commodities", []), timeframe, bars)
    forex = load_group_returns(store, groups.get("forex", []), timeframe, bars)

    print(f"\nCorrelation report ({timeframe}, log returns)")

    print_group("Indices", indices)
    print_group("Commodities", commodities)
    print_group("Forex", forex)

    print("\n=== Key Cross-Group Pairs ===")
    key_pairs = [
        ("#USSPX500", "GOLD"),
        ("#USSPX500", "EURUSD"),
        ("GOLD", "USDJPY"),
        ("#US30", "#USSPX500"),
    ]
    all_series = {**indices, **commodities, **forex}
    for a, b in key_pairs:
        if a in all_series and b in all_series:
            corr = correlation(all_series[a], all_series[b])
            print(f"  {a:12} vs {b:12}: {corr:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
