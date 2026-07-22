from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.research_agent import ResearchAgent
from src.core.config import load_config
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Technical feature report (Lesson 04)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override timeframe")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    agent = ResearchAgent(config, store)

    report = agent.analyze_symbol(args.symbol, args.timeframe)
    if report is None:
        print(f"ERROR: insufficient data for {args.symbol}")
        return 1

    print(f"\n=== Feature Report: {report.symbol} ({report.timeframe}) ===")
    print(f"  regime            : {report.regime.value}")
    print(f"  ann volatility    : {report.annualized_volatility:.2%}")
    print(f"  RSI               : {report.rsi:.2f}" if report.rsi else "  RSI               : n/a")
    print(f"  MACD DIFF         : {report.macd_diff:.6f}" if report.macd_diff else "  MACD DIFF         : n/a")
    print(f"  MACD Histogram    : {report.macd_histogram:.6f}" if report.macd_histogram else "  MACD Histogram    : n/a")
    print(f"  BB Position       : {report.bb_position:.3f}" if report.bb_position is not None else "  BB Position       : n/a")
    print(f"  ATR               : {report.atr:.6f}" if report.atr else "  ATR               : n/a")
    print(f"  ATR %             : {report.atr_pct:.4%}" if report.atr_pct else "  ATR %             : n/a")
    print(f"  MACD divergence   : {report.macd_divergence or 'none'}")
    print(f"  RSI divergence    : {report.rsi_divergence or 'none'}")
    print(f"  tail warning      : {report.tail_warning}")
    print("\n  Note: indicators are features, not standalone buy/sell rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
